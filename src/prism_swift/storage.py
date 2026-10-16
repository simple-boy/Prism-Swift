"""One atomic SQLite transaction for raw sources, scoped binary objects and indexes.

WAL + synchronous=FULL is deliberate. HTTP success must survive process restart.
The database is the only authority: there is no asynchronous indexing queue.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import sqlite3
import threading
import time

from .validation import ContractError, canonical, identifier, parse_content, restore, timestamp
from .retrieval import fts_query, query_plan, rank_evidence, tokens


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS receipts(
 user_id TEXT NOT NULL, request_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
 response TEXT NOT NULL, PRIMARY KEY(user_id,request_id));
CREATE TABLE IF NOT EXISTS scopes(user_id TEXT PRIMARY KEY, sequence INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS messages(
 row_id INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, user_id TEXT NOT NULL,
 session_id TEXT NOT NULL, request_id TEXT NOT NULL, sequence INTEGER NOT NULL,
 message_index INTEGER NOT NULL, role TEXT NOT NULL, timestamp INTEGER,
 created_at TEXT NOT NULL, source_json TEXT NOT NULL, text TEXT NOT NULL,
 tokens_json TEXT NOT NULL, media_bytes INTEGER NOT NULL, wire_bytes INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS message_scope ON messages(user_id,sequence);
CREATE INDEX IF NOT EXISTS message_time ON messages(user_id,timestamp);
CREATE INDEX IF NOT EXISTS message_session ON messages(user_id,session_id,sequence);
CREATE TABLE IF NOT EXISTS blobs(
 user_id TEXT NOT NULL, digest TEXT NOT NULL, data BLOB NOT NULL,
 width INTEGER NOT NULL, height INTEGER NOT NULL, features_json TEXT NOT NULL,
 r REAL, g REAL, b REAL, PRIMARY KEY(user_id,digest));
CREATE TABLE IF NOT EXISTS images(
 user_id TEXT NOT NULL, message_id TEXT NOT NULL, position INTEGER NOT NULL, digest TEXT NOT NULL,
 PRIMARY KEY(message_id,position),
 FOREIGN KEY(user_id,digest) REFERENCES blobs(user_id,digest),
 FOREIGN KEY(message_id) REFERENCES messages(id));
CREATE INDEX IF NOT EXISTS image_scope ON images(user_id,digest,message_id);
CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(
 message_id UNINDEXED, user_id UNINDEXED, text, tokenize='unicode61');
"""


class MemoryStore:
    def __init__(self, db_path, *, strategy="adaptive", max_pixels=16_000_000,
                 max_image_bytes=10*1024**2, max_media_bytes=30*1024**2,
                 response_bytes=64*1024**2, max_results=100, min_free_bytes=0, ablation="none", reranker=None):
        if strategy not in {"raw", "sparse", "adaptive"}:
            raise ValueError("Unknown strategy")
        if ablation not in {"none", "no_visual", "no_neighbours", "no_coverage"}:
            raise ValueError("Unknown ablation")
        if str(db_path) == ":memory:":
            raise ValueError("Durable database path required")
        if min(max_pixels, max_image_bytes, max_media_bytes, response_bytes, max_results) < 1:
            raise ValueError("Capacity settings must be positive")
        if max_image_bytes > 10*1024**2 or max_media_bytes > 30*1024**2 or max_results > 100:
            raise ValueError("Capacity exceeds AML image/top_k limits")
        self.db_path = str(Path(db_path).resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.strategy, self.max_pixels = strategy, max_pixels
        self.ablation = ablation
        self.reranker = reranker
        self.max_image_bytes, self.max_media_bytes = max_image_bytes, max_media_bytes
        self.response_bytes, self.max_results = response_bytes, max_results
        self.min_free_bytes = min_free_bytes
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {"add_requests": 0, "deduplicated_requests": 0, "messages": 0,
                       "source_bytes": 0, "unique_blob_bytes": 0, "image_occurrences": 0,
                       "parse_seconds": 0., "commit_seconds": 0., "lock_wait_seconds": 0.,
                       "local_rerank_calls": 0, "local_rerank_seconds": 0.}
        conn = self._connection()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        frozen = canonical({"schema": 1, "strategy": strategy, "ablation": ablation,
                            "reranker": reranker.config_dict() if reranker is not None else None,
                            "visual": "dhash64-rgb8" if strategy == "adaptive" else "none",
                            "tokenizer": "unicode-word-cjk-bigram-v1", "query": "coverage-neighbour-v1"})
        old = conn.execute("SELECT value FROM meta WHERE key='configuration'").fetchone()
        if old and old[0] != frozen:
            self.close()
            raise ValueError("Database configuration mismatch; use an independent database")
        conn.execute("INSERT OR IGNORE INTO meta VALUES('configuration',?)", (frozen,))
        self.configuration_hash = hashlib.sha256(frozen.encode()).hexdigest()
        # Keep the factory-thread connection alive for the service lifetime.
        # Otherwise closing the last request connection checkpoints/removes WAL
        # after every request, multiplying FULL fsync and startup costs.
        # Request-thread close() still releases its own connection; application
        # shutdown closes this initial connection on the factory thread.

    def _connection(self):
        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.connection = conn
        return self._local.connection

    def close(self):
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            del self._local.connection

    def health(self):
        conn = self._connection()
        conn.execute("SELECT 1").fetchone()
        return {"status": "ok", "service": "prism-swift", "version": "0.1.0",
                "strategy": self.strategy, "configuration_sha256": self.configuration_hash,
                "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0],
                "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0], "model_calls": 0,
                "local_reranker": self.reranker is not None}

    def metrics(self):
        with self._stats_lock:
            result = self._stats.copy()
        result.update(database_bytes=sum(p.stat().st_size for p in [Path(self.db_path), Path(self.db_path+"-wal")]
                                         if p.exists()), disk_free_bytes=shutil.disk_usage(Path(self.db_path).parent).free,
                      model_calls=0)
        return result

    def _insert_message(self, conn, values):
        return conn.execute("INSERT INTO messages(id,user_id,session_id,request_id,sequence,message_index,role,timestamp,created_at,"
                            "source_json,text,tokens_json,media_bytes,wire_bytes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)

    def add(self, body):
        if not isinstance(body, dict):
            raise ContractError("json_object_required")
        user = identifier(body.get("user_id"), "user_id")
        request = identifier(body.get("request_id"), "request_id")
        session = identifier(body.get("session_id"), "session_id")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ContractError("messages_must_be_nonempty_array")
        digest = hashlib.sha256(canonical({"session": session, "messages": messages}).encode()).hexdigest()
        conn = self._connection()
        old = conn.execute("SELECT payload_hash,response FROM receipts WHERE user_id=? AND request_id=?", (user, request)).fetchone()
        if old:
            if old[0] != digest:
                raise ContractError("request_id_reused_with_changed_payload", 409)
            with self._stats_lock:
                self._stats["deduplicated_requests"] += 1
            return json.loads(old[1])
        if shutil.disk_usage(Path(self.db_path).parent).free < self.min_free_bytes:
            raise ContractError("insufficient_storage_capacity", 507)
        parse_start = time.perf_counter()
        cache, prepared, media, source_size, occurrences = {}, [], 0, 0, 0
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or not isinstance(message.get("role"), str) or message["role"] not in {"user", "assistant"}:
                raise ContractError("role_must_be_user_or_assistant")
            if "timestamp" in message and message["timestamp"] is None:
                raise ContractError("timestamp_must_be_unix_milliseconds")
            stamp = timestamp(message.get("timestamp"))
            template, text, images = parse_content(message.get("content"), max_image_bytes=self.max_image_bytes,
                            max_pixels=self.max_pixels, features=self.strategy == "adaptive", cache=cache)
            size = sum(len(image.data) for image in images)
            media += size
            if media > self.max_media_bytes:
                raise ContractError("add_media_exceeds_declared_capacity", 413)
            wire_size = len(canonical(message["content"]).encode())
            if wire_size + 4096 > self.response_bytes:
                raise ContractError("single_source_exceeds_response_capacity", 413)
            source_size += wire_size
            occurrences += len(images)
            envelope = {**message, "content": template}
            mid = "swift_" + hashlib.sha256(canonical([user, request, index]).encode()).hexdigest()[:32]
            prepared.append((mid, index, stamp, envelope, text, images, size, wire_size))
        parse_seconds = time.perf_counter() - parse_start
        waiting = time.perf_counter()
        response = {"success": True, "request_id": request, "user_id": user, "session_id": session}
        with self._write_lock:
            lock_wait = time.perf_counter() - waiting
            commit_start = time.perf_counter()
            conn.execute("BEGIN IMMEDIATE")
            unique_bytes = 0
            try:
                # Another handler/process may have committed this request while validation ran.
                old = conn.execute("SELECT payload_hash,response FROM receipts WHERE user_id=? AND request_id=?", (user, request)).fetchone()
                if old:
                    if old[0] != digest:
                        raise ContractError("request_id_reused_with_changed_payload", 409)
                    conn.rollback()
                    with self._stats_lock:
                        self._stats["deduplicated_requests"] += 1
                    return json.loads(old[1])
                row = conn.execute("SELECT sequence FROM scopes WHERE user_id=?", (user,)).fetchone()
                sequence = row[0] if row else 0
                created = datetime.now(timezone.utc).isoformat()
                for mid, index, stamp, envelope, text, images, size, wire_size in prepared:
                    sequence += 1
                    self._insert_message(conn,
                        (mid,user,session,request,sequence,index,envelope["role"],stamp,created,canonical(envelope),text,
                         canonical(tokens(text)),size,wire_size))
                    for position, image in enumerate(images):
                        mean = image.features.get("mean", [None, None, None])
                        cursor = conn.execute("INSERT OR IGNORE INTO blobs VALUES(?,?,?,?,?,?,?,?,?)",
                            (user,image.digest,image.data,image.width,image.height,canonical(image.features),*mean))
                        if cursor.rowcount:
                            unique_bytes += len(image.data)
                        conn.execute("INSERT INTO images VALUES(?,?,?,?)", (user,mid,position,image.digest))
                    if self.strategy != "raw":
                        conn.execute("INSERT INTO text_fts(message_id,user_id,text) VALUES(?,?,?)",
                                     (mid,user," ".join(tokens(text))))
                conn.execute("INSERT INTO scopes VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET sequence=excluded.sequence", (user,sequence))
                conn.execute("INSERT INTO receipts VALUES(?,?,?,?)", (user,request,digest,canonical(response)))
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        with self._stats_lock:
            for key,value in {"add_requests":1,"messages":len(messages),"source_bytes":source_size,
                              "image_occurrences":occurrences,"unique_blob_bytes":unique_bytes,
                              "parse_seconds":parse_seconds,"commit_seconds":time.perf_counter()-commit_start,
                              "lock_wait_seconds":lock_wait}.items():
                self._stats[key] += value
        return response

    def _sparse(self, conn, user, words, limit):
        match = fts_query(words)
        if not match:
            return []
        # FTS5 bm25() uses collection-wide statistics even after a user_id filter.
        # It is deliberately NOT used: another user's writes must not change a ranking.
        rows = conn.execute("SELECT m.id,m.tokens_json,m.sequence FROM text_fts f JOIN messages m ON m.id=f.message_id "
                            "WHERE text_fts MATCH ? AND f.user_id=? AND m.user_id=?", (match,user,user)).fetchall()
        if not rows:
            return []
        count,average = conn.execute("SELECT count(*),avg(json_array_length(tokens_json)) FROM messages WHERE user_id=?", (user,)).fetchone()
        parsed = [(row[0],set(json.loads(row[1])),row[2]) for row in rows]
        terms = set(words[:128])
        frequency = {word:sum(word in vocabulary for _,vocabulary,_ in parsed) for word in terms}
        idf = {word:math.log(1+(count-frequency[word]+.5)/(frequency[word]+.5)) for word in terms}
        def score(row):
            denominator = 1+1.2*(.25+.75*len(row[1])/max(1,average))
            return sum(idf[word]*2.2/denominator for word in terms.intersection(row[1]))
        return [row[0] for row in sorted(parsed,key=lambda row:(-score(row),row[2],row[0]))[:limit]]

    def search(self, body):
        if not isinstance(body, dict):
            raise ContractError("json_object_required")
        user = identifier(body.get("user_id"), "user_id")
        top_k = body.get("top_k")
        if isinstance(top_k,bool) or not isinstance(top_k,int) or not 1 <= top_k <= self.max_results:
            raise ContractError("top_k_must_be_integer_1_to_100")
        options = body.get("options")
        if ("options" in body and not isinstance(options,list)) or (options is not None and any(not isinstance(x,str) or not x.strip() for x in options)):
            raise ContractError("options_must_be_nonempty_strings")
        _, text, images = parse_content(body.get("query"), max_image_bytes=self.max_image_bytes,
                                       max_pixels=self.max_pixels, features=self.strategy == "adaptive")
        if sum(len(image.data) for image in images) > self.max_media_bytes:
            raise ContractError("query_media_exceeds_declared_capacity", 413)
        query_images = [{"digest":image.digest, "features":image.features} for image in images]
        plan = query_plan(text, options, query_images)
        if self.ablation == "no_visual":
            plan["images"], plan["colors"] = [], []
        if self.ablation == "no_neighbours":
            plan["before"], plan["after"] = False, False
        conn = self._connection()
        # Use a consistent read snapshot across candidate selection, source links and blobs.
        conn.execute("BEGIN")
        try:
            sparse = self._sparse(conn,user,plan["words"],800) if self.strategy != "raw" else []
            option_ids = self._sparse(conn,user,plan["options"],200) if self.strategy != "raw" else []
            ids = set(sparse+option_ids)
            if self.strategy == "adaptive":
                for image in plan["images"]:
                    ids.update(row[0] for row in conn.execute("SELECT message_id FROM images WHERE user_id=? AND digest=? LIMIT 200",
                                                             (user,image["digest"])))
                targets = plan["colors"] + [image["features"]["mean"] for image in plan["images"] if "mean" in image["features"]]
                for red,green,blue in targets[:8]:
                    rows = conn.execute("SELECT i.message_id FROM blobs b JOIN images i ON i.user_id=b.user_id AND i.digest=b.digest "
                        "WHERE b.user_id=? AND b.r IS NOT NULL ORDER BY (b.r-?)*(b.r-?)+(b.g-?)*(b.g-?)+(b.b-?)*(b.b-?) LIMIT 200",
                        (user,red,red,green,green,blue,blue)).fetchall()
                    ids.update(row[0] for row in rows)
                for start,end in plan["dates"]:
                    ids.update(row[0] for row in conn.execute("SELECT id FROM messages WHERE user_id=? AND timestamp>=? AND timestamp<? LIMIT 200", (user,start,end)))
            if not ids:
                ids.update(row[0] for row in conn.execute("SELECT id FROM messages WHERE user_id=? ORDER BY sequence DESC LIMIT 200", (user,)))
            if self.strategy == "adaptive" and self.ablation != "no_neighbours" and (plan["before"] or plan["after"] or plan["multi"]):
                # Neighbours are fetched by source order, without a full per-user scan.
                for mid in sparse[:3]:
                    anchor = conn.execute("SELECT session_id,sequence,timestamp FROM messages WHERE user_id=? AND id=?", (user,mid)).fetchone()
                    if anchor:
                        for operator,order in (("<","DESC"),(">","ASC")):
                            if anchor[2] is not None:
                                rows = conn.execute(f"SELECT id FROM messages WHERE user_id=? AND session_id=? AND timestamp IS NOT NULL "
                                    f"AND (timestamp,sequence){operator}(?,?) ORDER BY timestamp {order},sequence {order} LIMIT 2", (user,anchor[0],anchor[2],anchor[1]))
                            else:
                                rows = conn.execute(f"SELECT id FROM messages WHERE user_id=? AND session_id=? AND sequence{operator}? ORDER BY sequence {order} LIMIT 2", (user,anchor[0],anchor[1]))
                            ids.update(row[0] for row in rows)
            names = sorted(ids)
            marks = ",".join("?" for _ in names)
            raw_rows = conn.execute(f"SELECT id,session_id,sequence,timestamp,tokens_json,media_bytes,wire_bytes,created_at FROM messages WHERE user_id=? AND id IN ({marks})", [user,*names]).fetchall()
            visual = {}
            if names and self.strategy == "adaptive":
                for row in conn.execute(f"SELECT i.message_id,b.digest,b.features_json FROM images i JOIN blobs b ON i.user_id=b.user_id AND i.digest=b.digest WHERE i.user_id=? AND i.message_id IN ({marks})", [user,*names]):
                    visual.setdefault(row[0],[]).append({"digest":row[1],"features":json.loads(row[2])})
            rows = [{"id":row["id"],"session":row["session_id"],"sequence":row["sequence"],"timestamp":row["timestamp"],
                     "tokens":json.loads(row["tokens_json"]),"images":visual.get(row["id"],[]),
                     "media_bytes":row["media_bytes"],"wire_bytes":row["wire_bytes"],"created_at":row["created_at"]} for row in raw_rows]
            visual_query = bool(images) or bool(plan["colors"]) or bool(re.search(
                r"\b(images?|photo(?:graph)?s?|pictures?|visual|look|colou?rs?|shapes?|objects?|wear|scene|shown|see)\b|图片|颜色|形状|穿",text,re.I))
            use_reranker = self.reranker is not None and visual_query
            candidate_limit = max(top_k,self.reranker.candidates) if use_reranker else top_k
            ranked = rank_evidence(rows,sparse,option_ids,plan,strategy=self.strategy,top_k=candidate_limit,ablation=self.ablation)
            if use_reranker:
                candidates, candidate_bytes = [],0
                for row,_ in ranked[:self.reranker.candidates]:
                    refs = conn.execute("SELECT b.data FROM images i JOIN blobs b ON i.user_id=b.user_id AND i.digest=b.digest "
                                        "WHERE i.user_id=? AND i.message_id=? ORDER BY i.position",(user,row["id"])).fetchall()
                    size = sum(len(ref[0]) for ref in refs)
                    if candidate_bytes+size > 30*1024**2:
                        break
                    if refs:
                        candidates.append({"id":row["id"],"image_bytes":[ref[0] for ref in refs]})
                        candidate_bytes += size
                rerank_start = time.perf_counter()
                scores = self.reranker.scores(text,[image.data for image in images],candidates)
                allowed = {row["id"] for row in candidates}
                if not isinstance(scores,dict) or any(mid not in allowed or isinstance(value,bool) or
                        not isinstance(value,(int,float)) or not math.isfinite(value) for mid,value in scores.items()):
                    raise ContractError("invalid_local_reranker_output",503)
                if scores:
                    scored = sorted((pair for pair in ranked if pair[0]["id"] in scores),
                                    key=lambda pair:(-scores[pair[0]["id"]],pair[0]["sequence"]))
                    rest = [pair for pair in ranked if pair[0]["id"] not in scores]
                    ranked = [(row,1/(index+1)) for index,(row,_) in enumerate(scored+rest)]
                with self._stats_lock:
                    self._stats["local_rerank_calls"] += 1
                    self._stats["local_rerank_seconds"] += time.perf_counter()-rerank_start
            ranked = ranked[:top_k]
            output,media,wire = [],0,12
            for row,score in ranked:
                if media+row["media_bytes"] > self.max_media_bytes or wire+row["wire_bytes"]+1024 > self.response_bytes:
                    break  # Preserve a whole ranked prefix; never remove a part from an evidence item.
                source = json.loads(conn.execute("SELECT source_json FROM messages WHERE user_id=? AND id=?", (user,row["id"])).fetchone()[0])
                def get_blob(digest):
                    value = conn.execute("SELECT data FROM blobs WHERE user_id=? AND digest=?", (user,digest)).fetchone()
                    if value is None:
                        raise sqlite3.DatabaseError("Missing scoped source object")
                    return value[0]
                content = restore(source["content"], get_blob)
                created_at = (datetime.fromtimestamp(row["timestamp"]/1000,timezone.utc).isoformat()
                              if row["timestamp"] is not None else row["created_at"])
                item = {"id":row["id"],"content":content,"score":score,"created_at":created_at}
                item_size = len(canonical(item).encode()) + 1
                if wire+item_size > self.response_bytes:
                    break
                output.append(item)
                media += row["media_bytes"]
                wire += item_size
            conn.commit()
            return {"data":output}
        except BaseException:
            conn.rollback()
            raise

    def backup(self, target):
        """Consistent operator-invoked backup; no evaluation content is printed."""
        path = Path(target).resolve()
        if str(path) == self.db_path or path.exists():
            raise ValueError("Choose a new independent backup path")
        with sqlite3.connect(path) as destination:
            self._connection().backup(destination)
        return path
