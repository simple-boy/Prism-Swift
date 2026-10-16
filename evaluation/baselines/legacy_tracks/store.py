"""Atomic raw evidence + derived indexes; extends the unchanged Chronicle V2 engine."""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from typing import Any

from memory_core import MemoryStore, extract_structure, index_text, terms
from records import PayloadTooLarge


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class TrackStore(MemoryStore):
    def __init__(self, db_path: str, processor, *, temporal=True):
        if db_path == ":memory:":
            raise ValueError("Use a temporary SQLite file for tests; service requires durable storage")
        self.processor = processor
        features = {"entity", "options"} | ({"temporal"} if temporal else set())
        # Code imperatives are source evidence; the prose forget/supersession path is not used.
        super().__init__(db_path, features=features)
        conn = self._connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS raw_messages(
                source_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                request_id TEXT NOT NULL, message_index INTEGER NOT NULL, content_json TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS raw_scope ON raw_messages(user_id,request_id);
            CREATE TABLE IF NOT EXISTS source_records(
                memory_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, source_id TEXT NOT NULL,
                content_json TEXT NOT NULL, features_json TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS record_scope ON source_records(user_id,memory_id);
            CREATE TABLE IF NOT EXISTS model_usage(
                user_id TEXT NOT NULL, request_id TEXT NOT NULL, usage_json TEXT NOT NULL,
                PRIMARY KEY(user_id,request_id));
        """)
        config = {"track": processor.name, "processor": processor.config_dict(), "temporal": temporal}
        model = getattr(processor, "model", None)
        if model is not None and hasattr(model, "config_dict"):
            config["model"] = model.config_dict()
        encoded = canonical(config)
        row = conn.execute("SELECT value FROM memory_meta WHERE key='track_config'").fetchone()
        if row and row["value"] != encoded:
            self.close()
            raise ValueError("Database is bound to a different track/config; use a separate database")
        conn.execute("INSERT OR IGNORE INTO memory_meta(key,value) VALUES('track_config',?)", (encoded,))
        conn.commit()

    def add_messages(self, *, request_id, user_id, session_id, messages, on_progress=None):
        def stage(name):
            if on_progress is not None:
                try:
                    on_progress(name)
                except Exception:
                    pass  # Diagnostics cannot change transaction success or idempotency.

        for value in (request_id, user_id, session_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("request_id, user_id, session_id must be nonempty strings")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty array")
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") not in {"user", "assistant"}:
                raise ValueError("message role must be user or assistant")
            if self.processor.name == "coding" and (not isinstance(msg.get("content"), str) or not msg["content"].strip()):
                raise ValueError("Coding content must be a nonempty string")
            stamp = msg.get("timestamp")
            if stamp is not None:
                if isinstance(stamp, bool) or not isinstance(stamp, int):
                    raise ValueError("timestamp must be Unix milliseconds")
                try:
                    datetime.fromtimestamp(stamp / 1000, tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    raise ValueError("timestamp outside supported datetime range") from None
        digest = hashlib.sha256(canonical(messages).encode()).hexdigest()
        # Check the receipt before calling a model: committed retries have zero model cost.
        stage("waiting_for_write_lock")
        with self._lock:
            stage("checking_receipt")
            conn = self._connection()
            receipt = conn.execute("SELECT session_id,payload_hash FROM add_requests WHERE user_id=? AND request_id=?",
                                   (user_id, request_id)).fetchone()
            if receipt:
                if receipt["session_id"] != session_id or receipt["payload_hash"] != digest:
                    raise ValueError("request_id was reused with a different payload")
                stage("deduplicated")
                return
            media = sum(self.processor.media_bytes(msg.get("content")) for msg in messages)
            if media > 30 * 1024 * 1024:
                raise PayloadTooLarge("Decoded media exceeds 30 MiB per Add")
            model = getattr(self.processor, "model", None)
            if model is not None and getattr(model, "enabled", True) and hasattr(model, "max_calls_per_add"):
                required = sum(self.processor.planned_calls(msg.get("content")) for msg in messages)
                if required > model.max_calls_per_add:
                    raise PayloadTooLarge("Add needs more index calls than the declared service capacity")
            if model is not None and hasattr(model, "start_add"):
                model.start_add()
            stage("preparing_index")
            prepared = [self.processor.prepare(msg.get("content")) for msg in messages]
            if any(not records for records in prepared):
                raise ValueError("Each message must produce nonempty evidence")
            stage("writing_database")
            conn.execute("BEGIN IMMEDIATE")
            try:
                seq = conn.execute("SELECT COALESCE(MAX(sequence_no),0) FROM memories WHERE user_id=? AND session_id=?",
                                   (user_id, session_id)).fetchone()[0]
                created = time.time()
                for mi, (msg, records) in enumerate(zip(messages, prepared)):
                    source_id = self._message_id(user_id, request_id, mi)
                    conn.execute("INSERT INTO raw_messages VALUES(?,?,?,?,?,?)",
                                 (source_id, user_id, session_id, request_id, mi, canonical(msg)))
                    for ri, record in enumerate(records):
                        memory_id = source_id + f"_{ri}"
                        if not isinstance(record.index_text, str) or not record.index_text.strip():
                            raise ValueError("Processor produced an empty index")
                        structure = extract_structure(record.index_text, msg.get("timestamp"))
                        event = structure["event"]
                        seq += 1
                        conn.execute("INSERT INTO memories(memory_id,user_id,session_id,request_id,content,created_at,"
                            "token_json,model_term_json,structure_json,event_type,event_time,event_time_key,"
                            "sequence_no,role,timestamp_ms,message_index) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (memory_id, user_id, session_id, f"{request_id}:msg:{mi}:chunk:{ri}", record.index_text,
                             created, canonical(terms(record.index_text)), "[]", canonical(structure),
                             structure.get("memory_type", "fact"), event.get("event_time"), event.get("event_time_key"),
                             seq, msg["role"], msg.get("timestamp"), mi))
                        self._replace_structure_indexes(conn, memory_id, user_id, structure)
                        if self._fts_enabled:
                            conn.execute("INSERT INTO memories_fts(memory_id,user_id,search_text) VALUES(?,?,?)",
                                         (memory_id, user_id, index_text(record.index_text)))
                        conn.execute("INSERT INTO source_records VALUES(?,?,?,?,?)",
                                     (memory_id, user_id, source_id, canonical(record.content), canonical(record.features)))
                conn.execute("INSERT INTO add_requests VALUES(?,?,?,?)", (user_id, request_id, session_id, digest))
                usage = model.usage() if model is not None and hasattr(model, "usage") else {"calls": 0}
                conn.execute("INSERT INTO model_usage VALUES(?,?,?)", (user_id, request_id, canonical(usage)))
                stage("committing")
                conn.commit()
                stage("committed")
            except Exception:
                conn.rollback()
                raise

    def search(self, *, user_id, query, top_k=100, options=None):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id is required")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        if options is not None and (not isinstance(options, list) or any(not isinstance(x, str) for x in options)):
            raise ValueError("options must be an array of strings")
        prepared = self.processor.query(query, options)
        base = super().search(user_id=user_id, query=prepared.text, top_k=100, include_source=False)
        candidate_rank = {row["id"]: rank for rank, row in enumerate(base)}
        conn = self._connection()
        # Domain ranking sees a wider lexical pool than the final Top-100 output.
        if self._fts_enabled:
            match = self._fts_query(prepared.text)
            if match:
                rows = conn.execute("SELECT m.memory_id FROM memories_fts f JOIN memories m ON m.memory_id=f.memory_id "
                    "WHERE f.user_id=? AND memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT 800",
                    (user_id, match)).fetchall()
                for rank, row in enumerate(rows):
                    candidate_rank.setdefault(row["memory_id"], 100 + rank)
        scores = {}
        # Only feature metadata is read for visual ranking; raw images are loaded for selected candidates.
        visual_enabled = (self.processor.name == "multimodal"
                          and self.processor.image_features and prepared.features.get("images"))
        if visual_enabled:
            rows = conn.execute("SELECT memory_id,features_json FROM source_records WHERE user_id=?", (user_id,))
            query_hashes = {image.get("sha256") for image in prepared.features["images"]}
            visual = []
            for row in rows:
                feature = json.loads(row["features_json"])
                similarity = self.processor.score(prepared.features, feature)
                exact = bool(query_hashes & {image.get("sha256") for image in feature.get("images", [])})
                visual.append((100 * similarity + int(exact), row["memory_id"]))
            for score, mid in sorted(visual, reverse=True)[:1000]:
                if score > 0:
                    scores[mid] = score
        for mid, rank in candidate_rank.items():
            row = conn.execute("SELECT features_json FROM source_records WHERE user_id=? AND memory_id=?", (user_id, mid)).fetchone()
            if row:
                if visual_enabled:
                    # Visual candidates are compared uniformly, independent of their lexical rank.
                    scores[mid] = scores.get(mid, 0) + 0.1 / (rank + 1)
                else:
                    scores[mid] = 1 / (rank + 1) + self.processor.score(prepared.features, json.loads(row[0]))
        output, media = [], 0
        for mid, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
            if not math.isfinite(score):
                continue
            row = conn.execute("SELECT r.content_json,m.session_id,m.role,m.sequence_no,m.timestamp_ms,m.created_at "
                               "FROM source_records r JOIN memories m ON m.memory_id=r.memory_id "
                               "WHERE r.user_id=? AND r.memory_id=?", (user_id, mid)).fetchone()
            if not row:
                continue
            content = json.loads(row["content_json"])
            size = self.processor.media_bytes(content)
            if media + size > 30 * 1024 * 1024:
                break  # Whole ranked prefix; no image or part is altered/dropped within an evidence record.
            media += size
            provenance = f"[role={row['role']}; session={row['session_id']}; order={row['sequence_no']}"
            if row["timestamp_ms"] is not None:
                provenance += f"; source_time={datetime.fromtimestamp(row['timestamp_ms']/1000, timezone.utc).isoformat()}"
            provenance += "]"
            if isinstance(content, str) and self.processor.name == "coding":
                content = provenance + "\n" + content
            # Multimodal parts remain byte-for-byte source objects, with no derived caption inserted.
            output.append({"id": mid, "content": content, "score": score,
                           "created_at": datetime.fromtimestamp(row["created_at"], timezone.utc).isoformat()})
            if len(output) >= top_k:
                break
        return output

    def delete_user(self, user_id):
        """Administrative whole-run/user purge; never exposed as an evaluation endpoint."""
        with self._lock:
            conn = self._connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                for table in ("memory_entities", "memory_relations", "source_records", "raw_messages", "add_requests", "model_usage", "memories"):
                    conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
                if self._fts_enabled:
                    conn.execute("DELETE FROM memories_fts WHERE user_id=?", (user_id,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            conn.execute("VACUUM")
            checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint and checkpoint[0]:
                raise RuntimeError("WAL checkpoint busy: retry purge when readers are idle")
