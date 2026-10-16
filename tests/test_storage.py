import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from prism_swift.storage import MemoryStore
from prism_swift.validation import ContractError, decode_image


def picture(color="red", size=(64,64), format="PNG"):
    output = io.BytesIO()
    with Image.new("RGB",size,color) as image:
        image.save(output,format)
    mime = {"PNG":"png","JPEG":"jpeg","WEBP":"webp"}[format]
    return "data:image/"+mime+";base64,"+base64.b64encode(output.getvalue()).decode()


def body(content="Alice gave Bob a book.", request="r", user="u", session="s", **extra):
    return {"request_id":request,"user_id":user,"session_id":session,
            "messages":[{"role":"user","content":content,**extra}]}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = str(Path(self.folder.name)/"memory.sqlite3")
        self.store = MemoryStore(self.path)
        self.addCleanup(self.store.close)

    def search(self, query, **extra):
        return self.store.search({"user_id":"u","query":query,"top_k":100,**extra})["data"]

    def test_ordered_original_evidence_and_user_scoped_binary_deduplication(self):
        uri = picture()
        parts = [{"type":"text","text":"Left caption Alice."},
                 {"type":"image_url","image_url":{"url":uri,"detail":"high"}},
                 {"type":"text","text":"Right caption Bob."},
                 {"type":"image_url","image_url":{"url":uri}}]
        self.store.add(body(parts))
        self.assertEqual(self.search("Alice")[0]["content"],parts)
        self.store.add(body(parts,request="r2"))
        self.assertEqual(self.store._connection().execute("SELECT count(*) FROM blobs").fetchone()[0],1)
        self.store.add(body(parts,user="other"))
        self.assertEqual(self.store._connection().execute("SELECT count(*) FROM blobs").fetchone()[0],2)
        self.assertEqual(len(self.search("Alice")),2)

    def test_committed_retry_has_stable_id_and_conflicting_reuse_is_rejected(self):
        payload=body()
        ack=self.store.add(payload)
        before=self.search("book")
        self.assertEqual(self.store.add(payload),ack)
        self.assertEqual(self.search("book"),before)
        for changed in (body("new"),body(session="new")):
            with self.assertRaises(ContractError) as error:
                self.store.add(changed)
            self.assertEqual(error.exception.status,409)

    def test_validate_whole_batch_before_writing(self):
        payload=body()
        payload["messages"].append({"role":"user","content":[{"type":"image_url","image_url":{"url":"https://invalid.invalid/private.png"}}]})
        with self.assertRaises(ContractError):
            self.store.add(payload)
        self.assertEqual(self.search("book"),[])
        self.assertEqual(self.store._connection().execute("SELECT count(*) FROM receipts").fetchone()[0],0)

    def test_sql_failure_rolls_back_raw_blob_fts_and_receipt(self):
        original=self.store._insert_message
        calls=0
        def fail_second(conn,values):
            nonlocal calls
            calls+=1
            if calls==2:
                raise sqlite3.OperationalError("synthetic failure")
            return original(conn,values)
        payload=body([{"type":"image_url","image_url":{"url":picture()}}])
        payload["messages"].append({"role":"assistant","content":"second"})
        with patch.object(self.store,"_insert_message",side_effect=fail_second):
            with self.assertRaises(sqlite3.OperationalError):
                self.store.add(payload)
        for table in ("messages","blobs","images","receipts","text_fts"):
            self.assertEqual(self.store._connection().execute("SELECT count(*) FROM "+table).fetchone()[0],0)
        self.assertTrue(self.store.add(payload)["success"])

    def test_process_crash_mid_transaction_cannot_leave_partial_sources(self):
        self.store.close()
        code="""import os,sys
from prism_swift.storage import MemoryStore
s=MemoryStore(sys.argv[1])
original=s._insert_message
def fail(conn,values):
    if values[6]=='assistant': os._exit(17)
    return original(conn,values)
s._insert_message=fail
s.add({'user_id':'u','request_id':'crash','session_id':'s','messages':[
 {'role':'user','content':'first'}, {'role':'assistant','content':'second'}]})
"""
        result=subprocess.run([sys.executable,"-B","-c",code,self.path],capture_output=True)
        self.assertEqual(result.returncode,17,result.stderr)
        self.assertEqual(self.store._connection().execute("SELECT count(*) FROM messages").fetchone()[0],0)
        self.assertEqual(self.store._connection().execute("PRAGMA integrity_check").fetchone()[0],"ok")

    def test_acknowledged_data_survives_process_exit_without_graceful_close(self):
        self.store.close()
        code="""import os,sys,json
from prism_swift.storage import MemoryStore
s=MemoryStore(sys.argv[1]); print(json.dumps(s.add(json.loads(sys.argv[2]))),flush=True); os._exit(0)
"""
        result=subprocess.run([sys.executable,"-B","-c",code,self.path,json.dumps(body())],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)["success"])
        self.assertIn("Bob",self.search("book")[0]["content"])
        health=self.store.health()
        self.assertEqual((health["synchronous"],health["journal_mode"]),(2,"wal"))

    def test_concurrent_committed_retry_is_single_record(self):
        def write(_):
            try:
                return self.store.add(body())
            finally:
                self.store.close()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(write,range(24)))
        self.assertTrue(all(result["success"] for result in results))
        self.assertEqual(len(self.search("book")),1)

    def test_concurrent_read_write_returns_complete_evidence(self):
        self.store.add(body("seed"))
        def work(index):
            try:
                if index%2:
                    self.store.add(body("event "+str(index),request=str(index)))
                else:
                    self.assertTrue(all(isinstance(row["content"],str) for row in self.search("event")))
            finally:
                self.store.close()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(work,range(40)))
        self.assertEqual(self.store._connection().execute("SELECT count(*) FROM messages").fetchone()[0],21)

    def test_config_freeze_and_consistent_backup(self):
        self.store.add(body())
        target=Path(self.folder.name)/"backup.sqlite3"
        self.store.backup(target)
        saved=MemoryStore(target)
        try:
            self.assertEqual(saved.search({"user_id":"u","query":"book","top_k":1})["data"][0]["id"],self.search("book")[0]["id"])
        finally:
            saved.close()
        with self.assertRaises(ValueError):
            MemoryStore(self.path,strategy="sparse")

    def test_image_formats_corruption_mime_and_pixel_capacity(self):
        for format in ("JPEG","PNG","WEBP"):
            value=picture(format=format)
            image,_,_=decode_image(value,max_image_bytes=10*1024**2,max_pixels=4096)
            self.assertEqual((image.width,image.height),(64,64))
        for value in (picture().replace("image/png","image/jpeg"),"data:image/png;base64,@@@@",
                      "data:image/gif;base64,R0lGODlh", "data:image/png;base64,YWJj"):
            with self.assertRaises(ContractError):
                decode_image(value,max_image_bytes=10*1024**2,max_pixels=4096)
        with self.assertRaises(ContractError) as error:
            decode_image(picture(),max_image_bytes=10*1024**2,max_pixels=100)
        self.assertEqual(error.exception.status,413)

    def test_media_budget_counts_duplicate_occurrences_not_unique_bytes(self):
        uri=picture()
        image,_,_=decode_image(uri,max_image_bytes=10*1024**2,max_pixels=4096)
        small=MemoryStore(str(Path(self.folder.name)/"small.sqlite3"),max_media_bytes=len(image.data)*2-1)
        try:
            with self.assertRaises(ContractError):
                small.add(body([{"type":"image_url","image_url":{"url":uri}}]*2))
            self.assertEqual(small._connection().execute("SELECT count(*) FROM blobs").fetchone()[0],0)
        finally:
            small.close()

    def test_response_budget_returns_whole_prefix_without_removing_images(self):
        uri=picture()
        part=[{"type":"text","text":"red source"},{"type":"image_url","image_url":{"url":uri}}]
        self.store.add(body(part))
        self.store.add(body(part,request="r2"))
        image,_,_=decode_image(uri,max_image_bytes=10*1024**2,max_pixels=4096)
        self.store.max_media_bytes=len(image.data)
        result=self.search("red")
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["content"],part)

    def test_cross_session_latest_and_entity_relation_sources(self):
        self.store.add(body("I prefer coffee.",timestamp=1700000000000))
        self.store.add(body("I prefer tea now.",request="r2",session="s2",timestamp=1700100000000))
        self.assertIn("tea",self.search("What is my latest preference?")[0]["content"])
        self.store.add(body("Alice gave Bob a book.",request="r3"))
        self.assertIn("Bob",self.search("Who received the book from Alice?")[0]["content"])

    def test_image_query_exact_match_and_color_text_query(self):
        for number,color in enumerate(("red","blue","green")):
            self.store.add(body([{"type":"image_url","image_url":{"url":picture(color)}}],request=str(number)))
        query=[{"type":"image_url","image_url":{"url":picture("blue")}}]
        self.assertEqual(self.search(query)[0]["content"],query)
        self.assertEqual(self.search("Show the blue object")[0]["content"],query)

    def test_ordered_temporal_neighbour_and_source_date_candidates(self):
        rows=[("Breakfast at hotel.",1704067200000),("Taxi to airport.",1704153600000),("Train to museum.",1704240000000)]
        self.store.add({"user_id":"u","request_id":"trip","session_id":"trip", "messages":[
            {"role":"user","content":text,"timestamp":stamp} for text,stamp in rows]})
        found=self.search("What happened before the airport visit?",top_k=2)
        self.assertTrue(any("Breakfast" in row["content"] for row in found))
        self.assertIn("museum",self.search("What happened on 2024-01-03?")[0]["content"])

    def test_long_text_chinese_and_top_k(self):
        self.store.add(body("今天我和小明去了图书馆。"+" long source"*100000))
        self.assertIn("图书馆",self.search("小明去了哪里？",top_k=1)[0]["content"])
        self.assertLessEqual(len(self.search("long",top_k=1)),1)
        for value in (0,101,True,"1"):
            with self.assertRaises(ContractError):
                self.search("long",top_k=value)

    def test_other_user_cannot_change_scoped_bm25_statistics(self):
        self.store.add(body("alpha",request="alpha"))
        self.store.add(body("beta",request="beta"))
        before=self.search("alpha beta")
        self.store.add({"user_id":"other","request_id":"many","session_id":"s","messages":[
            {"role":"user","content":"alpha"} for _ in range(100)]})
        self.assertEqual(self.search("alpha beta"),before)

    def test_temporal_neighbours_use_source_time_for_backfilled_history(self):
        self.store.add({"user_id":"u","request_id":"first","session_id":"s","messages":[
            {"role":"user","content":"Old hotel stop.","timestamp":10},
            {"role":"user","content":"Flight from airport.","timestamp":30},
            {"role":"user","content":"Dinner at home.","timestamp":40}]})
        self.store.add(body("Visited the library.",request="late",timestamp=20))
        found=self.search("What happened before the flight?",top_k=2)
        self.assertTrue(any("library" in row["content"] for row in found))

    def test_latest_uses_source_time_when_history_arrives_out_of_order(self):
        self.store.add(body("I prefer tea.",request="new",timestamp=1792022400000))
        self.store.add(body("I prefer coffee.",request="old",timestamp=1791936000000))
        latest=self.search("What is my latest preference?")[0]
        self.assertIn("tea",latest["content"])
        self.assertTrue(latest["created_at"].startswith("2026-10-15"))

    def test_all_strategies_reject_truncated_jpeg_and_max_date_does_not_crash(self):
        truncated=base64.b64decode(picture(format="JPEG").split(",",1)[1])[:-8]
        uri="data:image/jpeg;base64,"+base64.b64encode(truncated).decode()
        for strategy in ("raw","sparse","adaptive"):
            store=MemoryStore(str(Path(self.folder.name)/(strategy+".sqlite3")),strategy=strategy)
            try:
                with self.assertRaises(ContractError):
                    store.add(body([{"type":"image_url","image_url":{"url":uri}}]))
                self.assertEqual(store.search({"user_id":"u","query":"events on 9999-12-31","top_k":1}),{"data":[]})
            finally: store.close()


if __name__=="__main__":
    unittest.main()
