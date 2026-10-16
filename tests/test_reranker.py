import base64
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from prism_swift.storage import MemoryStore
from prism_swift.validation import ContractError


def source(color):
    output=io.BytesIO()
    with Image.new("RGB",(32,32),color) as image: image.save(output,"PNG")
    return [{"type":"text","text":"Gallery source."},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(output.getvalue()).decode()}}]


class FakeReranker:
    candidates=2
    def __init__(self): self.calls=[]
    def config_dict(self): return {"model":"synthetic-reranker-test","candidates":self.candidates}
    def scores(self,text,images,candidates):
        self.calls.append((text,images,candidates))
        return {row["id"]:float(index) for index,row in enumerate(candidates)}


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.reranker=FakeReranker()
        self.store=MemoryStore(str(Path(self.folder.name)/"memory.sqlite3"),reranker=self.reranker)
        self.addCleanup(self.store.close)

    def add(self,user,request,color):
        value=source(color)
        self.store.add({"user_id":user,"request_id":request,"session_id":"s","messages":[{"role":"user","content":value}]})
        return value

    def test_add_never_invokes_encoder_and_search_uses_only_scoped_sources(self):
        a=self.add("a","1","red"); b=self.add("b","2","blue")
        self.assertEqual(self.reranker.calls,[])
        result=self.store.search({"user_id":"a","query":"Find the gallery photo.","top_k":1})
        self.assertEqual(result["data"][0]["content"],a)
        candidates=self.reranker.calls[0][2]
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]["image_bytes"][0],base64.b64decode(a[1]["image_url"]["url"].split(",",1)[1]))
        self.assertNotEqual(candidates[0]["image_bytes"][0],base64.b64decode(b[1]["image_url"]["url"].split(",",1)[1]))

    def test_nonvisual_query_skips_reranker_and_no_derived_cache_is_written(self):
        self.add("a","1","red")
        self.store.search({"user_id":"a","query":"Recall gallery record","top_k":1})
        self.assertEqual(self.reranker.calls,[])
        conn=self.store._connection()
        before=conn.total_changes
        self.store.search({"user_id":"a","query":"gallery photo","top_k":1})
        self.assertEqual(conn.total_changes,before)

    def test_reranking_changes_order_preserves_content_and_honors_topk(self):
        first=self.add("a","1","red"); second=self.add("a","2","blue")
        result=self.store.search({"user_id":"a","query":"gallery photo","top_k":1})["data"]
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["content"],second)
        self.assertEqual(len(self.reranker.calls[0][2]),2)

    def test_invalid_reranker_output_rolls_back_read_transaction(self):
        self.add("a","1","red")
        self.reranker.scores=lambda *args: {"foreign-id":1.0}
        with self.assertRaises(ContractError) as error:
            self.store.search({"user_id":"a","query":"gallery photo","top_k":1})
        self.assertEqual(error.exception.status,503)
        self.assertFalse(self.store._connection().in_transaction)
        self.add("a","2","blue")

    def test_model_configuration_is_frozen_with_database(self):
        with self.assertRaises(ValueError):
            MemoryStore(self.store.db_path)

    def test_visual_routing_recognizes_photograph_plural_and_colour(self):
        self.add("a","1","red")
        for query in ("Find the photograph", "Find photographs", "red object", "What colour was it?"):
            before=len(self.reranker.calls)
            self.store.search({"user_id":"a","query":query,"top_k":1})
            self.assertEqual(len(self.reranker.calls),before+1)


if __name__=="__main__": unittest.main()
