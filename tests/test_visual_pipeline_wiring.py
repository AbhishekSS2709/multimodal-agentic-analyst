"""The visual half of the pipeline has to actually reach the visual store.

`load_image` returns a LoaderResult carrying both text_documents and
visual_assets, but the ingestion wrapper returned only `result.text_documents`,
so every visual asset was discarded at the door. `setup()` then had nothing to
index, `visual_search` queried an empty store, and `modality_match` was pinned
at 0.200 no matter how good retrieval was.
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSET = ROOT / "data" / "assets" / "quarterly_report_chart.png"


@unittest.skipUnless(ASSET.exists(),
                     "run scripts/make_demo_assets.py to generate the corpus")
class TestVisualAssetsSurviveIngestion(unittest.TestCase):

    def test_loader_produces_a_visual_asset(self):
        from src.ingestion.image_loader import load_image
        result = load_image(str(ASSET))
        self.assertTrue(result.visual_assets, "loader produced no visual asset")

    def test_wrapper_does_not_discard_the_asset(self):
        from src.ingestion.pipeline import _load_image_wrapper
        docs = _load_image_wrapper(str(ASSET))
        self.assertTrue(docs)
        carried = [a for d in docs for a in d.metadata.get("visual_assets", [])]
        self.assertTrue(carried, "visual asset was dropped by the wrapper")

    def test_asset_records_its_source_path(self):
        from src.ingestion.pipeline import _load_image_wrapper
        docs = _load_image_wrapper(str(ASSET))
        carried = [a for d in docs for a in d.metadata.get("visual_assets", [])]
        joined = " ".join(str(a) for a in carried)
        self.assertIn("quarterly_report_chart", joined)


class TestAssetCollection(unittest.TestCase):
    """setup() needs a way to gather assets back off the documents."""

    def test_collect_visual_assets_finds_them(self):
        from src.pipeline_orchestrator import collect_visual_assets

        class _Doc:
            def __init__(self, assets):
                self.metadata = {"visual_assets": assets} if assets else {}

        docs = [_Doc([{"path": "a.png"}]), _Doc(None), _Doc([{"path": "b.png"}])]
        found = collect_visual_assets(docs)
        self.assertEqual(len(found), 2)

    def test_collect_handles_documents_without_metadata(self):
        from src.pipeline_orchestrator import collect_visual_assets
        self.assertEqual(collect_visual_assets([object()]), [])

class TestAssetMetadataStaysSerialisable(unittest.TestCase):
    """Asset metadata rides into chunks and into checkpointed graph state.

    The loader's asset dict holds a live PIL image. Attaching that to document
    metadata made LangGraph's SQLite checkpointer fail with "Type is not
    msgpack serializable: Document", which killed the document subgraph and
    silently dropped every text finding.
    """

    @unittest.skipUnless(ASSET.exists(), "run scripts/make_demo_assets.py first")
    def test_no_pil_image_in_document_metadata(self):
        from src.ingestion.pipeline import _load_image_wrapper
        docs = _load_image_wrapper(str(ASSET))
        for doc in docs:
            for asset in doc.metadata.get("visual_assets", []):
                self.assertNotIn("image", asset,
                                 "live PIL image leaked into document metadata")

    @unittest.skipUnless(ASSET.exists(), "run scripts/make_demo_assets.py first")
    def test_metadata_is_plainly_serialisable(self):
        """A PIL handle raises here exactly as it does in the checkpointer."""
        import json
        from src.ingestion.pipeline import _load_image_wrapper
        docs = _load_image_wrapper(str(ASSET))
        for doc in docs:
            json.dumps(doc.metadata.get("visual_assets", []))

    @unittest.skipUnless(ASSET.exists(), "run scripts/make_demo_assets.py first")
    def test_asset_keeps_a_path_so_it_can_still_be_indexed(self):
        from src.ingestion.pipeline import _load_image_wrapper
        docs = _load_image_wrapper(str(ASSET))
        assets = [a for d in docs for a in d.metadata.get("visual_assets", [])]
        self.assertTrue(assets)
        self.assertTrue(all(a.get("original_path") for a in assets))


if __name__ == "__main__":
    unittest.main()
