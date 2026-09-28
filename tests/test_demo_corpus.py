"""The demo's corpus panel describes files that actually ship.

The panel silently skips a missing file, so a rename in ``data/`` would make
it drop an entry without anyone noticing. Read the list straight out of
``app.py`` (importing it would start Streamlit) and check every path.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _literal(name):
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in app.py")


class TestDemoCorpus(unittest.TestCase):

    def test_every_listed_file_exists(self):
        missing = [name for _, name, _ in _literal("CORPUS")
                   if not (ROOT / "data" / name).exists()]
        self.assertEqual(missing, [])

    def test_every_image_the_examples_mention_is_listed(self):
        listed = {Path(name).name for _, name, _ in _literal("CORPUS")}
        self.assertIn("scanned_supplier_notice.png", listed)
        self.assertIn("quarterly_report_chart.png", listed)


if __name__ == "__main__":
    unittest.main()
