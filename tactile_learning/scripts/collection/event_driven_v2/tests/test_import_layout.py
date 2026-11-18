import importlib.util
import sys
import unittest
from pathlib import Path

class ImportLayoutTests(unittest.TestCase):
    def test_project_root_is_derived_from_collector_location(self):
        collector = Path(__file__).resolve().parents[1] / "collector.py"
        project_root = collector.resolve().parents[3]
        self.assertTrue((project_root / "scripts" / "collection" / "event_driven_v2" / "core.py").is_file())

if __name__ == "__main__":
    unittest.main()
