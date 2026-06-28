import sys
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_py_install_req import resolve_pysm_root, resolve_search_target, resolve_target_python


class CliResolutionTests(unittest.TestCase):
    def test_default_python_is_current_interpreter(self):
        self.assertEqual(
            resolve_target_python(Namespace(inst_python_interpreter="")),
            Path(sys.executable).resolve(),
        )

    def test_explicit_missing_python_is_error(self):
        with self.assertRaises(FileNotFoundError):
            resolve_target_python(Namespace(inst_python_interpreter=r"Z:\missing\python.exe"))

    def test_default_search_target_is_pysm_root(self):
        self.assertEqual(
            resolve_search_target(Namespace(inst_search_path="")),
            resolve_pysm_root(),
        )
        self.assertTrue((resolve_pysm_root() / "requirements.txt").is_file())

    def test_explicit_missing_search_path_is_error(self):
        with self.assertRaises(FileNotFoundError):
            resolve_search_target(Namespace(inst_search_path=r"Z:\missing\requirements.txt"))


if __name__ == "__main__":
    unittest.main()
