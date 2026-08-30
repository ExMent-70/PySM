import contextlib
import io
import unittest

from scripts_utility.SYSTEM.pysm_updater.run_pysm_updater import build_argument_parser


class UpdateModeArgumentTests(unittest.TestCase):
    def test_default_mode_is_safe_plan(self):
        config = build_argument_parser().parse_args([])

        self.assertEqual(config.update_mode, "plan")

    def test_explicit_plan_mode_is_accepted(self):
        config = build_argument_parser().parse_args(["--update_mode", "plan"])

        self.assertEqual(config.update_mode, "plan")

    def test_explicit_apply_mode_is_accepted(self):
        config = build_argument_parser().parse_args(["--update_mode", "apply"])

        self.assertEqual(config.update_mode, "apply")

    def test_unknown_mode_is_rejected(self):
        parser = build_argument_parser()

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--update_mode", "unknown"])


if __name__ == "__main__":
    unittest.main()
