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


class CreateBackupArgumentTests(unittest.TestCase):
    def test_backup_is_enabled_by_default(self):
        config = build_argument_parser().parse_args([])

        self.assertTrue(config.create_backup)

    def test_backup_can_be_disabled_from_the_command_line(self):
        config = build_argument_parser().parse_args(["--no-create_backup"])

        self.assertFalse(config.create_backup)

    def test_legacy_no_backup_argument_is_rejected(self):
        parser = build_argument_parser()

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--no_backup"])


if __name__ == "__main__":
    unittest.main()
