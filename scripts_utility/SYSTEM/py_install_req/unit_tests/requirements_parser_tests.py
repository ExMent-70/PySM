import tempfile
import unittest
from pathlib import Path

from scripts_utility.SYSTEM.py_install_req.installer_lib.models import (
    CudaInfo,
    GpuInfo,
    PackageType,
    SystemInfo,
)
from scripts_utility.SYSTEM.py_install_req.installer_lib.requirements_parser import (
    RequirementsParser,
)


def nvidia_cuda_system() -> SystemInfo:
    return SystemInfo(
        gpu=GpuInfo(name="NVIDIA Test GPU", vendor="NVIDIA"),
        cuda=CudaInfo(
            is_available=True,
            driver_version="13.3",
            recommended_version="12.8",
            selected_version="12.8",
            selected_source="portable",
        ),
    )


class RequirementsParserTests(unittest.TestCase):
    def parse_text(self, text: str, filename: str = "requirements.txt"):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / filename
            path.write_text(text, encoding="utf-8")
            return RequirementsParser(nvidia_cuda_system()).parse(path)

    def test_pep508_extras_markers_and_gpu_categories(self):
        plan = self.parse_text(
            "\n".join(
                [
                    'requests[socks]>=2.31; python_version >= "3.11"',
                    "torch==2.10.0",
                    "onnxruntime",
                    "insightface==1.0.1",
                    "triton",
                ]
            )
        )

        self.assertEqual(plan.regular_packages[0].name, "requests")
        self.assertEqual(plan.regular_packages[0].extras, ["socks"])
        self.assertIn('python_version >= "3.11"', plan.regular_packages[0].to_spec())
        self.assertEqual(plan.torch_packages[0].package_type, PackageType.TORCH)
        self.assertEqual(plan.onnx_packages[0].package_type, PackageType.ONNXRUNTIME)
        self.assertEqual(plan.insightface_packages[0].package_type, PackageType.INSIGHTFACE)
        self.assertEqual(plan.insightface_packages[0].to_spec(), "insightface==1.0.1")
        self.assertEqual(plan.triton_packages[0].package_type, PackageType.TRITON)
        self.assertEqual(plan.torch_backend, "cu128")
        self.assertEqual(plan.onnx_package_name, "onnxruntime-gpu")

    def test_include_constraint_options_and_relative_find_links(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "wheels").mkdir()
            (root / "constraints.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
            (root / "extra.txt").write_text("Pillow\n", encoding="utf-8")
            requirements = root / "requirements.txt"
            requirements.write_text(
                "\n".join(
                    [
                        "-r extra.txt",
                        "-c constraints.txt",
                        "--find-links wheels",
                        "--pre",
                        "beautifulsoup4",
                    ]
                ),
                encoding="utf-8",
            )

            plan = RequirementsParser(nvidia_cuda_system()).parse(requirements)

        self.assertEqual([pkg.name for pkg in plan.regular_packages], ["Pillow", "beautifulsoup4"])
        self.assertIn("--constraint", plan.pip_options)
        self.assertTrue(any(value.endswith("constraints.txt") for value in plan.pip_options))
        self.assertIn("--find-links", plan.pip_options)
        self.assertTrue(any(value.endswith("wheels") for value in plan.pip_options))
        self.assertIn("--pre", plan.pip_options)
        self.assertEqual(len(plan.included_files), 2)
        self.assertIn("numpy", plan.package_constraints)
        self.assertEqual(plan.package_constraints["numpy"].to_spec(), "numpy==1.26.4")

    def test_direct_url_editable_local_path_and_hash_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "local_pkg").mkdir()
            requirements = root / "requirements.txt"
            requirements.write_text(
                "\n".join(
                    [
                        "demo @ https://example.invalid/demo-1.0.0.whl",
                        "-e ./local_pkg",
                        "./local_pkg",
                        "numpy==1.26.4 --hash=sha256:abc",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertLogs(level="WARNING"):
                plan = RequirementsParser(nvidia_cuda_system()).parse(requirements)

        specs = [pkg.to_spec() for pkg in plan.regular_packages]
        self.assertIn("demo @ https://example.invalid/demo-1.0.0.whl", specs)
        self.assertTrue(any(spec.startswith("-e ") and spec.endswith("local_pkg") for spec in specs))
        self.assertTrue(any(spec.endswith("local_pkg") for spec in specs))
        numpy_pkg = next(pkg for pkg in plan.regular_packages if pkg.name == "numpy")
        self.assertEqual(numpy_pkg.to_spec(), "numpy==1.26.4")
        self.assertTrue(any("--hash найден" in item for item in plan.diagnostics))

    def test_nested_constraint_file_is_parsed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "base_constraints.txt").write_text("-c nested/extra_constraints.txt\n", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "extra_constraints.txt").write_text("Pillow==12.2.0\n", encoding="utf-8")
            requirements = root / "requirements.txt"
            requirements.write_text("-c base_constraints.txt\nPillow\n", encoding="utf-8")

            plan = RequirementsParser(nvidia_cuda_system()).parse(requirements)

        self.assertIn("pillow", plan.package_constraints)
        self.assertEqual(plan.package_constraints["pillow"].to_spec(), "Pillow==12.2.0")
        self.assertEqual(len(plan.constraint_files), 2)

    def test_line_continuation_and_inline_comment(self):
        plan = self.parse_text(
            "transparent-background[gui] \\\n"
            '    ; python_version >= "3.11"  # keep marker\n'
            "regex # ordinary comment\n"
        )

        self.assertEqual([pkg.name for pkg in plan.regular_packages], ["transparent-background", "regex"])
        self.assertIn('python_version >= "3.11"', plan.regular_packages[0].to_spec())

    def test_environment_marker_filters_incompatible_requirement(self):
        with self.assertLogs(level="WARNING"):
            plan = self.parse_text(
                "\n".join(
                    [
                        'demo-skip; python_version < "0"',
                        'demo-keep; python_version >= "3"',
                    ]
                )
            )

        self.assertEqual([pkg.name for pkg in plan.regular_packages], ["demo-keep"])
        self.assertTrue(any("demo-skip" in item for item in plan.diagnostics))

    def test_prefer_binary_is_flag_option(self):
        plan = self.parse_text("--prefer-binary\nPillow\n")

        self.assertIn("--prefer-binary", plan.pip_options)
        self.assertEqual(plan.regular_packages[0].name, "Pillow")


if __name__ == "__main__":
    unittest.main()
