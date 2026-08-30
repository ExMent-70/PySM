import unittest
from pathlib import Path

from scripts_utility.SYSTEM.py_install_req.installer_lib.installation_manager import (
    InstallationManager,
)
from scripts_utility.SYSTEM.py_install_req.installer_lib.models import (
    CudaInfo,
    GpuInfo,
    InstallationPlan,
    PackageInfo,
    PackageType,
    SystemInfo,
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


def package(
    name: str,
    package_type: PackageType,
    spec: str | None = None,
    version: str | None = None,
    direct_reference: bool = False,
) -> PackageInfo:
    return PackageInfo(
        name=name,
        original_line=spec or name,
        package_type=package_type,
        version=version,
        spec=spec,
        direct_reference=direct_reference,
    )


def insightface_plan(
    insightface_version: str = "1.0.1",
    numpy_version: str = "2.4.6",
    include_torch: bool = False,
) -> InstallationPlan:
    regular_packages = [
        package(
            "numpy",
            PackageType.REGULAR,
            f"numpy=={numpy_version}",
            version=f"=={numpy_version}",
        )
    ]
    torch_packages = [package("torch", PackageType.TORCH, "torch")] if include_torch else []
    return InstallationPlan(
        regular_packages=regular_packages,
        torch_packages=torch_packages,
        onnx_packages=[package("onnxruntime-gpu", PackageType.ONNXRUNTIME, "onnxruntime-gpu")],
        insightface_packages=[
            package(
                "insightface",
                PackageType.INSIGHTFACE,
                f"insightface=={insightface_version}",
                version=f"=={insightface_version}",
            )
        ],
        torch_backend="cu128",
        torch_index_url="https://download.pytorch.org/whl/cu128",
        onnx_package_name="onnxruntime-gpu",
    )


class TestInstallationManager(InstallationManager):
    def __init__(
        self,
        plan: InstallationPlan,
        system_info: SystemInfo,
        use_uv: bool = True,
        force_upgrade: bool = False,
    ):
        self.plan = plan
        self.system_info = system_info
        self.python_executable = Path("python.exe")
        self.force_upgrade = force_upgrade
        self.plan_only = True
        self.failures = []
        self.category_results = {}
        self.use_uv = use_uv
        self.use_log_icons = False
        self.installed_packages = {}
        self.installed_torch_info = {"installed": False}
        self.torch_family_info = None
        self.captured_commands = []
        self.captured_constraint_contents = []
        self._numpy_constraint_path = None

    def _run_install_command(self, cmd, category_name):
        self.captured_commands.append((category_name, list(cmd)))
        for index, token in enumerate(cmd[:-1]):
            if token == "--constraint":
                constraint_path = Path(cmd[index + 1])
                if constraint_path.is_file():
                    self.captured_constraint_contents.append(
                        (constraint_path, constraint_path.read_text(encoding="utf-8"))
                    )
        return super()._run_install_command(cmd, category_name)

    def _get_installed_torch_family_info(self, packages):
        if self.torch_family_info is not None:
            return self.torch_family_info
        return super()._get_installed_torch_family_info(packages)


class GpuInstallationPlanTests(unittest.TestCase):
    def test_insightface_uses_requested_version_without_cpu_runtime_dependencies(self):
        plan = insightface_plan()
        manager = TestInstallationManager(plan, nvidia_cuda_system())

        manager._install_insightface_packages()

        category, command = manager.captured_commands[-1]
        self.assertEqual(category, "Insightface")
        self.assertIn("--no-deps", command)
        self.assertIn("insightface==1.0.1", command)
        self.assertNotIn("numpy==1.26.4", command)
        self.assertFalse(any(item == "onnxruntime" for item in command))

    def test_insightface_gpu_verification_rejects_cpu_runtime_collision(self):
        manager = TestInstallationManager(insightface_plan(), nvidia_cuda_system())
        manager._get_installed_insightface_info = lambda: {
            "installed": True,
            "insightface_version": "1.0.1",
            "numpy_version": "2.4.6",
            "gpu_runtime_version": "1.27.0",
            "cpu_runtime_version": "1.27.0",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        }

        with self.assertLogs(level="ERROR"):
            self.assertFalse(manager._verify_insightface_gpu_installation())

        self.assertIn("ONNX Runtime CPU/GPU package collision", manager.failures)

    def test_insightface_gpu_verification_accepts_cuda_provider(self):
        manager = TestInstallationManager(insightface_plan(), nvidia_cuda_system())
        manager._get_installed_insightface_info = lambda: {
            "installed": True,
            "insightface_version": "1.0.1",
            "numpy_version": "2.4.6",
            "gpu_runtime_version": "1.27.0",
            "cpu_runtime_version": None,
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        }

        self.assertTrue(manager._verify_insightface_gpu_installation())

    def test_dynamic_versions_drive_commands_constraints_and_postcheck(self):
        plan = insightface_plan("9.8.7", "6.5.4", include_torch=True)
        plan.regular_packages.append(package("requests", PackageType.REGULAR, "requests"))
        manager = TestInstallationManager(plan, nvidia_cuda_system())

        manager.execute_plan()

        commands_by_category = {
            category: command
            for category, command in manager.captured_commands
        }
        self.assertIn("numpy==6.5.4", commands_by_category["Обычные пакеты"])
        self.assertIn("insightface==9.8.7", commands_by_category["Insightface"])
        self.assertIn("--no-deps", commands_by_category["Insightface"])
        self.assertNotIn("--constraint", commands_by_category["Insightface"])
        self.assertNotIn("--no-deps", commands_by_category["Обычные пакеты"])
        self.assertNotIn("--no-deps", commands_by_category["Torch"])
        self.assertNotIn("--no-deps", commands_by_category["ONNX"])
        self.assertIn("--constraint", commands_by_category["Torch"])
        self.assertIn("--constraint", commands_by_category["ONNX"])
        self.assertIn("onnxruntime-gpu", commands_by_category["ONNX"])
        self.assertNotIn("onnxruntime", commands_by_category["ONNX"])
        self.assertTrue(manager.captured_constraint_contents)
        self.assertTrue(all(content == "numpy==6.5.4\n" for _, content in manager.captured_constraint_contents))
        self.assertTrue(all(not path.exists() for path, _ in manager.captured_constraint_contents))

        manager._get_installed_insightface_info = lambda: {
            "installed": True,
            "insightface_version": "9.8.7",
            "numpy_version": "6.5.4",
            "gpu_runtime_version": "1.27.0",
            "cpu_runtime_version": None,
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        }
        self.assertTrue(manager._verify_insightface_gpu_installation())

    def test_postcheck_rejects_versions_different_from_requirements(self):
        manager = TestInstallationManager(insightface_plan("9.8.7", "6.5.4"), nvidia_cuda_system())
        manager._get_installed_insightface_info = lambda: {
            "installed": True,
            "insightface_version": "1.0.1",
            "numpy_version": "6.5.4",
            "gpu_runtime_version": "1.27.0",
            "cpu_runtime_version": None,
            "providers": ["CUDAExecutionProvider"],
        }

        with self.assertLogs(level="ERROR"):
            self.assertFalse(manager._verify_insightface_gpu_installation())

        self.assertIn("InsightFace version verification", manager.failures)

    def test_cpu_onnxruntime_is_removed_before_gpu_runtime_reinstall(self):
        manager = TestInstallationManager(insightface_plan(), nvidia_cuda_system())
        manager.installed_packages = {
            "numpy": "2.4.6",
            "onnxruntime": "1.27.0",
            "onnxruntime-gpu": "1.27.0",
            "insightface": "1.0.1",
        }

        manager.execute_plan()

        uninstall_commands = [
            command
            for _, command in manager.captured_commands
            if "uninstall" in command
        ]
        gpu_reinstall_commands = [
            command
            for _, command in manager.captured_commands
            if "--reinstall-package" in command and "onnxruntime-gpu" in command
        ]
        self.assertEqual(len(uninstall_commands), 1)
        self.assertEqual(uninstall_commands[0][-1], "onnxruntime")
        self.assertEqual(len(gpu_reinstall_commands), 1)
        self.assertIn("--constraint", gpu_reinstall_commands[0])

    def test_onnx_distribution_inspection_error_blocks_real_installation(self):
        manager = TestInstallationManager(insightface_plan(), nvidia_cuda_system())
        manager.plan_only = False
        manager._get_onnx_distribution_info = lambda: {"error": "metadata unavailable"}

        with self.assertLogs(level="ERROR"):
            manager._repair_onnx_runtime_collision()

        self.assertIn("ONNX Runtime distribution inspection", manager.failures)
        self.assertEqual(manager.category_results["ONNX"]["status"], "FAIL")
        self.assertFalse(manager.captured_commands)

    def test_uv_base_command_includes_torch_backend_for_cuda(self):
        plan = InstallationPlan(
            torch_backend="cu128",
            torch_index_url="https://download.pytorch.org/whl/cu128",
        )
        manager = TestInstallationManager(plan, nvidia_cuda_system(), use_uv=True)

        self.assertEqual(
            manager._build_base_command(),
            ["python.exe", "-m", "uv", "pip", "install", "--torch-backend", "cu128"],
        )

    def test_pip_base_command_uses_extra_index_for_cuda_wheels(self):
        plan = InstallationPlan(
            torch_backend="cu128",
            torch_index_url="https://download.pytorch.org/whl/cu128",
        )
        manager = TestInstallationManager(plan, nvidia_cuda_system(), use_uv=False)

        self.assertEqual(
            manager._build_base_command(),
            [
                "python.exe",
                "-m",
                "pip",
                "install",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
        )

    def test_onnxruntime_is_normalized_to_gpu_package(self):
        plan = InstallationPlan(onnx_package_name="onnxruntime-gpu")
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        original = package(
            "onnxruntime",
            PackageType.ONNXRUNTIME,
            "onnxruntime>=1.20",
            version=">=1.20",
        )

        with self.assertLogs(level="WARNING"):
            normalized = manager._normalize_onnx_packages([original])

        self.assertEqual(normalized[0].name, "onnxruntime-gpu")
        self.assertEqual(normalized[0].to_spec(), "onnxruntime-gpu>=1.20")
        self.assertEqual(
            manager.plan.requirement_rewrites,
            ["onnxruntime>=1.20 -> onnxruntime-gpu>=1.20 (ONNX Runtime)"],
        )

    def test_onnxruntime_normalization_preserves_marker(self):
        plan = InstallationPlan(onnx_package_name="onnxruntime-gpu")
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        original = package(
            "onnxruntime",
            PackageType.ONNXRUNTIME,
            'onnxruntime>=1.20; python_version >= "3.11"',
            version=">=1.20",
        )

        with self.assertLogs(level="WARNING"):
            normalized = manager._normalize_onnx_packages([original])

        self.assertEqual(
            normalized[0].to_spec(),
            'onnxruntime-gpu>=1.20; python_version >= "3.11"',
        )

    def test_cpu_or_wrong_cuda_torch_requires_reinstall(self):
        plan = InstallationPlan(torch_backend="cu128")
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        torch_pkg = package("torch", PackageType.TORCH, "torch==2.10.0")

        manager.installed_torch_info = {
            "installed": True,
            "version": "2.12.0+cpu",
            "cuda_version": None,
            "cuda_available": False,
        }
        self.assertTrue(manager._needs_torch_cuda_reinstall(torch_pkg))

        manager.installed_torch_info = {
            "installed": True,
            "version": "2.12.0+cu126",
            "cuda_version": "12.6",
            "cuda_available": True,
        }
        self.assertTrue(manager._needs_torch_cuda_reinstall(torch_pkg))

        manager.installed_torch_info = {
            "installed": True,
            "version": "2.12.0+cu128",
            "cuda_version": "12.8",
            "cuda_available": True,
        }
        self.assertFalse(manager._needs_torch_cuda_reinstall(torch_pkg))

    def test_torch_companion_version_label_does_not_reuse_torch_version(self):
        plan = InstallationPlan(torch_backend="cu128")
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        torchvision_pkg = package(
            "torchvision",
            PackageType.TORCH,
            "torchvision==0.25.0",
            version="==0.25.0",
        )

        manager.installed_packages = {"torch": "2.12.0+cu128"}
        manager.installed_torch_info = {
            "installed": True,
            "version": "2.12.0+cu128",
            "cuda_version": "12.8",
            "cuda_available": True,
        }

        self.assertEqual(manager._get_current_version_label(torchvision_pkg), "не установлен")
        self.assertEqual(
            manager._format_package_version_details("UPDATE", torchvision_pkg),
            " (<b>0.25.0</b>)",
        )

    def test_torch_companion_cpu_or_wrong_cuda_requires_reinstall(self):
        plan = InstallationPlan(torch_backend="cu128")
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        torchvision_pkg = package("torchvision", PackageType.TORCH, "torchvision")
        manager.installed_torch_info = {
            "installed": True,
            "version": "2.12.0+cu128",
            "cuda_version": "12.8",
            "cuda_available": True,
        }

        manager.installed_packages = {"torchvision": "0.25.0+cpu"}
        self.assertTrue(manager._needs_torch_cuda_reinstall(torchvision_pkg))

        manager.installed_packages = {"torchvision": "0.25.0+cu126"}
        self.assertTrue(manager._needs_torch_cuda_reinstall(torchvision_pkg))

        manager.installed_packages = {"torchvision": "0.25.0+cu128"}
        self.assertFalse(manager._needs_torch_cuda_reinstall(torchvision_pkg))

        manager.installed_torch_info = {"installed": False}
        self.assertTrue(manager._needs_torch_cuda_reinstall(torchvision_pkg))

    def test_torch_cuda_verification_checks_requested_companion_packages(self):
        plan = InstallationPlan(torch_backend="cu128")
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        manager.torch_family_info = {
            "torch": {
                "installed": True,
                "version": "2.12.0+cu128",
                "cuda_version": "12.8",
                "cuda_available": True,
            },
            "torchvision": {"installed": True, "version": "0.25.0+cpu"},
        }

        with self.assertLogs(level="ERROR"):
            self.assertFalse(
                manager._verify_torch_cuda_installation([
                    package("torchvision", PackageType.TORCH, "torchvision")
                ])
            )
        self.assertIn("torchvision CUDA verification", manager.failures)

    def test_torch_cpu_pin_is_rewritten_to_selected_cuda_tag(self):
        plan = InstallationPlan(torch_backend="cu128")
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        torch_pkg = package("torch", PackageType.TORCH, "torch==2.10.0+cpu")

        with self.assertLogs(level="WARNING"):
            self.assertEqual(manager._normalize_torch_spec(torch_pkg), "torch==2.10.0+cu128")
        self.assertEqual(
            manager.plan.requirement_rewrites,
            ["torch==2.10.0+cpu -> torch==2.10.0+cu128 (CUDA 12.8)"],
        )
        self.assertEqual(manager._get_target_version_label(torch_pkg), "2.10.0+cu128")

    def test_torch_family_cpu_pins_are_rewritten_to_selected_cuda_tag(self):
        plan = InstallationPlan(torch_backend="cu128")
        manager = TestInstallationManager(plan, nvidia_cuda_system())

        with self.assertLogs(level="WARNING"):
            specs = manager._get_torch_specs([
                package("torchvision", PackageType.TORCH, "torchvision==0.25.0+cpu"),
                package("torchaudio", PackageType.TORCH, "torchaudio==2.10.0+cu126"),
            ])

        self.assertEqual(specs, ["torchvision==0.25.0+cu128", "torchaudio==2.10.0+cu128"])
        self.assertEqual(len(manager.plan.requirement_rewrites), 2)

    def test_torch_direct_reference_with_wrong_cuda_tag_is_blocked(self):
        plan = InstallationPlan(torch_backend="cu128")
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        torch_wheel = package(
            "torch",
            PackageType.TORCH,
            "https://download.pytorch.org/whl/cpu/torch-2.10.0+cpu-cp311-cp311-win_amd64.whl",
            direct_reference=True,
        )

        with self.assertLogs(level="ERROR"):
            self.assertFalse(manager._validate_torch_direct_references([torch_wheel]))
        self.assertIn("Torch direct reference CUDA mismatch", manager.failures)
        self.assertTrue(any("Автоматически переписать wheel/URL нельзя" in item for item in manager.plan.diagnostics))

    def test_reinstall_options_use_uv_reinstall_package(self):
        plan = InstallationPlan(torch_backend="cu128")
        manager = TestInstallationManager(plan, nvidia_cuda_system(), use_uv=True)
        cmd = ["python.exe", "-m", "uv", "pip", "install"]

        result = manager._add_reinstall_options(
            cmd,
            [package("torch", PackageType.TORCH), package("torchvision", PackageType.TORCH)],
        )

        self.assertIn("--reinstall-package", result)
        self.assertIn("torch", result)
        self.assertIn("torchvision", result)

    def test_editable_package_is_split_into_cli_arguments(self):
        plan = InstallationPlan()
        manager = TestInstallationManager(plan, nvidia_cuda_system(), use_uv=True)
        editable = package(
            "local-pkg",
            PackageType.REGULAR,
            r"-e D:\work\local_pkg",
            direct_reference=True,
        )

        self.assertEqual(manager._package_install_args(editable), ["-e", r"D:\work\local_pkg"])

    def test_package_plan_action_shows_current_and_target_versions(self):
        plan = InstallationPlan()
        manager = TestInstallationManager(plan, nvidia_cuda_system(), force_upgrade=True)
        manager.installed_packages = {"packaging": "23.2", "pillow": "10.0.0", "numpy": "1.26.4"}

        exact = manager._format_package_plan_action(
            "UPDATE",
            "Обновить",
            package("packaging", PackageType.REGULAR, "packaging==24.0", version="==24.0"),
        )
        ranged = manager._format_package_plan_action(
            "UPDATE",
            "Обновить",
            package("Pillow", PackageType.REGULAR, "Pillow>=10.1", version=">=10.1"),
        )
        unpinned = manager._format_package_plan_action(
            "INSTALL",
            "Установить",
            package("tqdm", PackageType.REGULAR, "tqdm"),
        )
        same_exact = manager._format_package_plan_action(
            "UPDATE",
            "Обновить",
            package("numpy", PackageType.REGULAR, "numpy==1.26.4", version="==1.26.4"),
        )

        self.assertEqual(
            exact,
            "  [U] Обновить: <i>packaging==24.0</i> (<b>23.2</b> -> <b>24.0</b>)",
        )
        self.assertEqual(
            ranged,
            "  [U] Обновить: <i>Pillow>=10.1</i> (<b>10.0.0</b>)",
        )
        self.assertEqual(
            unpinned,
            "  [+] Установить: <i>tqdm</i>",
        )
        self.assertEqual(
            same_exact,
            "  [U] Обновить: <i>numpy==1.26.4</i> (<b>1.26.4</b>)",
        )

    def test_matching_package_plan_action_is_compact(self):
        plan = InstallationPlan()
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        manager.installed_packages = {"beautifulsoup4": "4.15.0"}

        message = manager._format_package_plan_action(
            "OK",
            "Соответствует",
            package("beautifulsoup4", PackageType.REGULAR, "beautifulsoup4"),
        )

        self.assertEqual(
            message,
            "  [OK] Соответствует: <i>beautifulsoup4</i> (<b>4.15.0</b>)",
        )
        self.assertNotIn("цель:", message)

    def test_runtime_status_lines_use_info_prefix(self):
        manager = TestInstallationManager(InstallationPlan(), nvidia_cuda_system())

        self.assertEqual(manager._status_prefix("DIRECT"), "[i]")

    def test_runtime_status_lines_can_use_html_info_icon(self):
        manager = TestInstallationManager(InstallationPlan(), nvidia_cuda_system())
        manager.use_log_icons = True

        prefix = manager._status_prefix("DIRECT")

        self.assertTrue(prefix == "[i]" or prefix.startswith("<img "))

    def test_constraint_conflict_is_added_to_install_plan(self):
        constrained = package("numpy", PackageType.REGULAR, "numpy")
        constraint = package("numpy", PackageType.REGULAR, "numpy==1.26.4", version="==1.26.4")
        plan = InstallationPlan(
            regular_packages=[constrained],
            package_constraints={"numpy": constraint},
        )
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        manager.installed_packages = {"numpy": "2.0.0"}

        with self.assertLogs(level="ERROR") as captured:
            result = manager._filter_packages_to_install([constrained])

        self.assertEqual(result, [constrained])
        output = "\n".join(captured.output)
        self.assertIn("Конфликт constraints", output)
        self.assertIn("(<b>2.0.0</b> -> <b>1.26.4</b>)", output)

    def test_upgrade_with_exact_constraint_shows_target_version(self):
        constrained = package("beautifulsoup4", PackageType.REGULAR, "beautifulsoup4")
        constraint = package(
            "beautifulsoup4",
            PackageType.REGULAR,
            "beautifulsoup4==4.15.0",
            version="==4.15.0",
        )
        plan = InstallationPlan(
            regular_packages=[constrained],
            package_constraints={"beautifulsoup4": constraint},
        )
        manager = TestInstallationManager(plan, nvidia_cuda_system(), force_upgrade=True)
        manager.installed_packages = {"beautifulsoup4": "4.14.3"}

        with self.assertLogs(level="INFO") as captured:
            result = manager._filter_packages_to_install([constrained])

        self.assertEqual(result, [constrained])
        output = "\n".join(captured.output)
        self.assertIn("Обновить", output)
        self.assertIn("beautifulsoup4", output)
        self.assertIn("(<b>4.14.3</b> -> <b>4.15.0</b>)", output)

    def test_execute_plan_reports_blocking_direct_reference_in_plan_only(self):
        plan = InstallationPlan(
            torch_backend="cu128",
            torch_index_url="https://download.pytorch.org/whl/cu128",
            torch_packages=[
                package(
                    "torch",
                    PackageType.TORCH,
                    "https://download.pytorch.org/whl/cpu/torch-2.10.0+cpu-cp311-cp311-win_amd64.whl",
                    direct_reference=True,
                )
            ],
        )
        manager = TestInstallationManager(plan, nvidia_cuda_system())

        with self.assertLogs(level="ERROR") as captured:
            manager.execute_plan()

        output = "\n".join(captured.output)
        self.assertIn("Режим плана: обнаружены блокирующие проблемы", output)
        self.assertIn("Torch direct reference CUDA mismatch", manager.failures)
        self.assertEqual(manager.category_results["Torch"]["status"], "FAIL")

    def test_execute_plan_records_category_summary_for_plan_only(self):
        plan = InstallationPlan(
            regular_packages=[package("beautifulsoup4", PackageType.REGULAR, "beautifulsoup4")]
        )
        manager = TestInstallationManager(plan, nvidia_cuda_system())
        manager.installed_packages = {}

        with self.assertLogs(level="INFO") as captured:
            manager.execute_plan()

        self.assertEqual(manager.category_results["Обычные пакеты"]["status"], "PLAN")
        self.assertEqual(manager.category_results["Torch"]["status"], "SKIP")
        self.assertEqual(manager.category_results["Torch"]["details"], "нет требований")
        output = "\n".join(captured.output)
        self.assertIn("ИТОГ ПЛАНА УСТАНОВКИ", output)
        self.assertIn("[PLAN] Обычные пакеты", output)
        self.assertIn("[SKIP] Torch: нет требований", output)


if __name__ == "__main__":
    unittest.main()
