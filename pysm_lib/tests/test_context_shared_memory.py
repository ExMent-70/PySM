from __future__ import annotations

import unittest
import sys
from types import SimpleNamespace

from pysm_lib.context_shared_memory import (
    SharedMemoryContextError,
    SharedMemoryContextStore,
    calculate_segment_size,
    encoded_payload_size,
    make_context_shm_name,
)
from pysm_lib.pysm_context import PySMContext
from pysm_lib.set_runner_orchestrator import SetRunnerOrchestrator


class SharedMemoryContextStoreTests(unittest.TestCase):
    def test_shared_memory_roundtrip_between_handles(self) -> None:
        name = make_context_shm_name("pysm_test_context")
        store = SharedMemoryContextStore.create(
            name,
            1024 * 1024,
            {"alpha": {"type": "int", "value": 1}},
        )
        opened = None
        try:
            opened = SharedMemoryContextStore.open(name)
            self.assertEqual(opened.load()["alpha"]["value"], 1)

            opened.save({"alpha": {"type": "int", "value": 2}})
            self.assertEqual(store.load()["alpha"]["value"], 2)
        finally:
            if opened:
                opened.close()
            store.unlink()
            store.close()

    def test_generation_changes_after_save(self) -> None:
        name = make_context_shm_name("pysm_test_context")
        store = SharedMemoryContextStore.create(name, 1024 * 1024, {})
        try:
            first_generation = store.generation
            store.save({"value": {"type": "string", "value": "x"}})
            self.assertGreater(store.generation, first_generation)
        finally:
            store.unlink()
            store.close()

    def test_payload_telemetry_tracks_saved_snapshot(self) -> None:
        name = make_context_shm_name("pysm_test_context")
        snapshot = {"value": {"type": "string", "value": "hello"}}
        store = SharedMemoryContextStore.create(name, 1024 * 1024, snapshot)
        try:
            self.assertEqual(store.payload_size, encoded_payload_size(snapshot))
            self.assertGreater(store.usage_ratio, 0)
        finally:
            store.unlink()
            store.close()

    def test_overflow_fails_current_write(self) -> None:
        name = make_context_shm_name("pysm_test_context")
        store = SharedMemoryContextStore.create(name, 512, {})
        try:
            with self.assertRaises(SharedMemoryContextError):
                store.save({"blob": {"type": "string", "value": "x" * 2048}})
            self.assertEqual(store.load(), {})
        finally:
            store.unlink()
            store.close()

    def test_segment_size_honors_minimum(self) -> None:
        size = calculate_segment_size({}, min_size_mb=1, max_size_mb=2)
        self.assertEqual(size, 1024 * 1024)

    def test_pysm_context_public_api_writes_shared_memory(self) -> None:
        name = make_context_shm_name("pysm_test_context")
        store = SharedMemoryContextStore.create(name, 1024 * 1024, {})
        old_argv = sys.argv[:]
        context = None
        try:
            sys.argv = [
                "script.py",
                "--pysm-context-shm-name",
                name,
                "--pysm-context-mode",
                "shared_memory",
            ]
            context = PySMContext()
            context._send_ipc_update = lambda *args, **kwargs: None
            context.set("answer", 42)

            snapshot = store.load()
            self.assertEqual(snapshot["answer"]["value"], 42)
            self.assertEqual(snapshot["answer"]["type"], "int")
        finally:
            sys.argv = old_argv
            if context and getattr(context, "_context_store", None):
                context._context_store.close()
            store.unlink()
            store.close()

    def test_on_save_exit_does_not_checkpoint_when_run_finalizes(self) -> None:
        orchestrator = SetRunnerOrchestrator.__new__(SetRunnerOrchestrator)
        orchestrator.config_manager = SimpleNamespace(
            config=SimpleNamespace(
                runtime_context=SimpleNamespace(checkpoint_policy="on_save_exit")
            )
        )

        self.assertFalse(orchestrator._should_write_context_file_on_finalize())

    def test_after_each_script_checkpoints_when_run_finalizes(self) -> None:
        orchestrator = SetRunnerOrchestrator.__new__(SetRunnerOrchestrator)
        orchestrator.config_manager = SimpleNamespace(
            config=SimpleNamespace(
                runtime_context=SimpleNamespace(checkpoint_policy="after_each_script")
            )
        )

        self.assertTrue(orchestrator._should_write_context_file_on_finalize())

    def test_orchestrator_resizes_shared_memory_after_high_usage(self) -> None:
        name = make_context_shm_name("pysm_test_context")
        snapshot = {"blob": {"type": "string", "value": "x" * 850}}
        store = SharedMemoryContextStore.create(name, 1024, snapshot)
        logs = []
        orchestrator = SetRunnerOrchestrator.__new__(SetRunnerOrchestrator)
        orchestrator.config_manager = SimpleNamespace(
            config=SimpleNamespace(
                runtime_context=SimpleNamespace(
                    backend="shared_memory",
                    shared_memory_min_size_mb=1,
                    shared_memory_max_size_mb=1,
                )
            )
        )
        orchestrator._context_store = store
        orchestrator._context_backend = "shared_memory"
        orchestrator._context_shm_name = name
        orchestrator._context_segment_size = 1024
        orchestrator._context_snapshot_cache = snapshot
        orchestrator._context_resize_warning_logged = False
        orchestrator.log_message = SimpleNamespace(emit=lambda *args: logs.append(args))

        try:
            self.assertTrue(orchestrator._maybe_resize_runtime_context_store())
            self.assertNotEqual(orchestrator._context_shm_name, name)
            self.assertEqual(
                orchestrator._context_store.load()["blob"]["value"],
                snapshot["blob"]["value"],
            )
            self.assertTrue(any("Shared memory resized" in entry[1] for entry in logs))
        finally:
            orchestrator._context_store.unlink()
            orchestrator._context_store.close()

    def test_orchestrator_resizes_when_snapshot_write_overflows(self) -> None:
        name = make_context_shm_name("pysm_test_context")
        store = SharedMemoryContextStore.create(name, 512, {})
        snapshot = {"blob": {"type": "string", "value": "x" * 900}}
        logs = []
        orchestrator = SetRunnerOrchestrator.__new__(SetRunnerOrchestrator)
        orchestrator.config_manager = SimpleNamespace(
            config=SimpleNamespace(
                runtime_context=SimpleNamespace(
                    backend="shared_memory",
                    shared_memory_min_size_mb=1,
                    shared_memory_max_size_mb=1,
                )
            )
        )
        orchestrator._context_store = store
        orchestrator._context_backend = "shared_memory"
        orchestrator._context_shm_name = name
        orchestrator._context_segment_size = 512
        orchestrator._context_snapshot_cache = {}
        orchestrator._context_resize_warning_logged = False
        orchestrator.log_message = SimpleNamespace(emit=lambda *args: logs.append(args))

        try:
            orchestrator._save_snapshot_to_runtime_store(snapshot)
            self.assertNotEqual(orchestrator._context_shm_name, name)
            self.assertEqual(
                orchestrator._context_store.load()["blob"]["value"],
                snapshot["blob"]["value"],
            )
        finally:
            orchestrator._context_store.unlink()
            orchestrator._context_store.close()


if __name__ == "__main__":
    unittest.main()
