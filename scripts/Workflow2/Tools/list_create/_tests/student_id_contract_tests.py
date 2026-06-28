"""Изолированные проверки контракта student_id редактора списков."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from domain import (  # noqa: E402
    Student,
    StudentIdAllocator,
    generate_list_id,
    parse_student_id,
)
import io_services  # noqa: E402


class StudentIdAllocatorTests(unittest.TestCase):
    def test_generated_list_id_uses_compact_alphabet(self) -> None:
        list_id = generate_list_id()

        self.assertEqual(4, len(list_id))
        self.assertNotRegex(list_id, r"[IO01]")
        StudentIdAllocator(list_id=list_id)

    def test_ids_are_sequential_and_deleted_number_is_not_reused(self) -> None:
        allocator = StudentIdAllocator("A7K3")

        first = allocator.allocate()
        second = allocator.allocate()
        third = allocator.allocate()

        self.assertEqual("A7K3-S001", first)
        self.assertEqual("A7K3-S002", second)
        self.assertEqual("A7K3-S003", third)
        self.assertEqual(4, allocator.next_student_number)

    def test_name_change_does_not_change_id(self) -> None:
        student = Student(student_id="A7K3-S001", surname="Иванов", name="Иван")

        student.surname = "Петров"
        student.name = "Пётр"

        self.assertEqual("A7K3-S001", student.student_id)

    def test_duplicate_id_is_rejected(self) -> None:
        allocator = StudentIdAllocator("A7K3", 2)
        students = [
            Student(student_id="A7K3-S001", surname="Иванов", name="Иван"),
            Student(student_id="A7K3-S001", surname="Петров", name="Пётр"),
        ]

        with self.assertRaisesRegex(ValueError, "повторяется"):
            allocator.validate_students(students)

    def test_foreign_list_id_is_rejected(self) -> None:
        allocator = StudentIdAllocator("A7K3", 2)
        student = Student(student_id="B8M4-S001", surname="Иванов", name="Иван")

        with self.assertRaisesRegex(ValueError, "относится к списку"):
            allocator.validate_students([student])

    def test_counter_must_stay_above_assigned_numbers(self) -> None:
        allocator = StudentIdAllocator("A7K3", 1)
        student = Student(student_id="A7K3-S001", surname="Иванов", name="Иван")

        with self.assertRaisesRegex(ValueError, "next_student_number"):
            allocator.validate_students([student])

    def test_zero_student_number_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "S001–S999"):
            parse_student_id("A7K3-S000")


class SessionIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allocator = StudentIdAllocator("A7K3", 3)
        self.students = [
            Student(
                student_id="A7K3-S001",
                surname="Иванов",
                name="Иван",
                shoot_order=2,
                info={"Цитата": "Первая"},
            ),
            Student(
                student_id="A7K3-S002",
                surname="Петров",
                name="Пётр",
                shoot_order=1,
                info={"Цитата": "Вторая"},
            ),
        ]

    def test_save_load_roundtrip_preserves_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "class.list"
            io_services.save_session(
                path,
                "Класс",
                "Альбом",
                self.students,
                ["Цитата"],
                self.allocator,
            )

            metadata, loaded = io_services.load_session(path)

        self.assertEqual("A7K3", metadata["list_id"])
        self.assertEqual(3, metadata["next_student_number"])
        self.assertEqual(
            ["A7K3-S001", "A7K3-S002"],
            [student.student_id for student in loaded],
        )

    def test_old_record_without_id_is_rejected(self) -> None:
        payload = {
            "list_id": "A7K3",
            "next_student_number": 2,
            "students": [{"surname": "Иванов", "name": "Иван"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old.list"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "не содержит обязательный student_id"):
                io_services.load_session(path)

    def test_txt_contains_only_ids_in_shoot_order(self) -> None:
        without_order = Student(
            student_id="A7K3-S003", surname="Сидоров", name="Сидор"
        )
        allocator = StudentIdAllocator("A7K3", 4)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_children.txt"
            io_services.export_to_txt(
                path, [*self.students, without_order], allocator
            )
            content = path.read_text(encoding="utf-8")

        self.assertEqual("A7K3-S002\nA7K3-S001", content)

    def test_csv_contains_student_id_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "class.csv"
            io_services.export_to_csv(path, self.students, ["Цитата"])
            content = path.read_text(encoding="utf-16")

        self.assertTrue(content.startswith("student_id;shoot_order"))
        self.assertIn("A7K3-S001", content)


class AiEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.students = [
            Student(student_id="A7K3-S001", surname="Иванов", name="Иван"),
            Student(student_id="A7K3-S002", surname="Петров", name="Пётр"),
        ]

    def test_reference_contains_names_and_ids(self) -> None:
        reference = io_services.build_ai_student_reference(self.students)

        self.assertEqual("A7K3-S001", reference[0]["student_id"])
        self.assertEqual("Иванов", reference[0]["surname"])

    def test_response_is_addressed_by_student_id(self) -> None:
        payload = {
            "matched": [
                {
                    "student_id": "A7K3-S002",
                    "source_person": "Петров",
                    "info": {"Цитата": "Текст"},
                }
            ],
            "unresolved": [],
        }

        updates, unresolved = io_services.validate_ai_enrichment_response(
            payload, self.students
        )

        self.assertEqual({"A7K3-S002": {"Цитата": "Текст"}}, updates)
        self.assertEqual([], unresolved)

    def test_unresolved_entry_does_not_create_update(self) -> None:
        payload = {
            "matched": [],
            "unresolved": [
                {
                    "source_person": "Иванов",
                    "reason": "Несколько кандидатов",
                    "candidates": ["A7K3-S001", "A7K3-S002"],
                }
            ],
        }

        updates, unresolved = io_services.validate_ai_enrichment_response(
            payload, self.students
        )

        self.assertEqual({}, updates)
        self.assertEqual(1, len(unresolved))

    def test_unknown_id_is_rejected(self) -> None:
        payload = {
            "matched": [{"student_id": "A7K3-S999", "info": {}}],
            "unresolved": [],
        }

        with self.assertRaisesRegex(ValueError, "неизвестный student_id"):
            io_services.validate_ai_enrichment_response(payload, self.students)


if __name__ == "__main__":
    unittest.main()
