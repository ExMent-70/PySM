"""Coordinate background face search and explicit, main-thread assignments."""

from collections import OrderedDict
import html
import logging
from threading import Event

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QDialog, QMessageBox, QProgressDialog

from .editor_dialogs import FaceSelectorDialog
from .similar_faces import FaceEmbeddingIndex, SearchCancelled, face_address


logger = logging.getLogger(__name__)
DEFAULT_SIMILARITY_DISTANCE = 0.30


def russian_question(parent, title: str, text: str) -> int:
    """Use Russian button captions even without a Qt translation catalog."""

    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Question)
    dialog.setWindowTitle(title)
    dialog.setText(text)
    dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    dialog.button(QMessageBox.StandardButton.Yes).setText("Да")
    dialog.button(QMessageBox.StandardButton.No).setText("Нет")
    dialog.setDefaultButton(QMessageBox.StandardButton.No)
    try:
        return dialog.exec()
    finally:
        dialog.deleteLater()


class SimilarFacesWorker(QObject):
    """Read only immutable addresses and vectors outside the GUI thread."""

    finished = Signal(object)
    progress = Signal(int)

    def __init__(self, index, directory, seeds, candidates, threshold, cancel):
        super().__init__()
        self.args = index, directory, seeds, candidates, threshold, cancel

    @Slot()
    def run(self) -> None:
        index, directory, seeds, candidates, threshold, cancel = self.args
        try:
            index.load(directory, cancel)
            hits, skipped = index.search(seeds, candidates, threshold, cancel, self.progress.emit)
            result = (hits, skipped, "")
        except SearchCancelled:
            result = ([], 0, "cancelled")
        except Exception as exc:
            logger.exception("Similar face search failed")
            result = ([], 0, str(exc))
        self.finished.emit(result)


class SimilarFacesController(QObject):
    """Keep search lifetime, session identity and confirmation queue together."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.thread = None
        self.worker = None
        self.busy = False
        self.cancel = Event()
        self.index = FaceEmbeddingIndex()
        self.manager = None
        self.progress_dialog = None
        self.result = None

    def reset_session(self) -> None:
        """Discard the previous session's embedding cache."""

        self.cancel.set()
        self.index = FaceEmbeddingIndex()

    def cancel_search(self) -> None:
        """Request cancellation without terminating the worker's thread."""

        self.cancel.set()

    def _status(self, text: str) -> None:
        self.window.status_bar.setRange(0, 1)
        self.window.status_bar.setValue(0)
        self.window.status_bar.setFormat(text)

    def start(self, target_id: str, seeds: dict[str, int]) -> None:
        """Snapshot current faces after the original manual batch has committed."""

        if self.busy or not seeds:
            return
        w = self.window
        manager = w.data_manager
        target = w._get_cluster_item_data_by_id(target_id)
        if not target or not target.get("student_id"):
            return
        try:
            seed_addresses = tuple(
                face_address(fname, pos, manager.records[fname].faces[pos])
                for fname, pos in seeds.items()
            )
            candidates, identities = [], {}
            for fname, record in manager.records.items():
                if fname in seeds or any(f.extra_data.get("is_reference") for f in record.faces):
                    continue
                is_group = (
                    record.face_count > 1 or record.image_type == "group"
                    or not any(f.cluster_label is not None for f in record.faces)
                )
                if not is_group:
                    continue
                for position, face in enumerate(record.faces):
                    if face.is_trash or face.extra_data.get("matched_portrait_cluster_label") is not None:
                        continue
                    address = face_address(fname, position, face)
                    candidates.append(address)
                    identities[(fname, position)] = face
        except (ValueError, KeyError, IndexError) as exc:
            QMessageBox.warning(w, "Поиск похожих лиц", str(exc))
            return
        if not candidates:
            self._status("Других неопознанных лиц для поиска нет.")
            return
        self.busy = True
        self.manager = manager
        self.target_id = target_id
        self.student_id = target["student_id"]
        self.target_name = target["name"]
        self.identities = identities
        self.cancel = Event()
        self.result = None
        self.progress_dialog = QProgressDialog(
            "Поиск похожих неопознанных лиц…", "Отмена", 0, 100, w
        )
        self.progress_dialog.setWindowTitle("Поиск похожих лиц")
        self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.canceled.connect(self.cancel_search)
        self.thread = QThread(self)
        self.worker = SimilarFacesWorker(
            self.index, manager.embeddings_dir, seed_addresses, tuple(candidates),
            w.spin_similarity_distance.value(), self.cancel,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self._receive_result)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.progress_dialog.show()
        self.thread.start()

    @Slot(object)
    def _receive_result(self, result) -> None:
        self.result = result

    @Slot()
    def _thread_finished(self) -> None:
        """Review only after the worker has stopped, including on window close."""

        self.thread.deleteLater()
        self.thread = self.worker = None
        self.progress_dialog.canceled.disconnect(self.cancel_search)
        self.progress_dialog.close()
        self.progress_dialog.deleteLater()
        self.progress_dialog = None
        try:
            if self.cancel.is_set() or self.window.data_manager is not self.manager:
                self._status("Поиск похожих лиц отменён. Ручное назначение сохранено в памяти.")
                return
            hits, skipped, error = self.result or ([], 0, "Поиск не вернул результат.")
            if error:
                if error != "cancelled":
                    QMessageBox.warning(
                        self.window, "Поиск похожих лиц",
                        error + "\nРучное назначение сохранено в памяти.",
                    )
                return
            self._review(hits, skipped)
        except Exception as exc:
            logger.exception("Similar face confirmation failed")
            QMessageBox.warning(
                self.window, "Проверка похожих лиц",
                f"Не удалось завершить проверку: {exc}\n"
                "Ранее подтверждённые назначения сохранены в памяти.",
            )
        finally:
            self.busy = False
            self.identities = {}
            self.manager = None

    def _current_face(self, address):
        """Reject stale results instead of assigning a different face at that index."""

        if self.window.data_manager is not self.manager:
            return None
        record = self.manager.records.get(address.filename)
        if record is None or not 0 <= address.position < len(record.faces):
            return None
        face = record.faces[address.position]
        if (
            face is not self.identities.get((address.filename, address.position))
            or face.is_trash
            or face.extra_data.get("matched_portrait_cluster_label") is not None
        ):
            return None
        return face

    def _review(self, hits, skipped: int) -> None:
        """Confirm one face per photograph without recursively extending samples."""

        w = self.window
        groups = OrderedDict()
        for hit in hits:
            if self._current_face(hit.address) is not None:
                groups.setdefault(hit.address.filename, []).append(hit)
        note = f" Пропущено лиц без корректного эмбеддинга: {skipped}." if skipped else ""
        if not groups:
            self._status("Похожих неопознанных лиц не найдено." + note)
            return
        reply = russian_question(
            w, "Похожие лица",
            f"Найдены похожие неопознанные лица на {len(groups)} фотографиях.\n"
            f"Проверить их для назначения кластеру «{self.target_name}»?" + note,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        assigned = 0
        for number, (filename, group) in enumerate(groups.items(), start=1):
            if self.cancel.is_set() or w.data_manager is not self.manager:
                break
            group = [hit for hit in group if self._current_face(hit.address) is not None]
            if not group:
                continue
            path = w._get_image_path(filename)
            if not path.is_file():
                continue
            instruction = (
                f"Фото {number} из {len(groups)}. Кто на фото — "
                f"<b>{html.escape(self.target_name)}</b>?<br>"
                "Показаны похожие неопознанные лица. Подтвердите выбор."
            )
            record = self.manager.records[filename]
            if any(str(f.extra_data.get("matched_portrait_cluster_label")) == self.target_id for f in record.faces):
                instruction += "<br><b>На фото уже есть лицо целевого кластера.</b>"
            dialog = FaceSelectorDialog(
                path, [self._current_face(hit.address) for hit in group], w, instruction,
                image_cache=w.image_cache, image_loader=w.image_loader,
                review_distances=[hit.distance for hit in group],
            )
            outcome = dialog.exec()
            selected = dialog.get_selected_index()
            dialog.deleteLater()
            if outcome == FaceSelectorDialog.FINISH_REVIEW:
                break
            if outcome != QDialog.DialogCode.Accepted:
                continue
            if not 0 <= selected < len(group):
                continue
            hit = group[selected]
            target = w._get_cluster_item_data_by_id(self.target_id)
            if (
                not target or target.get("student_id") != self.student_id
                or self._current_face(hit.address) is None
            ):
                continue
            record = self.manager.records[filename]
            if any(str(f.extra_data.get("matched_portrait_cluster_label")) == self.target_id for f in record.faces):
                if russian_question(
                    w, "Повторное назначение",
                    f"Фотография: {filename}\n"
                    f"Ученик (целевой кластер): {self.target_name}\n\n"
                    "На этой фотографии другое лицо уже назначено этому ученику.\n"
                    "Вы собираетесь назначить ему ещё и выбранное лицо.\n\n"
                    "«Да» — добавить назначение. Несколько разных лиц на фото будут\n"
                    "отмечены как один и тот же ученик.\n"
                    "«Нет» — пропустить выбранное лицо, оставив его неопознанным.\n\n"
                    "Существующее назначение сохранится в обоих случаях.\n"
                    "Если оно ошибочно, его нужно исправить отдельно.\n\n"
                    "Назначить выбранное лицо этому же ученику?",
                ) != QMessageBox.StandardButton.Yes:
                    continue
            if self.cancel.is_set() or self._current_face(hit.address) is None:
                break
            self.manager.assign_manual_match(
                filename, self.target_id, self.student_id, hit.address.position
            )
            assigned += 1
        w._refresh_left_panel()
        if w.active_cluster_id:
            w._render_gallery(w.active_cluster_id, preserve_state=True)
        QMessageBox.information(
            w, "Проверка похожих лиц",
            f"Найдено фотографий: {len(groups)}. Назначено: {assigned}.\n"
            f"Осталось без назначения: {len(groups) - assigned}.\n"
            "Для записи изменений на диск нажмите «Сохранить»." + note,
        )
