"""Read-only review and publication of personal photo assignments."""

from ..._common.assignment_window import AssignmentWindow
from ..._common.window_base import run_window


class AssignmentsWindow(AssignmentWindow):
    """Publish assignments without editing selections or copying photographs."""

    OPERATION = "build"
    ACTION_TITLE = "Создать план вёрстки"
    WINDOW_STATE_VAR = "win_state.photo_selection04_assignments"


def run_application(config):
    """Open the assignment stage with resolved paths."""
    print("<b>ФОРМИРОВАНИЕ СПИСКА ФАЙЛОВ ДЛЯ ГЕНЕРАЦИИ АЛЬБОМА</b>", flush=True)
    return run_window(AssignmentsWindow, config)
