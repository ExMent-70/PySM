# pysm_lib/gui/widgets/_editors/__init__.py

# Этот файл импортирует все модули с редакторами.
# Сам факт импорта приведет к выполнению декораторов @register_editor
# и заполнит глобальный реестр EDITOR_REGISTRY.

from . import checkbox_editor
from . import choices_editor
from . import date_editor
from . import datetime_editor
from . import dialog_editor
from . import instance_editor
from . import line_edit_editor
from . import list_editor