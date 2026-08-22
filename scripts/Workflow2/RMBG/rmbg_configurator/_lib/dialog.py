"""PySide6 dialog for the first stable RMBG configuration schema."""

from __future__ import annotations

from html import escape
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from _common.config_schema import (
    AUTO_PROFILE_MODELS,
    BackgroundFitMode,
    BackgroundMode,
    BackgroundPosition,
    DeviceName,
    ImageFormat,
    ModelName,
    ModelSelection,
    PrecisionName,
    ProfilePreset,
    RefinementMode,
    RmbgSettings,
    SDMatteVariant,
    default_settings,
)
from _common.image_io import discover_images
from _common.model_registry import ModelRegistry
from _common.path_contract import normalize_model_dir_value, resolve_model_dir_value

from .preview_store import PreviewStore
from .template_store import TemplateStore
from .window_state import RmbgWindowStateStore


PROFILE_LABELS = {
    ProfilePreset.GENERAL_HQ: "Универсальный HQ",
    ProfilePreset.PORTRAIT_HQ: "Портрет и волосы HQ",
    ProfilePreset.TRANSPARENT_HQ: "Прозрачные объекты HQ",
    ProfilePreset.CUSTOM: "Пользовательский",
}


PARAMETER_HINTS = {
    "profile_name": (
        "<b>Назначение.</b> Произвольное понятное название набора настроек, "
        "например «Портреты класса» или «Предметная съёмка».<br><br>"
        "<b>Влияние.</b> Название сохраняется в JSON-контексте и manifest.json, "
        "но само по себе не меняет модель, качество маски или скорость обработки."
    ),
    "profile_preset": (
        "<b>Назначение.</b> Тип изображений определяет автоматическую модель, "
        "если в поле «Модель» выбран автоматический режим.<br><br>"
        "<b>Соответствие.</b> Универсальный HQ → RMBG-2.0; портрет и волосы → "
        "BiRefNet-portrait; прозрачные объекты → Lucida; пользовательский → "
        "BiRefNet-general. При ручном выборе модели этот параметр её не меняет."
    ),
    "model_name": (
        "<b>Назначение.</b> Основная нейросеть создаёт исходную полутоновую "
        "маску объекта.<br><br><b>Автоматически.</b> Модель выбирается по типу "
        "изображений. Любая конкретная модель в списке имеет приоритет над "
        "типом изображений. Для портретов обычно подходит BiRefNet-portrait, "
        "для смешанных объектов — RMBG-2.0 или BiRefNet-general."
    ),
    "model_dir": (
        "<b>Назначение.</b> Корневая папка, в которой хранятся и автоматически "
        "скачиваются модели RMBG, BiRefNet, Lucida и SDMatte.<br><br>"
        "<b>Контракт.</b> Путь является частью JSON-профиля и шаблонов, поэтому "
        "Configurator и Process всегда используют одно и то же хранилище без "
        "отдельного параметра запуска. Папки внутри PySM сохраняются переносимым "
        "путём относительно корня программы, например _BIN/models/RMBG. Нужные "
        "подкаталоги будут созданы при первом скачивании модели."
    ),
    "process_resolution": (
        "<b>Назначение.</b> Рабочее разрешение основной RMBG-модели. Чем оно "
        "выше, тем лучше могут сохраняться тонкие волосы и мелкие детали, но тем "
        "больше время обработки и расход VRAM.<br><br><b>Рекомендация.</b> "
        "Оставьте «Автоматически»: будет использовано проверенное разрешение "
        "выбранной модели. Допустимый ручной диапазон показан выше."
    ),
    "device": (
        "<b>Назначение.</b> Вычислительное устройство для основной модели. "
        "«Автоматически» использует CUDA при доступной совместимой видеокарте, "
        "иначе CPU.<br><br><b>Важно.</b> CUDA значительно быстрее. Для CPU "
        "поддерживается только FP32. SDMatte требует CUDA и на CPU не запускается."
    ),
    "precision": (
        "<b>Назначение.</b> Точность вычислений и хранения тензоров модели. "
        "FP16 обычно быстрее и экономнее по VRAM на CUDA; FP32 расходует больше "
        "памяти, но обеспечивает максимальную численную совместимость; BF16 "
        "зависит от модели и видеокарты.<br><br><b>Рекомендация.</b> "
        "«Автоматически» выбирает FP16 для CUDA и FP32 для CPU. SDMatte "
        "поддерживает FP16/FP32, но не BF16."
    ),
    "sensitivity": (
        "<b>Назначение.</b> Усиливает слабые значения исходной alpha-маски, "
        "помогая сохранить полупрозрачные и неуверенно найденные части объекта."
        "<br><br><b>Как менять.</b> 1,00 не изменяет маску. Уменьшение значения "
        "делает маску чувствительнее и может вернуть тонкие детали, но также "
        "способно захватить остатки фона."
    ),
    "blur": (
        "<b>Назначение.</b> Гауссово размытие применяется к маске до смещения "
        "края и морфологической очистки. Оно сглаживает шум, ступеньки и мелкие "
        "неровности по всей маске.<br><br><b>Как менять.</b> 0 отключает операцию. "
        "Большие значения сильнее сглаживают, но могут потерять волосы и другие "
        "тонкие детали. Для мягкого финального перехода используйте растушёвку."
    ),
    "offset": (
        "<b>Назначение.</b> Сдвигает границу маски морфологическим расширением "
        "или сжатием.<br><br><b>Как менять.</b> Положительное значение расширяет "
        "объект и помогает вернуть срезанный край; отрицательное сжимает маску "
        "и убирает ореол фона. Слишком большой сдвиг разрушает тонкие детали."
    ),
    "feather": (
        "<b>Назначение.</b> Финально смягчает уже очищенную и смещённую границу "
        "маски, создавая плавный переход между объектом и фоном.<br><br>"
        "<b>Отличие от размытия.</b> Размытие выполняется раньше и сглаживает "
        "исходную маску; растушёвка выполняется последней и определяет мягкость "
        "края итогового alpha. 0 сохраняет край без дополнительного смягчения."
    ),
    "fill_holes": (
        "<b>Назначение.</b> Заполняет замкнутые участки фона внутри объекта, "
        "например случайные провалы на одежде или корпусе предмета.<br><br>"
        "<b>Важно.</b> Отверстия, соединённые с внешним фоном, не заполняются. "
        "Для кружева, колец и других объектов с настоящими отверстиями режим "
        "может быть нежелателен."
    ),
    "max_hole_area": (
        "<b>Назначение.</b> Максимальная площадь замкнутого отверстия в пикселях, "
        "которое разрешено заполнить.<br><br><b>Как менять.</b> Меньший порог "
        "исправляет только мелкие дефекты. Значение 0 означает «без ограничения» "
        "и заполняет все замкнутые отверстия независимо от размера. Поле "
        "учитывается только при включённом заполнении отверстий."
    ),
    "remove_small_regions": (
        "<b>Назначение.</b> Удаляет отдельные небольшие компоненты переднего плана, "
        "которые модель ошибочно приняла за объект.<br><br><b>Важно.</b> Режим "
        "может удалить настоящие, но изолированные мелкие детали — блики, ремешки "
        "или отделившиеся пряди. Результат зависит от порога площади ниже."
    ),
    "min_region_area": (
        "<b>Назначение.</b> Минимальная площадь связной области переднего плана "
        "в пикселях. Все отдельные области меньшего размера удаляются.<br><br>"
        "<b>Как менять.</b> Увеличивайте для агрессивной очистки мусора, уменьшайте "
        "для сохранения мелких деталей. Значение 0 фактически отключает порог."
    ),
    "invert": (
        "<b>Назначение.</b> Меняет местами объект и фон в самом конце обработки: "
        "белое становится чёрным, чёрное — белым, полутоновые значения также "
        "инвертируются.<br><br><b>Использование.</b> Включайте, если нужна маска "
        "фона вместо маски объекта. Инверсия влияет на все сохраняемые результаты."
    ),
    "refinement": (
        "<b>Назначение.</b> Дополнительное уточнение выполняется после основной "
        "RMBG-модели и до обычной постобработки.<br><br><b>Режимы.</b> "
        "«Автоматически» сейчас выбирает быстрое CPU-уточнение; «Без refinement» "
        "оставляет исходную маску; «Быстрый» укрепляет уверенный фон и объект "
        "без дополнительной модели; SDMatte HQ строит более точный alpha matte, "
        "но требует CUDA и загрузки checkpoint около 5,2 ГБ."
    ),
    "sdmatte_variant": (
        "<b>Назначение.</b> Выбирает checkpoint модели SDMatte. Стандартный "
        "вариант — базовый и проверенный режим. Plus использует альтернативные "
        "веса и может иначе обрабатывать сложные или полупрозрачные края."
        "<br><br><b>Рекомендация.</b> Начинайте со стандартного варианта; Plus "
        "имеет смысл сравнивать на нескольких типичных изображениях вашего набора."
    ),
    "sdmatte_resolution": (
        "<b>Назначение.</b> Отдельное рабочее разрешение этапа SDMatte; оно не "
        "заменяет разрешение основной RMBG-модели.<br><br><b>Как влияет.</b> "
        "Большее значение может точнее восстановить волосы и полупрозрачные "
        "границы, но заметно увеличивает время и расход VRAM. 1024 — рекомендуемый "
        "баланс; 512 быстрее, 1536–2048 следует использовать только при наличии "
        "достаточного запаса видеопамяти."
    ),
    "sdmatte_transparent_object": (
        "<b>Назначение.</b> Сообщает SDMatte, что сам объект может содержать "
        "прозрачные или полупрозрачные области.<br><br><b>Включайте</b> для стекла, "
        "сетки, дыма, вуали, тонкой ткани и подобных материалов. Для обычного "
        "непрозрачного портрета флажок можно снять, чтобы не создавать лишнюю "
        "полупрозрачность внутри объекта."
    ),
    "sdmatte_constraint": (
        "<b>Назначение.</b> Определяет, насколько строго SDMatte обязан сохранять "
        "уверенные участки исходной RMBG-маски.<br><br><b>Как менять.</b> При 0,90 "
        "только почти уверенный фон и объект фиксируются, а широкая пограничная "
        "зона остаётся доступной для уточнения. Уменьшение значения сильнее "
        "ограничивает SDMatte исходной маской; увеличение даёт модели больше "
        "свободы, но может изменить уже правильно найденные участки."
    ),
    "save_cutout": (
        "<b>Результат.</b> Сохраняет отделённый объект в папку Cutout. PNG и "
        "lossless WebP содержат alpha-канал; JPEG прозрачность не поддерживает, "
        "поэтому объект накладывается на выбранный цвет.<br><br><b>Важно.</b> "
        "Отключение этого флажка не отключает создание маски или composite."
    ),
    "save_mask": (
        "<b>Результат.</b> Сохраняет полутоновую alpha-маску в папку Masks: белый "
        "означает объект, чёрный — фон, серый — частичную прозрачность.<br><br>"
        "<b>Формат.</b> Маска всегда записывается как 16-битный PNG независимо "
        "от выбранного формата итоговых изображений."
    ),
    "save_composite": (
        "<b>Результат.</b> Сохраняет объект, уже наложенный на новый фон, в папку "
        "Composite.<br><br><b>Фон.</b> После включения выберите сплошной цвет или "
        "конкретное изображение из background_dir. Cutout и composite являются "
        "независимыми результатами и могут сохраняться одновременно."
    ),
    "background_mode": (
        "<b>Назначение.</b> Определяет источник фона только для composite. "
        "«Сплошной цвет» использует цвет ниже; «Изображение из background_dir» "
        "использует один выбранный файл для всей пакетной обработки.<br><br>"
        "Параметр не влияет на PNG/WebP cutout с прозрачностью."
    ),
    "background_color": (
        "<b>Назначение.</b> Цвет в формате #RRGGBB. Он используется как сплошной "
        "фон composite, как цвет свободных полей в режиме contain и как фон "
        "непрозрачного JPEG cutout.<br><br><b>Пример.</b> #FFFFFF — белый, "
        "#000000 — чёрный. Кнопка «Выбрать» открывает стандартную палитру."
    ),
    "background_image": (
        "<b>Назначение.</b> Выбирает один файл внутри background_dir, который "
        "используется как фон для всех исходных изображений текущей пакетной "
        "обработки.<br><br><b>Хранение.</b> В профиль записывается относительный "
        "путь, поэтому RMBG Process должен получить совместимую папку "
        "background_dir с этим файлом."
    ),
    "background_fit": (
        "<b>Назначение.</b> Определяет подгонку фонового изображения к размеру "
        "исходника.<br><br><b>Cover</b> сохраняет пропорции, полностью заполняет "
        "кадр и обрезает лишнее. <b>Contain</b> показывает фон целиком, а свободные "
        "поля заполняет выбранным цветом. <b>Растянуть</b> заполняет кадр без "
        "сохранения пропорций. Во всех режимах используется Lanczos."
    ),
    "background_position": (
        "<b>Назначение.</b> Выбирает, какая часть фонового изображения сохраняется "
        "при обрезке в режиме cover: центр, верх, низ, левый или правый край."
        "<br><br>В режимах contain и «Растянуть» параметр не используется и "
        "автоматически отключается."
    ),
    "image_suffix": (
        "<b>Назначение.</b> Текст, добавляемый к имени исходного файла при "
        "сохранении cutout в папку Cutout. Например, photo.jpg при суффиксе "
        "_rmbg станет photo_rmbg.png или photo_rmbg.jpg.<br><br>"
        "Суффикс не должен содержать запрещённые в Windows символы пути."
    ),
    "mask_suffix": (
        "<b>Назначение.</b> Текст, добавляемый к имени 16-битной маски в папке "
        "Masks. Например, photo.jpg при суффиксе _mask станет photo_mask.png."
        "<br><br>Используйте отдельный суффикс, чтобы маска не конфликтовала "
        "с другими результатами."
    ),
    "composite_suffix": (
        "<b>Назначение.</b> Текст, добавляемый к имени изображения с новым фоном "
        "в папке Composite. Например, photo.jpg при суффиксе _composite станет "
        "photo_composite.png.<br><br>Суффикс не меняет содержимое изображения."
    ),
    "image_format": (
        "<b>Назначение.</b> Формат cutout и composite. PNG сохраняет alpha и "
        "поддерживает регулируемое lossless-сжатие; WebP lossless обычно даёт "
        "меньший файл и также сохраняет alpha; JPEG использует сжатие с потерями "
        "и не поддерживает прозрачность.<br><br>Полутоновая маска при любом "
        "выборе остаётся 16-битным PNG."
    ),
    "png_compress_level": (
        "<b>Назначение.</b> Уровень lossless-сжатия PNG от 0 до 9. Он меняет "
        "время записи и размер файла, но не качество пикселей.<br><br>"
        "<b>Рекомендация.</b> 3 — быстрый сбалансированный вариант. 0 записывает "
        "быстрее и создаёт крупные файлы; 9 сжимает сильнее, но заметно медленнее. "
        "Поле активно только для PNG."
    ),
    "jpeg_quality": (
        "<b>Назначение.</b> Качество JPEG от 1 до 100. Более высокое значение "
        "лучше сохраняет волосы и контрастные края, но увеличивает файл."
        "<br><br><b>Рекомендация.</b> 95 — высококачественный вариант. JPEG "
        "использует сжатие с потерями; настройка не применяется к 16-битной маске."
    ),
    "io_workers": (
        "<b>Назначение.</b> Количество CPU-потоков для чтения, постобработки и "
        "записи файлов. Нейросеть по-прежнему обрабатывает изображения "
        "последовательно с GPU batch=1.<br><br><b>Рекомендация.</b> 4 — обычный "
        "баланс. 2 экономит RAM; 6–8 может ускорить PNG на многоядерном CPU. "
        "Слишком большое значение повышает расход памяти и не всегда ускоряет работу."
    ),
}


class RmbgConfiguratorDialog(QDialog):
    """Edit the supported settings while preserving one validated JSON object."""

    def __init__(
        self,
        settings: RmbgSettings,
        registry: ModelRegistry,
        *,
        upstream_label: str,
        background_dir: Path | None = None,
        test_root: Path | None = None,
        window_state_store: RmbgWindowStateStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._window_state_store = window_state_store
        self._background_dir = background_dir.resolve() if background_dir else None
        self._background_images = self._discover_background_images()
        self._accepted_settings: RmbgSettings | None = None
        script_root = Path(__file__).resolve().parents[1]
        project_root = Path(__file__).resolve().parents[5]
        self._template_store = TemplateStore(script_root / "rmbg_templates.json")
        self._preview_store = PreviewStore(
            test_root.resolve()
            if test_root is not None
            else project_root.parents[1] / "tmp" / "Masks"
        )
        self.setWindowTitle("RMBG Configurator")
        self.setMinimumSize(680, 620)

        root_layout = QVBoxLayout(self)
        title = QLabel("Высококачественное удаление фона и сегментация")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        root_layout.addWidget(title)

        version_label = QLabel(upstream_label)
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setWordWrap(True)
        root_layout.addWidget(version_label)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)
        self._build_main_tab()
        self._build_mask_tab()
        self._build_output_tab()
        self._build_performance_tab()
        self._install_parameter_tooltips()

        buttons_layout = QHBoxLayout()
        self.templates_button = QPushButton("Шаблоны…")
        self.templates_button.setToolTip(
            "Сохранить все текущие параметры как именованный шаблон с описанием "
            "или загрузить ранее сохранённый шаблон."
        )
        self.templates_button.clicked.connect(self._open_templates)
        buttons_layout.addWidget(self.templates_button)
        self.test_masks_button = QPushButton("Тест масок…")
        self.test_masks_button.setToolTip(
            "Создать нумерованные тестовые наборы из текущих параметров и "
            "сравнить маски на одинаковых исходных изображениях."
        )
        self.test_masks_button.clicked.connect(self._open_test_masks)
        buttons_layout.addWidget(self.test_masks_button)
        self.reset_button = QPushButton("Рекомендуемые настройки")
        self.reset_button.setToolTip(
            "Восстанавливает рекомендуемый профиль RMBG во всех вкладках. "
            "Изменения попадут в контекст только после нажатия «Сохранить»."
        )
        self.reset_button.clicked.connect(self._reset_defaults)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addStretch(1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setText(
            "Сохранить"
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(
            "Отмена"
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setToolTip(
            "Проверить параметры и сохранить весь RMBG-профиль в JSON-переменную "
            "контекста PySM."
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setToolTip(
            "Закрыть конфигуратор, не изменяя RMBG-профиль в контексте PySM."
        )
        self.button_box.accepted.connect(self._validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        buttons_layout.addWidget(self.button_box)
        root_layout.addLayout(buttons_layout)

        self._load_settings(settings)
        if self._window_state_store is not None:
            self._window_state_store.restore("configurator", self)

    def _build_main_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        profile_group = QGroupBox("Профиль")
        profile_form = QFormLayout(profile_group)
        self.profile_name = QLineEdit()
        profile_form.addRow("Название профиля:", self.profile_name)
        self.profile_preset = QComboBox()
        for value, label in PROFILE_LABELS.items():
            self.profile_preset.addItem(label, value.value)
        profile_form.addRow("Тип изображений:", self.profile_preset)
        layout.addWidget(profile_group)

        model_group = QGroupBox("Модель и вычисления")
        model_form = QFormLayout(model_group)
        self.model_name = QComboBox()
        self.model_name.addItem("Автоматически по типу изображений", "auto")
        for descriptor in self._registry.descriptors():
            self.model_name.addItem(descriptor.display_name, descriptor.model_id.value)
        model_form.addRow("Модель:", self.model_name)
        self.effective_model = QLabel()
        self.effective_model.setWordWrap(True)
        model_form.addRow("Будет использована:", self.effective_model)

        model_dir_widget = QWidget()
        model_dir_layout = QHBoxLayout(model_dir_widget)
        model_dir_layout.setContentsMargins(0, 0, 0, 0)
        self.model_dir = QLineEdit()
        self.model_dir_button = QPushButton("Выбрать…")
        self.model_dir_button.clicked.connect(self._choose_model_dir)
        model_dir_layout.addWidget(self.model_dir, 1)
        model_dir_layout.addWidget(self.model_dir_button)
        self.model_dir_row = model_dir_widget
        model_form.addRow("Папка моделей:", model_dir_widget)

        self.process_resolution = QSpinBox()
        self.process_resolution.setRange(0, 4096)
        self.process_resolution.setSingleStep(32)
        self.process_resolution.setSpecialValueText("Автоматически")
        model_form.addRow("Разрешение обработки:", self.process_resolution)

        self.device = QComboBox()
        self.device.addItem("Автоматически", DeviceName.AUTO.value)
        self.device.addItem("CUDA", DeviceName.CUDA.value)
        self.device.addItem("CPU", DeviceName.CPU.value)
        model_form.addRow("Устройство:", self.device)

        self.precision = QComboBox()
        self.precision.addItem("Автоматически", PrecisionName.AUTO.value)
        self.precision.addItem("FP32", PrecisionName.FP32.value)
        self.precision.addItem("FP16", PrecisionName.FP16.value)
        self.precision.addItem("BF16", PrecisionName.BF16.value)
        model_form.addRow("Точность:", self.precision)

        layout.addWidget(model_group)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Основное")

        self.profile_preset.currentIndexChanged.connect(
            self._update_effective_model_label
        )
        self.model_name.currentIndexChanged.connect(
            self._update_effective_model_label
        )

    def _build_mask_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)

        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.0, 1.0)
        self.sensitivity.setSingleStep(0.05)
        self.sensitivity.setDecimals(2)
        form.addRow("Чувствительность:", self.sensitivity)

        self.blur = QSpinBox()
        self.blur.setRange(0, 64)
        form.addRow("Размытие:", self.blur)

        self.offset = QSpinBox()
        self.offset.setRange(-20, 20)
        form.addRow("Смещение края:", self.offset)

        self.feather = QSpinBox()
        self.feather.setRange(0, 64)
        form.addRow("Растушёвка:", self.feather)

        self.fill_holes = QCheckBox("Заполнять отверстия")
        form.addRow("", self.fill_holes)
        self.max_hole_area = QSpinBox()
        self.max_hole_area.setRange(0, 1_000_000)
        self.max_hole_area.setSpecialValueText("Без ограничения")
        form.addRow("Макс. площадь отверстия:", self.max_hole_area)
        self.remove_small_regions = QCheckBox("Удалять мелкие области")
        form.addRow("", self.remove_small_regions)
        self.min_region_area = QSpinBox()
        self.min_region_area.setRange(0, 1_000_000)
        form.addRow("Мин. площадь области:", self.min_region_area)
        self.invert = QCheckBox("Инвертировать маску")
        form.addRow("", self.invert)

        self.refinement = QComboBox()
        self.refinement.addItem(
            "Автоматически (быстрое уточнение)",
            RefinementMode.AUTO.value,
        )
        self.refinement.addItem("Без refinement", RefinementMode.NONE.value)
        self.refinement.addItem("Быстрый", RefinementMode.FAST.value)
        self.refinement.addItem("SDMatte HQ", RefinementMode.SDMATTE.value)
        form.addRow("Уточнение края:", self.refinement)

        self.sdmatte_variant = QComboBox()
        self.sdmatte_variant.addItem(
            "SDMatte — стандартная",
            SDMatteVariant.STANDARD.value,
        )
        self.sdmatte_variant.addItem(
            "SDMatte Plus",
            SDMatteVariant.PLUS.value,
        )
        self.sdmatte_variant_label = QLabel("Вариант SDMatte:")
        form.addRow(self.sdmatte_variant_label, self.sdmatte_variant)

        self.sdmatte_resolution = QSpinBox()
        self.sdmatte_resolution.setRange(256, 2048)
        self.sdmatte_resolution.setSingleStep(8)
        self.sdmatte_resolution_label = QLabel("Разрешение SDMatte:")
        form.addRow(self.sdmatte_resolution_label, self.sdmatte_resolution)

        self.sdmatte_transparent_object = QCheckBox(
            "Учитывать прозрачные объекты"
        )
        self.sdmatte_transparent_object_label = QLabel("")
        form.addRow(
            self.sdmatte_transparent_object_label,
            self.sdmatte_transparent_object,
        )

        self.sdmatte_constraint = QDoubleSpinBox()
        self.sdmatte_constraint.setRange(0.1, 1.0)
        self.sdmatte_constraint.setSingleStep(0.05)
        self.sdmatte_constraint.setDecimals(2)
        self.sdmatte_constraint_label = QLabel("Строгость исходной маски:")
        form.addRow(self.sdmatte_constraint_label, self.sdmatte_constraint)

        sdmatte_note = QLabel(
            "SDMatte уточняет маску, созданную основной RMBG-моделью. "
            "Первый запуск автоматически скачает веса и компоненты с progress bar."
        )
        sdmatte_note.setWordWrap(True)
        self.sdmatte_note = sdmatte_note
        self.sdmatte_note_label = QLabel("")
        form.addRow(self.sdmatte_note_label, self.sdmatte_note)
        self.tabs.addTab(tab, "Маска")

        self.refinement.currentIndexChanged.connect(
            self._update_mask_controls
        )

    def _build_output_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.save_cutout = QCheckBox("Изображение с прозрачностью")
        self.save_mask = QCheckBox("Полутоновая маска")
        self.save_composite = QCheckBox("Изображение с новым фоном")
        form.addRow("Сохранять:", self.save_cutout)
        form.addRow("", self.save_mask)
        form.addRow("", self.save_composite)

        self.background_mode = QComboBox()
        self.background_mode.addItem("Сплошной цвет", BackgroundMode.SOLID.value)
        self.background_mode.addItem(
            "Изображение из background_dir", BackgroundMode.IMAGE.value
        )
        form.addRow("Режим фона:", self.background_mode)

        color_widget = QWidget()
        color_row = QHBoxLayout(color_widget)
        color_row.setContentsMargins(0, 0, 0, 0)
        self.background_color = QLineEdit()
        self.color_button = QPushButton("Выбрать")
        self.color_button.clicked.connect(self._choose_color)
        color_row.addWidget(self.background_color, 1)
        color_row.addWidget(self.color_button)
        self.background_color_row = color_widget
        self.background_color_label = QLabel("Цвет фона:")
        form.addRow(self.background_color_label, color_widget)

        self.background_image = QComboBox()
        self.background_image.addItem("— Выберите изображение —", "")
        for relative_path in self._background_images:
            self.background_image.addItem(relative_path, relative_path)
        form.addRow("Фоновое изображение:", self.background_image)

        self.background_fit = QComboBox()
        self.background_fit.addItem(
            "Заполнить с обрезкой (cover)", BackgroundFitMode.COVER.value
        )
        self.background_fit.addItem(
            "Вписать полностью (contain)", BackgroundFitMode.CONTAIN.value
        )
        self.background_fit.addItem(
            "Растянуть", BackgroundFitMode.STRETCH.value
        )
        form.addRow("Размещение фона:", self.background_fit)

        self.background_position = QComboBox()
        self.background_position.addItem("Центр", BackgroundPosition.CENTER.value)
        self.background_position.addItem("Верх", BackgroundPosition.TOP.value)
        self.background_position.addItem("Низ", BackgroundPosition.BOTTOM.value)
        self.background_position.addItem("Лево", BackgroundPosition.LEFT.value)
        self.background_position.addItem("Право", BackgroundPosition.RIGHT.value)
        form.addRow("Положение обрезки:", self.background_position)

        background_note = QLabel(
            "Выбранное фоновое изображение применяется ко всей пакетной обработке."
        )
        background_note.setWordWrap(True)
        form.addRow("", background_note)

        self.image_suffix = QLineEdit()
        self.mask_suffix = QLineEdit()
        self.composite_suffix = QLineEdit()
        form.addRow("Суффикс изображения:", self.image_suffix)
        form.addRow("Суффикс маски:", self.mask_suffix)
        form.addRow("Суффикс composite:", self.composite_suffix)
        self.image_format = QComboBox()
        self.image_format.addItem("PNG", ImageFormat.PNG.value)
        self.image_format.addItem("WebP lossless", ImageFormat.WEBP.value)
        self.image_format.addItem("JPEG", ImageFormat.JPEG.value)
        form.addRow("Формат итоговых изображений:", self.image_format)
        self.png_compress_level = QSpinBox()
        self.png_compress_level.setRange(0, 9)
        form.addRow("Сжатие PNG (0–9):", self.png_compress_level)
        self.jpeg_quality = QSpinBox()
        self.jpeg_quality.setRange(1, 100)
        form.addRow("Качество JPEG (1–100):", self.jpeg_quality)
        self.tabs.addTab(tab, "Результат")

        self.save_cutout.toggled.connect(self._update_output_controls)
        self.save_composite.toggled.connect(self._update_output_controls)
        self.background_mode.currentIndexChanged.connect(
            self._update_output_controls
        )
        self.background_fit.currentIndexChanged.connect(
            self._update_output_controls
        )
        self.image_format.currentIndexChanged.connect(
            self._update_output_controls
        )

    def _build_performance_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.io_workers = QSpinBox()
        self.io_workers.setRange(1, 32)
        form.addRow("Потоки чтения/записи:", self.io_workers)
        runtime_note = QLabel(
            "GPU inference выполняется последовательно с batch=1. В каждом "
            "запуске используется одна основная модель, которая автоматически "
            "выгружается после завершения."
        )
        runtime_note.setWordWrap(True)
        form.addRow("", runtime_note)
        self.tabs.addTab(tab, "Производительность")

    @staticmethod
    def _apply_tooltip(widget: QWidget, hint: str) -> None:
        """Apply a readable rich tooltip long enough for detailed guidance."""

        rich_hint = (
            "<qt><table width='520' cellspacing='0' cellpadding='0'><tr><td>"
            f"{hint}</td></tr></table></qt>"
        )
        widget.setToolTip(rich_hint)
        widget.setToolTipDuration(60_000)

    def _apply_parameter_tooltip(
        self,
        widget: QWidget,
        hint: str,
        *,
        form_field: QWidget | None = None,
    ) -> None:
        """Add the same hint to a parameter control and its form label."""

        field = form_field if form_field is not None else widget
        self._apply_tooltip(widget, hint)
        if field is not widget:
            self._apply_tooltip(field, hint)
        parent = field.parentWidget()
        layout = parent.layout() if parent is not None else None
        if isinstance(layout, QFormLayout):
            label = layout.labelForField(field)
            if label is not None:
                self._apply_tooltip(label, hint)

    def _install_parameter_tooltips(self) -> None:
        """Cover every editable schema parameter with detailed GUI guidance."""

        background_hint = PARAMETER_HINTS["background_image"]
        if self._background_dir is None:
            background_hint += (
                "<br><br><b>Текущее состояние.</b> Параметр background_dir не "
                "передан конфигуратору, поэтому выбрать файл сейчас нельзя."
            )
        elif not self._background_images:
            background_hint += (
                "<br><br><b>Текущее состояние.</b> В указанной папке не найдены "
                "поддерживаемые изображения: "
                f"{escape(str(self._background_dir))}"
            )

        for name, hint in PARAMETER_HINTS.items():
            if name in {"background_color", "background_image", "model_dir"}:
                continue
            self._apply_parameter_tooltip(getattr(self, name), hint)

        color_hint = PARAMETER_HINTS["background_color"]
        self._apply_parameter_tooltip(
            self.background_color,
            color_hint,
            form_field=self.background_color_row,
        )
        self._apply_tooltip(self.color_button, color_hint)
        self._apply_parameter_tooltip(self.background_image, background_hint)
        model_dir_hint = PARAMETER_HINTS["model_dir"]
        self._apply_parameter_tooltip(
            self.model_dir,
            model_dir_hint,
            form_field=self.model_dir_row,
        )
        self._apply_tooltip(self.model_dir_button, model_dir_hint)

        self._apply_tooltip(
            self.effective_model,
            "<b>Информация.</b> Здесь показана модель, которая фактически будет "
            "использована с учётом автоматического профиля или ручного выбора, "
            "а также её рекомендуемое и допустимое разрешение.",
        )

    def _load_settings(self, settings: RmbgSettings) -> None:
        # Task type and segmentation are reserved by schema v1 for later model
        # stages and have no controls in this dialog yet. Preserve them when a
        # template or context profile is loaded instead of silently resetting
        # hidden values when the visible settings are saved.
        self._task_type = settings.task.type
        self._segmentation_settings = settings.segmentation.model_copy(deep=True)
        self.profile_name.setText(settings.profile_name)
        self._set_combo(self.profile_preset, settings.task.preset.value)
        selected_model = (
            "auto"
            if settings.model.selection == ModelSelection.AUTO
            else settings.model.name.value
        )
        self._set_combo(self.model_name, selected_model)
        self.model_dir.setText(settings.model.model_dir)
        self.process_resolution.setValue(settings.model.process_resolution)
        self._set_combo(self.device, settings.model.device.value)
        self._set_combo(self.precision, settings.model.precision.value)
        self._update_effective_model_label()

        self.sensitivity.setValue(settings.mask.sensitivity)
        self.blur.setValue(settings.mask.blur)
        self.offset.setValue(settings.mask.offset)
        self.feather.setValue(settings.mask.feather)
        self.fill_holes.setChecked(settings.mask.fill_holes)
        self.max_hole_area.setValue(settings.mask.max_hole_area)
        self.remove_small_regions.setChecked(settings.mask.remove_small_regions)
        self.min_region_area.setValue(settings.mask.min_region_area)
        self.invert.setChecked(settings.mask.invert)
        self._set_combo(self.refinement, settings.mask.refinement.value)
        self._set_combo(
            self.sdmatte_variant,
            settings.mask.sdmatte_variant.value,
        )
        self.sdmatte_resolution.setValue(settings.mask.sdmatte_resolution)
        self.sdmatte_transparent_object.setChecked(
            settings.mask.sdmatte_transparent_object
        )
        self.sdmatte_constraint.setValue(settings.mask.sdmatte_constraint)
        self._update_mask_controls()

        self.save_cutout.setChecked(settings.output.save_cutout)
        self.save_mask.setChecked(settings.output.save_mask)
        self.save_composite.setChecked(settings.output.save_composite)
        background_mode = settings.output.background_mode
        if background_mode not in {BackgroundMode.SOLID, BackgroundMode.IMAGE}:
            background_mode = BackgroundMode.SOLID
        self._set_combo(self.background_mode, background_mode.value)
        self.background_color.setText(settings.output.background_color)
        self._set_combo(self.background_image, settings.output.background_image)
        self._set_combo(self.background_fit, settings.output.background_fit.value)
        self._set_combo(
            self.background_position, settings.output.background_position.value
        )
        self.image_suffix.setText(settings.output.image_suffix)
        self.mask_suffix.setText(settings.output.mask_suffix)
        self.composite_suffix.setText(settings.output.composite_suffix)
        self._set_combo(self.image_format, settings.output.image_format.value)
        self.png_compress_level.setValue(settings.output.png_compress_level)
        self.jpeg_quality.setValue(settings.output.jpeg_quality)

        self.io_workers.setValue(settings.performance.io_workers)
        self._update_output_controls()

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _reset_defaults(self) -> None:
        self._load_settings(default_settings())

    def _open_templates(self) -> None:
        """Open the persistent named-template manager without committing context."""

        from .template_dialog import TemplateManagerDialog

        dialog = TemplateManagerDialog(
            self._template_store,
            self.build_settings,
            window_state_store=self._window_state_store,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.selected_settings is not None:
                self._load_settings(dialog.selected_settings)

    def _open_test_masks(self) -> None:
        """Open cached test generation and visual comparison for current settings."""

        from .preview_dialog import TestMaskManagerDialog

        TestMaskManagerDialog(
            self._preview_store,
            self._template_store,
            self.build_settings,
            self._load_settings,
            window_state_store=self._window_state_store,
            parent=self,
        ).exec()

    def _update_effective_model_label(self) -> None:
        """Explain the effective model and the precedence rule directly in GUI."""

        selected_model = self.model_name.currentData()
        preset_data = self.profile_preset.currentData()
        if selected_model is None or preset_data is None:
            return
        if selected_model == "auto":
            preset = ProfilePreset(preset_data)
            model_id = AUTO_PROFILE_MODELS[preset]
            source = "автоматически по типу изображений"
        else:
            model_id = ModelName(selected_model)
            source = "выбрана вручную; тип изображений модель не меняет"
        descriptor = self._registry.get(model_id)
        self.effective_model.setText(
            f"{descriptor.display_name} — {source}. Рекомендуется "
            f"{descriptor.default_resolution}; допустимо "
            f"{descriptor.min_resolution}–{descriptor.max_resolution}."
        )

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self.background_color.setText(color.name().upper())

    def _choose_model_dir(self) -> None:
        """Select the shared model store persisted in the RMBG profile."""

        initial = self.model_dir.text().strip()
        try:
            initial_dir = (
                str(resolve_model_dir_value(initial))
                if initial
                else str(Path.cwd())
            )
        except ValueError:
            initial_dir = str(Path(__file__).resolve().parents[5])
        selected = QFileDialog.getExistingDirectory(
            self,
            "Папка хранения моделей RMBG",
            initial_dir,
        )
        if selected:
            try:
                portable = normalize_model_dir_value(selected)
            except ValueError as exc:
                QMessageBox.warning(self, "Некорректная папка моделей", str(exc))
                return
            self.model_dir.setText(portable)

    def _discover_background_images(self) -> tuple[str, ...]:
        """Return GUI choices as portable paths relative to background_dir."""

        if self._background_dir is None or not self._background_dir.is_dir():
            return ()
        return tuple(
            path.resolve().relative_to(self._background_dir).as_posix()
            for path in discover_images(self._background_dir, recursive=True)
        )

    def _update_output_controls(self) -> None:
        """Enable only controls that affect the selected composite mode."""

        composite_enabled = self.save_composite.isChecked()
        image_mode = self.background_mode.currentData() == BackgroundMode.IMAGE.value
        fit_mode = self.background_fit.currentData()
        use_color = not image_mode or fit_mode == BackgroundFitMode.CONTAIN.value
        image_format = self.image_format.currentData()
        jpeg_selected = image_format == ImageFormat.JPEG.value
        jpeg_cutout = self.save_cutout.isChecked() and jpeg_selected
        color_enabled = (composite_enabled and use_color) or jpeg_cutout

        self.background_mode.setEnabled(composite_enabled)
        self.background_image.setEnabled(composite_enabled and image_mode)
        self.background_fit.setEnabled(composite_enabled and image_mode)
        self.background_position.setEnabled(
            composite_enabled
            and image_mode
            and fit_mode == BackgroundFitMode.COVER.value
        )
        self.background_color.setEnabled(color_enabled)
        self.color_button.setEnabled(color_enabled)
        self.save_cutout.setText(
            "Изображение без прозрачности (JPG)"
            if jpeg_selected
            else "Изображение с прозрачностью"
        )
        self.background_color_label.setText(
            "Цвет фона JPG / свободных полей:"
            if jpeg_cutout
            and composite_enabled
            and image_mode
            and fit_mode == BackgroundFitMode.CONTAIN.value
            else (
                "Цвет фона JPG:"
                if jpeg_cutout
                else (
                    "Цвет свободных полей:"
                    if image_mode and fit_mode == BackgroundFitMode.CONTAIN.value
                    else "Цвет фона:"
                )
            )
        )
        self.png_compress_level.setEnabled(image_format == ImageFormat.PNG.value)
        self.jpeg_quality.setEnabled(jpeg_selected)

    def _update_mask_controls(self) -> None:
        """Show SDMatte-specific controls only for HQ refinement."""

        visible = self.refinement.currentData() == RefinementMode.SDMATTE.value
        for widget in (
            self.sdmatte_variant_label,
            self.sdmatte_variant,
            self.sdmatte_resolution_label,
            self.sdmatte_resolution,
            self.sdmatte_transparent_object_label,
            self.sdmatte_transparent_object,
            self.sdmatte_constraint_label,
            self.sdmatte_constraint,
            self.sdmatte_note_label,
            self.sdmatte_note,
        ):
            widget.setVisible(visible)

    def _validate_and_accept(self) -> None:
        try:
            self._accepted_settings = self.build_settings()
        except Exception as exc:
            QMessageBox.warning(self, "Некорректные настройки", str(exc))
            return
        self.accept()

    def done(self, result: int) -> None:
        """Persist the complete dialog family when the main window closes."""

        if self._window_state_store is not None:
            self._window_state_store.save(
                "configurator",
                self,
                commit=True,
            )
        super().done(result)

    @property
    def accepted_settings(self) -> RmbgSettings:
        if self._accepted_settings is None:
            raise RuntimeError("Настройки не были подтверждены пользователем.")
        return self._accepted_settings

    def build_settings(self) -> RmbgSettings:
        selected_model = self.model_name.currentData()
        model_selection = (
            ModelSelection.AUTO.value
            if selected_model == "auto"
            else ModelSelection.MANUAL.value
        )
        model_name = None if selected_model == "auto" else selected_model
        if (
            self.save_composite.isChecked()
            and self.background_mode.currentData() == BackgroundMode.IMAGE.value
            and not self.background_image.currentData()
        ):
            raise ValueError(
                "Для изображения с новым фоном выберите файл из background_dir."
            )
        settings = RmbgSettings.model_validate(
            {
                "schema_version": 1,
                "profile_name": self.profile_name.text(),
                "task": {
                    "type": self._task_type.value,
                    "preset": self.profile_preset.currentData(),
                },
                "model": {
                    "selection": model_selection,
                    "name": model_name,
                    "model_dir": self.model_dir.text(),
                    "process_resolution": self.process_resolution.value(),
                    "device": self.device.currentData(),
                    "precision": self.precision.currentData(),
                    "unload_after_run": True,
                },
                "segmentation": self._segmentation_settings.model_dump(mode="json"),
                "mask": {
                    "sensitivity": self.sensitivity.value(),
                    "blur": self.blur.value(),
                    "offset": self.offset.value(),
                    "feather": self.feather.value(),
                    "fill_holes": self.fill_holes.isChecked(),
                    "max_hole_area": self.max_hole_area.value(),
                    "remove_small_regions": self.remove_small_regions.isChecked(),
                    "min_region_area": self.min_region_area.value(),
                    "invert": self.invert.isChecked(),
                    "refinement": self.refinement.currentData(),
                    "sdmatte_variant": self.sdmatte_variant.currentData(),
                    "sdmatte_resolution": self.sdmatte_resolution.value(),
                    "sdmatte_transparent_object": (
                        self.sdmatte_transparent_object.isChecked()
                    ),
                    "sdmatte_constraint": self.sdmatte_constraint.value(),
                },
                "output": {
                    "save_cutout": self.save_cutout.isChecked(),
                    "save_mask": self.save_mask.isChecked(),
                    "save_composite": self.save_composite.isChecked(),
                    "background_mode": self.background_mode.currentData(),
                    "background_color": self.background_color.text(),
                    "background_image": self.background_image.currentData() or "",
                    "background_fit": self.background_fit.currentData(),
                    "background_position": self.background_position.currentData(),
                    "image_suffix": self.image_suffix.text(),
                    "mask_suffix": self.mask_suffix.text(),
                    "composite_suffix": self.composite_suffix.text(),
                    "image_format": self.image_format.currentData(),
                    "png_compress_level": self.png_compress_level.value(),
                    "jpeg_quality": self.jpeg_quality.value(),
                },
                "performance": {
                    "io_workers": self.io_workers.value(),
                },
            }
        )
        descriptor = self._registry.get(settings.resolved_model_name())
        resolution = settings.model.process_resolution
        if resolution and not (
            descriptor.min_resolution <= resolution <= descriptor.max_resolution
        ):
            raise ValueError(
                f"Разрешение {resolution} не поддерживается моделью "
                f"{descriptor.display_name}; допустимо "
                f"{descriptor.min_resolution}–{descriptor.max_resolution}, "
                "либо выберите «Автоматически»."
            )
        return settings
