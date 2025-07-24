// fc_lib/report_script.js

// --- Глобальные переменные для модального окна ---
let modal = null;
let modalImg = null;
let captionText = null; // Теперь это <pre>
let prevBtn = null;
let nextBtn = null;

let currentGroupKey = null;
let currentGroupType = null; // 'portrait', 'group'
let currentIndex = 0;
let currentImageList = [];

// Переменные для масштабирования
let isZoomed = false;
let currentZoomScale = 1;
let startX, startY, initialX, initialY, translateX = 0, translateY = 0;
let isDragging = false;

// --- Инициализация при загрузке DOM ---
document.addEventListener("DOMContentLoaded", function() {
    console.log("DOM Loaded. Initializing UI states and modal...");

    // --- ИНИЦИАЛИЗАЦИЯ ЭЛЕМЕНТОВ МОДАЛЬНОГО ОКНА ---
    modal = document.getElementById("imageModal");
    modalImg = document.getElementById("modalImage");
    captionText = document.getElementById("modalCaption"); // Получаем <pre>
    prevBtn = document.querySelector(".modal-prev");
    nextBtn = document.querySelector(".modal-next");

    // --- ИНИЦИАЛИЗАЦИЯ LAZYLOAD ---
    if (typeof LazyLoad !== 'undefined') {
        window.lazyLoadInstance = new LazyLoad({
            elements_selector: ".lazy",
            threshold: 300
        });
        console.log("LazyLoad instance created.");
    } else {
        console.error("LazyLoad library not found!");
    }


    // --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ СЕКЦИЙ ---
    var sections = ["portrait-section", "match-section", "visualization-section", "config-section", "noise-section"];
    sections.forEach(id => {
        var content = document.getElementById(id);
        var header = document.getElementById(id + "-header");
        var arrow = header ? header.querySelector('.arrow') : null;
        if (content) content.style.display = "none";
        if (arrow) arrow.textContent = '►';
        if (header) header.classList.remove('active');
    });
     console.log("Sections initialized.");

    // --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ СТРОК ТАБЛИЦ ---
    var buttons = document.querySelectorAll('button.toggle-button');
    buttons.forEach(button => {
        button.textContent = "Развернуть";
         var rowId = button.id.replace('-button', '');
         var content = document.getElementById(rowId + '-content');
         if (content) {
              content.style.display = "none";
              content.classList.remove('visible-row-content');
         }
    });
    console.log("Table rows initialized.");

    // Первоначальное обновление LazyLoad
    if (window.lazyLoadInstance) {
        window.lazyLoadInstance.update();
        console.log("Initial LazyLoad update done.");
    }


     // --- ОБРАБОТЧИКИ СОБЫТИЙ МОДАЛЬНОГО ОКНА ---
     if (modal) {
        // Закрытие по клику вне изображения
        modal.addEventListener('click', function(event) {
            if (event.target === modal) { closeModal(); }
        });
        // Обработка клика для зума
        if (modalImg) {
            modalImg.addEventListener('click', toggleZoom);
            // Обработка колеса мыши для зума
            modalImg.addEventListener('wheel', handleWheelZoom, { passive: false }); // passive: false для preventDefault
            // Обработка перетаскивания для панорамирования (mousedown)
            modalImg.addEventListener('mousedown', startDrag);
        }

     } else {
         console.error("Modal element not found!");
     }

     // Глобальный обработчик отпускания кнопки (на случай, если ушли с картинки)
     document.addEventListener('mouseup', stopDrag);
     document.addEventListener('mouseleave', stopDrag); // Также при выходе за пределы окна

}); // Конец DOMContentLoaded


// --- Функции управления UI ---
function toggleRow(rowId) {
    var content = document.getElementById(rowId + '-content');
    var button = document.getElementById(rowId + '-button');
    if (content && button) {
        var isHidden = !content.classList.contains('visible-row-content');
        content.classList.toggle('visible-row-content', isHidden);
        content.style.display = isHidden ? "block" : "none"; // Используем block, а не table-cell
        button.textContent = isHidden ? "Свернуть" : "Развернуть";
        if (isHidden && window.lazyLoadInstance) {
            // Обновляем lazyload для открытого содержимого
            var images = content.querySelectorAll('img.lazy');
            window.lazyLoadInstance.update(images);
            console.log(`LazyLoad updated for row ${rowId}`);
        }
    } else { console.error("Element not found for rowId:", rowId); }
}

function toggleSection(sectionId) {
    var content = document.getElementById(sectionId);
    var header = document.getElementById(sectionId + "-header");
    var arrow = header ? header.querySelector('.arrow') : null;
    if (content && header && arrow) {
        var isHidden = content.style.display === "none" || content.style.display === "";
        content.style.display = isHidden ? "block" : "none";
        header.classList.toggle('active', isHidden);
        arrow.textContent = isHidden ? '▼' : '►';
         if (isHidden && window.lazyLoadInstance) {
            var images = content.querySelectorAll('img.lazy');
            window.lazyLoadInstance.update(images);
            console.log(`LazyLoad updated for section ${sectionId}`);
        }
    } else { console.error("Elements not found for sectionId:", sectionId); }
}

// --- Функции модального окна ---
function openModal(groupKey, type, index) {
    if (!modal || !modalImg || !captionText || !prevBtn || !nextBtn) { console.error("Modal elements not initialized!"); return; }
    if (typeof portraitData === 'undefined' || typeof matchData === 'undefined') { console.error("Data (portraitData/matchData) not available!"); return; }

    resetZoomAndPan();
    currentGroupKey = groupKey;
    currentGroupType = type;
    currentIndex = index;
    currentImageList = []; // Очищаем перед заполнением

    try {
        // Получаем список изображений для текущего кластера/сопоставления
        if (type === 'portrait' && portraitData[groupKey] && portraitData[groupKey].files) {
             currentImageList = portraitData[groupKey].files;
        } else if (type === 'group' && matchData[groupKey] && matchData[groupKey].group_photos) {
             currentImageList = matchData[groupKey].group_photos;
        } else if (type === 'portrait' && groupKey === '-1' && portraitData['-1'] && portraitData['-1'].files) { // Обработка шума
             currentImageList = portraitData['-1'].files;
        }
        console.log(`Opening modal for ${type} ${groupKey}, index ${index}. Images found: ${currentImageList.length}`);
    } catch (e) {
        console.error("Error accessing data for modal:", e);
    }


    if (currentImageList.length > 0 && index >= 0 && index < currentImageList.length) {
        modal.style.display = "block";
        loadImageWithNav(currentIndex); // Загружаем изображение
        document.addEventListener('keydown', handleKeyPress); // Добавляем обработчик клавиш
    } else {
         console.warn("No images or invalid index for modal:", groupKey, type, index, currentImageList.length);
         closeModal(); // Закрываем, если нет данных
    }
}

function openSingleImageModal(imageSrc, imageCaption) {
    if (!modal || !modalImg || !captionText || !prevBtn || !nextBtn) { console.error("Modal elements not initialized!"); return; }
     resetZoomAndPan();
     currentImageList = []; // Нет навигации
     modal.style.display = "block";
     modalImg.src = imageSrc;
     captionText.textContent = imageCaption || ""; // Используем textContent для <pre>
     prevBtn.style.display = "none";
     nextBtn.style.display = "none";
     document.addEventListener('keydown', handleKeyPress);
}

function loadImageWithNav(index) {
     if (!modalImg || !captionText || !prevBtn || !nextBtn || !currentImageList) return;
     resetZoomAndPan();

     if (index >= 0 && index < currentImageList.length) {
          var imgData = currentImageList[index];
          modalImg.src = imgData.rel_path; // Используем rel_path

          // --- Формируем новую подпись ---
          let caption = (imgData.filename || "N/A") + "\n"; // Имя файла
          // Добавляем атрибуты, если они есть (проверяем наличие)
          caption += `Пол(O): ${imgData.gender_onnx !== undefined && imgData.gender_onnx !== null ? imgData.gender_onnx : 'N/A'}`
          caption += `, Возраст(O): ${imgData.age_onnx !== undefined && imgData.age_onnx !== null ? imgData.age_onnx : 'N/A'}\n`;
          caption += `Эмоция: ${imgData.emotion_onnx !== undefined && imgData.emotion_onnx !== null ? imgData.emotion_onnx : 'N/A'}`
          caption += `, Глаза: ${imgData.eye_state_combined !== undefined && imgData.eye_state_combined !== null ? imgData.eye_state_combined : 'N/A'}\n`;
          caption += `Привлекательность: ${imgData.beauty_onnx !== undefined && imgData.beauty_onnx !== null ? imgData.beauty_onnx : 'N/A'}`
          // Добавляем Det Score, если он есть в данных (для портретов и шума)
          if (imgData.det_score !== undefined) {
               caption += `, Det Score: ${imgData.det_score}`;
          }
          // Добавляем инфо для групповых фото, если есть
          if (imgData.num_faces !== undefined) {
               caption += `\nЛиц: ${imgData.num_faces}, Мин.Расст: ${imgData.confidence !== undefined && imgData.confidence !== null ? Number(imgData.confidence).toFixed(4) : 'N/A'}`;
          }
          // --- Конец формирования подписи ---

          captionText.textContent = caption; // Устанавливаем текст для <pre>
          currentIndex = index;
          var showNav = currentImageList.length > 1;
          prevBtn.style.display = (index > 0 && showNav) ? "block" : "none";
          nextBtn.style.display = (index < currentImageList.length - 1 && showNav) ? "block" : "none";
     } else {
         console.warn("loadImageWithNav: Invalid index or image list empty", index, currentImageList.length);
     }
}

function changeImage(delta) {
    if (currentImageList.length > 1) {
        var newIndex = currentIndex + delta;
        if (newIndex >= 0 && newIndex < currentImageList.length) {
            loadImageWithNav(newIndex);
        }
    }
}

function closeModal() {
    if (modal) {
         modal.style.display = "none";
         document.removeEventListener('keydown', handleKeyPress);
         resetZoomAndPan();
         currentGroupKey = null;
         currentGroupType = null;
         currentIndex = 0;
         currentImageList = [];
         if(modalImg) modalImg.src = ""; // Очищаем src
         if(captionText) captionText.textContent = "";
    }
}

function handleKeyPress(event) {
     if (modal && modal.style.display === 'block') {
         if (event.key === "Escape") { closeModal(); }
         else if (event.key === "ArrowLeft") { changeImage(-1); }
         else if (event.key === "ArrowRight") { changeImage(1); }
         else if (event.key === "+" || event.key === "=") { applyZoom(1.2); event.preventDefault(); }
         else if (event.key === "-") { applyZoom(1 / 1.2); event.preventDefault(); }
     }
}

// --- Функции масштабирования и панорамирования (без изменений) ---
function applyZoom(factor) {
    if (!modalImg) return;
    const maxZoom = 5; const minZoom = 0.5;
    let newScale = currentZoomScale * factor;
    newScale = Math.max(minZoom, Math.min(maxZoom, newScale));
    if (Math.abs(newScale - 1) < 0.01) { resetZoomAndPan(); }
    else { currentZoomScale = newScale; updateTransform(); isZoomed = true; modalImg.classList.add('zoomed'); modalImg.style.cursor = 'grab'; }
}
function handleWheelZoom(event) { if (!modalImg) return; event.preventDefault(); const delta = event.deltaY > 0 ? 1 / 1.1 : 1.1; applyZoom(delta); }
function toggleZoom() { if (!modalImg) return; if (isZoomed) { resetZoomAndPan(); } else { applyZoom(2); } }
function resetZoomAndPan() { if (!modalImg) return; isZoomed = false; currentZoomScale = 1; translateX = 0; translateY = 0; modalImg.style.transform = `scale(1) translate(0px, 0px)`; modalImg.style.transformOrigin = 'center center'; modalImg.classList.remove('zoomed'); modalImg.style.cursor = 'zoom-in'; }
function startDrag(event) { if (!isZoomed || !modalImg) return; event.preventDefault(); isDragging = true; startX = event.clientX - translateX; startY = event.clientY - translateY; modalImg.style.cursor = 'grabbing'; document.addEventListener('mousemove', dragImage); document.addEventListener('mouseup', stopDrag); document.addEventListener('mouseleave', stopDrag); }
function dragImage(event) { if (!isDragging || !modalImg) return; event.preventDefault(); translateX = event.clientX - startX; translateY = event.clientY - startY; updateTransform(); }
function stopDrag(event) { if (isDragging) { isDragging = false; if(modalImg) { modalImg.style.cursor = isZoomed ? 'grab' : 'zoom-in'; } document.removeEventListener('mousemove', dragImage); document.removeEventListener('mouseup', stopDrag); document.removeEventListener('mouseleave', stopDrag); } }
function updateTransform() { if (!modalImg) return; modalImg.style.transformOrigin = 'center center'; modalImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentZoomScale})`; }