// analize/generate_report/templates/report_script.js

// --- Глобальные переменные для модального окна ---
let modal = null;
let modalImg = null;
let captionText = null;
let prevBtn = null;
let nextBtn = null;

let currentGroupKey = null;
let currentGroupType = null;
let currentIndex = 0;
let currentImageList = [];

// Переменные для масштабирования
let isZoomed = false;
let currentZoomScale = 1;
let startX, startY, translateX = 0, translateY = 0;
let isDragging = false;

// --- Инициализация при загрузке DOM ---
document.addEventListener("DOMContentLoaded", function() {
    modal = document.getElementById("imageModal");
    modalImg = document.getElementById("modalImage");
    captionText = document.getElementById("modalCaption");
    prevBtn = document.querySelector(".modal-prev");
    nextBtn = document.querySelector(".modal-next");

    if (typeof LazyLoad !== 'undefined') {
        window.lazyLoadInstance = new LazyLoad({
            elements_selector: ".lazy",
            threshold: 300
        });
    } else {
        console.error("LazyLoad library not found!");
    }

    // Инициализация состояния UI
    document.querySelectorAll(".section-content").forEach(el => el.style.display = "none");
    document.querySelectorAll('button.toggle-button').forEach(button => {
        button.textContent = "Развернуть";
        let content = document.getElementById(button.id.replace('-button', '') + '-content');
        if (content) content.style.display = "none";
    });

    if (window.lazyLoadInstance) window.lazyLoadInstance.update();

    // Обработчики модального окна
    if (modal) {
        modal.addEventListener('click', (event) => { if (event.target === modal) closeModal(); });
        if (modalImg) {
            modalImg.addEventListener('click', toggleZoom);
            modalImg.addEventListener('wheel', handleWheelZoom, { passive: false });
            modalImg.addEventListener('mousedown', startDrag);
        }
    }
    document.addEventListener('mouseup', stopDrag);
    document.addEventListener('mouseleave', stopDrag);
});


// --- Функции управления UI ---
function toggleRow(rowId) {
    const content = document.getElementById(rowId + '-content');
    const button = document.getElementById(rowId + '-button');
    if (content && button) {
        const isHidden = content.style.display === "none";
        content.style.display = isHidden ? "block" : "none";
        button.textContent = isHidden ? "Свернуть" : "Развернуть";
        // --- ИЗМЕНЕНИЕ: Глобальное обновление LazyLoad ---
        if (isHidden && window.lazyLoadInstance) {
            window.lazyLoadInstance.update();
        }
    }
}

function toggleSection(sectionId) {
    const content = document.getElementById(sectionId);
    const header = document.getElementById(sectionId + "-header");
    const arrow = header ? header.querySelector('.arrow') : null;
    if (content && header && arrow) {
        const isHidden = content.style.display === "none";
        content.style.display = isHidden ? "block" : "none";
        header.classList.toggle('active', isHidden);
        arrow.textContent = isHidden ? '▼' : '►';
        // --- ИЗМЕНЕНИЕ: Глобальное обновление LazyLoad ---
        if (isHidden && window.lazyLoadInstance) {
            window.lazyLoadInstance.update();
        }
    }
}

// --- Функции модального окна (без изменений, взяты из вашего рабочего варианта) ---
function openModal(groupKey, type, index) {
    if (!modal || typeof portraitData === 'undefined' || typeof matchData === 'undefined') return;

    resetZoomAndPan();
    currentGroupKey = groupKey;
    currentGroupType = type;
    currentIndex = index;
    currentImageList = [];

    if (type === 'portrait' && portraitData[groupKey]) {
        currentImageList = portraitData[groupKey].files;
    } else if (type === 'group' && matchData[groupKey]) {
        currentImageList = matchData[groupKey].group_photos;
    }

    if (currentImageList.length > 0 && index < currentImageList.length) {
        modal.style.display = "block";
        loadImageWithNav(currentIndex);
        document.addEventListener('keydown', handleKeyPress);
    }
}

function openSingleImageModal(imageSrc, imageCaption) {
    if (!modal) return;
    resetZoomAndPan();
    currentImageList = [];
    modal.style.display = "block";
    modalImg.src = imageSrc;
    captionText.textContent = imageCaption || "";
    prevBtn.style.display = "none";
    nextBtn.style.display = "none";
    document.addEventListener('keydown', handleKeyPress);
}

function loadImageWithNav(index) {
    if (!modalImg || index < 0 || index >= currentImageList.length) return;
    resetZoomAndPan();
    var imgData = currentImageList[index];
    modalImg.src = imgData.rel_path;

    let caption = `${imgData.filename || "N/A"}\n`;
    caption += `Пол(O): ${imgData.gender_onnx || 'N/A'}, Возраст(O): ${imgData.age_onnx || 'N/A'}\n`;
    caption += `Эмоция: ${imgData.emotion_onnx || 'N/A'}, Глаза: ${imgData.eye_state_combined || 'N/A'}\n`;
    caption += `Привлекательность: ${imgData.beauty_onnx || 'N/A'}`;
    if (imgData.det_score !== undefined) caption += `, Det Score: ${imgData.det_score}`;
    if (imgData.num_faces !== undefined) caption += `\nЛиц: ${imgData.num_faces}, Мин.Расст: ${(imgData.confidence !== undefined && imgData.confidence !== null) ? Number(imgData.confidence).toFixed(4) : 'N/A'}`;
    
    captionText.textContent = caption;
    currentIndex = index;
    let showNav = currentImageList.length > 1;
    prevBtn.style.display = (index > 0 && showNav) ? "block" : "none";
    nextBtn.style.display = (index < currentImageList.length - 1 && showNav) ? "block" : "none";
}

function changeImage(delta) {
    let newIndex = currentIndex + delta;
    if (newIndex >= 0 && newIndex < currentImageList.length) loadImageWithNav(newIndex);
}

function closeModal() {
    if (modal) {
        modal.style.display = "none";
        document.removeEventListener('keydown', handleKeyPress);
        resetZoomAndPan();
    }
}

function handleKeyPress(event) {
    if (modal && modal.style.display === 'block') {
        if (event.key === "Escape") closeModal();
        else if (event.key === "ArrowLeft") changeImage(-1);
        else if (event.key === "ArrowRight") changeImage(1);
        else if (event.key === "+" || event.key === "=") { applyZoom(1.2); event.preventDefault(); }
        else if (event.key === "-") { applyZoom(1 / 1.2); event.preventDefault(); }
    }
}

// --- Функции зума и панорамирования ---
function applyZoom(factor) { if (!modalImg) return; let newScale = Math.max(0.5, Math.min(5, currentZoomScale * factor)); if (Math.abs(newScale - 1) < 0.01) { resetZoomAndPan(); } else { currentZoomScale = newScale; updateTransform(); isZoomed = true; modalImg.style.cursor = 'grab'; } }
function handleWheelZoom(event) { if (!modalImg) return; event.preventDefault(); applyZoom(event.deltaY > 0 ? 1 / 1.1 : 1.1); }
function toggleZoom() { if (!modalImg) return; isZoomed ? resetZoomAndPan() : applyZoom(2); }
function resetZoomAndPan() { if (!modalImg) return; isZoomed = false; currentZoomScale = 1; translateX = 0; translateY = 0; modalImg.style.transform = `scale(1) translate(0px, 0px)`; modalImg.style.cursor = 'zoom-in'; }
function startDrag(event) { if (!isZoomed || !modalImg) return; event.preventDefault(); isDragging = true; startX = event.clientX - translateX; startY = event.clientY - translateY; modalImg.style.cursor = 'grabbing'; document.addEventListener('mousemove', dragImage); document.addEventListener('mouseup', stopDrag); document.addEventListener('mouseleave', stopDrag); }
function dragImage(event) { if (!isDragging || !modalImg) return; event.preventDefault(); translateX = event.clientX - startX; translateY = event.clientY - startY; updateTransform(); }
function stopDrag() { if (isDragging) { isDragging = false; if (modalImg) modalImg.style.cursor = isZoomed ? 'grab' : 'zoom-in'; document.removeEventListener('mousemove', dragImage); document.removeEventListener('mouseup', stopDrag); document.removeEventListener('mouseleave', stopDrag); } }
function updateTransform() { if (!modalImg) return; modalImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentZoomScale})`; }