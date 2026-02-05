/**
 * DINO-SAM Labeling Agent - Frontend Application
 */

// ========================================
// State Management
// ========================================

const state = {
    currentImageId: null,
    currentImage: null,
    currentImageObjectUrl: null,  // createObjectURL로 생성, 메모리 해제용
    labelingResult: null,
    isProcessing: false,
    pointSegmentMode: false,  // 클릭 세그멘테이션 모드
    labelingRequestId: 0,     // 요청 순서 추적 (race condition 방지)
    uploadMode: 'single',     // 'single' | 'batch'
    batchJobId: null,         // 배치 작업 ID
    batchImageIds: [],        // 배치 이미지 ID 목록
    batchResults: [],         // 배치 라벨링 결과 [{ image_id, result }, ...]
    batchResultIndex: 0,      // 현재 보고 있는 배치 결과 인덱스
};

const DEBUG_MASK_RENDER = true;

// ========================================
// DOM Elements
// ========================================

const elements = {
    // Upload
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    uploadZoneText: document.getElementById('uploadZoneText'),
    batchProgress: document.getElementById('batchProgress'),
    batchStage: document.getElementById('batchStage'),
    batchProgressFill: document.getElementById('batchProgressFill'),
    batchProgressText: document.getElementById('batchProgressText'),
    batchLabelBtn: document.getElementById('batchLabelBtn'),
    batchNav: document.getElementById('batchNav'),
    batchPrevBtn: document.getElementById('batchPrevBtn'),
    batchNextBtn: document.getElementById('batchNextBtn'),
    batchNavText: document.getElementById('batchNavText'),
    batchResultsHeader: document.getElementById('batchResultsHeader'),
    batchResultsSummary: document.getElementById('batchResultsSummary'),

    // Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabPanes: document.querySelectorAll('.tab-pane'),
    btnTabResult: document.getElementById('btnTabResult'),

    // Prompt & Settings
    promptInput: document.getElementById('promptInput'),
    confidenceSlider: document.getElementById('confidenceSlider'),
    confidenceValue: document.getElementById('confidenceValue'),
    formatSelect: document.getElementById('formatSelect'),
    clientRenderCheckbox: document.getElementById('clientRenderCheckbox'),
    
    // Buttons
    labelBtn: document.getElementById('labelBtn'),
    approveBtn: document.getElementById('approveBtn'),
    rejectBtn: document.getElementById('rejectBtn'),
    exportBtn: document.getElementById('exportBtn'),
    exportCurrentBtn: document.getElementById('exportCurrentBtn'),

    // Canvas
    canvasContainer: document.getElementById('canvasContainer'),
    canvasPlaceholder: document.getElementById('canvasPlaceholder'),
    imageCanvas: document.getElementById('imageCanvas'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    
    // HITL
    hitlControls: document.getElementById('hitlControls'),
    hitlCount: document.getElementById('hitlCount'),
    
    // Results
    resultsList: document.getElementById('resultsList'),
    queueList: document.getElementById('queueList'),
    
    // Stats
    statTotal: document.getElementById('statTotal'),
    statPending: document.getElementById('statPending'),
    statApproved: document.getElementById('statApproved'),
    statRejected: document.getElementById('statRejected'),
    
    // Toast
    toast: document.getElementById('toast'),
};

// ========================================
// Initialization
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    if (DEBUG_MASK_RENDER) {
        console.log('[mask-debug] app.js loaded');
    }
    updateUploadZoneText();
    initializeEventListeners();
    loadStats();
    loadPendingQueue();
});

function initializeEventListeners() {
    // Tabs
    initializeTabs();

    // Upload Mode Toggle
    document.querySelectorAll('input[name="uploadMode"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.uploadMode = e.target.value;
            updateUploadZoneText();
            resetBatchState();
        });
    });

    // Upload Zone
    elements.uploadZone.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag & Drop
    elements.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadZone.classList.add('dragover');
    });
    elements.uploadZone.addEventListener('dragleave', () => {
        elements.uploadZone.classList.remove('dragover');
    });
    elements.uploadZone.addEventListener('drop', handleDrop);
    
    // Confidence Slider
    elements.confidenceSlider.addEventListener('input', (e) => {
        elements.confidenceValue.textContent = e.target.value;
    });
    
    // Label Buttons
    elements.labelBtn.addEventListener('click', runLabeling);
    elements.batchLabelBtn?.addEventListener('click', runBatchLabeling);
    elements.batchPrevBtn?.addEventListener('click', () => showBatchResultIndex(state.batchResultIndex - 1));
    elements.batchNextBtn?.addEventListener('click', () => showBatchResultIndex(state.batchResultIndex + 1));
    
    // HITL Buttons
    elements.approveBtn.addEventListener('click', () => submitFeedback('approved'));
    elements.rejectBtn.addEventListener('click', () => submitFeedback('rejected'));
    
    // Export Buttons
    elements.exportBtn.addEventListener('click', () => exportLabels(true));
    elements.exportCurrentBtn?.addEventListener('click', () => exportLabels(false));
    
    // Canvas click for point segmentation
    elements.imageCanvas.addEventListener('click', handleCanvasClick);
}

function initializeTabs() {
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            if (btn.disabled) return;
            switchTab(targetTab);
        });
    });
}

function switchTab(tabId) {
    // Update Buttons
    elements.tabBtns.forEach(btn => {
        if (btn.dataset.tab === tabId) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Update Panes
    elements.tabPanes.forEach(pane => {
        if (pane.id === tabId) {
            pane.classList.add('active');
        } else {
            pane.classList.remove('active');
        }
    });
}

// ========================================
// Image Upload
// ========================================

function updateUploadZoneText() {
    if (!elements.uploadZoneText) return;
    if (state.uploadMode === 'batch') {
        elements.uploadZoneText.textContent = 'Drag & drop zip or multiple images (2+)';
        elements.fileInput.setAttribute('accept', 'image/*,.zip');
        elements.fileInput.removeAttribute('multiple');
        elements.fileInput.setAttribute('multiple', 'multiple');
        if (elements.labelBtn) elements.labelBtn.style.display = 'none';
        if (elements.batchLabelBtn) elements.batchLabelBtn.style.display = '';
    } else {
        elements.uploadZoneText.textContent = 'Drag & drop images here';
        elements.fileInput.setAttribute('accept', 'image/*');
        elements.fileInput.removeAttribute('multiple');
        if (elements.labelBtn) elements.labelBtn.style.display = '';
        if (elements.batchLabelBtn) elements.batchLabelBtn.style.display = 'none';
    }
}

function resetBatchState() {
    state.batchJobId = null;
    state.batchImageIds = [];
    state.batchResults = [];
    state.batchResultIndex = 0;
    elements.batchLabelBtn?.setAttribute('disabled', 'disabled');
    elements.batchProgress?.setAttribute('hidden', 'hidden');
    elements.batchNav?.setAttribute('hidden', 'hidden');
    elements.batchResultsHeader?.setAttribute('hidden', 'hidden');
    updateExportButtonText(0);
    if (elements.exportCurrentBtn) {
        elements.exportCurrentBtn.style.display = 'none';
        elements.exportCurrentBtn.disabled = true;
    }
    // Disable Result Tab if no single result either
    if (!state.labelingResult) {
        elements.btnTabResult.disabled = true;
    }
}

function updateExportButtonText(batchCount) {
    if (!elements.exportBtn) return;
    elements.exportBtn.textContent = batchCount > 0
        ? `Export All (${batchCount})`
        : 'Export Labels';
    if (elements.exportCurrentBtn) {
        elements.exportCurrentBtn.style.display = batchCount > 0 ? '' : 'none';
        elements.exportCurrentBtn.disabled = false;
    }
}

async function showBatchResultIndex(index) {
    if (!state.batchResults.length) return;
    const i = Math.max(0, Math.min(index, state.batchResults.length - 1));
    state.batchResultIndex = i;
    const item = state.batchResults[i];
    if (!item) return;

    state.currentImageId = item.image_id;
    state.labelingResult = item.result;

    if (item.result.overlay_image) {
        drawResults(item.result);
    } else {
        try {
            const imgResp = await fetch(`/api/image/${item.image_id}`);
            if (imgResp.ok) {
                const blob = await imgResp.blob();
                const url = URL.createObjectURL(blob);
                if (state.currentImageObjectUrl) URL.revokeObjectURL(state.currentImageObjectUrl);
                state.currentImageObjectUrl = url;
                state.currentImage = null;
                drawResults(item.result);
            }
        } catch (e) {
            console.error('Failed to load batch image:', e);
        }
    }

    updateResultsList(item.result);
    if (elements.batchNavText) elements.batchNavText.textContent = `${i + 1} / ${state.batchResults.length}`;
    if (elements.batchPrevBtn) elements.batchPrevBtn.disabled = i <= 0;
    if (elements.batchNextBtn) elements.batchNextBtn.disabled = i >= state.batchResults.length - 1;
}

function updateBatchProgress(stage, current, total, message) {
    if (!elements.batchProgress) return;
    elements.batchProgress.removeAttribute('hidden');
    if (elements.batchStage) elements.batchStage.textContent = message || stage;
    const pct = total > 0 ? Math.round((current / total) * 100) : 0;
    if (elements.batchProgressFill) elements.batchProgressFill.style.width = pct + '%';
    if (elements.batchProgressText) elements.batchProgressText.textContent = `${current} / ${total}`;
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length === 0) return;
    if (state.uploadMode === 'batch') {
        batchUpload(files);
    } else {
        uploadImage(files[0]);
    }
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length === 0) return;
    if (state.uploadMode === 'batch') {
        batchUpload(files);
    } else {
        uploadImage(files[0]);
    }
}

async function uploadImage(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file', 'error');
        return;
    }
    
    try {
        showLoading(true);
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }
        
        state.currentImageId = data.image_id;
        state.currentImage = file;
        
        // Display image
        displayImage(file);
        
        // Enable labeling
        elements.labelBtn.disabled = false;
        
        showToast('Image uploaded successfully', 'success');
        
    } catch (error) {
        console.error('Upload error:', error);
        showToast(error.message, 'error');
    } finally {
        showLoading(false);
    }
}

async function batchUpload(files) {
    if (state.uploadMode !== 'batch') return;
    const isZip = files.length === 1 && (files[0].name || '').toLowerCase().endsWith('.zip');
    const isMultiImage = files.length >= 2 && Array.from(files).every(f => (f.type || '').startsWith('image/'));
    if (!isZip && !isMultiImage) {
        showToast('배치: zip 파일 1개 또는 이미지 2장 이상을 업로드하세요', 'error');
        return;
    }

    showLoading(true);
    resetBatchState();
    updateBatchProgress('upload', 0, 1, '업로드 중...');

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    try {
        const response = await fetch('/api/batch/upload', {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Upload failed (${response.status})`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let jobId = null;
        let imageIds = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const block of lines) {
                if (!block.trim()) continue;
                const eventMatch = block.match(/^event:\s*(.+)$/m);
                const dataMatch = block.match(/^data:\s*(.+)$/m);
                const event = eventMatch ? eventMatch[1].trim() : 'message';
                const dataStr = dataMatch ? dataMatch[1].trim() : '{}';
                let data = {};
                try {
                    data = JSON.parse(dataStr);
                } catch (_) {}

                if (event === 'stage') {
                    const msg = data.message || data.stage || '';
                    updateBatchProgress(data.stage || 'stage', 0, 1, msg);
                } else if (event === 'progress') {
                    updateBatchProgress(data.stage, data.current || 0, data.total || 1, `${data.stage || ''}: ${(data.current || 0)} / ${data.total || 1}`);
                } else if (event === 'done') {
                    jobId = data.job_id;
                    imageIds = data.image_ids || [];
                    state.batchJobId = jobId;
                    state.batchImageIds = imageIds;
                    updateBatchProgress('done', imageIds.length, imageIds.length, `업로드 완료: ${imageIds.length}장`);
                    elements.batchLabelBtn?.removeAttribute('disabled');
                    showToast(`배치 업로드 완료: ${imageIds.length}장`, 'success');
                    if (imageIds.length > 0) {
                        state.currentImageId = imageIds[0];
                        const imgResp = await fetch(`/api/image/${imageIds[0]}`);
                        if (imgResp.ok) {
                            const blob = await imgResp.blob();
                            displayImage(blob);
                        }
                    }
                } else if (event === 'error') {
                    throw new Error(data.message || 'Batch upload failed');
                }
            }
        }
    } catch (error) {
        console.error('Batch upload error:', error);
        showToast(error.message || 'Batch upload failed', 'error');
        resetBatchState();
    } finally {
        showLoading(false);
    }
}

async function runBatchLabeling() {
    if (!state.batchJobId) {
        showToast('배치 업로드를 먼저 완료하세요', 'error');
        return;
    }
    const prompt = elements.promptInput.value.trim();
    if (!prompt) {
        showToast('Please enter a detection prompt', 'error');
        return;
    }

    showLoading(true);
    elements.batchLabelBtn?.setAttribute('disabled', 'disabled');
    updateBatchProgress('labeling', 0, state.batchImageIds.length, '라벨링 시작...');

    try {
        const response = await fetch('/api/batch/label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: state.batchJobId,
                prompt,
                confidence_threshold: parseFloat(elements.confidenceSlider?.value || 0.35),
                use_client_render: elements.clientRenderCheckbox?.checked ?? false,
            }),
        });
        if (!response.ok) {
            throw new Error(`Labeling failed (${response.status})`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let completed = 0;
        const total = state.batchImageIds.length;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const block of lines) {
                if (!block.trim()) continue;
                const eventMatch = block.match(/^event:\s*(.+)$/m);
                const dataMatch = block.match(/^data:\s*(.+)$/m);
                const event = eventMatch ? eventMatch[1].trim() : 'message';
                const dataStr = dataMatch ? dataMatch[1].trim() : '{}';
                let data = {};
                try {
                    data = JSON.parse(dataStr);
                } catch (_) {}

                if (event === 'progress') {
                    completed = data.current || completed;
                    const msg = data.skipped ? `건너뜀: ${data.image_id}` : `라벨링: ${data.current} / ${total}`;
                    updateBatchProgress('labeling', completed, total, msg);
                } else if (event === 'done') {
                    const results = data.results || [];
                    const count = data.count || results.length;
                    state.batchResults = results;
                    state.batchResultIndex = 0;
                    updateBatchProgress('done', total, total, `라벨링 완료: ${count}장`);
                    elements.exportBtn?.removeAttribute('disabled');
                    showToast(`Batch labeling 완료: ${count}장`, 'success');
                    if (results.length > 0) {
                        elements.batchNav?.removeAttribute('hidden');
                        elements.batchResultsHeader?.removeAttribute('hidden');
                        if (elements.batchResultsSummary) elements.batchResultsSummary.textContent = `${results.length} images labeled`;
                        updateExportButtonText(results.length);
                        showBatchResultIndex(0);
                        
                        // Switch to Result Tab
                        elements.btnTabResult.disabled = false;
                        switchTab('tab-result');
                    }
                }
            }
        }
    } catch (error) {
        console.error('Batch labeling error:', error);
        showToast(error.message || 'Batch labeling failed', 'error');
    } finally {
        showLoading(false);
        elements.batchLabelBtn?.removeAttribute('disabled');
    }
}

function displayImage(file) {
    if (state.currentImageObjectUrl) {
        URL.revokeObjectURL(state.currentImageObjectUrl);
        state.currentImageObjectUrl = null;
    }
    const url = URL.createObjectURL(file);
    state.currentImageObjectUrl = url;
    const img = new Image();
    img.onload = () => {
        const canvas = elements.imageCanvas;
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.style.display = 'block';
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0);
        canvas.dataset.originalWidth = img.width;
        canvas.dataset.originalHeight = img.height;
        if (DEBUG_MASK_RENDER) {
            console.log('[mask-debug] image size (원본 유지):', { width: img.width, height: img.height });
        }
        elements.canvasPlaceholder.style.display = 'none';
    };
    img.onerror = () => {
        URL.revokeObjectURL(url);
        state.currentImageObjectUrl = null;
        showToast('Failed to load image', 'error');
    };
    img.src = url;
}

// ========================================
// Labeling
// ========================================

async function runLabeling() {
    if (!state.currentImageId) {
        showToast('Please upload an image first', 'error');
        return;
    }
    
    const prompt = elements.promptInput.value.trim();
    if (!prompt) {
        showToast('Please enter a detection prompt', 'error');
        return;
    }
    
    showLoading(true);
    state.isProcessing = true;
    const requestId = ++state.labelingRequestId;

    try {
        const response = await fetch('/api/label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: state.currentImageId,
                prompt: prompt,
                confidence_threshold: parseFloat(elements.confidenceSlider.value),
                use_client_render: elements.clientRenderCheckbox?.checked ?? false,
            }),
        });
        
        let data;
        try {
            data = await response.json();
        } catch (jsonError) {
            throw new Error(`Server error (${response.status}): Unable to parse response`);
        }
        
        if (!response.ok) {
            throw new Error(data.detail || `Labeling failed (${response.status})`);
        }
        if (requestId !== state.labelingRequestId) return; // 새 요청이 이미 시작됨 (race 방지)

        state.labelingResult = data.result;
        
        // Draw results on canvas
        drawResults(data.result);
        
        // Update results list
        updateResultsList(data.result);
        
        // Show HITL controls
        showHITLControls(data.result);
        
        // Refresh stats
        loadStats();
        loadPendingQueue();
        
        showToast(`Found ${data.result.num_objects} objects`, 'success');
        
        // Enable and Switch to Result Tab
        elements.btnTabResult.disabled = false;
        switchTab('tab-result');
        
    } catch (error) {
        console.error('Labeling error:', error);
        showToast(error.message || 'Labeling failed', 'error');
    } finally {
        showLoading(false);
        state.isProcessing = false;
    }
}

function drawResults(result) {
    if (!result) return;
    
    const canvas = elements.imageCanvas;
    const ctx = canvas.getContext('2d');
    
    // 서버에서 생성한 오버레이 이미지가 있으면 직접 표시 (원본 해상도 유지)
    if (result.overlay_image) {
        console.log('[overlay] 서버 생성 오버레이 이미지 사용:', result.overlay_image);
        
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(img, 0, 0);
            canvas.dataset.originalWidth = img.width;
            canvas.dataset.originalHeight = img.height;
            console.log('[overlay] 오버레이 이미지 표시 완료 (원본 유지):', [img.width, img.height]);
        };
        img.onerror = () => {
            console.error('[overlay] 오버레이 이미지 로드 실패:', result.overlay_image);
            showToast('Failed to load overlay image', 'error');
        };
        img.src = result.overlay_image;
        return;
    }
    
    // 폴백: 원본 방식 (클라이언트 렌더링, 또는 검출 없음 시 원본 이미지 표시)
    const imageUrl = state.currentImageObjectUrl || (state.currentImage && URL.createObjectURL(state.currentImage));
    if (imageUrl) {
        const img = new Image();
        img.onload = () => {
            if (canvas.width !== img.width || canvas.height !== img.height) {
                canvas.width = img.width;
                canvas.height = img.height;
            }
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            if (result.masks_raw && result.masks_raw.length > 0) {
                if (DEBUG_MASK_RENDER) {
                    console.log('[mask-debug] drawMasks called', {
                        masksCount: result.masks_raw.length,
                        canvasWidth: canvas.width,
                        canvasHeight: canvas.height,
                    });
                }
                drawMasks(ctx, result, canvas);
            } else {
                drawBoxes(ctx, result, canvas);
            }
        };
        img.onerror = () => showToast('Failed to load image for overlay', 'error');
        img.src = imageUrl;
    }
}

function drawMasks(ctx, result, canvas) {
    const masks = result.masks_raw || [];
    const maskSize = result.mask_size || [canvas.height, canvas.width];
    const height = maskSize[0];
    const width = maskSize[1];
    const threshold = 0.5;

    if (DEBUG_MASK_RENDER) {
        console.log('[mask-debug] drawMasks', {
            masksCount: masks.length,
            maskSize,
            canvasSize: [canvas.width, canvas.height],
            imageSize: [result.image_width, result.image_height],
        });
    }

    masks.forEach((maskFlat, index) => {
        if (!Array.isArray(maskFlat) || maskFlat.length !== width * height) {
            if (DEBUG_MASK_RENDER) {
                console.log('[mask-debug] mask size mismatch', {
                    index,
                    expected: width * height,
                    actual: maskFlat ? maskFlat.length : 0,
                });
            }
            return;
        }
        ctx.save();

        const color = getClassColor(index);
        const rgb = hexToRgb(color);

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.imageSmoothingEnabled = false;

        const imageData = tempCtx.createImageData(width, height);
        const data = imageData.data;
        
        // row-major order: iterate by row then column
        let activeCount = 0;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const maskIdx = y * width + x;
                const prob = maskFlat[maskIdx];
                const dataIdx = (y * width + x) * 4;
                if (prob > threshold) {
                    data[dataIdx] = rgb.r;
                    data[dataIdx + 1] = rgb.g;
                    data[dataIdx + 2] = rgb.b;
                    data[dataIdx + 3] = 100;
                    activeCount++;
                } else {
                    data[dataIdx + 3] = 0;
                }
            }
        }

        if (DEBUG_MASK_RENDER) {
            console.log('[mask-debug] mask rendered', {
                index,
                activePixels: activeCount,
                totalPixels: width * height,
                coverage: (activeCount / (width * height)).toFixed(4),
            });
        }

        tempCtx.putImageData(imageData, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
        ctx.restore();
    });

    drawBoxes(ctx, result, canvas);
}


function drawBoxes(ctx, result, canvas) {
    // 이미지 크기에 따른 스케일 (큰 이미지에서도 label text가 보이도록)
    const minDim = Math.min(canvas.width, canvas.height);
    const scale = Math.max(0.5, Math.min(6, minDim / 400));
    const fontSize = Math.max(14, Math.min(72, Math.round(16 * scale)));
    const labelHeight = Math.max(20, Math.min(80, Math.round(24 * scale)));
    const padding = Math.max(4, Math.round(8 * scale));
    const lineWidth = Math.max(2, Math.round(3 * scale));

    result.boxes_percent.forEach((box, index) => {
        const x = (box.x / 100) * canvas.width;
        const y = (box.y / 100) * canvas.height;
        const w = (box.width / 100) * canvas.width;
        const h = (box.height / 100) * canvas.height;

        const color = getClassColor(index);

        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x, y, w, h);

        const label = result.labels[index];
        const score = (result.scores[index] * 100).toFixed(1);
        const text = `${label} ${score}%`;

        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        const textWidth = ctx.measureText(text).width;

        const labelY = Math.max(y - labelHeight, 0);
        ctx.fillStyle = color;
        ctx.fillRect(x, labelY, textWidth + padding * 2, labelHeight);

        ctx.fillStyle = '#fff';
        ctx.fillText(text, x + padding, labelY + labelHeight - padding);
    });
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : { r: 0, g: 0, b: 0 };
}

function getClassColor(index) {
    const colors = [
        '#6366f1', // Indigo
        '#22c55e', // Green
        '#f59e0b', // Amber
        '#ef4444', // Red
        '#06b6d4', // Cyan
        '#ec4899', // Pink
        '#8b5cf6', // Violet
        '#14b8a6', // Teal
    ];
    return colors[index % colors.length];
}

function updateResultsList(result) {
    if (!result || result.num_objects === 0) {
        elements.resultsList.innerHTML = '<div class="results-empty">No objects detected</div>';
        return;
    }
    
    let html = '';
    result.labels.forEach((label, index) => {
        const score = (result.scores[index] * 100).toFixed(1);
        const color = getClassColor(index);
        
        html += `
            <div class="result-item" style="border-left: 3px solid ${color}">
                <span class="result-label">${label}</span>
                <span class="result-score">${score}%</span>
            </div>
        `;
    });
    
    elements.resultsList.innerHTML = html;
}

function showHITLControls(result) {
    elements.hitlControls.hidden = false;
    elements.hitlCount.textContent = `${result.num_objects} objects detected`;
    elements.exportBtn.disabled = false;
}

// ========================================
// HITL Feedback
// ========================================

async function submitFeedback(status) {
    if (!state.currentImageId) return;
    
    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: state.currentImageId,
                status: status,
            }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Feedback submission failed');
        }
        
        // Refresh stats and queue
        loadStats();
        loadPendingQueue();
        
        // Reset state for next image
        resetForNextImage();
        
        showToast(`Marked as ${status}`, 'success');
        
    } catch (error) {
        console.error('Feedback error:', error);
        showToast(error.message, 'error');
    }
}

function resetForNextImage() {
    state.currentImageId = null;
    state.currentImage = null;
    if (state.currentImageObjectUrl) {
        URL.revokeObjectURL(state.currentImageObjectUrl);
        state.currentImageObjectUrl = null;
    }
    state.labelingResult = null;
    
    elements.imageCanvas.style.display = 'none';
    elements.canvasPlaceholder.style.display = 'flex';
    elements.hitlControls.hidden = true;
    elements.labelBtn.disabled = true;
    elements.exportBtn.disabled = true;
    elements.resultsList.innerHTML = '<div class="results-empty">No results yet</div>';
}

// ========================================
// Export
// ========================================

async function exportLabels(exportAll = true) {
    // exportAll: true=전체 배치, false=현재 보고 있는 이미지만
    let imageIds;
    if (exportAll) {
        imageIds = state.batchResults?.length
            ? state.batchResults.map((r) => r.image_id)
            : state.batchImageIds?.length
                ? state.batchImageIds
                : state.currentImageId
                    ? [state.currentImageId]
                    : [];
    } else {
        imageIds = state.currentImageId ? [state.currentImageId] : [];
    }
    if (!imageIds.length) return;

    try {
        const format = elements.formatSelect.value;

        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_ids: imageIds,
                format: format,
            }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Export failed');
        }
        
        showToast(`Exported to ${data.output_path}`, 'success');
        
    } catch (error) {
        console.error('Export error:', error);
        showToast(error.message, 'error');
    }
}

// ========================================
// Stats & Queue
// ========================================

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats;
            elements.statTotal.textContent = stats.total || 0;
            elements.statPending.textContent = stats.by_status?.pending || 0;
            elements.statApproved.textContent = stats.by_status?.approved || 0;
            elements.statRejected.textContent = stats.by_status?.rejected || 0;
        }
    } catch (error) {
        console.error('Stats error:', error);
    }
}

async function loadPendingQueue() {
    try {
        const response = await fetch('/api/feedback/pending?limit=10');
        const data = await response.json();
        
        if (data.success && data.items.length > 0) {
            let html = '';
            data.items.forEach(item => {
                const name = item.image_id.substring(0, 8);
                html += `
                    <div class="queue-item" data-image-id="${item.image_id}">
                        <div class="queue-thumb">🖼️</div>
                        <div class="queue-info">
                            <div class="queue-name">${name}...</div>
                            <div class="queue-status">${item.prompt}</div>
                        </div>
                    </div>
                `;
            });
            elements.queueList.innerHTML = html;
        } else {
            elements.queueList.innerHTML = '<div class="queue-empty">No pending reviews</div>';
        }
    } catch (error) {
        console.error('Queue error:', error);
    }
}

// ========================================
// UI Helpers
// ========================================

function showLoading(show) {
    if (show) {
        elements.loadingOverlay.classList.add('active');
    } else {
        elements.loadingOverlay.classList.remove('active');
    }
}

function showToast(message, type = 'info') {
    elements.toast.textContent = message;
    elements.toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

// ========================================
// Point Segmentation (클릭 세그멘테이션)
// ========================================

async function handleCanvasClick(e) {
    // 이미지가 업로드되지 않았으면 무시
    if (!state.currentImageId) {
        return;
    }
    
    // 처리 중이면 무시
    if (state.isProcessing) {
        return;
    }
    
    const canvas = elements.imageCanvas;
    const rect = canvas.getBoundingClientRect();
    
    // 캔버스 좌표 계산
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    
    console.log('[point-segment] 클릭:', { x, y, canvasSize: [canvas.width, canvas.height] });
    
    try {
        showLoading(true);
        state.isProcessing = true;
        
        const response = await fetch('/api/segment-point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: state.currentImageId,
                point_x: x,
                point_y: y,
                point_label: 1,  // foreground
            }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Point segmentation failed');
        }
        
        console.log('[point-segment] 결과:', data);
        
        // 결과 이미지 표시
        if (data.overlay_image) {
            const img = new Image();
            img.onload = () => {
                const ctx = canvas.getContext('2d');
                ctx.imageSmoothingEnabled = false;
                ctx.drawImage(img, 0, 0);
            };
            img.src = data.overlay_image;
        }
        
        showToast(`Segmented: ${data.mask_pixels} pixels (score: ${data.score.toFixed(2)})`, 'success');
        
    } catch (error) {
        console.error('[point-segment] 에러:', error);
        showToast(error.message || 'Point segmentation failed', 'error');
    } finally {
        showLoading(false);
        state.isProcessing = false;
    }
}
