/**
 * DINO-SAM Labeling Agent - Frontend Application
 */

// ========================================
// State Management
// ========================================

const state = {
    currentImageId: null,
    currentImage: null,
    labelingResult: null,
    isProcessing: false,
    pointSegmentMode: false,  // 클릭 세그멘테이션 모드
};

const DEBUG_MASK_RENDER = true;

// ========================================
// DOM Elements
// ========================================

const elements = {
    // Upload
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    
    // Prompt & Settings
    promptInput: document.getElementById('promptInput'),
    confidenceSlider: document.getElementById('confidenceSlider'),
    confidenceValue: document.getElementById('confidenceValue'),
    formatSelect: document.getElementById('formatSelect'),
    
    // Buttons
    labelBtn: document.getElementById('labelBtn'),
    approveBtn: document.getElementById('approveBtn'),
    rejectBtn: document.getElementById('rejectBtn'),
    exportBtn: document.getElementById('exportBtn'),
    
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
    initializeEventListeners();
    loadStats();
    loadPendingQueue();
});

function initializeEventListeners() {
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
    
    // Label Button
    elements.labelBtn.addEventListener('click', runLabeling);
    
    // HITL Buttons
    elements.approveBtn.addEventListener('click', () => submitFeedback('approved'));
    elements.rejectBtn.addEventListener('click', () => submitFeedback('rejected'));
    
    // Export Button
    elements.exportBtn.addEventListener('click', exportLabels);
    
    // Canvas click for point segmentation
    elements.imageCanvas.addEventListener('click', handleCanvasClick);
}

// ========================================
// Image Upload
// ========================================

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadImage(files[0]);
    }
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
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

function displayImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            const canvas = elements.imageCanvas;
            const ctx = canvas.getContext('2d');
            
            // Calculate canvas size
            const maxWidth = elements.canvasContainer.clientWidth - 48;
            const maxHeight = elements.canvasContainer.clientHeight - 48;
            
            let width = img.width;
            let height = img.height;
            
            if (width > maxWidth) {
                height = (height * maxWidth) / width;
                width = maxWidth;
            }
            if (height > maxHeight) {
                width = (width * maxHeight) / height;
                height = maxHeight;
            }
            
            canvas.width = width;
            canvas.height = height;
            canvas.style.display = 'block';
            
            // Avoid smoothing artifacts on scaled masks/images
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, 0, 0, width, height);
            
            // Store original dimensions
            canvas.dataset.originalWidth = img.width;
            canvas.dataset.originalHeight = img.height;
            
            if (DEBUG_MASK_RENDER) {
                console.log('[mask-debug] image size', {
                    originalWidth: img.width,
                    originalHeight: img.height,
                    canvasWidth: canvas.width,
                    canvasHeight: canvas.height,
                });
            }

            elements.canvasPlaceholder.style.display = 'none';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
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
    
    try {
        const response = await fetch('/api/label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: state.currentImageId,
                prompt: prompt,
                confidence_threshold: parseFloat(elements.confidenceSlider.value),
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
    
    // 서버에서 생성한 오버레이 이미지가 있으면 직접 표시
    if (result.overlay_image) {
        console.log('[overlay] 서버 생성 오버레이 이미지 사용:', result.overlay_image);
        
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, 0, 0);
            
            console.log('[overlay] 오버레이 이미지 표시 완료:', {
                width: img.width,
                height: img.height
            });
        };
        img.onerror = () => {
            console.error('[overlay] 오버레이 이미지 로드 실패:', result.overlay_image);
            showToast('Failed to load overlay image', 'error');
        };
        img.src = result.overlay_image;
        return;
    }
    
    // 폴백: 원본 방식 (클라이언트 렌더링)
    if (!result.boxes_percent) return;
    
    if (state.currentImage) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                // Draw masks first (underneath boxes)
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
                    // If masks are loaded, draw boxes after; otherwise draw immediately
                    drawBoxes(ctx, result, canvas);
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(state.currentImage);
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
    // 이미지 크기에 따른 스케일 (기준 600px)
    const refSize = 600;
    const minDim = Math.min(canvas.width, canvas.height);
    const scale = Math.max(0.5, Math.min(2, minDim / refSize));
    const fontSize = Math.max(10, Math.round(14 * scale));
    const labelHeight = Math.max(18, Math.round(24 * scale));
    const padding = Math.max(4, Math.round(6 * scale));
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

async function exportLabels() {
    if (!state.currentImageId) return;
    
    try {
        const format = elements.formatSelect.value;
        
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_ids: [state.currentImageId],
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
