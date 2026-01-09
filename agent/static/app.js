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
};

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
            
            ctx.drawImage(img, 0, 0, width, height);
            
            // Store original dimensions
            canvas.dataset.originalWidth = img.width;
            canvas.dataset.originalHeight = img.height;
            
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
    if (!result || !result.boxes_percent) return;
    
    const canvas = elements.imageCanvas;
    const ctx = canvas.getContext('2d');
    
    // Redraw original image first
    if (state.currentImage) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                // Draw boxes
                result.boxes_percent.forEach((box, index) => {
                    const x = (box.x / 100) * canvas.width;
                    const y = (box.y / 100) * canvas.height;
                    const w = (box.width / 100) * canvas.width;
                    const h = (box.height / 100) * canvas.height;
                    
                    // Get color for this class
                    const color = getClassColor(index);
                    
                    // Draw box
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, w, h);
                    
                    // Draw label background
                    const label = result.labels[index];
                    const score = (result.scores[index] * 100).toFixed(1);
                    const text = `${label} ${score}%`;
                    
                    ctx.font = 'bold 12px Inter, sans-serif';
                    const textWidth = ctx.measureText(text).width;
                    
                    ctx.fillStyle = color;
                    ctx.fillRect(x, y - 20, textWidth + 8, 20);
                    
                    // Draw label text
                    ctx.fillStyle = '#fff';
                    ctx.fillText(text, x + 4, y - 6);
                });
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(state.currentImage);
    }
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
