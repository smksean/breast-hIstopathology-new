// State
let currentMode = 'single';
let selectedFiles = [];
const API_URL = 'https://breast-histopathology-new-production.up.railway.app';

// Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');
const previewGrid = document.getElementById('previewGrid');
const fileCount = document.getElementById('fileCount');
const actionButtons = document.getElementById('actionButtons');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const resultsContainer = document.getElementById('resultsContainer');
const uploadTitle = document.getElementById('uploadTitle');
const uploadDesc = document.getElementById('uploadDesc');
const analyzeText = document.getElementById('analyzeText');

// Check API health
async function checkAPI() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        const apiStatus = document.getElementById('apiStatus');
        const apiStatusText = document.getElementById('apiStatusText');
        const apiError = document.getElementById('apiError');
        
        if (data.status === 'healthy') {
            apiStatus.className = 'api-status healthy';
            apiStatusText.textContent = 'API Connected';
            apiError.style.display = 'none';
            return true;
        } else {
            apiStatus.className = 'api-status offline';
            apiStatusText.textContent = 'API Offline';
            apiError.style.display = 'flex';
            return false;
        }
    } catch (error) {
        const apiStatus = document.getElementById('apiStatus');
        const apiStatusText = document.getElementById('apiStatusText');
        const apiError = document.getElementById('apiError');
        
        apiStatus.className = 'api-status offline';
        apiStatusText.textContent = 'API Offline';
        apiError.style.display = 'flex';
        return false;
    }
}

// Mode selector
document.querySelectorAll('.mode-option').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-option').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        currentMode = btn.dataset.mode;
        selectedFiles = [];
        updateUI();
        
        if (currentMode === 'single') {
            fileInput.removeAttribute('multiple');
            uploadTitle.textContent = 'Upload Image';
            uploadDesc.textContent = 'Upload a histopathology image for diagnosis';
        } else {
            fileInput.setAttribute('multiple', 'multiple');
            uploadTitle.textContent = 'Upload Multiple Images';
            uploadDesc.textContent = 'Upload multiple images from the same patient';
        }
    });
});

// Drag & Drop
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('active');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('active');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('active');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    const fileArray = Array.from(files);
    
    if (currentMode === 'single' && fileArray.length > 1) {
        alert('Single image mode: Please select only one image.');
        return;
    }
    
    selectedFiles = fileArray;
    updateUI();
}

function updateUI() {
    if (selectedFiles.length === 0) {
        filePreview.style.display = 'none';
        actionButtons.style.display = 'none';
        resultsContainer.innerHTML = '';
        return;
    }
    
    // Show preview
    filePreview.style.display = 'block';
    actionButtons.style.display = 'flex';
    fileCount.textContent = `${selectedFiles.length} ${selectedFiles.length === 1 ? 'file' : 'files'} selected`;
    
    // Update analyze button
    analyzeText.textContent = currentMode === 'single' ? 'Analyze Image' : `Analyze ${selectedFiles.length} Images`;
    
    // Render previews
    previewGrid.innerHTML = '';
    selectedFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'preview-item';
        
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.onload = () => URL.revokeObjectURL(img.src);
        
        const info = document.createElement('div');
        info.className = 'preview-info';
        info.innerHTML = `
            <div class="file-name">${file.name}</div>
            <div class="file-size">${(file.size / 1024).toFixed(1)} KB</div>
        `;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => removeFile(index);
        
        item.appendChild(img);
        item.appendChild(info);
        item.appendChild(removeBtn);
        previewGrid.appendChild(item);
    });
    
    lucide.createIcons();
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateUI();
}

clearBtn.addEventListener('click', () => {
    selectedFiles = [];
    fileInput.value = '';
    updateUI();
});

// Analyze
analyzeBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;
    
    const apiHealthy = await checkAPI();
    if (!apiHealthy) {
        alert('API server is not running. Please start it with: python api.py');
        return;
    }
    
    loading.style.display = 'block';
    actionButtons.style.display = 'none';
    resultsContainer.innerHTML = '';
    
    try {
        if (currentMode === 'single') {
            await analyzeSingle();
        } else {
            await analyzeMultiple();
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loading.style.display = 'none';
        actionButtons.style.display = 'flex';
    }
});

async function analyzeSingle() {
    const formData = new FormData();
    formData.append('file', selectedFiles[0]);
    
    const response = await fetch(`${API_URL}/predict/single`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) throw new Error('Prediction failed');
    
    const data = await response.json();
    displaySingleResult(data);
}

async function analyzeMultiple() {
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    const response = await fetch(`${API_URL}/predict/folder`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) throw new Error('Batch prediction failed');
    
    const data = await response.json();
    displayMultipleResults(data);
}

function displaySingleResult(data) {
    const isBenign = data.prediction === 'benign';
    const colorClass = isBenign ? 'benign' : 'malignant';
    
    resultsContainer.innerHTML = `
        <div class="diagnosis-card ${colorClass}">
            <div class="diagnosis-header">
                <i data-lucide="${isBenign ? 'check-circle' : 'alert-triangle'}" style="width: 48px; height: 48px;"></i>
                <div>
                    <h2>Diagnosis: ${data.prediction.toUpperCase()}</h2>
                    <p class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
                </div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Patches</div>
                <div class="metric-value">${data.num_patches}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Benign Patches</div>
                <div class="metric-value">${data.patch_breakdown.benign_patches}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Malignant Patches</div>
                <div class="metric-value">${data.patch_breakdown.malignant_patches}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Class Probabilities</h3>
            <canvas id="probChart" width="400" height="200"></canvas>
        </div>
    `;
    
    lucide.createIcons();
    
    // Create chart
    const ctx = document.getElementById('probChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Benign', 'Malignant'],
            datasets: [{
                label: 'Probability (%)',
                data: [
                    (data.probabilities.benign * 100).toFixed(2),
                    (data.probabilities.malignant * 100).toFixed(2)
                ],
                backgroundColor: ['#4CAF50', '#f44336']
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

function displayMultipleResults(data) {
    const agg = data.aggregated_diagnosis;
    const isBenign = agg.prediction === 'benign';
    const colorClass = isBenign ? 'benign' : 'malignant';
    
    resultsContainer.innerHTML = `
        <div class="diagnosis-card ${colorClass}">
            <div class="diagnosis-header">
                <i data-lucide="${isBenign ? 'check-circle' : 'alert-triangle'}" style="width: 48px; height: 48px;"></i>
                <div>
                    <h2>Aggregated Diagnosis: ${agg.prediction.toUpperCase()}</h2>
                    <p class="confidence">Confidence: ${(agg.confidence * 100).toFixed(2)}%</p>
                    <p>Based on analysis of ${data.num_images} images</p>
                </div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Images</div>
                <div class="metric-value">${data.num_images}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Benign Images</div>
                <div class="metric-value">${data.image_breakdown.benign_images}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Malignant Images</div>
                <div class="metric-value">${data.image_breakdown.malignant_images}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Average Probabilities Across All Images</h3>
            <canvas id="probChart" width="400" height="200"></canvas>
        </div>
    `;
    
    lucide.createIcons();
    
    // Create chart
    const ctx = document.getElementById('probChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Benign', 'Malignant'],
            datasets: [{
                label: 'Probability (%)',
                data: [
                    (agg.probabilities.benign * 100).toFixed(2),
                    (agg.probabilities.malignant * 100).toFixed(2)
                ],
                backgroundColor: ['#4CAF50', '#f44336']
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// Initialize
checkAPI();

