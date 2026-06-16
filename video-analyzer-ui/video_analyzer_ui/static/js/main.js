// Global state
let currentSession = null;
let outputEventSource = null;
let currentVideoPath = null;
let currentVideoId = null;
let serverVideos = [];
let lastStreamEventId = 0;
let reconnectTimer = null;
let statusPollTimer = null;
let analysisFinished = false;
let disconnectNoticeShown = false;
let analysisDefaults = {};

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const configSection = document.getElementById('configSection');
const outputSection = document.getElementById('outputSection');
const analysisForm = document.getElementById('analysisForm');
const outputText = document.getElementById('outputText');
const commandPreview = document.getElementById('commandPreview');
const downloadResults = document.getElementById('downloadResults');
const newAnalysis = document.getElementById('newAnalysis');
const clientSelect = document.getElementById('client');
const ollamaSettings = document.getElementById('ollamaSettings');
const openaiSettings = document.getElementById('openaiSettings');
const uploadedVideoPath = document.getElementById('uploadedVideoPath');
const uploadProgress = document.getElementById('uploadProgress');
const uploadProgressFill = document.getElementById('uploadProgressFill');
const uploadPercent = document.getElementById('uploadPercent');
const uploadStatus = document.getElementById('uploadStatus');
const serverVideoSelect = document.getElementById('serverVideoSelect');
const serverVideoMeta = document.getElementById('serverVideoMeta');
const refreshVideos = document.getElementById('refreshVideos');
const useServerVideo = document.getElementById('useServerVideo');
const deleteServerVideo = document.getElementById('deleteServerVideo');
const analysisConnection = document.getElementById('analysisConnection');
const analysisState = document.getElementById('analysisState');
const analysisStep = document.getElementById('analysisStep');
const analysisFrameProgress = document.getElementById('analysisFrameProgress');
const analysisProgressFill = document.getElementById('analysisProgressFill');
const analysisStatusMeta = document.getElementById('analysisStatusMeta');
const analysisDefaultsSummary = document.getElementById('analysisDefaultsSummary');
const analysisContextWindow = document.getElementById('analysisContextWindow');
const analysisLegacyHistory = document.getElementById('analysisLegacyHistory');
const analysisChunkSummary = document.getElementById('analysisChunkSummary');
const analysisChunkMaxFrames = document.getElementById('analysisChunkMaxFrames');
const analysisAudioWindow = document.getElementById('analysisAudioWindow');
const analysisMaxImageSide = document.getElementById('analysisMaxImageSide');
const analysisJpegQuality = document.getElementById('analysisJpegQuality');
const analysisLongVideoThreshold = document.getElementById('analysisLongVideoThreshold');
const resultsPreview = document.getElementById('resultsPreview');
const previewFramesProcessed = document.getElementById('previewFramesProcessed');
const previewAudioLanguage = document.getElementById('previewAudioLanguage');
const previewChunkCount = document.getElementById('previewChunkCount');
const previewSceneCount = document.getElementById('previewSceneCount');
const previewBeatCount = document.getElementById('previewBeatCount');
const previewCharacterCount = document.getElementById('previewCharacterCount');
const previewAudioSummary = document.getElementById('previewAudioSummary');
const previewSceneCards = document.getElementById('previewSceneCards');
const previewStoryBeats = document.getElementById('previewStoryBeats');
const previewCharacterTimeline = document.getElementById('previewCharacterTimeline');
const previewScriptGuidance = document.getElementById('previewScriptGuidance');

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
analysisForm.addEventListener('submit', handleAnalysis);
analysisForm.addEventListener('input', updateCommandPreview);
clientSelect.addEventListener('change', toggleClientSettings);
downloadResults.addEventListener('click', downloadAnalysisResults);
newAnalysis.addEventListener('click', resetUI);
refreshVideos.addEventListener('click', loadServerVideos);
useServerVideo.addEventListener('click', selectExistingVideo);
deleteServerVideo.addEventListener('click', deleteExistingVideo);
serverVideoSelect.addEventListener('change', updateSelectedVideoMeta);

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (isValidVideoFile(file)) {
        handleFile(file);
    } else {
        alert('Please upload a valid video file (MP4, AVI, MOV, or MKV)');
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && isValidVideoFile(file)) {
        handleFile(file);
    }
}

function isValidVideoFile(file) {
    const validTypes = ['.mp4', '.avi', '.mov', '.mkv'];
    return validTypes.some(type => file.name.toLowerCase().endsWith(type));
}

async function handleFile(file) {
    const formData = new FormData();
    formData.append('video', file);

    showUploadProgress();
    setUploadProgress(0, 'Uploading video...');

    try {
        const data = await uploadFileWithProgress(formData);
        currentSession = data.session_id;
        currentVideoPath = data.video_path || null;
        currentVideoId = data.video_id || null;
        setUploadProgress(100, 'Upload complete');
        await loadServerVideos();
        showConfigSection();
    } catch (error) {
        setUploadProgress(0, 'Upload failed');
        alert(`Error uploading file: ${error.message}`);
    }
}

function uploadFileWithProgress(formData) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload');

        xhr.upload.addEventListener('progress', (event) => {
            if (!event.lengthComputable) {
                return;
            }
            const percent = Math.round((event.loaded / event.total) * 100);
            setUploadProgress(percent, `Uploading video... ${percent}%`);
        });

        xhr.addEventListener('load', () => {
            let data = {};
            try {
                data = JSON.parse(xhr.responseText || '{}');
            } catch (error) {
                reject(new Error('Invalid upload response'));
                return;
            }

            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(data);
                return;
            }

            reject(new Error(data.error || 'Upload failed'));
        });

        xhr.addEventListener('error', () => reject(new Error('Network error during upload')));
        xhr.send(formData);
    });
}

function showUploadProgress() {
    uploadProgress.style.display = 'block';
}

function setUploadProgress(percent, statusText) {
    const safePercent = Math.max(0, Math.min(100, percent));
    uploadProgressFill.style.width = `${safePercent}%`;
    uploadPercent.textContent = `${safePercent}%`;
    uploadStatus.textContent = statusText;
}

async function loadServerVideos() {
    serverVideoSelect.innerHTML = '<option value="">Loading...</option>';

    try {
        const response = await fetch('/videos');
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to load server videos');
        }

        serverVideos = data.videos || [];
        renderServerVideoOptions();
    } catch (error) {
        serverVideos = [];
        serverVideoSelect.innerHTML = '<option value="">Failed to load server videos</option>';
        serverVideoMeta.textContent = error.message;
    }
}

function renderServerVideoOptions() {
    serverVideoSelect.innerHTML = '';
    if (serverVideos.length === 0) {
        serverVideoSelect.innerHTML = '<option value="">No server videos available</option>';
        serverVideoMeta.textContent = 'Upload a new video or wait for server videos to appear.';
        return;
    }

    for (const video of serverVideos) {
        const option = document.createElement('option');
        option.value = video.video_id;
        option.textContent = `${video.filename} (${formatFileSize(video.size_bytes)})`;
        serverVideoSelect.appendChild(option);
    }

    if (currentVideoId) {
        serverVideoSelect.value = currentVideoId;
    }

    updateSelectedVideoMeta();
}

function updateSelectedVideoMeta() {
    const selected = serverVideos.find((video) => video.video_id === serverVideoSelect.value);
    if (!selected) {
        serverVideoMeta.textContent = 'Select an existing video on the server to continue.';
        return;
    }

    const modifiedAt = new Date(selected.modified_at * 1000).toLocaleString();
    serverVideoMeta.textContent = `Path: ${selected.video_path} | Modified: ${modifiedAt}`;
}

async function selectExistingVideo() {
    const videoId = serverVideoSelect.value;
    if (!videoId) {
        alert('Please select an existing server video first.');
        return;
    }

    try {
        const response = await fetch('/videos/select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ video_id: videoId })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to select existing video');
        }

        currentSession = data.session_id;
        currentVideoPath = data.video_path || null;
        currentVideoId = videoId;
        showConfigSection();
    } catch (error) {
        alert(`Error selecting video: ${error.message}`);
    }
}

async function deleteExistingVideo() {
    const videoId = serverVideoSelect.value;
    if (!videoId) {
        alert('Please select a server video to delete.');
        return;
    }

    if (!window.confirm('Delete the selected server video and its analysis results?')) {
        return;
    }

    try {
        const response = await fetch(`/videos/${videoId}`, {
            method: 'DELETE'
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to delete server video');
        }

        if (currentVideoId === videoId) {
            currentSession = null;
            currentVideoId = null;
            currentVideoPath = null;
            uploadedVideoPath.value = '';
            updateCommandPreview();
        }

        serverVideos = data.videos || [];
        renderServerVideoOptions();
    } catch (error) {
        alert(`Error deleting video: ${error.message}`);
    }
}

function formatFileSize(sizeBytes) {
    if (!sizeBytes) {
        return '0 B';
    }

    const units = ['B', 'KB', 'MB', 'GB'];
    let size = sizeBytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
    }

    return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

// Analysis Handlers
async function handleAnalysis(e) {
    e.preventDefault();
    if (!currentSession) return;
    
    const formData = new FormData(analysisForm);
    showOutputSection();
    
    // Close any existing event source
    if (outputEventSource) {
        outputEventSource.close();
    }
    clearReconnectTimer();
    clearStatusPollTimer();
    lastStreamEventId = 0;
    analysisFinished = false;
    disconnectNoticeShown = false;
    
    // Clear previous output and show loading
    outputText.textContent = '';
    resetAnalysisStatusPanel();
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.innerHTML = '<div class="loading-text">Analyzing video...</div>';
    outputText.parentElement.appendChild(loadingDiv);
    
    try {
        // Make POST request to start analysis
        const response = await fetch(`/analyze/${currentSession}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to start analysis');
        }
        const data = await response.json();
        applyStatusPayload(data.status || {});
    } catch (error) {
        outputText.textContent = `Error starting analysis: ${error.message}\n`;
        document.querySelector('.output-actions').style.display = 'flex';
        downloadResults.style.display = 'none';
        loadingDiv.remove();
        return;
    }

    // Start SSE connection for output
    connectAnalysisStream(loadingDiv);
}

function connectAnalysisStream(loadingDiv) {
    if (!currentSession || analysisFinished) {
        return;
    }

    if (outputEventSource) {
        outputEventSource.close();
    }

    updateConnectionState('Connecting');
    outputEventSource = new EventSource(`/analyze/${currentSession}/stream?last_event_id=${lastStreamEventId}`);

    outputEventSource.onopen = () => {
        updateConnectionState('Connected');
        disconnectNoticeShown = false;
        clearStatusPollTimer();
    };

    outputEventSource.addEventListener('log', (event) => {
        if (loadingDiv.parentElement) {
            loadingDiv.remove();
        }
        handleServerEvent('log', event);
    });

    outputEventSource.addEventListener('status', (event) => {
        if (loadingDiv.parentElement) {
            loadingDiv.remove();
        }
        handleServerEvent('status', event);
    });

    outputEventSource.addEventListener('heartbeat', (event) => {
        handleServerEvent('heartbeat', event);
        updateConnectionState('Connected');
    });

    outputEventSource.onerror = async (error) => {
        console.error('SSE Error:', error);
        if (outputEventSource) {
            outputEventSource.close();
        }

        if (analysisFinished) {
            return;
        }

        if (loadingDiv.parentElement) {
            loadingDiv.remove();
        }

        updateConnectionState('Reconnecting');
        if (!disconnectNoticeShown) {
            disconnectNoticeShown = true;
            outputText.textContent += '\nConnection lost, attempting to reconnect while the backend continues running...\n';
            outputText.scrollTop = outputText.scrollHeight;
        }

        await pollAnalysisStatus();
        if (analysisFinished) {
            return;
        }
        clearReconnectTimer();
        reconnectTimer = setTimeout(() => connectAnalysisStream(loadingDiv), 3000);
    };
}

function handleServerEvent(eventType, event) {
    if (event.lastEventId) {
        lastStreamEventId = Math.max(lastStreamEventId, Number(event.lastEventId));
    }

    let payload = {};
    try {
        payload = JSON.parse(event.data || '{}');
    } catch (error) {
        payload = {};
    }

    if (eventType === 'log' && payload.line) {
        appendOutputLine(payload.line);
    } else {
        applyStatusPayload(payload);
    }
}

async function pollAnalysisStatus() {
    if (!currentSession || analysisFinished) {
        return;
    }

    clearStatusPollTimer();
    try {
        const response = await fetch(`/analyze/${currentSession}/status?last_event_id=${lastStreamEventId}`);
        const data = await response.json();
        if (!response.ok) {
            if (response.status === 404 || response.status === 410 || data.code === 'session_expired') {
                handleExpiredSession(data.error || 'Session expired. Please re-select or upload the video again.');
                return;
            }
            throw new Error(data.error || 'Failed to fetch analysis status');
        }

        for (const event of data.events || []) {
            lastStreamEventId = Math.max(lastStreamEventId, Number(event.id));
            if (event.type === 'log' && event.payload?.line) {
                appendOutputLine(event.payload.line);
            } else {
                applyStatusPayload(event.payload || {});
            }
        }

        applyStatusPayload(data.status || {});

        if (!analysisFinished) {
            statusPollTimer = setTimeout(pollAnalysisStatus, 5000);
        }
    } catch (error) {
        updateConnectionState('Disconnected');
        analysisStatusMeta.textContent = `Status polling failed: ${error.message}`;
        statusPollTimer = setTimeout(pollAnalysisStatus, 5000);
    }
}

function handleExpiredSession(message) {
    analysisFinished = true;
    clearReconnectTimer();
    clearStatusPollTimer();
    if (outputEventSource) {
        outputEventSource.close();
    }
    updateConnectionState('Expired');
    analysisState.textContent = 'expired';
    analysisStep.textContent = 'Session expired';
    analysisStatusMeta.textContent = message;
    appendOutputLine(message);
    currentSession = null;
    document.querySelector('.output-actions').style.display = 'flex';
    downloadResults.style.display = 'none';
}

function appendOutputLine(line) {
    outputText.textContent += `${line}\n`;
    outputText.scrollTop = outputText.scrollHeight;
}

function applyStatusPayload(status) {
    if (!status || Object.keys(status).length === 0) {
        return;
    }

    analysisState.textContent = status.state || 'Unknown';
    analysisStep.textContent = status.step_label || status.step || 'Waiting';
    analysisFrameProgress.textContent = `${status.analyzed_frames || 0} / ${status.total_frames || 0}`;
    analysisProgressFill.style.width = `${status.progress_percent || 0}%`;
    analysisStatusMeta.textContent = status.last_log || status.error || 'Waiting for backend updates...';

    if (status.state === 'completed') {
        finalizeAnalysis(true, status.results_ready);
    } else if (status.state === 'failed') {
        finalizeAnalysis(false, status.results_ready);
    }
}

function updateConnectionState(text) {
    analysisConnection.textContent = text;
}

async function finalizeAnalysis(success, resultsReady = null) {
    if (analysisFinished) {
        if (success && resultsReady) {
            downloadResults.style.display = 'inline-block';
        }
        return;
    }

    analysisFinished = true;
    updateConnectionState(success ? 'Completed' : 'Closed');
    clearReconnectTimer();
    clearStatusPollTimer();
    if (outputEventSource) {
        outputEventSource.close();
    }

    document.querySelector('.output-actions').style.display = 'flex';
    downloadResults.style.display = success && resultsReady ? 'inline-block' : 'none';
    if (success && resultsReady) {
        await loadResultsPreview();
    }
    if (!success) {
        appendOutputLine('Analysis failed. Please check the output above for errors.');
    }
}

function resetAnalysisStatusPanel() {
    analysisConnection.textContent = 'Starting';
    analysisState.textContent = 'queued';
    analysisStep.textContent = 'Waiting to start';
    analysisFrameProgress.textContent = '0 / 0';
    analysisProgressFill.style.width = '0%';
    analysisStatusMeta.textContent = 'Preparing backend task...';
}

function clearReconnectTimer() {
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
}

function clearStatusPollTimer() {
    if (statusPollTimer) {
        clearTimeout(statusPollTimer);
        statusPollTimer = null;
    }
}

// UI Updates
function showConfigSection() {
    dropZone.style.display = 'none';
    configSection.style.display = 'block';
    uploadedVideoPath.value = currentVideoPath || '';
    renderAnalysisDefaults();
    updateCommandPreview();
}

function showOutputSection() {
    configSection.style.display = 'none';
    outputSection.style.display = 'block';
    document.querySelector('.output-actions').style.display = 'none';
}

function toggleClientSettings() {
    const client = clientSelect.value;
    if (client === 'ollama') {
        ollamaSettings.style.display = 'block';
        openaiSettings.style.display = 'none';
    } else {
        ollamaSettings.style.display = 'none';
        openaiSettings.style.display = 'block';
    }
    updateCommandPreview();
}

function updateCommandPreview() {
    const formData = new FormData(analysisForm);
    const videoPath = currentVideoPath || '<video_path>';
    let command = `video-analyzer ${videoPath}`;
    
    for (const [key, value] of formData.entries()) {
        if (value) {
            if (key === 'keep-frames') {
                command += ` --${key}`;
            } else {
                command += ` --${key} ${value}`;
            }
        }
    }
    
    commandPreview.textContent = command;
}

function formatBoolean(value) {
    return value ? 'Enabled' : 'Disabled';
}

function renderAnalysisDefaults() {
    const defaults = analysisDefaults || {};
    analysisContextWindow.textContent = defaults.context_window ?? '-';
    analysisLegacyHistory.textContent = formatBoolean(Boolean(defaults.legacy_full_history));
    analysisChunkSummary.textContent = formatBoolean(Boolean(defaults.enable_chunk_summary));
    analysisChunkMaxFrames.textContent = defaults.chunk_max_frames ?? '-';
    analysisAudioWindow.textContent = defaults.audio_alignment_window ?? '-';
    analysisMaxImageSide.textContent = defaults.max_image_side ?? '-';
    analysisJpegQuality.textContent = defaults.image_jpeg_quality ?? '-';
    analysisLongVideoThreshold.textContent = defaults.long_video_threshold_minutes ?? '-';

    const defaultItems = [
        `context_window=${defaults.context_window ?? '-'}`,
        `legacy_full_history=${Boolean(defaults.legacy_full_history)}`,
        `chunk_summary=${Boolean(defaults.enable_chunk_summary)}`,
        `audio_alignment_window=${defaults.audio_alignment_window ?? '-'}`,
        `max_image_side=${defaults.max_image_side ?? '-'}`,
        `image_jpeg_quality=${defaults.image_jpeg_quality ?? '-'}`,
    ];
    analysisDefaultsSummary.textContent = `Loaded from /api/config: ${defaultItems.join(', ')}`;
}

function truncateText(text, maxLength = 360) {
    if (!text) {
        return '';
    }
    if (text.length <= maxLength) {
        return text;
    }
    return `${text.slice(0, maxLength)}...`;
}

function formatPreviewJson(value) {
    return JSON.stringify(value, null, 2);
}

async function loadResultsPreview() {
    if (!currentSession) {
        return;
    }

    try {
        const response = await fetch(`/results/${currentSession}/preview`);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to load structured result preview');
        }
        renderResultsPreview(data);
    } catch (error) {
        resultsPreview.style.display = 'block';
        previewAudioSummary.textContent = `Failed to load structured result preview: ${error.message}`;
        previewSceneCards.textContent = '[]';
        previewStoryBeats.textContent = '[]';
        previewCharacterTimeline.textContent = '[]';
        previewScriptGuidance.textContent = '{}';
    }
}

function renderResultsPreview(data) {
    const metadata = data.metadata || {};
    const transcript = data.transcript || {};
    const audioSummary = data.audio_summary || {};
    const chunkSummaries = data.chunk_summaries || [];
    const sceneCards = data.scene_cards || [];
    const storyBeats = data.story_beats || [];
    const characterTimeline = data.character_timeline || [];
    const scriptGuidance = data.script_guidance || {};

    resultsPreview.style.display = 'block';
    previewFramesProcessed.textContent = metadata.frames_processed ?? metadata.frames_extracted ?? '-';
    previewAudioLanguage.textContent = metadata.audio_language || audioSummary.language || transcript.language || '-';
    previewChunkCount.textContent = chunkSummaries.length;
    previewSceneCount.textContent = sceneCards.length;
    previewBeatCount.textContent = storyBeats.length;
    previewCharacterCount.textContent = characterTimeline.length;

    const audioBlockSummary = (audioSummary.time_blocks || [])
        .slice(0, 2)
        .map((block) => `${block.start}-${block.end}s: ${block.summary}`)
        .join('\n');
    previewAudioSummary.textContent = truncateText(
        audioBlockSummary || (audioSummary.key_dialogues || []).join(' | ') || 'No audio summary available.',
        420
    );
    previewSceneCards.textContent = formatPreviewJson(sceneCards.slice(0, 3));
    previewStoryBeats.textContent = formatPreviewJson(storyBeats.slice(0, 3));
    previewCharacterTimeline.textContent = formatPreviewJson(characterTimeline.slice(0, 3));
    previewScriptGuidance.textContent = formatPreviewJson(scriptGuidance);
}

// Results Handling
async function downloadAnalysisResults() {
    if (!currentSession) return;
    
    try {
        const response = await fetch(`/results/${currentSession}`);
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to fetch results');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'analysis.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        alert(`Error downloading results: ${error.message}`);
        console.error('Download error:', error);
    }
}

function resetUI() {
    // Clean up current session
    if (currentSession) {
        fetch(`/cleanup/${currentSession}`, { method: 'POST' })
            .catch(error => console.error('Cleanup error:', error));
    }
    
    // Reset state
    currentSession = null;
    currentVideoPath = null;
    currentVideoId = null;
    if (outputEventSource) {
        outputEventSource.close();
    }
    clearReconnectTimer();
    clearStatusPollTimer();
    analysisFinished = false;
    disconnectNoticeShown = false;
    lastStreamEventId = 0;
    
    // Reset form
    analysisForm.reset();
    
    // Reset UI
    dropZone.style.display = 'block';
    configSection.style.display = 'none';
    outputSection.style.display = 'none';
    outputText.textContent = '';
    resultsPreview.style.display = 'none';
    previewFramesProcessed.textContent = '-';
    previewAudioLanguage.textContent = '-';
    previewChunkCount.textContent = '-';
    previewSceneCount.textContent = '-';
    previewBeatCount.textContent = '-';
    previewCharacterCount.textContent = '-';
    previewAudioSummary.textContent = 'No audio summary available.';
    previewSceneCards.textContent = '[]';
    previewStoryBeats.textContent = '[]';
    previewCharacterTimeline.textContent = '[]';
    previewScriptGuidance.textContent = '{}';
    fileInput.value = '';
    uploadedVideoPath.value = '';
    setUploadProgress(0, 'Uploading video...');
    uploadProgress.style.display = 'none';
    resetAnalysisStatusPanel();
    
    // Reset client settings
    toggleClientSettings();
    loadServerVideos();
}

// Initialize UI
toggleClientSettings();
loadServerVideos();
resetAnalysisStatusPanel();
populateClientConfigFromServer();

async function populateClientConfigFromServer() {
    // Pull the `clients` node from the analyzer config so the form mirrors
    // whatever the CLI would actually use (user config > default config).
    try {
        const response = await fetch('/api/config');
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data || data.available !== true) {
            return;
        }

        const clients = data.clients || {};
        analysisDefaults = data.analysis || {};
        renderAnalysisDefaults();
        const defaultClient = clients.default;
        const knownClients = new Set(['ollama', 'openai_api']);

        if (defaultClient && knownClients.has(defaultClient)) {
            clientSelect.value = defaultClient;
            toggleClientSettings();
        }

        const ollama = clients.ollama || {};
        const openaiApi = clients.openai_api || {};

        const ollamaUrlInput = document.getElementById('ollama-url');
        if (ollamaUrlInput && typeof ollama.url === 'string' && ollama.url) {
            ollamaUrlInput.value = ollama.url;
        }

        const apiKeyInput = document.getElementById('api-key');
        if (apiKeyInput && typeof openaiApi.api_key === 'string') {
            apiKeyInput.value = openaiApi.api_key;
        }

        const apiUrlInput = document.getElementById('api-url');
        if (apiUrlInput && typeof openaiApi.api_url === 'string' && openaiApi.api_url) {
            apiUrlInput.value = openaiApi.api_url;
        }

        const activeClient = clientSelect.value;
        const activeConfig = activeClient === 'openai_api' ? openaiApi : ollama;
        const modelInput = document.getElementById('model');
        if (modelInput && activeConfig && typeof activeConfig.model === 'string' && activeConfig.model) {
            modelInput.value = activeConfig.model;
        }

        updateCommandPreview();
    } catch (error) {
        // Non-fatal: leave the form defaults in place.
        console.warn('Failed to load client config from server:', error);
        analysisDefaultsSummary.textContent = 'Failed to load analysis defaults from server config.';
    }
}
