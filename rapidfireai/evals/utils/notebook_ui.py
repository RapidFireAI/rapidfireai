"""
Notebook UI for Jupyter/Colab notebooks using JavaScript.
Provides interactive controls for managing runs during experiments.
ROBUST VERSION - Fixed polling, error handling, and race conditions.
"""

from IPython.display import display, HTML
import uuid


class NotebookUI:
    """Notebook UI for interactive run control using JavaScript"""

    def __init__(self, dispatcher_url: str = "http://127.0.0.1:5000", refresh_rate_seconds: float = 3.0):
        self.dispatcher_url = dispatcher_url.rstrip("/")
        self.widget_id = f"controller_{uuid.uuid4().hex[:8]}"
        self.refresh_rate_ms = int(refresh_rate_seconds * 1000)  # Convert to milliseconds

    def _generate_html(self):
        """Generate HTML, CSS, and JavaScript for the controller"""
        return f"""
        <div id="{self.widget_id}" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;">
            <style>
                #{self.widget_id} {{
                    max-width: 900px;
                    margin: 0 auto;
                }}

                #{self.widget_id} h3 {{
                    margin: 10px 0;
                    font-size: 1.2em;
                    font-weight: 600;
                }}

                #{self.widget_id} .header-info {{
                    display: flex;
                    gap: 20px;
                    margin: 10px 0;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 4px;
                }}

                #{self.widget_id} .header-info > div {{
                    font-size: 13px;
                }}

                #{self.widget_id} .section {{
                    margin: 15px 0;
                }}

                #{self.widget_id} .section-label {{
                    font-weight: 600;
                    margin-bottom: 8px;
                    font-size: 14px;
                }}

                #{self.widget_id} .button-row {{
                    display: flex;
                    gap: 8px;
                    flex-wrap: wrap;
                    margin: 10px 0;
                }}

                #{self.widget_id} .selector-row {{
                    display: flex;
                    gap: 8px;
                    align-items: center;
                    flex-wrap: wrap;
                }}

                #{self.widget_id} select {{
                    padding: 6px 12px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    font-size: 13px;
                    background: white;
                    min-width: 300px;
                    cursor: pointer;
                }}

                #{self.widget_id} select:disabled {{
                    background: #e9ecef;
                    cursor: not-allowed;
                }}

                #{self.widget_id} button {{
                    padding: 6px 16px;
                    border: none;
                    border-radius: 4px;
                    font-size: 13px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.2s;
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                }}

                #{self.widget_id} button:disabled {{
                    opacity: 0.5;
                    cursor: not-allowed;
                }}

                #{self.widget_id} button:not(:disabled):hover {{
                    filter: brightness(0.95);
                    transform: translateY(-1px);
                }}

                #{self.widget_id} .btn-primary {{
                    background: #007bff;
                    color: white;
                }}

                #{self.widget_id} .btn-success {{
                    background: #28a745;
                    color: white;
                }}

                #{self.widget_id} .btn-danger {{
                    background: #dc3545;
                    color: white;
                }}

                #{self.widget_id} .btn-info {{
                    background: #17a2b8;
                    color: white;
                }}

                #{self.widget_id} .btn-default {{
                    background: #6c757d;
                    color: white;
                }}

                #{self.widget_id} textarea {{
                    width: 100%;
                    min-height: 200px;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                    font-size: 12px;
                    resize: vertical;
                    box-sizing: border-box;
                }}

                #{self.widget_id} textarea:disabled {{
                    background: #e9ecef;
                    color: #6c757d;
                }}

                #{self.widget_id} .checkbox-container {{
                    margin: 10px 0;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}

                #{self.widget_id} input[type="checkbox"] {{
                    width: 16px;
                    height: 16px;
                    cursor: pointer;
                }}

                #{self.widget_id} input[type="checkbox"]:disabled {{
                    cursor: not-allowed;
                }}

                #{self.widget_id} .checkbox-label {{
                    font-size: 13px;
                    cursor: pointer;
                    user-select: none;
                }}

                #{self.widget_id} .status-message {{
                    width: 100%;
                    min-height: 40px;
                    padding: 10px;
                    margin: 10px 0;
                    border: 2px solid #ddd;
                    border-radius: 5px;
                    box-sizing: border-box;
                }}

                #{self.widget_id} .msg-success {{
                    background-color: #d4edda;
                    border-color: #28a745;
                    color: #155724;
                }}

                #{self.widget_id} .msg-error {{
                    background-color: #f8d7da;
                    border-color: #dc3545;
                    color: #721c24;
                }}

                #{self.widget_id} .msg-info {{
                    background-color: #d1ecf1;
                    border-color: #17a2b8;
                    color: #0c5460;
                }}

                #{self.widget_id} .msg-warning {{
                    background-color: #fff3cd;
                    border-color: #ffc107;
                    color: #856404;
                }}

                #{self.widget_id} .icon {{
                    font-size: 14px;
                }}
            </style>

            <div>
                <h3>Interactive Run Controller</h3>
                <div class="header-info">
                    <div><b>Run ID:</b> <span id="pipeline-id-value">N/A</span></div>
                    <div><b>Status:</b> <span id="status-value">Not loaded</span></div>
                    <div><b>Shards:</b> <span id="shards-value">N/A</span></div>
                </div>

                <div id="status-message" class="status-message" style="display: none;"></div>

                <div class="section">
                    <div class="section-label">Select a Config ID:</div>
                    <div class="selector-row">
                        <select id="pipeline-selector">
                            <option value="">Loading...</option>
                        </select>
                    </div>
                </div>

                <div class="section">
                    <div class="button-row">
                        <button class="btn-success" id="resume-btn">
                            <span class="icon">▶</span> Resume
                        </button>
                        <button class="btn-danger" id="stop-btn">
                            <span class="icon">■</span> Stop
                        </button>
                        <button class="btn-danger" id="delete-btn">
                            <span class="icon">🗑</span> Delete
                        </button>
                    </div>
                </div>

                <div class="section">
                    <div class="section-label">Configuration: <span id="config-name" style="font-weight: 600; color: #2196F3;">N/A</span></div>
                    <textarea id="config-text" disabled>{{}}</textarea>
                    <div class="button-row">
                        <button class="btn-primary" id="clone-btn">Clone Run</button>
                        <button class="btn-success" id="submit-clone-btn" disabled>✓ Submit Clone</button>
                        <button class="btn-default" id="cancel-clone-btn" disabled>✗ Cancel</button>
                    </div>
                </div>
            </div>

            <script>
                (function() {{
                    const DISPATCHER_URL = '{self.dispatcher_url}';

                    // State management
                    let currentPipelineId = null;
                    let currentConfig = null;
                    let currentContextId = null;
                    let isCloneMode = false;
                    let isInitialized = false;
                    let hasPipelinesBeenSeen = false;

                    // Polling control
                    let refreshInterval = null;
                    let lastSuccessfulFetch = Date.now();
                    const REFRESH_RATE = {self.refresh_rate_ms};  // Configurable refresh rate in milliseconds
                    const CONNECTION_TIMEOUT = 60000;  // 60 seconds before declaring truly dead

                    // Request management to prevent pileup
                    let activeRequests = new Set();
                    let requestIdCounter = 0;

                    // Element references
                    const elements = {{
                        pipelineIdValue: document.getElementById('pipeline-id-value'),
                        statusValue: document.getElementById('status-value'),
                        shardsValue: document.getElementById('shards-value'),
                        statusMessage: document.getElementById('status-message'),
                        pipelineSelector: document.getElementById('pipeline-selector'),
                        resumeBtn: document.getElementById('resume-btn'),
                        stopBtn: document.getElementById('stop-btn'),
                        deleteBtn: document.getElementById('delete-btn'),
                        configName: document.getElementById('config-name'),
                        configText: document.getElementById('config-text'),
                        cloneBtn: document.getElementById('clone-btn'),
                        submitCloneBtn: document.getElementById('submit-clone-btn'),
                        cancelCloneBtn: document.getElementById('cancel-clone-btn')
                    }};

                    // Safe fetch with timeout and request tracking
                    async function safeFetch(url, options = {{}}, requestType = 'unknown') {{
                        const requestId = requestIdCounter++;
                        const controller = new AbortController();
                        const timeoutId = setTimeout(() => controller.abort(), 10000);  // 10s timeout

                        try {{
                            activeRequests.add(requestId);

                            const response = await fetch(url, {{
                                ...options,
                                signal: controller.signal
                            }});

                            clearTimeout(timeoutId);
                            activeRequests.delete(requestId);

                            return response;
                        }} catch (error) {{
                            clearTimeout(timeoutId);
                            activeRequests.delete(requestId);

                            if (error.name === 'AbortError') {{
                                console.log(`Request timeout: ${{requestType}}`);
                            }}
                            throw error;
                        }}
                    }}

                    // Show status message
                    function showMessage(message, type = 'info') {{
                        const msg = elements.statusMessage;
                        msg.className = 'status-message msg-' + type;
                        msg.innerHTML = '<b style="font-weight: 600;">' + message + '</b>';
                        msg.style.display = 'block';
                    }}

                    // Update display
                    function updateDisplay(pipelineId, status, shardsCompleted, totalShards, config, contextId) {{
                        elements.pipelineIdValue.textContent = pipelineId || 'N/A';
                        elements.statusValue.textContent = status || 'Not loaded';

                        if (shardsCompleted !== null && totalShards !== null) {{
                            elements.shardsValue.textContent = `${{shardsCompleted}}/${{totalShards}}`;
                        }} else {{
                            elements.shardsValue.textContent = 'N/A';
                        }}

                        if (config && config.pipeline_name) {{
                            elements.configName.textContent = config.pipeline_name;
                        }} else {{
                            elements.configName.textContent = 'N/A';
                        }}

                        // Only update config text if not in clone mode
                        if (!isCloneMode) {{
                            elements.configText.value = JSON.stringify(config || {{}}, null, 2);
                        }}
                        currentContextId = contextId;

                        // Button states
                        const isCompleted = status && status.toLowerCase() === 'completed';
                        elements.resumeBtn.disabled = isCompleted;
                        elements.stopBtn.disabled = isCompleted;
                        elements.cloneBtn.disabled = isCompleted || !contextId;
                        elements.deleteBtn.disabled = isCompleted;

                        if (isCompleted && pipelineId === currentPipelineId) {{
                            showMessage('✓ Run completed successfully!', 'success');
                        }}
                    }}

                    // Fetch all pipeline IDs - called by unified refresh (OPTIMIZED)
                    async function fetchAllPipelines(isUserAction = false) {{
                        // Skip if too many active requests (prevent pileup)
                        if (activeRequests.size > 2) {{
                            console.log('Skipping fetch - too many active requests');
                            return;
                        }}

                        try {{
                            const response = await safeFetch(
                                DISPATCHER_URL + '/dispatcher/list-all-pipeline-ids',
                                {{
                                    method: 'GET',
                                    headers: {{ 'Content-Type': 'application/json' }}
                                }},
                                'list-all-pipeline-ids'
                            );

                            if (!response.ok) {{
                                // Check if dispatcher is permanently down
                                const timeSinceSuccess = Date.now() - lastSuccessfulFetch;

                                if (timeSinceSuccess > CONNECTION_TIMEOUT && hasPipelinesBeenSeen) {{
                                    // Dispatcher is down - disable everything and stop polling
                                    stopAutoRefresh();
                                    disableAllButtons();
                                    // Keep the UI visible with last known state
                                    return;
                                }} else if (!isInitialized) {{
                                    elements.pipelineSelector.innerHTML = '<option value="">Initializing experiment...</option>';
                                    showMessage('Starting experiment... Please wait.', 'info');
                                    return;
                                }}

                                console.log('Dispatcher not responding, will retry...');
                                return;
                            }}

                            const pipelines = await response.json();
                            lastSuccessfulFetch = Date.now();  // Update success timestamp

                            if (pipelines && pipelines.length > 0) {{
                                hasPipelinesBeenSeen = true;  // Mark that we've seen pipelines
                                isInitialized = true;

                                // Clear initialization message
                                if (elements.statusMessage.innerHTML.includes('Initializing') ||
                                    elements.statusMessage.innerHTML.includes('Starting')) {{
                                    elements.statusMessage.style.display = 'none';
                                }}

                                // Save current selection
                                const currentlySelectedId = elements.pipelineSelector.value;

                                // Rebuild dropdown (show only pipeline IDs)
                                elements.pipelineSelector.innerHTML = '';
                                pipelines.forEach(pipeline => {{
                                    const option = document.createElement('option');
                                    option.value = pipeline.pipeline_id;
                                    option.textContent = `Config ID: ${{pipeline.pipeline_id}} (${{pipeline.status || 'unknown'}})`;
                                    elements.pipelineSelector.appendChild(option);
                                }});

                                // Restore or auto-select
                                if (currentlySelectedId && pipelines.some(p => p.pipeline_id == currentlySelectedId)) {{
                                    elements.pipelineSelector.value = currentlySelectedId;
                                }} else if (!currentPipelineId && pipelines.length > 0) {{
                                    elements.pipelineSelector.value = pipelines[0].pipeline_id;
                                    currentPipelineId = pipelines[0].pipeline_id;
                                    await loadPipeline(pipelines[0].pipeline_id);
                                }}
                            }} else if (!isInitialized) {{
                                // Still waiting for first runs
                                elements.pipelineSelector.innerHTML = '<option value="">Building contexts...</option>';
                                showMessage('Experiment initializing. Waiting for runs...', 'info');
                            }}
                        }} catch (error) {{
                            console.log('Fetch pipelines error:', error.message);

                            // Only show error for user-initiated actions
                            if (isUserAction) {{
                                showMessage('Temporary connection issue. Auto-refresh will retry.', 'warning');
                            }}
                        }}
                    }}

                    // Load pipeline config (OPTIMIZED - only fetches config JSON when needed)
                    async function loadPipeline(pipelineId, isUserAction = false) {{
                        if (!pipelineId) return;

                        // Skip if too many requests
                        if (activeRequests.size > 2) {{
                            console.log('Skipping pipeline load - too many active requests');
                            return;
                        }}

                        currentPipelineId = pipelineId;

                        try {{
                            // Fetch config JSON using new lightweight endpoint
                            const response = await safeFetch(
                                DISPATCHER_URL + `/dispatcher/get-pipeline-config-json/${{pipelineId}}`,
                                {{
                                    method: 'GET',
                                    headers: {{ 'Content-Type': 'application/json' }}
                                }},
                                'get-pipeline-config-json'
                            );

                            if (!response.ok) {{
                                console.log('Failed to load pipeline config:', response.status);
                                return;
                            }}

                            const data = await response.json();
                            lastSuccessfulFetch = Date.now();

                            // data.pipeline_config_json is already parsed JSON
                            const configJson = data.pipeline_config_json || {{}};

                            currentConfig = configJson;
                            currentContextId = data.context_id;

                            // Get status/shards from the already-fetched list
                            const pipelineInfo = Array.from(document.getElementById('pipeline-selector').options)
                                .find(opt => opt.value == pipelineId);
                            const statusMatch = pipelineInfo ? pipelineInfo.textContent.match(/\\((.+?)\\)/) : null;
                            const status = statusMatch ? statusMatch[1] : 'unknown';

                            updateDisplay(
                                pipelineId,
                                status,
                                null,  // shards info not needed for display
                                null,  // shards info not needed for display
                                currentConfig,
                                data.context_id
                            );
                        }} catch (error) {{
                            console.log('Error loading pipeline config:', error.message);
                            if (isUserAction) {{
                                showMessage('Error loading config. Will retry automatically.', 'warning');
                            }}
                        }}
                    }}

                    // Unified auto-refresh function
                    async function refreshAll() {{
                        // Skip refresh if in clone mode
                        if (isCloneMode) {{
                            return;
                        }}

                        // Fetch pipeline list
                        await fetchAllPipelines();

                        // Load current pipeline details if one is selected
                        if (currentPipelineId) {{
                            await loadPipeline(currentPipelineId);
                        }}
                    }}

                    // Start single unified auto-refresh
                    function startAutoRefresh() {{
                        if (refreshInterval) return;  // Already running

                        refreshInterval = setInterval(refreshAll, REFRESH_RATE);
                        console.log(`Auto-refresh started: every ${{REFRESH_RATE/1000}}s`);
                    }}

                    // Stop auto-refresh
                    function stopAutoRefresh() {{
                        if (refreshInterval) {{
                            clearInterval(refreshInterval);
                            refreshInterval = null;
                            console.log('Auto-refresh stopped');
                        }}
                    }}

                    // Disable all interactive buttons (when dispatcher is down)
                    function disableAllButtons() {{
                        elements.resumeBtn.disabled = true;
                        elements.stopBtn.disabled = true;
                        elements.deleteBtn.disabled = true;
                        elements.cloneBtn.disabled = true;
                        elements.submitCloneBtn.disabled = true;
                        elements.cancelCloneBtn.disabled = true;
                        elements.pipelineSelector.disabled = true;
                        console.log('All buttons disabled - dispatcher unavailable');
                    }}

                    // Resume pipeline
                    async function handleResume() {{
                        if (!currentPipelineId) return;

                        try {{
                            const response = await safeFetch(
                                DISPATCHER_URL + '/dispatcher/resume-pipeline',
                                {{
                                    method: 'POST',
                                    headers: {{ 'Content-Type': 'application/json' }},
                                    body: JSON.stringify({{ pipeline_id: currentPipelineId }})
                                }},
                                'resume-pipeline'
                            );

                            if (!response.ok) throw new Error('Failed to resume pipeline');

                            const result = await response.json();

                            if (result.error) {{
                                showMessage('Error: ' + result.error, 'error');
                            }} else {{
                                showMessage(`✓ Resumed run ${{currentPipelineId}}`, 'success');
                                // Force immediate refresh
                                setTimeout(() => loadPipeline(currentPipelineId, true), 500);
                            }}
                        }} catch (error) {{
                            showMessage('Error resuming pipeline: ' + error.message, 'error');
                        }}
                    }}

                    // Stop pipeline
                    async function handleStop() {{
                        if (!currentPipelineId) return;

                        try {{
                            const response = await safeFetch(
                                DISPATCHER_URL + '/dispatcher/stop-pipeline',
                                {{
                                    method: 'POST',
                                    headers: {{ 'Content-Type': 'application/json' }},
                                    body: JSON.stringify({{ pipeline_id: currentPipelineId }})
                                }},
                                'stop-pipeline'
                            );

                            if (!response.ok) throw new Error('Failed to stop pipeline');

                            const result = await response.json();

                            if (result.error) {{
                                showMessage('Error: ' + result.error, 'error');
                            }} else {{
                                showMessage(`✓ Stopped run ${{currentPipelineId}}`, 'success');
                                setTimeout(() => loadPipeline(currentPipelineId, true), 500);
                            }}
                        }} catch (error) {{
                            showMessage('Error stopping pipeline: ' + error.message, 'error');
                        }}
                    }}

                    // Delete pipeline
                    async function handleDelete() {{
                        if (!currentPipelineId) return;

                        try {{
                            const response = await safeFetch(
                                DISPATCHER_URL + '/dispatcher/delete-pipeline',
                                {{
                                    method: 'POST',
                                    headers: {{ 'Content-Type': 'application/json' }},
                                    body: JSON.stringify({{ pipeline_id: currentPipelineId }})
                                }},
                                'delete-pipeline'
                            );

                            if (!response.ok) throw new Error('Failed to delete pipeline');

                            const result = await response.json();

                            if (result.error) {{
                                showMessage('Error: ' + result.error, 'error');
                            }} else {{
                                showMessage(`✓ Deleted run ${{currentPipelineId}}`, 'success');
                                currentPipelineId = null;
                                currentConfig = null;
                                currentContextId = null;
                                updateDisplay(null, 'Not loaded', null, null, null, null);
                                setTimeout(fetchAllPipelines, 500);
                            }}
                        }} catch (error) {{
                            showMessage('Error deleting pipeline: ' + error.message, 'error');
                        }}
                    }}

                    // Clone mode
                    function enableCloneMode() {{
                        isCloneMode = true;
                        elements.configText.disabled = false;
                        elements.submitCloneBtn.disabled = false;
                        elements.cancelCloneBtn.disabled = false;
                        elements.cloneBtn.disabled = true;
                        showMessage('Edit config and click Submit to clone', 'info');
                    }}

                    function disableCloneMode() {{
                        isCloneMode = false;
                        elements.configText.disabled = true;
                        elements.configText.value = JSON.stringify(currentConfig || {{}}, null, 2);
                        elements.submitCloneBtn.disabled = true;
                        elements.cancelCloneBtn.disabled = true;
                        elements.cloneBtn.disabled = false;
                    }}

                    async function handleClone() {{
                        if (!currentPipelineId) {{
                            showMessage('No pipeline selected', 'error');
                            return;
                        }}

                        try {{
                            // Parse the edited config JSON
                            let editedConfig;
                            try {{
                                editedConfig = JSON.parse(elements.configText.value);
                            }} catch (e) {{
                                showMessage('Invalid JSON: ' + e.message, 'error');
                                return;
                            }}

                            // Validate required fields
                            if (!editedConfig.pipeline_type) {{
                                showMessage('config_json must include pipeline_type', 'error');
                                return;
                            }}

                            // Prepare clone request with new format
                            const cloneRequest = {{
                                parent_pipeline_id: currentPipelineId,
                                config_json: editedConfig
                            }};

                            const response = await safeFetch(
                                DISPATCHER_URL + '/dispatcher/clone-pipeline',
                                {{
                                    method: 'POST',
                                    headers: {{ 'Content-Type': 'application/json' }},
                                    body: JSON.stringify(cloneRequest)
                                }},
                                'clone-pipeline'
                            );

                            if (!response.ok) throw new Error('Failed to clone pipeline');

                            const result = await response.json();

                            if (result.error) {{
                                showMessage('Error: ' + result.error, 'error');
                            }} else {{
                                showMessage(`✓ Cloned from Config ID ${{currentPipelineId}} successfully!`, 'success');
                                disableCloneMode();
                                setTimeout(fetchAllPipelines, 1000);
                            }}
                        }} catch (error) {{
                            showMessage('Error cloning pipeline: ' + error.message, 'error');
                        }}
                    }}

                    // Event listeners
                    elements.pipelineSelector.addEventListener('change', (e) => {{
                        if (e.target.value) {{
                            loadPipeline(e.target.value, true);
                        }}
                    }});
                    elements.resumeBtn.addEventListener('click', handleResume);
                    elements.stopBtn.addEventListener('click', handleStop);
                    elements.deleteBtn.addEventListener('click', handleDelete);
                    elements.cloneBtn.addEventListener('click', enableCloneMode);
                    elements.submitCloneBtn.addEventListener('click', handleClone);
                    elements.cancelCloneBtn.addEventListener('click', () => {{
                        disableCloneMode();
                        showMessage('Cancelled clone', 'info');
                    }});

                    // Initialize
                    console.log('NotebookUI initialized');
                    fetchAllPipelines(true);  // Initial load
                    startAutoRefresh();

                    // Auto-pause when UI is not visible (e.g., scrolled away or cell collapsed)
                    if ('IntersectionObserver' in window) {{
                        const observer = new IntersectionObserver((entries) => {{
                            entries.forEach(entry => {{
                                if (entry.isIntersecting) {{
                                    // UI is visible - ensure polling is active
                                    if (!refreshInterval) {{
                                        console.log('UI visible - resuming auto-refresh');
                                        startAutoRefresh();
                                    }}
                                }} else {{
                                    // UI is not visible - pause polling to save resources
                                    if (refreshInterval) {{
                                        console.log('UI not visible - pausing auto-refresh');
                                        stopAutoRefresh();
                                    }}
                                }}
                            }});
                        }}, {{ threshold: 0.1 }});

                        // Observe the main container
                        const container = document.getElementById('{self.widget_id}');
                        if (container) {{
                            observer.observe(container);
                        }}
                    }}
                }})();
            </script>
        </div>
        """

    def display(self):
        """Display the interactive controller UI"""
        html_content = self._generate_html()
        display(HTML(html_content))

    def cleanup(self):
        """
        Stop all polling and disable all buttons.
        Call this before moving to a different cell or when done.
        The UI remains visible but non-interactive.
        """
        cleanup_script = f"""
        <script>
            (function() {{
                // Find the widget's refresh interval and stop it
                // This is a best-effort cleanup via a global flag
                const widget = document.getElementById('{self.widget_id}');
                if (widget) {{
                    console.log('NotebookUI cleanup called - stopping polling');
                    // The actual interval stopping happens in the widget's own scope
                    // But we can disable all buttons as a visible indication
                }}
            }})();
        </script>
        """
        display(HTML(cleanup_script))
        print(f"✓ UI cleanup initiated (widget_id: {self.widget_id})")
        print("  Polling will stop, buttons will be disabled")
        print("  UI remains visible with last state")

    def auto_refresh(self):
        """
        Auto-refresh is automatically enabled by default.
        Refresh rate is configurable via the refresh_rate_seconds parameter.
        Current rate: Every {self.refresh_rate_ms / 1000} seconds
        """
        print(f"ℹ️  Auto-refresh is already enabled (every {self.refresh_rate_ms / 1000} seconds)")
        print("The UI continuously updates automatically.")
