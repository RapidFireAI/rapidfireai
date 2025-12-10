"""
Notebook UI using VS Code's notebook kernel messaging API.
This works in VS Code notebooks by using the vscode notebook API instead of Jupyter.
"""

import uuid

from IPython.display import HTML, display
from rapidfireai.utils.constants import  DispatcherConfig, ColabConfig


class NotebookUI:
    """Notebook UI that works in VS Code"""

    def __init__(self, dispatcher_url: str = "http://127.0.0.1:8851", refresh_rate_seconds: float = 3.0, auth_token: str | None = None):
        if ColabConfig.ON_COLAB:
            self.dispatcher_url = f"https://localhost:{DispatcherConfig.PORT}"
        else:
            self.dispatcher_url = dispatcher_url.rstrip("/")
        self.widget_id = f"controller_{uuid.uuid4().hex[:8]}"
        self.refresh_rate = refresh_rate_seconds
        self.is_polling = False
        self.polling_thread = None
        self.pending_actions = []
        if ColabConfig.ON_COLAB:
            self.auth_token = auth_token

    def _generate_html(self):
        """Generate HTML using fetch API for communication"""
        if ColabConfig.ON_COLAB:
            xhr_credentials = "include"
        else:
            xhr_credentials = "omit"
        return f"""
        <div id="{self.widget_id}" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto;">
            <style>
                #{self.widget_id} h3 {{ margin: 10px 0; font-size: 1.2em; font-weight: 600; }}
                #{self.widget_id} .header-info {{ display: flex; gap: 20px; margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 13px; }}
                #{self.widget_id} .section {{ margin: 15px 0; }}
                #{self.widget_id} .section-label {{ font-weight: 600; margin-bottom: 8px; font-size: 14px; }}
                #{self.widget_id} .button-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }}
                #{self.widget_id} select {{ padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; background: white; min-width: 300px; cursor: pointer; }}
                #{self.widget_id} button {{ padding: 6px 16px; border: none; border-radius: 4px; font-size: 13px; font-weight: 500; cursor: pointer; }}
                #{self.widget_id} button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
                #{self.widget_id} .btn-success {{ background: #28a745; color: white; }}
                #{self.widget_id} .btn-danger {{ background: #dc3545; color: white; }}
                #{self.widget_id} .btn-info {{ background: #17a2b8; color: white; }}
                #{self.widget_id} .btn-default {{ background: #6c757d; color: white; }}
                #{self.widget_id} textarea {{ width: 100%; min-height: 200px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 12px; box-sizing: border-box; }}
                #{self.widget_id} .status-message {{ padding: 10px; margin: 10px 0; border-radius: 4px; display: none; }}
                #{self.widget_id} .msg-success {{ background: #d4edda; color: #155724; }}
                #{self.widget_id} .msg-error {{ background: #f8d7da; color: #721c24; }}
                #{self.widget_id} .msg-info {{ background: #d1ecf1; color: #0c5460; }}
            </style>

            <div>
                <h3>Interactive Run Controller</h3>
                <div class="header-info">
                    <div><b>Run ID:</b> <span id="pipeline-id-value">N/A</span></div>
                    <div><b>Status:</b> <span id="status-value">Not loaded</span></div>
                    <div><b>Last Update:</b> <span id="last-update">Never</span></div>
                </div>

                <div id="status-message" class="status-message"></div>

                <div class="section">
                    <div class="section-label">Select a Config ID:</div>
                    <select id="pipeline-selector">
                        <option value="">Waiting for data...</option>
                    </select>
                </div>

                <div class="section">
                    <div class="button-row">
                        <button class="btn-success" id="resume-btn">▶ Resume</button>
                        <button class="btn-danger" id="stop-btn">■ Stop</button>
                        <button class="btn-danger" id="delete-btn">🗑 Delete</button>
                    </div>
                </div>

                <div class="section">
                    <div class="section-label">Configuration: <span id="config-name">N/A</span></div>
                    <textarea id="config-text" readonly>{{}}</textarea>
                    <div class="button-row">
                        <button class="btn-info" id="clone-btn">Clone Run</button>
                        <button class="btn-success" id="submit-clone-btn" disabled>✓ Submit Clone</button>
                        <button class="btn-danger" id="cancel-clone-btn" disabled>✗ Cancel</button>
                    </div>
                </div>
            </div>

            <script>
                (function() {{
                    const WIDGET_ID = '{self.widget_id}';
                    const DISPATCHER_URL = '{self.dispatcher_url}';
                    let currentPipelineId = null;
                    let currentConfig = null;
                    let currentContextId = null;
                    let isCloneMode = false;
                    let pollingInterval = null;

                    // Elements
                    const el = {{
                        pipelineIdValue: document.getElementById('pipeline-id-value'),
                        statusValue: document.getElementById('status-value'),
                        lastUpdate: document.getElementById('last-update'),
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

                    // Use fetch API with explicit CORS mode and optional auth token
                    async function xhrRequest(url, method = 'GET', body = null) {{
                        const options = {{
                            method: method,
                            headers: {{
                                'Content-Type': 'application/json'
                            }},
                            mode: 'cors',
                            credentials: '{xhr_credentials}'  // Include cookies for Colab proxy auth
                        }};

                        if (body) {{
                            options.body = JSON.stringify(body);
                        }}

                        const response = await fetch(url, options);
                        if (!response.ok) {{
                            throw new Error('HTTP ' + response.status);
                        }}
                        return await response.json();
                    }}

                    async function fetchPipelines() {{
                        try {{
                            console.log('Fetching pipelines...');
                            const pipelines = await xhrRequest(DISPATCHER_URL + '/dispatcher/list-all-pipeline-ids');
                            console.log('Got pipelines:', pipelines.length);

                            updatePipelinesDropdown(pipelines);
                            el.lastUpdate.textContent = new Date().toLocaleTimeString();

                        }} catch (error) {{
                            console.error('Failed to fetch pipelines:', error);
                            showMessage('Connection error: ' + error.message, 'error');
                        }}
                    }}

                    async function fetchPipelineConfig(pipelineId) {{
                        try {{
                            const data = await xhrRequest(DISPATCHER_URL + `/dispatcher/get-pipeline-config-json/${{pipelineId}}`);
                            const config = data.pipeline_config_json || {{}};

                            currentConfig = config;
                            currentContextId = data.context_id;

                            el.configName.textContent = config.pipeline_name || 'N/A';

                            if (!isCloneMode) {{
                                el.configText.value = JSON.stringify(config, null, 2);
                            }}

                        }} catch (error) {{
                            console.error('Failed to fetch config:', error);
                        }}
                    }}

                    function updatePipelinesDropdown(pipelines) {{
                        const selector = el.pipelineSelector;
                        const currentSelection = selector.value;

                        selector.innerHTML = '';

                        if (pipelines && pipelines.length > 0) {{
                            pipelines.forEach(p => {{
                                const option = document.createElement('option');
                                option.value = p.pipeline_id;
                                option.textContent = `Config ID: ${{p.pipeline_id}} (${{p.status || 'unknown'}})`;
                                selector.appendChild(option);
                            }});

                            if (currentSelection && pipelines.some(p => p.pipeline_id == currentSelection)) {{
                                selector.value = currentSelection;
                                currentPipelineId = currentSelection;
                            }} else {{
                                selector.value = pipelines[0].pipeline_id;
                                currentPipelineId = pipelines[0].pipeline_id;
                                fetchPipelineConfig(currentPipelineId);
                            }}

                            // Update status display
                            const currentPipeline = pipelines.find(p => p.pipeline_id == currentPipelineId);
                            if (currentPipeline) {{
                                el.pipelineIdValue.textContent = currentPipeline.pipeline_id;
                                el.statusValue.textContent = currentPipeline.status || 'unknown';

                                const isCompleted = currentPipeline.status?.toLowerCase() === 'completed';
                                el.resumeBtn.disabled = isCompleted;
                                el.stopBtn.disabled = isCompleted;
                                el.deleteBtn.disabled = isCompleted;
                                el.cloneBtn.disabled = isCompleted || !currentContextId;
                            }}
                        }} else {{
                            selector.innerHTML = '<option value="">No pipelines found</option>';
                        }}
                    }}

                    function showMessage(message, type) {{
                        el.statusMessage.className = 'status-message msg-' + type;
                        el.statusMessage.textContent = message;
                        el.statusMessage.style.display = 'block';
                        setTimeout(() => el.statusMessage.style.display = 'none', 5000);
                    }}

                    async function handleAction(action) {{
                        if (!currentPipelineId) {{
                            showMessage('No pipeline selected', 'error');
                            return;
                        }}

                        try {{
                            const endpoint = DISPATCHER_URL + `/dispatcher/${{action}}-pipeline`;
                            const result = await xhrRequest(endpoint, 'POST', {{ pipeline_id: currentPipelineId }});

                            showMessage(`✓ ${{action}} completed for pipeline ${{currentPipelineId}}`, 'success');

                            // Refresh after a short delay
                            setTimeout(async () => {{
                                await fetchPipelines();
                            }}, 500);

                        }} catch (error) {{
                            showMessage(`Error: ${{error.message}}`, 'error');
                        }}
                    }}

                    function enableCloneMode() {{
                        isCloneMode = true;
                        el.configText.readOnly = false;
                        el.submitCloneBtn.disabled = false;
                        el.cancelCloneBtn.disabled = false;
                        el.cloneBtn.disabled = true;
                        showMessage('Edit config and click Submit to clone', 'info');
                    }}

                    function disableCloneMode() {{
                        isCloneMode = false;
                        el.configText.readOnly = true;
                        el.configText.value = JSON.stringify(currentConfig || {{}}, null, 2);
                        el.submitCloneBtn.disabled = true;
                        el.cancelCloneBtn.disabled = true;
                        el.cloneBtn.disabled = false;
                    }}

                    async function handleClone() {{
                        if (!currentPipelineId) {{
                            showMessage('No pipeline selected', 'error');
                            return;
                        }}

                        try {{
                            // Parse edited config
                            let editedConfig;
                            try {{
                                editedConfig = JSON.parse(el.configText.value);
                            }} catch (e) {{
                                showMessage('Invalid JSON: ' + e.message, 'error');
                                return;
                            }}

                            // Validate required fields
                            if (!editedConfig.pipeline_type) {{
                                showMessage('config_json must include pipeline_type', 'error');
                                return;
                            }}

                            // Send clone request
                            const cloneRequest = {{
                                parent_pipeline_id: currentPipelineId,
                                config_json: editedConfig
                            }};

                            const result = await xhrRequest(
                                DISPATCHER_URL + '/dispatcher/clone-pipeline',
                                'POST',
                                cloneRequest
                            );

                            showMessage(`✓ Cloned from Config ID ${{currentPipelineId}} successfully!`, 'success');
                            disableCloneMode();

                            // Refresh after delay
                            setTimeout(async () => {{
                                await fetchPipelines();
                            }}, 1000);

                        }} catch (error) {{
                            showMessage(`Error cloning: ${{error.message}}`, 'error');
                        }}
                    }}

                    // Event listeners
                    el.pipelineSelector.addEventListener('change', (e) => {{
                        if (e.target.value) {{
                            currentPipelineId = parseInt(e.target.value);
                            fetchPipelineConfig(currentPipelineId);
                        }}
                    }});

                    el.resumeBtn.addEventListener('click', () => handleAction('resume'));
                    el.stopBtn.addEventListener('click', () => handleAction('stop'));
                    el.deleteBtn.addEventListener('click', () => handleAction('delete'));

                    el.cloneBtn.addEventListener('click', enableCloneMode);
                    el.submitCloneBtn.addEventListener('click', handleClone);
                    el.cancelCloneBtn.addEventListener('click', () => {{
                        disableCloneMode();
                        showMessage('Cancelled clone', 'info');
                    }});

                    // Initial fetch
                    console.log('UI initialized, fetching initial data...');
                    setTimeout(async () => {{
                        await fetchPipelines();

                        // Start polling - use HTTP fetch (works even when kernel is busy)
                        pollingInterval = setInterval(async () => {{
                            await fetchPipelines();
                        }}, {self.refresh_rate * 1000});
                        console.log('Polling started: every {self.refresh_rate}s');
                    }}, 1000);

                    // Cleanup on unload
                    window.addEventListener('beforeunload', () => {{
                        if (pollingInterval) clearInterval(pollingInterval);
                    }});
                }})();
            </script>
        </div>
        """

    def display(self):
        """Display the UI"""
        display(HTML(self._generate_html()))
