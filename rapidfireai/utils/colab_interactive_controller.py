"""
Interactive Controller for Jupyter/Colab notebooks using HTML/JavaScript.
Provides UI controls for managing training runs with direct REST API calls.
This approach avoids ipywidgets compatibility issues in Google Colab.
"""

from IPython.display import HTML, display


class ColabInteractiveController:
    """Interactive run controller using HTML/JavaScript for notebooks"""

    def __init__(self, dispatcher_url: str = "http://127.0.0.1:8081"):
        self.dispatcher_url = dispatcher_url.rstrip("/")

    def display(self):
        """Display the interactive controller UI"""
        html_interface = f"""
<style>
    .rf-controller {{
        margin: 20px 0;
        padding: 25px;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .rf-header {{
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 2px solid #ddd;
    }}
    .rf-header h3 {{
        margin: 0 0 10px 0;
        color: #2c3e50;
    }}
    .rf-status-bar {{
        display: flex;
        gap: 20px;
        margin: 10px 0;
        font-size: 14px;
    }}
    .rf-status-item {{
        padding: 5px 10px;
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .rf-status-item strong {{
        color: #555;
    }}
    .rf-action-buttons {{
        margin: 20px 0;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }}
    .rf-btn {{
        padding: 12px 24px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        border: none;
        border-radius: 6px;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .rf-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }}
    .rf-btn:active {{
        transform: translateY(0);
    }}
    .rf-btn-resume {{ background-color: #4CAF50; }}
    .rf-btn-stop {{ background-color: #f44336; }}
    .rf-btn-delete {{ background-color: #ff9800; }}
    .rf-btn-clone {{ background-color: #2196F3; }}
    .rf-btn-refresh {{ background-color: #9C27B0; }}
    .rf-btn-submit {{
        background-color: #1976D2;
        padding: 12px 32px;
        font-size: 16px;
    }}
    .rf-input-section {{
        margin: 20px 0;
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    .rf-input-label {{
        display: block;
        margin-bottom: 8px;
        color: #555;
        font-weight: 600;
    }}
    .rf-input {{
        padding: 10px;
        font-size: 14px;
        border: 2px solid #ddd;
        border-radius: 6px;
        width: 200px;
        transition: border-color 0.3s;
    }}
    .rf-input:focus {{
        outline: none;
        border-color: #2196F3;
    }}
    .rf-textarea {{
        width: 100%;
        min-height: 200px;
        padding: 12px;
        font-size: 13px;
        font-family: 'Courier New', monospace;
        border: 2px solid #ddd;
        border-radius: 6px;
        resize: vertical;
    }}
    .rf-checkbox {{
        margin: 15px 0;
    }}
    .rf-checkbox input {{
        margin-right: 8px;
        width: 18px;
        height: 18px;
        cursor: pointer;
    }}
    .rf-checkbox label {{
        cursor: pointer;
        font-weight: 600;
        color: #555;
    }}
    .rf-clone-section {{
        margin-top: 15px;
        padding: 20px;
        background-color: #e3f2fd;
        border-radius: 8px;
        display: none;
    }}
    .rf-log {{
        margin-top: 20px;
        padding: 15px;
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 8px;
        max-height: 400px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }}
    .rf-log-entry {{
        padding: 8px 12px;
        margin: 5px 0;
        border-left: 4px solid #2196F3;
        background-color: white;
        border-radius: 4px;
    }}
    .rf-log-success {{
        border-left-color: #4CAF50;
        background-color: #e8f5e9;
    }}
    .rf-log-error {{
        border-left-color: #f44336;
        background-color: #ffebee;
    }}
    .rf-status-msg {{
        margin: 15px 0;
        padding: 12px;
        border-radius: 6px;
        font-weight: 600;
        display: none;
    }}
    .rf-status-msg.success {{
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #4CAF50;
    }}
    .rf-status-msg.error {{
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #f44336;
    }}
</style>

<div class="rf-controller">
    <div class="rf-header">
        <h3>🎮 Interactive Run Controller</h3>
        <div class="rf-status-bar" id="statusBar">
            <div class="rf-status-item">
                <strong>Run ID:</strong> <span id="currentRunId">N/A</span>
            </div>
            <div class="rf-status-item">
                <strong>Status:</strong> <span id="currentStatus">Not loaded</span>
            </div>
            <div class="rf-status-item">
                <strong>Chunk:</strong> <span id="currentChunk">N/A</span>
            </div>
        </div>
    </div>

    <div class="rf-input-section">
        <label class="rf-input-label" for="runIdInput">Run ID:</label>
        <input type="number" class="rf-input" id="runIdInput" placeholder="Enter run ID" min="1">
        <button class="rf-btn rf-btn-refresh" onclick="loadRun()">🔄 Load Run</button>
    </div>

    <div class="rf-action-buttons">
        <button class="rf-btn rf-btn-resume" onclick="performAction('resume')">▶ Resume</button>
        <button class="rf-btn rf-btn-stop" onclick="performAction('stop')">⏹ Stop</button>
        <button class="rf-btn rf-btn-delete" onclick="performAction('delete')">🗑 Delete</button>
        <button class="rf-btn rf-btn-clone" onclick="showCloneSection()">📋 Clone</button>
    </div>

    <div class="rf-clone-section" id="cloneSection">
        <h4 style="margin-top: 0;">Clone Run with Modifications</h4>
        <label class="rf-input-label">Configuration (JSON):</label>
        <textarea class="rf-textarea" id="configText" placeholder="Edit configuration JSON here...">{{}}</textarea>

        <div class="rf-checkbox">
            <input type="checkbox" id="warmStartCheck">
            <label for="warmStartCheck">Warm Start (continue from previous checkpoint)</label>
        </div>

        <div style="margin-top: 15px;">
            <button class="rf-btn rf-btn-submit" onclick="submitClone()">✓ Submit Clone</button>
            <button class="rf-btn" style="background-color: #757575;" onclick="cancelClone()">✗ Cancel</button>
        </div>
    </div>

    <div class="rf-status-msg" id="statusMessage"></div>

    <div class="rf-log" id="logOutput">
        <strong>📋 Activity Log:</strong><br>
        <em style="color: #999;">Ready to manage runs...</em>
    </div>
</div>

<script>
    const DISPATCHER_URL = '{self.dispatcher_url}';
    let logCounter = 0;
    let currentRunData = null;

    function addLog(message, isError = false) {{
        const logDiv = document.getElementById('logOutput');
        const timestamp = new Date().toLocaleTimeString();
        const logClass = isError ? 'rf-log-entry rf-log-error' : 'rf-log-entry rf-log-success';

        if (logCounter === 0) {{
            logDiv.innerHTML = '<strong>📋 Activity Log:</strong><br>';
        }}

        logCounter++;
        const logEntry = document.createElement('div');
        logEntry.className = logClass;
        logEntry.innerHTML = `<strong>[${{timestamp}}]</strong> ${{message}}`;
        logDiv.appendChild(logEntry);
        logDiv.scrollTop = logDiv.scrollHeight;
    }}

    function showStatus(message, isError = false) {{
        const statusMsg = document.getElementById('statusMessage');
        statusMsg.textContent = message;
        statusMsg.className = 'rf-status-msg ' + (isError ? 'error' : 'success');
        statusMsg.style.display = 'block';

        setTimeout(() => {{
            statusMsg.style.display = 'none';
        }}, 5000);
    }}

    async function loadRun() {{
        const runId = document.getElementById('runIdInput').value;

        if (!runId) {{
            showStatus('❌ Please enter a run ID', true);
            return;
        }}

        try {{
            addLog(`🔍 Loading run ${{runId}}...`);

            const response = await fetch(`${{DISPATCHER_URL}}/dispatcher/get-run`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ run_id: parseInt(runId) }})
            }});

            const data = await response.json();

            if (response.ok && !data.error) {{
                currentRunData = data;

                // Update status bar
                document.getElementById('currentRunId').textContent = runId;
                document.getElementById('currentStatus').textContent = data.status || 'Unknown';
                document.getElementById('currentChunk').textContent = data.num_chunks_visited || 0;

                // Update config textarea
                document.getElementById('configText').value = JSON.stringify(data.config || {{}}, null, 2);

                addLog(`✅ Run ${{runId}} loaded successfully`);
                showStatus(`✓ Run ${{runId}} loaded`);
            }} else {{
                const errorMsg = data.error || data.err_msg || 'Unknown error';
                addLog(`❌ Failed to load run: ${{errorMsg}}`, true);
                showStatus(`✗ Error: ${{errorMsg}}`, true);
            }}
        }} catch (error) {{
            addLog(`❌ Network error: ${{error.message}}`, true);
            showStatus(`✗ Network error: ${{error.message}}`, true);
        }}
    }}

    async function performAction(action) {{
        const runId = document.getElementById('currentRunId').textContent;

        if (runId === 'N/A') {{
            showStatus('❌ Please load a run first', true);
            return;
        }}

        const endpoints = {{
            'resume': '/dispatcher/resume-run',
            'stop': '/dispatcher/stop-run',
            'delete': '/dispatcher/delete-run'
        }};

        const endpoint = endpoints[action];
        if (!endpoint) return;

        try {{
            addLog(`🚀 Sending ${{action}} request for run ${{runId}}...`);

            const response = await fetch(`${{DISPATCHER_URL}}${{endpoint}}`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ run_id: parseInt(runId) }})
            }});

            const data = await response.json();

            if (response.ok && !data.error) {{
                addLog(`✅ ${{action.toUpperCase()}} completed for run ${{runId}}`);
                showStatus(`✓ ${{action.toUpperCase()}} successful`);

                // Reload run data after action
                if (action !== 'delete') {{
                    setTimeout(() => loadRun(), 1000);
                }} else {{
                    // Clear display after delete
                    document.getElementById('currentRunId').textContent = 'N/A';
                    document.getElementById('currentStatus').textContent = 'Not loaded';
                    document.getElementById('currentChunk').textContent = 'N/A';
                }}
            }} else {{
                const errorMsg = data.error || data.err_msg || 'Unknown error';
                addLog(`❌ ${{action.toUpperCase()}} failed: ${{errorMsg}}`, true);
                showStatus(`✗ Error: ${{errorMsg}}`, true);
            }}
        }} catch (error) {{
            addLog(`❌ Network error: ${{error.message}}`, true);
            showStatus(`✗ Network error: ${{error.message}}`, true);
        }}
    }}

    function showCloneSection() {{
        const runId = document.getElementById('currentRunId').textContent;

        if (runId === 'N/A') {{
            showStatus('❌ Please load a run first', true);
            return;
        }}

        document.getElementById('cloneSection').style.display = 'block';
        addLog(`📝 Clone mode enabled - edit config and submit`);
    }}

    function cancelClone() {{
        document.getElementById('cloneSection').style.display = 'none';
        document.getElementById('warmStartCheck').checked = false;
        addLog(`❌ Clone cancelled`);
    }}

    async function submitClone() {{
        const runId = document.getElementById('currentRunId').textContent;
        const configText = document.getElementById('configText').value;
        const warmStart = document.getElementById('warmStartCheck').checked;

        if (runId === 'N/A') {{
            showStatus('❌ Please load a run first', true);
            return;
        }}

        // Validate JSON
        let config;
        try {{
            config = JSON.parse(configText);
        }} catch (e) {{
            showStatus(`✗ Invalid JSON: ${{e.message}}`, true);
            addLog(`❌ Invalid JSON configuration`, true);
            return;
        }}

        try {{
            addLog(`🚀 Submitting clone request for run ${{runId}}...`);

            const response = await fetch(`${{DISPATCHER_URL}}/dispatcher/clone-modify-run`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    run_id: parseInt(runId),
                    config: config,
                    warm_start: warmStart
                }})
            }});

            const data = await response.json();

            if (response.ok && data.result !== false) {{
                addLog(`✅ Clone successful! New run created.`);
                showStatus(`✓ Clone successful`);
                cancelClone();
            }} else {{
                const errorMsg = data.error || data.err_msg || 'Unknown error';
                addLog(`❌ Clone failed: ${{errorMsg}}`, true);
                showStatus(`✗ Clone error: ${{errorMsg}}`, true);
            }}
        }} catch (error) {{
            addLog(`❌ Network error: ${{error.message}}`, true);
            showStatus(`✗ Network error: ${{error.message}}`, true);
        }}
    }}
</script>
"""
        display(HTML(html_interface))
