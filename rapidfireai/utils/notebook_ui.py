"""
Browser-side Interactive Run Controller (IC Ops panel) for notebooks.

The panel is plain HTML + JavaScript. Once displayed it talks to the RapidFire
dispatcher directly from the browser over HTTP (polling for run status and sending
stop/resume/delete/clone requests), so it never needs the notebook kernel. That keeps
it live while a blocking ``run_fit()`` / ``run_evals()`` call occupies the kernel,
which is what makes it usable in Google Colab, where the MLflow-backed dashboard does
not fit in the free tier's RAM.

The same panel serves both modes; only the dispatcher endpoints differ:

- ``mode="fit"``: runs, via ``/dispatcher/get-all-runs``, ``/dispatcher/get-run``,
  ``/dispatcher/{stop,resume,delete}-run`` and ``/dispatcher/clone-modify-run``
  (with optional warm start).
- ``mode="evals"``: pipelines (configs), via ``/dispatcher/list-all-pipeline-ids``,
  ``/dispatcher/get-pipeline-config-json/<id>``, ``/dispatcher/{stop,resume,delete}-pipeline``
  and ``/dispatcher/clone-pipeline``.

It also provides :func:`end_experiment_button`, the Colab "Click to End Experiment" button
that blocks until clicked so "Run all" does not end the experiment early.
"""

import json
import uuid
from typing import Any

from IPython.display import HTML, display

from rapidfireai.utils.constants import ColabConfig, DispatcherConfig

_MODES = {
    "fit": {"item_label": "Run ID", "warm_start": True},
    "evals": {"item_label": "Config ID", "warm_start": False},
}

_TEMPLATE = """
<div id="__WIDGET_ID__" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto;">
    <style>
        #__WIDGET_ID__ h3 { margin: 10px 0; font-size: 1.2em; font-weight: 600; }
        #__WIDGET_ID__ .header-info { display: flex; gap: 20px; margin: 10px 0; padding: 10px; background: #f8f9fa; color: #212529; border-radius: 4px; font-size: 13px; }
        #__WIDGET_ID__ .section { margin: 15px 0; }
        #__WIDGET_ID__ .section-label { font-weight: 600; margin-bottom: 8px; font-size: 14px; }
        #__WIDGET_ID__ .button-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }
        #__WIDGET_ID__ select { padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; background: white; color: #212529; min-width: 300px; cursor: pointer; }
        #__WIDGET_ID__ button { padding: 6px 16px; border: none; border-radius: 4px; font-size: 13px; font-weight: 500; cursor: pointer; }
        #__WIDGET_ID__ button:disabled { opacity: 0.5; cursor: not-allowed; }
        #__WIDGET_ID__ .btn-success { background: #28a745; color: white; }
        #__WIDGET_ID__ .btn-danger { background: #dc3545; color: white; }
        #__WIDGET_ID__ .btn-info { background: #17a2b8; color: white; }
        #__WIDGET_ID__ textarea { width: 100%; min-height: 200px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 12px; box-sizing: border-box; }
        #__WIDGET_ID__ .warm-start { display: flex; align-items: center; gap: 6px; margin: 10px 0; font-size: 13px; }
        #__WIDGET_ID__ .status-message { padding: 10px; margin: 10px 0; border-radius: 4px; display: none; }
        #__WIDGET_ID__ .msg-success { background: #d4edda; color: #155724; }
        #__WIDGET_ID__ .msg-error { background: #f8d7da; color: #721c24; }
        #__WIDGET_ID__ .msg-info { background: #d1ecf1; color: #0c5460; }
    </style>

    <div>
        <h3>Interactive Run Controller</h3>
        <div class="header-info">
            <div><b>__ITEM_LABEL__:</b> <span data-el="itemIdValue">N/A</span></div>
            <div><b>Status:</b> <span data-el="statusValue">Not loaded</span></div>
            <div><b>Last Update:</b> <span data-el="lastUpdate">Never</span></div>
        </div>

        <div data-el="statusMessage" class="status-message"></div>

        <div class="section">
            <div class="section-label">Select a __ITEM_LABEL__:</div>
            <select data-el="selector">
                <option value="">Waiting for data...</option>
            </select>
        </div>

        <div class="section">
            <div class="button-row">
                <button class="btn-success" data-el="resumeBtn">&#9654; Resume</button>
                <button class="btn-danger" data-el="stopBtn">&#9632; Stop</button>
                <button class="btn-danger" data-el="deleteBtn">&#128465; Delete</button>
            </div>
        </div>

        <div class="section">
            <div class="section-label">Configuration: <span data-el="configName">N/A</span></div>
            <textarea data-el="configText" readonly>{}</textarea>
            <label class="warm-start" data-el="warmStartRow" style="display: none;">
                <input type="checkbox" data-el="warmStart" disabled> Warm start (continue from the parent run's checkpoint)
            </label>
            <div class="button-row">
                <button class="btn-info" data-el="cloneBtn">Clone Run</button>
                <button class="btn-success" data-el="submitCloneBtn" disabled>&#10003; Submit Clone</button>
                <button class="btn-danger" data-el="cancelCloneBtn" disabled>&#10007; Cancel</button>
            </div>
        </div>
    </div>

    <script>
        (function() {
            const CFG = __CONFIG_JSON__;
            const root = document.getElementById(CFG.widgetId);
            const $ = (name) => root.querySelector('[data-el="' + name + '"]');
            const el = {
                itemIdValue: $('itemIdValue'),
                statusValue: $('statusValue'),
                lastUpdate: $('lastUpdate'),
                statusMessage: $('statusMessage'),
                selector: $('selector'),
                resumeBtn: $('resumeBtn'),
                stopBtn: $('stopBtn'),
                deleteBtn: $('deleteBtn'),
                configName: $('configName'),
                configText: $('configText'),
                warmStartRow: $('warmStartRow'),
                warmStart: $('warmStart'),
                cloneBtn: $('cloneBtn'),
                submitCloneBtn: $('submitCloneBtn'),
                cancelCloneBtn: $('cancelCloneBtn')
            };

            let currentId = null;
            let currentConfig = null;
            let canClone = false;
            let isCloneMode = false;
            let pollingInterval = null;
            let messageTimer = null;

            // Plain fetch to the dispatcher; on Colab, cookies are included for the proxy auth.
            async function request(path, method = 'GET', body = null) {
                const options = {
                    method: method,
                    headers: { 'Content-Type': 'application/json' },
                    mode: 'cors',
                    credentials: CFG.credentials
                };
                if (body) {
                    options.body = JSON.stringify(body);
                }
                const response = await fetch(CFG.dispatcherUrl + path, options);
                let data = null;
                try {
                    data = await response.json();
                } catch (e) {
                    data = null;
                }
                if (!response.ok || (data && data.error)) {
                    const detail = data && (data.error || data.err_msg);
                    throw new Error(detail ? String(detail).split('\\n')[0] : 'HTTP ' + response.status);
                }
                return data;
            }

            // Mode-specific dispatcher API. Everything else in the panel is shared.
            const API = CFG.mode === 'fit' ? {
                list: async () => (await request('/dispatcher/get-all-runs'))
                    .map(r => ({ id: r.run_id, status: r.status })),
                detail: async (id) => {
                    const data = await request('/dispatcher/get-run', 'POST', { run_id: id });
                    return { config: data.config || {}, name: null, canClone: true };
                },
                action: (action, id) => request('/dispatcher/' + action + '-run', 'POST', { run_id: id }),
                clone: (id, config, warmStart) => request('/dispatcher/clone-modify-run', 'POST',
                    { run_id: id, config: config, warm_start: warmStart })
            } : {
                list: async () => (await request('/dispatcher/list-all-pipeline-ids'))
                    .map(p => ({ id: p.pipeline_id, status: p.status })),
                detail: async (id) => {
                    const data = await request('/dispatcher/get-pipeline-config-json/' + id);
                    const config = data.pipeline_config_json || {};
                    return { config: config, name: config.pipeline_name || null, canClone: !!data.context_id };
                },
                action: (action, id) => request('/dispatcher/' + action + '-pipeline', 'POST', { pipeline_id: id }),
                clone: (id, config) => {
                    if (!config.pipeline_type) {
                        throw new Error('config_json must include pipeline_type');
                    }
                    return request('/dispatcher/clone-pipeline', 'POST',
                        { parent_pipeline_id: id, config_json: config });
                }
            };

            function showMessage(message, type) {
                el.statusMessage.className = 'status-message msg-' + type;
                el.statusMessage.textContent = message;
                el.statusMessage.style.display = 'block';
                if (messageTimer) clearTimeout(messageTimer);
                messageTimer = setTimeout(() => { el.statusMessage.style.display = 'none'; }, 5000);
            }

            function updateButtons(status) {
                const isCompleted = (status || '').toLowerCase() === 'completed';
                el.resumeBtn.disabled = isCompleted;
                el.stopBtn.disabled = isCompleted;
                el.deleteBtn.disabled = isCompleted;
                el.cloneBtn.disabled = isCompleted || isCloneMode || !canClone;
            }

            async function fetchDetail(id) {
                try {
                    const detail = await API.detail(id);
                    if (id !== currentId) return;  // selection changed while loading
                    currentConfig = detail.config;
                    canClone = detail.canClone;
                    el.configName.textContent = detail.name || ('#' + id);
                    if (!isCloneMode) {
                        el.configText.value = JSON.stringify(currentConfig, null, 2);
                    }
                    updateButtons(el.statusValue.textContent);
                } catch (error) {
                    console.error('Failed to fetch config:', error);
                    showMessage('Failed to load config: ' + error.message, 'error');
                }
            }

            function renderList(items) {
                const selector = el.selector;
                const previous = currentId;
                selector.innerHTML = '';

                if (!items || items.length === 0) {
                    selector.innerHTML = '<option value="">No runs yet</option>';
                    currentId = null;
                    el.itemIdValue.textContent = 'N/A';
                    el.statusValue.textContent = 'Not loaded';
                    return;
                }

                items.forEach(item => {
                    const option = document.createElement('option');
                    option.value = item.id;
                    option.dataset.status = item.status || 'unknown';
                    option.textContent = CFG.itemLabel + ': ' + item.id + ' (' + (item.status || 'unknown') + ')';
                    selector.appendChild(option);
                });

                const kept = items.find(item => String(item.id) === String(previous));
                const current = kept || items[0];
                currentId = current.id;
                selector.value = String(current.id);
                if (!kept) {
                    canClone = false;
                    fetchDetail(currentId);
                }

                el.itemIdValue.textContent = current.id;
                el.statusValue.textContent = current.status || 'unknown';
                updateButtons(current.status);
            }

            async function refresh() {
                try {
                    renderList(await API.list());
                    el.lastUpdate.textContent = new Date().toLocaleTimeString();
                } catch (error) {
                    console.error('Failed to fetch runs:', error);
                    showMessage('Connection error: ' + error.message, 'error');
                }
            }

            async function handleAction(action) {
                if (currentId === null) {
                    showMessage('Nothing selected', 'error');
                    return;
                }
                try {
                    await API.action(action, currentId);
                    showMessage('\\u2713 ' + action + ' requested for ' + CFG.itemLabel + ' ' + currentId, 'success');
                    setTimeout(refresh, 500);
                } catch (error) {
                    showMessage('Error: ' + error.message, 'error');
                }
            }

            function setCloneMode(enabled) {
                isCloneMode = enabled;
                el.configText.readOnly = !enabled;
                el.warmStart.disabled = !enabled;
                el.submitCloneBtn.disabled = !enabled;
                el.cancelCloneBtn.disabled = !enabled;
                if (!enabled) {
                    el.warmStart.checked = false;
                    el.configText.value = JSON.stringify(currentConfig || {}, null, 2);
                }
                updateButtons(el.statusValue.textContent);
            }

            async function handleClone() {
                if (currentId === null) {
                    showMessage('Nothing selected', 'error');
                    return;
                }
                let editedConfig;
                try {
                    editedConfig = JSON.parse(el.configText.value);
                } catch (e) {
                    showMessage('Invalid JSON: ' + e.message, 'error');
                    return;
                }
                try {
                    await API.clone(currentId, editedConfig, el.warmStart.checked);
                    showMessage('\\u2713 Cloned from ' + CFG.itemLabel + ' ' + currentId, 'success');
                    setCloneMode(false);
                    setTimeout(refresh, 1000);
                } catch (error) {
                    showMessage('Error cloning: ' + error.message, 'error');
                }
            }

            if (CFG.warmStart) {
                el.warmStartRow.style.display = 'flex';
            }

            el.selector.addEventListener('change', (e) => {
                if (e.target.value === '') return;
                const option = e.target.options[e.target.selectedIndex];
                currentId = Number(e.target.value);
                canClone = false;
                el.itemIdValue.textContent = currentId;
                el.statusValue.textContent = option.dataset.status;
                if (isCloneMode) setCloneMode(false);
                updateButtons(option.dataset.status);
                fetchDetail(currentId);
            });
            el.resumeBtn.addEventListener('click', () => handleAction('resume'));
            el.stopBtn.addEventListener('click', () => handleAction('stop'));
            el.deleteBtn.addEventListener('click', () => handleAction('delete'));
            el.cloneBtn.addEventListener('click', () => {
                setCloneMode(true);
                showMessage('Edit the config and click Submit Clone', 'info');
            });
            el.submitCloneBtn.addEventListener('click', handleClone);
            el.cancelCloneBtn.addEventListener('click', () => {
                setCloneMode(false);
                showMessage('Clone cancelled', 'info');
            });

            // Poll over HTTP: works even while the kernel is busy running run_fit()/run_evals().
            setTimeout(async () => {
                await refresh();
                pollingInterval = setInterval(refresh, CFG.refreshMs);
            }, 1000);
            window.addEventListener('beforeunload', () => {
                if (pollingInterval) clearInterval(pollingInterval);
            });
        })();
    </script>
</div>
"""


class NotebookUI:
    """Interactive Run Controller panel for fit and evals experiments."""

    def __init__(
        self,
        mode: str,
        dispatcher_url: str = "http://127.0.0.1:8851",
        refresh_rate_seconds: float = 3.0,
    ):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
        self.mode = mode
        if ColabConfig.ON_COLAB:
            # Colab proxies localhost ports for requests made from notebook output frames
            self.dispatcher_url = f"https://localhost:{DispatcherConfig.PORT}"
        else:
            self.dispatcher_url = dispatcher_url.rstrip("/")
        self.widget_id = f"controller_{uuid.uuid4().hex[:8]}"
        self.refresh_rate = refresh_rate_seconds

    def _generate_html(self) -> str:
        """Generate the panel HTML (markup, styles, and the polling script)."""
        mode_cfg = _MODES[self.mode]
        config = {
            "widgetId": self.widget_id,
            "mode": self.mode,
            "itemLabel": mode_cfg["item_label"],
            "warmStart": mode_cfg["warm_start"],
            "dispatcherUrl": self.dispatcher_url,
            "credentials": "include" if ColabConfig.ON_COLAB else "omit",
            "refreshMs": int(self.refresh_rate * 1000),
        }
        return (
            _TEMPLATE.replace("__WIDGET_ID__", self.widget_id)
            .replace("__ITEM_LABEL__", mode_cfg["item_label"])
            .replace("__CONFIG_JSON__", json.dumps(config))
        )

    def display(self) -> None:
        """Display the panel in the notebook."""
        display(HTML(self._generate_html()))


_END_BUTTON_JS = """
new Promise((resolve) => {
    const button = document.getElementById("__BUTTON_ID__");
    button.onclick = () => {
        button.disabled = true;
        button.innerText = "Continuing...";
        resolve("clicked");
    };
})
"""


def end_experiment_button(experiment: Any) -> None:
    """Show a "Click to End Experiment" button, block until it is clicked, then end the experiment.

    Colab only. Blocking keeps "Run all" from ending the experiment, and from running the
    cells after this one, before you are done with it.
    """
    try:
        from google.colab import output
    except ImportError as e:
        raise RuntimeError(
            "end_experiment_button() needs Google Colab; elsewhere call experiment.end() when you are done."
        ) from e

    button_id = f"end_experiment_{uuid.uuid4().hex[:8]}"
    display(
        HTML(
            f'<button id="{button_id}" style="padding: 10px 20px; font-size: 16px;">Click to End Experiment</button>'
        )
    )

    # eval_js blocks until the Promise resolves, i.e. until the button is clicked
    output.eval_js(_END_BUTTON_JS.replace("__BUTTON_ID__", button_id))

    # Actually end the experiment after the button is clicked
    experiment.end()
    print("Done!")

