"""
Interactive Controller for Jupyter/Colab notebooks.
Provides UI controls for managing training runs similar to the frontend.
"""

import json
import threading
import time

import requests
from IPython.display import display

try:
    import ipywidgets as widgets
except ImportError as e:
    raise ImportError("ipywidgets is required for InteractiveController. Install with: pip install ipywidgets") from e


class InteractiveController:
    """Interactive run controller for notebooks"""

    def __init__(self, dispatcher_url: str = "http://127.0.0.1:8851"):
        self.dispatcher_url = dispatcher_url.rstrip("/")
        self.run_id: int | None = None
        self.config: dict | None = None
        self.status: str = "Unknown"
        self.chunk_number: int = 0

        # Create UI widgets
        self._create_widgets()

    def _create_widgets(self):
        """Create ipywidgets UI components"""
        # Run selector
        self.run_selector = widgets.Dropdown(
            options=[], description="", disabled=False, layout=widgets.Layout(width="300px")
        )
        self.load_btn = widgets.Button(
            description="Load Run", button_style="primary", tooltip="Load the selected run", icon="download"
        )
        self.refresh_selector_btn = widgets.Button(
            description="Refresh List",
            button_style="info",
            tooltip="Refresh the list of available runs",
            icon="refresh",
        )

        # Status display
        self.status_label = widgets.HTML(value="<b>Status:</b> Not loaded")
        self.chunk_label = widgets.HTML(value="<b>Chunk:</b> N/A")
        self.run_id_label = widgets.HTML(value="<b>Run ID:</b> N/A")

        # Action buttons
        self.resume_btn = widgets.Button(
            description="Resume",
            button_style="success",
            tooltip="Resume this run",
            icon="play",
        )
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", tooltip="Stop this run", icon="stop")
        self.delete_btn = widgets.Button(
            description="Delete",
            button_style="danger",
            tooltip="Delete this run",
            icon="trash",
        )
        self.refresh_btn = widgets.Button(
            description="Refresh Status",
            button_style="info",
            tooltip="Refresh current run status and metrics",
            icon="sync",
        )

        # Config editor (for clone/modify)
        self.config_text = widgets.Textarea(
            value="{}",
            placeholder="Run configuration (JSON)",
            disabled=True,
            layout=widgets.Layout(width="100%", height="200px"),
        )
        self.warm_start_checkbox = widgets.Checkbox(
            value=False,
            description="Warm Start (continue from previous checkpoint)",
            disabled=True,
            style={"description_width": "initial"},
            layout=widgets.Layout(margin="10px 0px"),
        )
        self.clone_btn = widgets.Button(
            description="Clone",
            button_style="primary",
            tooltip="Clone this run with modifications",
        )
        self.submit_clone_btn = widgets.Button(description="✓ Submit Clone", button_style="success", disabled=True)
        self.cancel_clone_btn = widgets.Button(description="✗ Cancel", button_style="", disabled=True)

        # Status message box
        self.status_message = widgets.HTML(
            value="",
            layout=widgets.Layout(
                width="100%",
                min_height="40px",
                padding="10px",
                margin="10px 0px",
                border="2px solid #ddd",
                border_radius="5px",
            ),
        )

        # Experiment status display (live progress)
        # self.experiment_status = widgets.HTML(
        #     value='<div style="padding: 10px; background-color: #f8f9fa; border: 2px solid #dee2e6; border-radius: 5px;">'
        #           '<b>Experiment Status:</b> Loading...'
        #           '</div>',
        #     layout=widgets.Layout(
        #         width='100%',
        #         margin='10px 0px'
        #     )
        # )

        # Bind button callbacks
        self.refresh_selector_btn.on_click(lambda b: self.fetch_all_runs())
        self.load_btn.on_click(lambda b: self._handle_load())
        self.resume_btn.on_click(lambda b: self._handle_resume())
        self.stop_btn.on_click(lambda b: self._handle_stop())
        self.delete_btn.on_click(lambda b: self._handle_delete())
        self.refresh_btn.on_click(lambda b: self.load_run(self.run_id) if self.run_id else None)
        self.clone_btn.on_click(lambda b: self._enable_clone_mode())
        self.submit_clone_btn.on_click(lambda b: self._handle_clone())
        self.cancel_clone_btn.on_click(lambda b: self._handle_cancel_clone())

        # Auto-load run when dropdown selection changes
        self.run_selector.observe(self._on_run_selected, names="value")

    def _show_message(self, message: str, message_type: str = "info"):
        """Display a status message with styling"""
        colors = {
            "success": {"bg": "#d4edda", "border": "#28a745", "text": "#155724"},
            "error": {"bg": "#f8d7da", "border": "#dc3545", "text": "#721c24"},
            "info": {"bg": "#d1ecf1", "border": "#17a2b8", "text": "#0c5460"},
            "warning": {"bg": "#fff3cd", "border": "#ffc107", "text": "#856404"},
        }

        style = colors.get(message_type, colors["info"])

        self.status_message.value = f"""
            <div style="
                background-color: {style["bg"]};
                border: 2px solid {style["border"]};
                color: {style["text"]};
                padding: 10px;
                border-radius: 5px;
                font-weight: 600;
            ">
                {message}
            </div>
        """

    def _update_experiment_status(self):
        """Update experiment status display with live progress"""
        try:
            response = requests.get(
                f"{self.dispatcher_url}/dispatcher/get-all-runs",
                timeout=5,
            )
            response.raise_for_status()
            runs = response.json()

            if runs:
                total_runs = len(runs)
                completed_runs = sum(1 for r in runs if r.get("status") == "COMPLETED")
                ongoing_runs = sum(1 for r in runs if r.get("status") == "ONGOING")

                # Determine status color and icon
                if completed_runs == total_runs:
                    bg_color = "#d4edda"
                    border_color = "#28a745"
                    text_color = "#155724"
                    icon = "✓"
                    status_text = "All runs completed"
                elif ongoing_runs > 0:
                    bg_color = "#d1ecf1"
                    border_color = "#17a2b8"
                    text_color = "#0c5460"
                    icon = "🔄"
                    status_text = "Training in progress"
                else:
                    bg_color = "#fff3cd"
                    border_color = "#ffc107"
                    text_color = "#856404"
                    icon = "⏸"
                    status_text = "Training paused or stopped"

                self.experiment_status.value = (
                    f'<div style="padding: 10px; background-color: {bg_color}; '
                    f'border: 2px solid {border_color}; border-radius: 5px; color: {text_color};">'
                    f"<b>{icon} Experiment Status:</b> {status_text}<br>"
                    f"<b>Progress:</b> {completed_runs}/{total_runs} runs completed"
                    "</div>"
                )
            else:
                self.experiment_status.value = (
                    '<div style="padding: 10px; background-color: #f8f9fa; '
                    'border: 2px solid #dee2e6; border-radius: 5px;">'
                    "<b>Experiment Status:</b> No runs found"
                    "</div>"
                )

        except requests.RequestException:
            # Silently fail - don't update status if request fails
            pass

    def fetch_all_runs(self):
        """Fetch all runs and populate dropdown"""
        try:
            response = requests.get(
                f"{self.dispatcher_url}/dispatcher/get-all-runs",
                timeout=5,
            )
            response.raise_for_status()
            runs = response.json()

            if runs:
                # Create options as (label, value) tuples
                options = [(f"Run {run['run_id']} - {run.get('status', 'Unknown')}", run["run_id"]) for run in runs]
                self.run_selector.options = options
                self._show_message(f"Found {len(runs)} runs", "success")
            else:
                self.run_selector.options = []
                self._show_message("No runs found", "info")

            # Update experiment status
            # COMMENTED OUT
            # self._update_experiment_status()

        except requests.RequestException as e:
            self._show_message(f"Error fetching runs: {e}", "error")

    def _on_run_selected(self, change):
        """Handle dropdown selection change - auto-load run"""
        if change["new"] is not None:
            self.load_run(change["new"])

    def _handle_load(self):
        """Handle load button click"""
        if self.run_selector.value is not None:
            self.load_run(self.run_selector.value)
        else:
            self._show_message("Please select a run first", "warning")

    def load_run(self, run_id: int):
        """Load run details from dispatcher API"""
        self.run_id = run_id
        try:
            response = requests.post(
                f"{self.dispatcher_url}/dispatcher/get-run",
                json={"run_id": run_id},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()

            # Update state
            self.config = data.get("config", {})
            self.status = data.get("status", "Unknown")
            self.chunk_number = data.get("num_chunks_visited", 0)

            # Update UI
            self._update_display()
            self._show_message(f"Loaded run {run_id}", "success")

            # Update experiment status
            # COMMENTED OUT
            # self._update_experiment_status()

        except requests.RequestException as e:
            self._show_message(f"Error loading run: {e}", "error")

    def _update_display(self):
        """Update widget values"""
        self.run_id_label.value = f"<b>Run ID:</b> {self.run_id}"
        self.status_label.value = f"<b>Status:</b> {self.status}"
        self.chunk_label.value = f"<b>Chunk:</b> {self.chunk_number}"
        self.config_text.value = json.dumps(self.config, indent=2)

        # Disable buttons if completed
        is_completed = self.status.lower() == "completed"
        self.resume_btn.disabled = is_completed
        self.stop_btn.disabled = is_completed
        self.clone_btn.disabled = is_completed
        self.delete_btn.disabled = is_completed

    def _handle_resume(self):
        """Resume the run"""
        try:
            response = requests.post(
                f"{self.dispatcher_url}/dispatcher/resume-run",
                json={"run_id": self.run_id},
                timeout=5,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("error"):
                self._show_message(f"Error: {result['error']}", "error")
            else:
                self._show_message(f"Resumed run {self.run_id}", "success")
                self.load_run(self.run_id)
        except requests.RequestException as e:
            self._show_message(f"Error resuming run: {e}", "error")

    def _handle_stop(self):
        """Stop the run"""
        try:
            response = requests.post(
                f"{self.dispatcher_url}/dispatcher/stop-run",
                json={"run_id": self.run_id},
                timeout=5,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("error"):
                self._show_message(f"Error: {result['error']}", "error")
            else:
                self._show_message(f"Stopped run {self.run_id}", "success")
                self.load_run(self.run_id)
        except requests.RequestException as e:
            self._show_message(f"Error stopping run: {e}", "error")

    def _handle_delete(self):
        """Delete the run"""
        try:
            response = requests.post(
                f"{self.dispatcher_url}/dispatcher/delete-run",
                json={"run_id": self.run_id},
                timeout=5,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("error"):
                self._show_message(f"Error: {result['error']}", "error")
            else:
                self._show_message(f"Deleted run {self.run_id}", "success")
        except requests.RequestException as e:
            self._show_message(f"Error deleting run: {e}", "error")

    def _enable_clone_mode(self):
        """Enable config editing for clone/modify"""
        self.config_text.disabled = False
        self.warm_start_checkbox.disabled = False
        self.submit_clone_btn.disabled = False
        self.cancel_clone_btn.disabled = False
        self.clone_btn.disabled = True
        self._show_message("Edit config and click Submit to clone", "info")

    def _disable_clone_mode(self):
        """Disable config editing"""
        self.config_text.disabled = True
        self.config_text.value = json.dumps(self.config, indent=2)
        self.warm_start_checkbox.disabled = True
        self.warm_start_checkbox.value = False
        self.submit_clone_btn.disabled = True
        self.cancel_clone_btn.disabled = True
        self.clone_btn.disabled = False

    def _handle_cancel_clone(self):
        """Handle cancel clone button click"""
        self._disable_clone_mode()
        self._show_message("Cancelled clone", "info")

    def _enable_colab_widgets(self):
        """Enable custom widget manager for Google Colab"""
        try:
            # Try to import google.colab to detect if we're in Colab
            import google.colab

            # Enable custom widget manager for ipywidgets to work in Colab
            from google.colab import output

            output.enable_custom_widget_manager()
        except ImportError:
            # Not in Colab, no action needed
            pass

    def _handle_clone(self):
        """Clone/modify the run"""
        try:
            # Parse config
            try:
                new_config = json.loads(self.config_text.value)
            except json.JSONDecodeError as e:
                self._show_message(f"Invalid JSON: {e}", "error")
                return

            response = requests.post(
                f"{self.dispatcher_url}/dispatcher/clone-modify-run",
                json={
                    "run_id": self.run_id,
                    "config": new_config,
                    "warm_start": self.warm_start_checkbox.value,
                },
                timeout=5,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("error") or (result.get("result") is False):
                error_msg = result.get("err_msg") or result.get("error")
                self._show_message(f"Error: {error_msg}", "error")
            else:
                self._show_message(f"Cloned run {self.run_id}", "success")
                self._disable_clone_mode()

        except requests.RequestException as e:
            self._show_message(f"Error cloning run: {e}", "error")

    def display(self):
        """Display the interactive controller UI"""
        # Enable custom widget manager for Google Colab
        self._enable_colab_widgets()

        # Layout
        header = widgets.VBox(
            [
                widgets.HTML("<h3>Interactive Run Controller</h3>"),
                widgets.HBox([self.run_id_label, self.status_label, self.chunk_label]),
            ]
        )

        # Run selector section
        selector_section = widgets.VBox(
            [
                widgets.HTML("<b>Select a Run:</b>"),
                widgets.HBox([self.run_selector, self.load_btn, self.refresh_selector_btn]),
            ]
        )

        actions = widgets.HBox([self.resume_btn, self.stop_btn, self.delete_btn, self.refresh_btn])

        config_section = widgets.VBox(
            [
                widgets.HTML("<b>Configuration:</b>"),
                self.config_text,
                self.warm_start_checkbox,
                widgets.HBox([self.clone_btn, self.submit_clone_btn, self.cancel_clone_btn]),
            ]
        )

        # COMMENTED OUT - Displaying experiment status in cell
        # ui = widgets.VBox([header, self.experiment_status, self.status_message, selector_section, actions, config_section])
        ui = widgets.VBox([header, self.status_message, selector_section, actions, config_section])

        display(ui)

        # Automatically fetch available runs
        self.fetch_all_runs()

        # Load initial data if run_id set
        if self.run_id:
            self.load_run(self.run_id)

    def auto_refresh(self, interval: int = 5):
        """Auto-refresh status every N seconds (run in background)"""

        def refresh_loop():
            while True:
                if self.run_id:
                    self.load_run(self.run_id)
                time.sleep(interval)

        thread = threading.Thread(target=refresh_loop, daemon=True)
        thread.start()
