"""MLflow Gateway client for provisioning secrets, model definitions, and endpoints."""

from __future__ import annotations

import logging
from typing import Any

import requests as http_requests

from rapidfireai.utils.constants import DispatcherConfig, MLflowConfig

logger = logging.getLogger(__name__)


class GatewayError(Exception):
    """Base exception for MLflow Gateway operations."""


class GatewayConnectionError(GatewayError):
    """Raised when the MLflow Gateway server is unreachable."""


class GatewayProvisioningError(GatewayError):
    """Raised when a gateway resource cannot be created or fetched."""


class MLflowGatewayClient:
    """Client for the MLflow Gateway REST API.

    Provides idempotent get-or-create operations for secrets, model definitions,
    and endpoints.  All requests go through the ajax gateway API at
    ``{mlflow_url}/ajax-api/3.0/mlflow/gateway/``.
    """

    _REQUEST_TIMEOUT = 30  # seconds

    def __init__(self, gateway_url: str | None = None):
        if gateway_url is None:
            gateway_url = f"{MLflowConfig.URL}/ajax-api/3.0/mlflow/gateway"
        self._base_url = gateway_url.rstrip("/")

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}/{path}"
        try:
            resp = http_requests.post(url, json=payload, timeout=self._REQUEST_TIMEOUT)
        except http_requests.ConnectionError as exc:
            raise GatewayConnectionError(
                f"Cannot reach MLflow Gateway at {self._base_url}. "
                f"Ensure the MLflow server is running (`rapidfireai start`). "
                f"Details: {exc}"
            ) from exc
        except http_requests.Timeout as exc:
            raise GatewayConnectionError(
                f"Request to {url} timed out after {self._REQUEST_TIMEOUT}s."
            ) from exc

        return self._handle_response(resp, path)

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self._base_url}/{path}"
        try:
            resp = http_requests.get(url, params=params, timeout=self._REQUEST_TIMEOUT)
        except http_requests.ConnectionError as exc:
            raise GatewayConnectionError(
                f"Cannot reach MLflow Gateway at {self._base_url}. "
                f"Ensure the MLflow server is running (`rapidfireai start`). "
                f"Details: {exc}"
            ) from exc
        except http_requests.Timeout as exc:
            raise GatewayConnectionError(
                f"Request to {url} timed out after {self._REQUEST_TIMEOUT}s."
            ) from exc

        return self._handle_response(resp, path)

    @staticmethod
    def _handle_response(resp: http_requests.Response, path: str) -> dict:
        if not resp.ok:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            if 400 <= resp.status_code < 500:
                raise GatewayProvisioningError(
                    f"Gateway rejected request to {path} (HTTP {resp.status_code}): {body}"
                )
            raise GatewayError(
                f"Gateway server error for {path} (HTTP {resp.status_code}): {body}"
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise GatewayError(
                f"Invalid JSON response from {path}: {resp.text[:200]}"
            ) from exc

    @staticmethod
    def _find_by_name(items: list[dict], name: str) -> dict | None:
        return next(
            (x for x in items if x.get("name") == name or x.get("secret_name") == name),
            None,
        )

    # ── Experiment lookup ─────────────────────────────────────────────────────

    @staticmethod
    def _get_mlflow_experiment_id() -> str | None:
        """Fetch the MLflow experiment ID for the running RapidFire experiment.

        Queries the dispatcher's ``/get-running-experiment`` endpoint and
        returns the ``metric_experiment_id`` (the MLflow experiment ID).
        Returns ``None`` if the dispatcher is unreachable or no experiment
        is running.
        """
        try:
            resp = http_requests.get(
                f"{DispatcherConfig.URL}/dispatcher/get-running-experiment",
                timeout=5,
            )
            if resp.ok:
                return resp.json().get("metric_experiment_id")
        except Exception:
            logger.debug("Could not reach dispatcher to fetch MLflow experiment ID")
        return None

    # ── Secret operations ─────────────────────────────────────────────────────

    def list_secrets(self) -> list[dict]:
        return self._get("secrets/list").get("secrets", [])

    def get_or_create_secret(
        self,
        secret_name: str,
        provider: str,
        api_key: str | None,
        api_base_url: str = "",
        api_version: str = "",
        verbose: bool = True,
    ) -> str:
        """Get existing secret by name or create a new one.

        If the secret already exists in the gateway, ``api_key`` is not
        required.  It is only validated when a new secret must be created.

        Returns:
            The ``secret_id`` string.

        Raises:
            ValueError: If the secret does not exist and ``api_key`` is
                empty or not provided.
        """
        existing = self._find_by_name(self.list_secrets(), secret_name)
        if existing:
            if verbose:
                print(f'  Using existing API key: "{secret_name}"')
            logger.debug("Reusing existing secret '%s' (%s)", secret_name, existing["secret_id"])
            return existing["secret_id"]

        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                f"Secret '{secret_name}' does not exist and no api_key was provided. "
                f"Provide a valid API key in endpoint_config['api_key']."
            )

        secret_value = {"api_key": api_key}

        auth_config = {}
        if api_base_url:
            auth_config["api_base"] = api_base_url
        if api_version:
            auth_config["api_version"] = api_version

        payload = {
            "secret_name": secret_name,
            "provider": provider,
            "secret_value": secret_value,
        }
        if auth_config:
            payload["auth_config"] = auth_config

        result = self._post("secrets/create", payload)
        secret = result.get("secret", {})
        secret_id = secret.get("secret_id")
        if not secret_id:
            raise GatewayProvisioningError(
                f"Secret creation for '{secret_name}' returned no secret_id. "
                f"Response: {result}"
            )
        if verbose:
            print(f'  Created new API key: "{secret_name}"')
        logger.info(
            "Created gateway secret '%s' (id=%s, provider=%s)",
            secret_name, secret_id, provider,
        )
        return secret_id

    # ── Model definition operations ───────────────────────────────────────────

    def list_model_definitions(self) -> list[dict]:
        return self._get("model-definitions/list").get("model_definitions", [])

    def get_or_create_model_definition(
        self, name: str, secret_id: str, provider: str, model_name: str,
    ) -> str:
        """Get existing model definition by name or create a new one.

        Returns:
            The ``model_definition_id`` string.
        """
        existing = self._find_by_name(self.list_model_definitions(), name)
        if existing:
            logger.debug(
                "Reusing existing model definition '%s' (%s)", name, existing["model_definition_id"],
            )
            return existing["model_definition_id"]

        result = self._post("model-definitions/create", {
            "name": name,
            "secret_id": secret_id,
            "provider": provider,
            "model_name": model_name,
        })
        model_def_id = result["model_definition"]["model_definition_id"]
        logger.info("Created gateway model definition '%s' (%s)", name, model_def_id)
        return model_def_id

    # ── Endpoint operations ───────────────────────────────────────────────────

    def list_endpoints(self) -> list[dict]:
        return self._get("endpoints/list").get("endpoints", [])

    def get_or_create_endpoint(
        self,
        name: str,
        model_definition_id: str,
        usage_tracking: bool = True,
        experiment_id: str | None = None,
        verbose: bool = True,
    ) -> dict:
        """Get existing endpoint by name or create a new one.

        When the endpoint already exists and ``experiment_id`` is provided,
        the endpoint's tracing experiment is updated if it differs from
        the current one.  This ensures that reused endpoints always log
        traces under the currently running experiment.

        Args:
            name: Endpoint name.
            model_definition_id: ID of the model definition to attach.
            usage_tracking: Enable token usage tracking (default ``True``).
                When true, MLflow logs traces for endpoint invocations.
            experiment_id: MLflow experiment ID for tracing.  Typically
                auto-detected via the dispatcher.  When provided and the
                endpoint already exists with a different ``experiment_id``,
                the endpoint is updated to point to this experiment.

        Returns:
            The full endpoint dict from the gateway (includes ``endpoint_id``,
            ``model_mappings``, etc.).
        """
        existing = self._find_by_name(self.list_endpoints(), name)
        if existing:
            if experiment_id and existing.get("experiment_id") != experiment_id:
                self._update_endpoint(
                    existing["endpoint_id"], experiment_id=experiment_id,
                )
                logger.info(
                    "Updated endpoint '%s' experiment_id to %s", name, experiment_id,
                )
                existing["experiment_id"] = experiment_id
            else:
                logger.debug("Reusing existing endpoint '%s' (%s)", name, existing["endpoint_id"])
            return existing

        payload: dict[str, Any] = {
            "name": name,
            "model_configs": [{
                "model_definition_id": model_definition_id,
                "weight": 1.0,
                "linkage_type": "PRIMARY",
            }],
            "usage_tracking": usage_tracking,
        }
        if experiment_id is not None:
            payload["experiment_id"] = experiment_id

        result = self._post("endpoints/create", payload)
        endpoint = result["endpoint"]
        logger.info("Created gateway endpoint '%s' (%s)", name, endpoint["endpoint_id"])
        return endpoint

    def _update_endpoint(self, endpoint_id: str, **kwargs) -> dict:
        """Update an existing gateway endpoint.

        Calls ``POST endpoints/update`` with the given keyword arguments
        (e.g. ``experiment_id``, ``usage_tracking``).

        Returns:
            The updated endpoint dict from the gateway.
        """
        payload = {"endpoint_id": endpoint_id, **kwargs}
        result = self._post("endpoints/update", payload)
        return result.get("endpoint", {})

    # ── High-level provisioning ───────────────────────────────────────────────

    def provision_endpoints(self, endpoint_config: dict[str, Any], verbose: bool = True) -> list[dict]:
        """Provision all gateway resources described by an endpoint configuration.

        Creates or reuses the shared secret, then for each endpoint dict
        creates or reuses its model definition and gateway endpoint.

        When an endpoint already exists, its API key (secret) is compared
        against the provided ``api_key_name``.  A mismatch raises
        ``GatewayProvisioningError`` so the user can resolve it via the
        MLflow dashboard.

        ``endpoint_config["endpoint"]`` may be a single ``dict`` or a
        ``list[dict]``; both are handled transparently.

        Args:
            endpoint_config: Must contain ``provider``, ``api_key_name``,
                ``api_key``, and ``endpoint`` (dict or list of dicts).

        Returns:
            List of resolved endpoint dicts, each containing at minimum
            ``name`` and ``endpoint_id``.

        Raises:
            GatewayConnectionError: MLflow server is unreachable.
            GatewayProvisioningError: A resource could not be created or
                an existing endpoint uses a different API key.
            ValueError: Required fields are missing.
        """
        provider = endpoint_config.get("provider")
        api_key_name = endpoint_config.get("api_key_name")
        api_key = endpoint_config.get("api_key")
        api_base_url = endpoint_config.get("api_base_url", "")
        api_version = endpoint_config.get("api_version", "")
        endpoint_raw = endpoint_config.get("endpoint")

        if not provider:
            raise ValueError("endpoint_config must include 'provider'")
        if not api_key_name:
            raise ValueError("endpoint_config must include 'api_key_name'")
        if not endpoint_raw:
            raise ValueError("endpoint_config must include 'endpoint'")

        if isinstance(endpoint_raw, dict):
            endpoint_dicts = [endpoint_raw]
        elif isinstance(endpoint_raw, list):
            endpoint_dicts = endpoint_raw
        elif hasattr(endpoint_raw, "values"):
            endpoint_dicts = endpoint_raw.values
        else:
            raise TypeError(
                f"endpoint_config['endpoint'] must be a dict or list-like. "
                f"Got: {type(endpoint_raw).__name__}"
            )

        if verbose:
            print(f"Provisioning MLflow Gateway resources ({provider})...")

        # Auto-detect the MLflow experiment ID for usage tracing
        experiment_id = self._get_mlflow_experiment_id()

        # Step 1: shared secret (one per api_key_name)
        secret_id = self.get_or_create_secret(
            api_key_name, provider, api_key, api_base_url, api_version, verbose=verbose,
        )

        # Cache list calls to avoid repeated round-trips
        existing_model_defs = {
            md["name"]: md for md in self.list_model_definitions()
        }
        # Also build a model_definition_id → model_def lookup for key checks
        model_def_by_id = {
            md["model_definition_id"]: md for md in existing_model_defs.values()
        }

        resolved: list[dict] = []
        for ep in endpoint_dicts:
            name = ep.get("name")
            if not name:
                raise ValueError(f"Each endpoint dict must have a 'name' key. Got: {ep}")

            model = ep.get("model")
            extra = {k: v for k, v in ep.items() if k not in ("name", "model")}

            # Check if endpoint already exists
            existing_ep = self._find_by_name(self.list_endpoints(), name)
            if existing_ep:
                # Validate that the existing endpoint uses the same API key.
                # Trace: endpoint → model_mappings → model_definition → secret_name
                self._validate_endpoint_key(
                    existing_ep, model_def_by_id, api_key_name,
                )

                # Update experiment_id if it changed (traces go to new experiment)
                updated_tracing = False
                if experiment_id and existing_ep.get("experiment_id") != experiment_id:
                    self._update_endpoint(
                        existing_ep["endpoint_id"], experiment_id=experiment_id,
                    )
                    existing_ep["experiment_id"] = experiment_id
                    updated_tracing = True

                if verbose:
                    if updated_tracing:
                        print(f'  Using existing endpoint: "{name}" (updated tracing to current experiment)')
                    else:
                        print(f'  Using existing endpoint: "{name}"')
                logger.debug(
                    "Endpoint '%s' already exists (%s), reusing",
                    name, existing_ep["endpoint_id"],
                )
                resolved.append({
                    "name": name,
                    "endpoint_id": existing_ep["endpoint_id"],
                    "model": model,
                    **extra,
                })
                continue

            # Endpoint doesn't exist — model is required to create it
            if not model:
                raise GatewayProvisioningError(
                    f"Endpoint '{name}' does not exist in the gateway and no 'model' was "
                    f"specified in the endpoint config.  Either create the endpoint manually "
                    f"or add a 'model' key to this endpoint dict."
                )

            # Step 2: model definition (one per endpoint)
            model_def_name = f"{name}-model-def"
            if model_def_name in existing_model_defs:
                model_def_id = existing_model_defs[model_def_name]["model_definition_id"]
                logger.debug("Reusing existing model definition '%s' (%s)", model_def_name, model_def_id)
            else:
                model_def_id = self.get_or_create_model_definition(
                    name=model_def_name,
                    secret_id=secret_id,
                    provider=provider,
                    model_name=model,
                )

            # Step 3: endpoint
            usage_tracking = ep.get("usage_tracking", True)
            gw_endpoint = self.get_or_create_endpoint(
                name, model_def_id,
                usage_tracking=usage_tracking,
                experiment_id=experiment_id,
                verbose=verbose,
            )
            if verbose:
                print(f'  Created new endpoint: "{name}"')

            resolved.append({
                "name": name,
                "endpoint_id": gw_endpoint["endpoint_id"],
                "model": model,
                **extra,
            })

        return resolved

    def _validate_endpoint_key(
        self,
        endpoint: dict,
        model_def_by_id: dict[str, dict],
        expected_key_name: str,
    ) -> None:
        """Raise if an existing endpoint's API key doesn't match ``expected_key_name``.

        Traces endpoint → model_mappings → model_definition → secret_name
        and compares against the key name the user provided.
        """
        mappings = endpoint.get("model_mappings") or []
        for mapping in mappings:
            md_id = mapping.get("model_definition_id")
            if not md_id:
                continue
            model_def = model_def_by_id.get(md_id)
            if not model_def:
                continue
            existing_secret_name = model_def.get("secret_name", "")
            if existing_secret_name and existing_secret_name != expected_key_name:
                raise GatewayProvisioningError(
                    f"Endpoint '{endpoint.get('name')}' already exists but uses "
                    f'API key "{existing_secret_name}" instead of "{expected_key_name}". '
                    f"Please update the endpoint configuration using the MLflow "
                    f"dashboard at {self._base_url.replace('/ajax-api/3.0/mlflow/gateway', '')}/#/gateway "
                    f"or use the same api_key_name."
                )
