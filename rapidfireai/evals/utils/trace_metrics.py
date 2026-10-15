"""Aggregate MLflow autolog trace metrics (latency / tokens / spend) for API RAG runs.

RAG pipelines backed by ``RFAPIModelConfig`` route every LLM call through the
MLflow AI gateway with ``mlflow.openai.autolog()`` enabled (see
``rapidfireai/evals/utils/mlflow_utils.py``), and each pipeline opens
``mlflow.start_run(run_id=metric_run_id)`` before batches run. As a result every
``AsyncOpenAI().chat.completions.create()`` call lands as its own ``CHAT_MODEL``
span, tagged with the pipeline's MLflow run id, under a per-batch trace.

This module reads those spans back for a given run and derives three cumulative
"system" metrics, computed *as of the current set of traces*:

* ``query_latency_avg_seconds`` -- sum of each query's ``CHAT_MODEL`` span
  duration divided by the number of queries (true per-query latency, since
  ``asyncio.gather`` fans out concurrent calls that overlap in wall-clock time).
* ``total_tokens`` -- input + output tokens summed across all queries.
* ``token_spend_usd`` -- cost summed across all queries (LiteLLM static pricing).

Because the controller re-aggregates after every shard, a naive
"rescan-everything" implementation would re-fetch and re-deserialize the entire
(monotonically growing) trace history on each shard, making the aggregation cost
scale with the *square* of completed batches on long API RAG runs. To avoid that,
``aggregate_api_trace_metrics`` accepts a per-run ``cache`` dict: it persists the
running sums plus the set of already-aggregated trace ids and a ``timestamp_ms``
cursor across calls, and only fetches traces at/after ``cursor - _CURSOR_LOOKBACK_MS``.
Per-shard work is then proportional to the *new* traces plus one lookback window,
so the whole run stays effectively linear. Dedup by trace id keeps re-scanned
(already-counted) traces from being double-counted; traces whose id resolves
to ``None`` are deduped on a synthetic key built from their ``CHAT_MODEL`` span
ids (see ``_scan_new_traces``) so they don't get re-counted on every lookback.

The lookback margin exists because ``timestamp_ms`` is a trace's *start* time, not
its export time: RAG batches fan out concurrent queries via ``asyncio.gather``, so
a "straggler" call that starts before, but finishes (and is exported) after,
another call in the same or a later batch would otherwise have a ``timestamp_ms``
below the cursor forever once that later call's scan advances it — silently and
permanently dropping the straggler from every future aggregation. Rewinding the
filter by ``_CURSOR_LOOKBACK_MS`` on every scan re-examines that trailing window
so such stragglers are still picked up once they land, at the cost of re-fetching
(but not re-counting, thanks to dedup) traces already seen within the window.

Everything is defensive: any MLflow/LiteLLM failure returns ``None`` so metric
aggregation never breaks a running pipeline. vLLM pipelines produce no OpenAI
autolog spans, so callers gate on ``RFAPIModelConfig`` and this returns ``None``
for them anyway.
"""

from __future__ import annotations

from typing import Any

# Cap the number of traces scanned per aggregation. A single RAG run produces one
# trace per batch; this is a safety valve so an unexpectedly large run can't turn
# the per-shard aggregation into an unbounded scan.
_MAX_TRACES = 100_000

# How far back (in ms) to rewind the incremental-scan cursor on every call, to
# catch stragglers exported out of start-time order (see module docstring).
# Generous on purpose: LLM calls can take a long time under rate-limit backoff
# / retries, and under-counting spend/tokens is worse than a bit of re-fetch
# work. Trace-id dedup makes re-scanning this window every call safe (no
# double counting), just not free -- 30 min bounds the extra work per shard to
# "traces from the last half hour" rather than "the whole run".
_CURSOR_LOOKBACK_MS = 30 * 60 * 1000

# ``_scan_new_traces`` status codes. ``_SCAN_OK`` means the scan completed (it
# may have folded in zero new traces); ``_SCAN_UNSUPPORTED`` means the backing
# store permanently rejected the ``filter_string`` / ``order_by`` clause; any
# other failure is ``_SCAN_ERROR`` (transient -- timeout / network / server
# 5xx). Only ``_SCAN_UNSUPPORTED`` is allowed to flip ``query_options_supported``
# off, so a one-off blip can't force full-history rescans on every later shard.
_SCAN_OK = "ok"
_SCAN_UNSUPPORTED = "unsupported"
_SCAN_ERROR = "error"

# --- System-metric projection bounds (per-query value_range upper bounds) ----
#
# The three ``system/*`` metrics are projected to full-dataset scale through
# the aggregator's ``online_strategy`` (see ``project_system_metrics``). The
# strategy's confidence interval uses a uniform-on-[0, max] variance
# approximation, so each metric needs a per-query upper bound. These are
# derived from the pipeline config where possible (see
# ``system_metric_value_ranges``); the constants below are the fallbacks when a
# field is missing or unparseable. They are deliberately generous so the CI
# stays finite and the projection clamp (``min(b * population, estimate)``)
# does not silently cap real spend/tokens.
_DEFAULT_MAX_LATENCY_S = 300.0
_PROMPT_OVERHEAD_TOKENS = 512
_DEFAULT_MAX_INPUT_TOKENS = 8192
_DEFAULT_MAX_COMPLETION_TOKENS = 150
_DEFAULT_MAX_SPEND_PER_QUERY_USD = 1.0


def _is_unsupported_query_error(exc: BaseException) -> bool:
    """True if ``exc`` signals the store permanently rejects the filter/order-by clause.

    MLflow raises ``MlflowException`` with ``error_code == 'INVALID_PARAMETER_VALUE'``
    when ``search_traces`` is handed a ``filter_string`` / ``order_by`` clause the
    backing store can't honour (e.g. a store that doesn't implement
    ``timestamp_ms`` filtering). That is a deterministic, permanent property of
    the store, so it is the one failure we are willing to permanently disable
    the incremental-scan clause for. Everything else -- request timeouts,
    connection errors, server-side 5xx surfaced under a different error code --
    is transient and must NOT flip ``query_options_supported``.

    ``MlflowException.error_code`` is always the *string* name of the code
    (``ErrorCode.Name(code)``, e.g. ``'INVALID_PARAMETER_VALUE'``), never the
    raw int, so compare against the resolved name rather than the proto int
    constant (which is ``1000`` and would silently never match).
    """
    try:
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode

        unsupported_name = ErrorCode.Name(INVALID_PARAMETER_VALUE)
    except Exception:
        return False
    return (
        isinstance(exc, MlflowException)
        and getattr(exc, "error_code", None) == unsupported_name
    )


def provider_model_for_pipeline(pipeline: Any) -> tuple[str | None, str]:
    """Return ``(model, provider)`` for an ``RFAPIModelConfig`` pipeline.

    ``model`` is the real provider model name (e.g. ``"gpt-4o-mini"``) needed for
    LiteLLM cost lookup -- NOT the gateway endpoint name, which the cost function
    explicitly skips (``_SKIP_COST_PREFIXES``). Returns ``(None, provider)`` when
    the model is not present on the endpoint config (e.g. a pre-existing gateway
    endpoint), in which case cost simply cannot be computed.
    """
    provider = "openai"
    model: str | None = None
    try:
        endpoint_config = getattr(pipeline, "endpoint_config", None) or {}
        provider = endpoint_config.get("provider", "openai") or "openai"
        endpoint = endpoint_config.get("endpoint")
        if isinstance(endpoint, dict):
            model = endpoint.get("model")
        else:
            # automl List (has ``.values``) or a plain list/tuple of endpoint
            # dicts: use the first leaf, matching ``RFAPIModelConfig.model_name``.
            values = getattr(endpoint, "values", None)
            if values is None and isinstance(endpoint, (list, tuple)):
                values = endpoint
            if values:
                first = values[0]
                if isinstance(first, dict):
                    model = first.get("model")
    except Exception:
        return None, provider
    return model, provider


def resolve_mlflow_handle(metric_manager: Any) -> tuple[Any | None, str | None]:
    """Return ``(client, experiment_id)`` for the MLflow backend, or ``(None, None)``.

    ``metric_manager`` may be an ``MLflowMetricLogger`` directly (exposes
    ``.client`` / ``.experiment_id``) or an ``RFMetricLogger`` fan-out wrapper
    that holds one in its ``.metric_loggers`` dict. Returns ``(None, None)`` when
    no MLflow backend is present (e.g. MLflow logging disabled, or only
    TensorBoard/Trackio configured).
    """
    if metric_manager is None:
        return None, None
    client = getattr(metric_manager, "client", None)
    if client is not None:
        return client, getattr(metric_manager, "experiment_id", None)
    sub_loggers = getattr(metric_manager, "metric_loggers", None)
    if isinstance(sub_loggers, dict):
        for sub in sub_loggers.values():
            sub_client = getattr(sub, "client", None)
            if sub_client is not None:
                return sub_client, getattr(sub, "experiment_id", None)
    return None, None


def aggregate_api_trace_metrics(
    client: Any,
    run_id: str,
    model: str | None,
    provider: str | None,
    experiment_id: str | None = None,
    cache: dict | None = None,
) -> dict | None:
    """Aggregate per-query latency / tokens / spend from a run's ``CHAT_MODEL`` spans.

    Args:
        client: An ``mlflow.tracking.MlflowClient`` instance.
        run_id: The MLflow run id whose autolog traces should be aggregated.
        model: Real provider model name for cost lookup (``None`` skips spend).
        provider: Provider name (e.g. ``"openai"``) for cost lookup.
        experiment_id: Experiment id containing the run. When ``None`` it is
            resolved from the run (``search_traces`` with a ``run_id`` requires
            the run's experiment to be in the searched locations).
        cache: Optional per-run accumulator dict, reused across calls for the
            same ``run_id`` so each call only scans traces from roughly the
            last ``_CURSOR_LOOKBACK_MS`` window forward instead of the whole
            history (see module docstring). Mutated in place. When ``None``
            every call performs a fresh full scan.

    Returns:
        ``{"query_latency_avg_seconds", "total_tokens", "token_spend_usd",
        "query_count"}`` (``token_spend_usd`` may be ``None`` when the model is
        unknown or not in LiteLLM's pricing table), or ``None`` when no
        ``CHAT_MODEL`` spans exist or any MLflow call fails.
    """
    if client is None or not run_id:
        return None

    try:
        from mlflow.entities import SpanType
        from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
    except Exception:
        return None

    try:
        if experiment_id is None:
            experiment_id = client.get_run(run_id).info.experiment_id
    except Exception:
        return None
    if experiment_id is None:
        return None

    # Running totals live in ``cache`` so successive per-shard calls accumulate
    # instead of recomputing from scratch. A local dict (cache is None) makes
    # this a one-shot full scan with identical results.
    state = cache if isinstance(cache, dict) else {}
    seen_trace_ids: set = state.setdefault("seen_trace_ids", set())

    # Autolog traces are exported on a background thread, so a just-finished
    # batch's spans may not be queryable yet. Flush pending async trace logging
    # first (best-effort) -- ``flush`` is NOT a kwarg on
    # ``MlflowClient.search_traces`` and passing it raises ``TypeError``.
    try:
        import mlflow

        mlflow.flush_trace_async_logging()
    except Exception:
        pass

    # Only fetch traces at/after (cursor - lookback). The lookback rewind (not a
    # bare ``>=`` on the raw cursor) is what catches stragglers exported out of
    # start-time order -- see module docstring. Trace-id dedup makes re-scanning
    # that trailing window on every call safe. ``query_options_supported`` is
    # unset until proven false so a store that rejects the filter/order-by
    # clause only pays the failed attempt once -- and "proven false" requires a
    # permanent ``INVALID_PARAMETER_VALUE`` rejection, not a transient blip.
    cursor_ms = state.get("cursor_ms")
    filter_string = (
        f"timestamp_ms >= {max(0, int(cursor_ms) - _CURSOR_LOOKBACK_MS)}"
        if cursor_ms is not None
        else None
    )
    use_query_options = state.get("query_options_supported", True)

    status = _scan_new_traces(
        client, run_id, experiment_id, filter_string, state, seen_trace_ids,
        SpanType, SpanAttributeKey, TokenUsageKey, use_query_options=use_query_options,
    )
    if status != _SCAN_OK and use_query_options:
        # The filtered/order-by scan failed. Retry as a plain unscoped scan so
        # we still emit a metric point this shard -- trace-id dedup keeps it
        # correct even if the filtered attempt made partial progress. But only
        # *permanently* drop the incremental clause when the store permanently
        # rejects it (``INVALID_PARAMETER_VALUE``). A transient blip (timeout /
        # network) must NOT flip ``query_options_supported``: otherwise this
        # one failure would force full-history rescans on every later shard,
        # undoing the linear-cost design. The next shard retries the cheap
        # filtered path with the clause still enabled.
        disable_permanently = status == _SCAN_UNSUPPORTED
        status = _scan_new_traces(
            client, run_id, experiment_id, None, state, seen_trace_ids,
            SpanType, SpanAttributeKey, TokenUsageKey, use_query_options=False,
        )
        if disable_permanently and status == _SCAN_OK:
            state["query_options_supported"] = False
    if status != _SCAN_OK:
        return None

    query_count = state.get("query_count", 0)
    if query_count == 0:
        return None

    input_tokens_sum = state.get("input_tokens_sum", 0)
    output_tokens_sum = state.get("output_tokens_sum", 0)
    total_tokens = input_tokens_sum + output_tokens_sum
    token_spend_usd = _compute_cost_usd(model, provider, input_tokens_sum, output_tokens_sum)

    return {
        "query_latency_avg_seconds": state.get("latency_sum_seconds", 0.0) / query_count,
        "total_tokens": total_tokens,
        "token_spend_usd": token_spend_usd,
        "query_count": query_count,
    }


def _scan_new_traces(
    client: Any,
    run_id: str,
    experiment_id: str,
    filter_string: str | None,
    state: dict,
    seen_trace_ids: set,
    SpanType: Any,
    SpanAttributeKey: Any,
    TokenUsageKey: Any,
    use_query_options: bool = True,
) -> str:
    """Fold not-yet-seen ``CHAT_MODEL`` spans into ``state``; return a status code.

    Iterates ``search_traces`` pages (oldest first so the cursor advances
    monotonically), skips any trace already in ``seen_trace_ids``, and for each
    new trace accumulates latency / token sums, records its id, and advances
    ``state["cursor_ms"]``.

    Returns one of:

    * ``_SCAN_OK`` -- the scan completed (possibly with zero new traces).
      Any partial progress made before a *transient* failure is left in
      ``state`` (correct thanks to trace-id dedup on the next retry).
    * ``_SCAN_UNSUPPORTED`` -- the store permanently rejected the
      ``filter_string`` / ``order_by`` clause (``MlflowException`` with
      ``INVALID_PARAMETER_VALUE``). The caller retries unscoped and disables
      the clause for future shards.
    * ``_SCAN_ERROR`` -- any other failure (timeout / network / server 5xx),
      treated as transient. The caller keeps the clause enabled so the next
      shard retries the cheap filtered path instead of permanently falling
      back to a full-history rescan.

    ``use_query_options`` gates the ``filter_string`` / ``order_by`` kwargs so the
    caller can retry with a plain unscoped scan on stores that reject them.
    """
    # Only forward the incremental scoping kwargs when enabled, so the retry path
    # degrades to the original unscoped ``search_traces`` call.
    query_kwargs: dict[str, Any] = {}
    if use_query_options:
        query_kwargs["order_by"] = ["timestamp_ms ASC"]
        if filter_string is not None:
            query_kwargs["filter_string"] = filter_string

    # ``_MAX_TRACES`` caps the number of traces scanned in *this* call (a
    # per-aggregation safety valve), NOT the cumulative ``state["query_count"]``:
    # that total persists across shards, so once a long run crosses the cap a
    # cumulative check would be permanently true and break after the first
    # lookback page on every later shard -- dropping all subsequently exported
    # traces and freezing cost/latency metrics. Count locally instead.
    scanned_this_call = 0
    try:
        page_token: str | None = None
        while True:
            traces = client.search_traces(
                run_id=run_id,
                locations=[experiment_id],
                include_spans=True,
                page_token=page_token,
                **query_kwargs,
            )
            for trace in traces:
                scanned_this_call += 1
                info = getattr(trace, "info", None)
                trace_id = getattr(info, "request_id", None) or getattr(info, "trace_id", None)
                if trace_id is not None and trace_id in seen_trace_ids:
                    continue

                # Accumulate this trace's deltas locally and only commit them
                # (plus mark the key seen) once the whole trace is processed.
                # If span processing raises mid-trace we leave state untouched
                # and the key unseen, so the next scan retries the trace
                # cleanly instead of permanently skipping it (which would
                # under-count latency / tokens / spend). Dedup on the
                # successful retry still prevents double-counting.
                trace_query_count = 0
                trace_latency_seconds = 0.0
                trace_input_tokens = 0
                trace_output_tokens = 0
                # CHAT_MODEL span ids are persisted and stable across re-fetches,
                # so we use them to synthesize a dedup key for traces whose trace
                # id resolves to None (see dedup_key below). Collected during the
                # same pass that computes the deltas.
                chat_span_ids: list[str] = []

                spans = getattr(getattr(trace, "data", None), "spans", None) or []
                for span in spans:
                    if span.span_type != SpanType.CHAT_MODEL:
                        continue
                    trace_query_count += 1
                    start_ns = span.start_time_ns
                    end_ns = span.end_time_ns
                    if start_ns is not None and end_ns is not None and end_ns >= start_ns:
                        trace_latency_seconds += (end_ns - start_ns) / 1e9
                    usage = span.get_attribute(SpanAttributeKey.CHAT_USAGE)
                    if isinstance(usage, dict):
                        trace_input_tokens += usage.get(TokenUsageKey.INPUT_TOKENS, 0) or 0
                        trace_output_tokens += usage.get(TokenUsageKey.OUTPUT_TOKENS, 0) or 0
                    span_id = getattr(span, "span_id", None)
                    if isinstance(span_id, str) and span_id:
                        chat_span_ids.append(span_id)

                # Dedup key: prefer the persisted trace id. When it is None we
                # can't dedup at the top of the loop (the early-skip check above
                # is gated on ``trace_id is not None``), so synthesize a key from
                # the CHAT_MODEL span ids. Without this, a None-id trace would be
                # committed on every lookback re-scan (its timestamp_ms sits at
                # or under the cursor) yet never recorded in seen_trace_ids,
                # inflating latency / tokens / spend across shards for the rest
                # of the run. A trace with no CHAT_MODEL spans contributes 0 to
                # every sum, so leaving it without a key is harmless.
                dedup_key = trace_id
                if dedup_key is None and chat_span_ids:
                    dedup_key = "\x00".join(chat_span_ids)
                if dedup_key is not None and dedup_key in seen_trace_ids:
                    continue

                # Commit deltas and advance cursor only after the trace is
                # fully (and successfully) folded.
                state["query_count"] = state.get("query_count", 0) + trace_query_count
                state["latency_sum_seconds"] = (
                    state.get("latency_sum_seconds", 0.0) + trace_latency_seconds
                )
                state["input_tokens_sum"] = state.get("input_tokens_sum", 0) + trace_input_tokens
                state["output_tokens_sum"] = (
                    state.get("output_tokens_sum", 0) + trace_output_tokens
                )
                timestamp_ms = getattr(info, "timestamp_ms", None)
                if isinstance(timestamp_ms, (int, float)):
                    prev = state.get("cursor_ms")
                    state["cursor_ms"] = timestamp_ms if prev is None else max(prev, timestamp_ms)
                if dedup_key is not None:
                    seen_trace_ids.add(dedup_key)
            page_token = getattr(traces, "token", None)
            if not page_token or scanned_this_call >= _MAX_TRACES:
                break
    except Exception as exc:
        # Classify the failure so the caller only permanently drops the
        # incremental clause for a *permanent* "store can't honour this clause"
        # rejection -- never for a transient timeout / network blip.
        return _SCAN_UNSUPPORTED if _is_unsupported_query_error(exc) else _SCAN_ERROR
    return _SCAN_OK


def _compute_cost_usd(
    model: str | None,
    provider: str | None,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Return summed spend in USD, or ``None`` if it can't be computed.

    Cost is linear in token counts for a fixed model/provider, so computing once
    from the summed input/output tokens equals the sum of per-query costs.
    """
    if not model or (input_tokens == 0 and output_tokens == 0):
        return None
    try:
        from mlflow.tracing.constant import TokenUsageKey
        from mlflow.tracing.utils import calculate_cost_by_model_and_token_usage

        cost = calculate_cost_by_model_and_token_usage(
            model,
            {
                TokenUsageKey.INPUT_TOKENS: input_tokens,
                TokenUsageKey.OUTPUT_TOKENS: output_tokens,
            },
            provider,
        )
    except Exception:
        return None
    if not isinstance(cost, dict):
        return None
    total = cost.get("total_cost")
    return float(total) if isinstance(total, (int, float)) else None


def _max_input_tokens_for_pipeline(pipeline: Any) -> int:
    """Upper bound on input (prompt) tokens for one query of ``pipeline``.

    Approximated as the retrieval context (``k`` chunks of ``chunk_size``) plus
    a fixed prompt overhead. Falls back to ``_DEFAULT_MAX_INPUT_TOKENS`` when the
    RAG spec / splitter / ``k`` are missing or unparseable -- defensive so a
    malformed config never breaks metric projection.
    """
    try:
        rag = getattr(pipeline, "rag", None)
        if rag is not None:
            search_kwargs = getattr(rag, "search_kwargs", None) or {}
            k = search_kwargs.get("k")
            splitter = getattr(rag, "text_splitter", None)
            chunk_size = getattr(splitter, "_chunk_size", None) if splitter else None
            if (
                isinstance(k, (int, float))
                and isinstance(chunk_size, (int, float))
                and k > 0
                and chunk_size > 0
            ):
                return int(k) * int(chunk_size) + _PROMPT_OVERHEAD_TOKENS
    except Exception:
        pass
    return _DEFAULT_MAX_INPUT_TOKENS


def system_metric_value_ranges(
    pipeline: Any,
    model: str | None,
    provider: str | None,
) -> dict[str, tuple[bool, tuple[float, float]]]:
    """Per-query ``value_range`` upper bounds for the three ``system/*`` metrics.

    Returns ``{metric_name: (is_distributive, (min, max))}``. ``is_distributive``
    is True for the cumulative-sum metrics (tokens / spend), which the online
    strategy projects to ``population_size * sample_mean``; False for the
    per-query average (latency), which the strategy scores as an algebraic
    average with a CI but no scaling.

    Bounds are derived from the pipeline config: latency from the OpenAI client
    ``timeout``, tokens from ``max_completion_tokens`` plus the retrieval
    context size, and spend by reusing ``_compute_cost_usd`` on those token
    counts. Every field degrades to a generous default so projection still works
    on partial / pre-existing endpoint configs.
    """
    try:
        client_config = getattr(pipeline, "client_config", None) or {}
        timeout = client_config.get("timeout")
        if isinstance(timeout, (int, float)) and timeout > 0:
            max_latency = float(timeout)
        else:
            max_latency = _DEFAULT_MAX_LATENCY_S
    except Exception:
        max_latency = _DEFAULT_MAX_LATENCY_S

    max_completion = getattr(pipeline, "max_completion_tokens", None)
    if not isinstance(max_completion, (int, float)) or max_completion <= 0:
        max_completion = _DEFAULT_MAX_COMPLETION_TOKENS
    max_completion = float(max_completion)

    max_input = float(_max_input_tokens_for_pipeline(pipeline))
    max_tokens_per_query = max_input + max_completion

    max_spend = _compute_cost_usd(
        model, provider, int(max_input), int(max_completion)
    )
    if not isinstance(max_spend, (int, float)) or max_spend <= 0:
        max_spend = _DEFAULT_MAX_SPEND_PER_QUERY_USD

    return {
        "system/query_latency_avg_seconds": (False, (0.0, max_latency)),
        "system/total_tokens": (True, (0.0, max_tokens_per_query)),
        "system/token_spend_usd": (True, (0.0, float(max_spend))),
    }


def project_system_metrics(
    strategy: Any,
    raw_metrics: dict,
    value_ranges: dict[str, tuple[bool, tuple[float, float]]],
    query_count: int,
) -> dict[str, tuple[float | None, float | None]] | None:
    """Project raw ``system/*`` metrics to full-dataset scale with a CI.

    Shapes the raw cumulative sums / average as ``add_confidence_interval_info``
    expects (``is_distributive`` / ``is_algebraic`` + ``value_range``) and runs
    them through the same ``online_strategy`` the eval metrics use. Returns
    ``{metric: (projected_value, confidence_interval)}`` -- the projected
    value is the distributive ``population_size * sample_mean`` estimate for
    tokens/spend (so a 1-shard run is comparable to a 4-shard run) and the
    unchanged per-query average for latency, each with a margin-of-error CI.

    Returns ``None`` when projection is not possible (no ``query_count``, no
    ``total_population_size`` on the strategy, or no shapable raw values) so
    the caller can fall back to logging raw floats.
    """
    if not isinstance(query_count, (int, float)) or query_count <= 0:
        return None
    population_size = getattr(strategy, "total_population_size", None)
    if not isinstance(population_size, int) or population_size <= 0:
        return None

    shaped: dict[str, dict] = {}
    for name, (is_distributive, value_range) in value_ranges.items():
        # ``system/<x>`` -> raw key ``<x>`` used by aggregate_api_trace_metrics.
        raw_key = name.split("/", 1)[1] if "/" in name else name
        raw_val = raw_metrics.get(raw_key)
        if not isinstance(raw_val, (int, float)):
            continue
        shaped[name] = {
            "value": float(raw_val),
            "value_range": value_range,
            "is_distributive": is_distributive,
            "is_algebraic": not is_distributive,
        }
    if not shaped:
        return None

    projected = strategy.add_confidence_interval_info(shaped, sample_size=int(query_count))
    if not isinstance(projected, dict):
        return None

    out: dict[str, tuple[float | None, float | None]] = {}
    for name, data in projected.items():
        if not isinstance(data, dict):
            continue
        val = data.get("value")
        ci = data.get("confidence_interval")
        out[name] = (
            float(val) if isinstance(val, (int, float)) else None,
            float(ci) if isinstance(ci, (int, float)) else None,
        )
    return out or None
