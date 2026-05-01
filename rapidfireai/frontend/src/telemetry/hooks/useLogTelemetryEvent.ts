/**
 * Stub for telemetry hook — not available in the RapidFire fork.
 * The hook returns a function that callers invoke directly with an event payload,
 * matching the upstream MLflow signature: const logEvent = useLogTelemetryEvent(); logEvent({...});
 */
export const useLogTelemetryEvent = () => {
  return (..._args: any[]) => {};
};
