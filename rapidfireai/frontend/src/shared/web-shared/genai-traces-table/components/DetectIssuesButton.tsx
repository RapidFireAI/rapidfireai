/**
 * Detect Issues Button — Databricks-only feature.
 *
 * Stubbed in the RapidFire OSS fork. Callers gate rendering on
 * `shouldEnableIssueDetection()` which returns false in OSS, so this
 * stub is never rendered. It exists so the import resolves.
 */
export interface DetectIssuesButtonProps {
  onClick?: () => void;
  disabled?: boolean;
}

export const DetectIssuesButton = (_props: DetectIssuesButtonProps): null => null;
