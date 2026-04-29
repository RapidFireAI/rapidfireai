/**
 * Issue Detection Modal — Databricks-only feature.
 *
 * Stubbed in the RapidFire OSS fork. The page that imports this component
 * gates rendering on `shouldEnableIssueDetection()` which returns false in
 * OSS, so this stub is never rendered. It exists so the import resolves.
 */
export interface IssueDetectionModalProps {
  componentId?: string;
  visible?: boolean;
  onClose?: () => void;
  experimentId?: string;
}

export const IssueDetectionModal = (_props: IssueDetectionModalProps): null => null;
