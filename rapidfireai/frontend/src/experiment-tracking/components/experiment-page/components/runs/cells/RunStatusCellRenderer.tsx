import { Theme } from '@emotion/react';
import React from 'react';
import { FormattedMessage } from 'react-intl';
import { Typography } from '@databricks/design-system';
import { RunStatusIcon } from '../../../../RunStatusIcon';

/**
 * Cell renderer for the run-status column in the experiment runs table.
 *
 * In the RapidFire fork, the dispatcher DB is the source of truth for
 * run/pipeline status text. The notebook progress table uses the
 * dispatcher's vocabulary (COMPLETED / STOPPED / ONGOING / NEW /
 * FAILED), while MLflow stores its own enum (FINISHED / KILLED /
 * RUNNING / SCHEDULED / FAILED). This cell -- mirroring
 * RunViewStatusBox in the run detail page -- relabels MLflow's words
 * to the dispatcher's, so the experiment table and the notebook table
 * read identically.
 *
 * Colors:
 * - FINISHED (COMPLETED) -> success/green
 * - KILLED   (STOPPED)   -> warning/amber (intentional halt, not an error)
 * - FAILED               -> error/red
 * - RUNNING  (ONGOING)   -> info/secondary (matches the neutral clock icon)
 * - SCHEDULED (NEW)      -> info/secondary (matches the neutral clock icon)
 *
 * NOTE on ONGOING/NEW color: we briefly tried forcing the text to blue
 * (`theme.colors.blue500`), but the matching ``ClockIcon`` in
 * ``RunStatusIcon.tsx`` has no explicit color and inherits the
 * design-system text color. That produced a blue-text / white-icon
 * combo that looked broken. The cleanest answer is to keep both at
 * ``color="info"`` (TextSecondary in the design system) so text and
 * icon match -- visually neutral, but consistent.
 */
export interface RunStatusCellRendererProps {
  value?: string;
}

const StatusLabel = ({ status }: { status?: string }) => {
  switch (status) {
    case 'FINISHED':
      return (
        <Typography.Text color="success">
          <FormattedMessage
            defaultMessage="COMPLETED"
            description="Experiment runs table > Status column > Value for finished state"
          />
        </Typography.Text>
      );
    case 'KILLED':
      return (
        <Typography.Text color="warning">
          <FormattedMessage
            defaultMessage="STOPPED"
            description="Experiment runs table > Status column > Value for killed state"
          />
        </Typography.Text>
      );
    case 'FAILED':
      return (
        <Typography.Text color="error">
          <FormattedMessage
            defaultMessage="FAILED"
            description="Experiment runs table > Status column > Value for failed state"
          />
        </Typography.Text>
      );
    case 'RUNNING':
      return (
        <Typography.Text color="info">
          <FormattedMessage
            defaultMessage="ONGOING"
            description="Experiment runs table > Status column > Value for running state"
          />
        </Typography.Text>
      );
    case 'SCHEDULED':
      return (
        <Typography.Text color="info">
          <FormattedMessage
            defaultMessage="NEW"
            description="Experiment runs table > Status column > Value for scheduled state"
          />
        </Typography.Text>
      );
    default:
      return <>{status ?? '-'}</>;
  }
};

export const RunStatusCellRenderer = React.memo(({ value }: RunStatusCellRendererProps) => {
  if (!value) {
    return <>-</>;
  }
  return (
    <span css={styles.cellWrapper}>
      <RunStatusIcon status={value} />
      <StatusLabel status={value} />
    </span>
  );
});

const styles = {
  cellWrapper: (theme: Theme) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
  }),
};
