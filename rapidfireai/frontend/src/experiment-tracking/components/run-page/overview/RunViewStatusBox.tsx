import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { RunInfoEntity } from '../../../types';
import { RunStatusIcon } from '../../RunStatusIcon';
import { FormattedMessage } from 'react-intl';
import type { MlflowRunStatus } from '../../../../graphql/__generated__/graphql';

/**
 * Displays run status cell in run detail overview.
 */
export const RunViewStatusBox = ({ status }: { status: RunInfoEntity['status'] | MlflowRunStatus | null }) => {
  const { theme } = useDesignSystemTheme();
  // KILLED (dispatcher STOPPED) is intentionally halted, not a failure, so
  // it gets the warning (amber/yellow) palette rather than the red
  // reserved for FAILED. See RunStatusIcon.tsx for the matching icon.
  const getTagColor = () => {
    if (status === 'FINISHED') {
      return theme.isDarkMode ? theme.colors.green800 : theme.colors.green100;
    }
    if (status === 'KILLED') {
      return theme.isDarkMode ? theme.colors.yellow800 : theme.colors.yellow100;
    }
    if (status === 'FAILED') {
      return theme.isDarkMode ? theme.colors.red800 : theme.colors.red100;
    }
    if (status === 'SCHEDULED' || status === 'RUNNING') {
      return theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100;
    }

    return undefined;
  };

  // In the RapidFire fork, the dispatcher DB is the source of truth for
  // run/pipeline status text shown to the user. The notebook progress
  // table uses the dispatcher's vocabulary (COMPLETED / STOPPED /
  // ONGOING / NEW / FAILED), while the underlying MLflow server is
  // constrained to its own enum (FINISHED / KILLED / RUNNING /
  // SCHEDULED / FAILED). This box relabels MLflow's native words to the
  // dispatcher's so the dashboard and the notebook tables read
  // identically across both fit and evals modes. See
  // DISPATCHER_TO_MLFLOW_STATUS in
  // rapidfireai/utils/metric_mlflow_manager.py for the mapping.
  const getStatusLabel = () => {
    if (status === 'FINISHED') {
      return (
        <Typography.Text color="success">
          <FormattedMessage
            defaultMessage="COMPLETED"
            description="Run page > Overview > Run status cell > Value for finished state"
          />
        </Typography.Text>
      );
    }
    if (status === 'KILLED') {
      return (
        <Typography.Text color="warning">
          <FormattedMessage
            defaultMessage="STOPPED"
            description="Run page > Overview > Run status cell > Value for killed state"
          />
        </Typography.Text>
      );
    }
    if (status === 'FAILED') {
      return (
        <Typography.Text color="error">
          <FormattedMessage
            defaultMessage="FAILED"
            description="Run page > Overview > Run status cell > Value for failed state"
          />
        </Typography.Text>
      );
    }
    // ONGOING / NEW are kept at ``color="info"`` (TextSecondary in the
    // design system) so they match the neutral, uncolored ``ClockIcon``
    // rendered by ``RunStatusIcon``. Forcing the text to blue while
    // leaving the icon at the inherited text color produced a
    // blue-text / white-icon mismatch -- so we deliberately render
    // both text and icon in the same neutral tone instead.
    if (status === 'RUNNING') {
      return (
        <Typography.Text color="info">
          <FormattedMessage
            defaultMessage="ONGOING"
            description="Run page > Overview > Run status cell > Value for running state"
          />
        </Typography.Text>
      );
    }
    if (status === 'SCHEDULED') {
      return (
        <Typography.Text color="info">
          <FormattedMessage
            defaultMessage="NEW"
            description="Run page > Overview > Run status cell > Value for scheduled state"
          />
        </Typography.Text>
      );
    }
    return status;
  };

  return (
    <Tag
      componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewstatusbox.tsx_81"
      css={{ backgroundColor: getTagColor() }}
    >
      {status && <RunStatusIcon status={status} />}{' '}
      <Typography.Text css={{ marginLeft: theme.spacing.sm }}>{getStatusLabel()}</Typography.Text>
    </Tag>
  );
};
