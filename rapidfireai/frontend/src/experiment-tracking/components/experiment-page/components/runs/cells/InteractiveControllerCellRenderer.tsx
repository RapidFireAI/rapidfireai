import { Button } from '@databricks/design-system';
import { css } from '@emotion/react';
import { InteractiveControllerIcon } from '../../../../../../common/components/InteractiveControllerIcon';
import { useIsExperimentRunning } from '../../../../../hooks/useExperimentLogs';

const styles = {
  cellWrapper: css`
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    position: relative;
    z-index: 10;
  `
};

export const InteractiveControllerCellRenderer = (props: {
  data: { runUuid: string; runName: string; experimentName?: { name: string } };
  onOpenController?: (runUuid: string, runName: string) => void;
}) => {
  const { runUuid, runName, experimentName: rowExperiment } = props.data;
  const { onOpenController } = props;
  const rowExperimentName = rowExperiment?.name ?? '';

  // Use the new per-experiment check - this calls the API for THIS row's experiment
  const { data: experimentStatus, isLoading } = useIsExperimentRunning(rowExperimentName, !!rowExperimentName);
  const isThisExperimentRunning = experimentStatus?.is_running ?? false;

  if (!onOpenController) {
    return <div css={styles.cellWrapper}>-</div>;
  }

  return (
    <div css={styles.cellWrapper}>
      <Button
        icon={<InteractiveControllerIcon />}
        onClick={() => onOpenController(runUuid, runName)}
        componentId={'interactive-controller-button'}
        disabled={isLoading || !isThisExperimentRunning}
      />
    </div>
  );
};
