import { Button } from '@databricks/design-system';
import { css } from '@emotion/react';
import { InteractiveControllerIcon } from '../../../../../../common/components/InteractiveControllerIcon';

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
  runningExperimentName?: string;
  context?: { runningExperimentName?: string };
}) => {
  const { runUuid, runName, experimentName: rowExperiment } = props.data;
  const { onOpenController, context } = props;
  // Get running experiment name from either direct prop or context (context is used by ag-grid)
  const runningExperimentName = props.runningExperimentName ?? context?.runningExperimentName;
  const rowExperimentName = rowExperiment?.name;

  if (!onOpenController) {
    return <div css={styles.cellWrapper}>-</div>;
  }

  return (
    <div css={styles.cellWrapper}>
      <Button
        icon={<InteractiveControllerIcon />}
        onClick={() => onOpenController(runUuid, runName)}
        componentId={'interactive-controller-button'}
        disabled={!runningExperimentName || Boolean(rowExperimentName && runningExperimentName !== rowExperimentName)}
      />
    </div>
  );
};
