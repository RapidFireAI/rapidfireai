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
  data: { runUuid: string; runName: string };
  onOpenController?: (runUuid: string, runName: string) => void;
  context?: { isExperimentRunning?: boolean };
}) => {
  const { runUuid, runName } = props.data;
  const { onOpenController, context } = props;
  const isExperimentRunning = context?.isExperimentRunning;

  if (!onOpenController) {
    return <div css={styles.cellWrapper}>-</div>;
  }

  return (
    <div css={styles.cellWrapper}>
      <Button
        icon={<InteractiveControllerIcon />}
        onClick={() => onOpenController(runUuid, runName)}
        componentId={'interactive-controller-button'}
        disabled={isExperimentRunning === false}
      />
    </div>
  );
};
