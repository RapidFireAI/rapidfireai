import { Button, Tooltip } from '@databricks/design-system';
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
  isExperimentEnded?: boolean;
}) => {
  const { runUuid, runName } = props.data;
  const { onOpenController, isExperimentEnded } = props;

  if (!onOpenController) {
    return <div css={styles.cellWrapper}>-</div>;
  }

  const button = (
    <Button
      icon={<InteractiveControllerIcon />}
      onClick={() => onOpenController(runUuid, runName)}
      componentId={'interactive-controller-button'}
      disabled={isExperimentEnded}
    />
  );

  return (
    <div css={styles.cellWrapper}>
      {isExperimentEnded ? (
        <Tooltip
          content="Experiment has ended"
          componentId="ic-ops-button-disabled-tooltip"
        >
          <span>{button}</span>
        </Tooltip>
      ) : (
        button
      )}
    </div>
  );
};
