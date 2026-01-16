import { useCallback, useMemo, useState } from 'react';
import { ExperimentEntity } from '../../../../types';
import { useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { DeleteExperimentModal } from '../../../modals/DeleteExperimentModal';
import { RenameExperimentModal } from '../../../modals/RenameExperimentModal';
import { useInvalidateExperimentList } from '../../hooks/useExperimentListQuery';

/**
 * Experiment page header part responsible for displaying menu
 * with rename and delete buttons
 */
export const ExperimentViewManagementMenu = ({
  experiment,
  setEditing,
  baseComponentId = 'mlflow.experiment_page.managementMenu',
}: {
  experiment: ExperimentEntity;
  setEditing?: (editing: boolean) => void;
  baseComponentId?: string;
}) => {
  const [showRenameExperimentModal, setShowRenameExperimentModal] = useState(false);
  const [showDeleteExperimentModal, setShowDeleteExperimentModal] = useState(false);
  const invalidateExperimentList = useInvalidateExperimentList();
  const navigate = useNavigate();

  return (
    <>
      <RenameExperimentModal
        experimentId={experiment.experimentId}
        experimentName={experiment.name}
        isOpen={showRenameExperimentModal}
        onClose={() => setShowRenameExperimentModal(false)}
        onExperimentRenamed={invalidateExperimentList}
      />
      <DeleteExperimentModal
        experimentId={experiment.experimentId}
        experimentName={experiment.name}
        isOpen={showDeleteExperimentModal}
        onClose={() => setShowDeleteExperimentModal(false)}
        onExperimentDeleted={() => {
          invalidateExperimentList();
          navigate(Routes.experimentsObservatoryRoute);
        }}
      />
    </>
  );
};
