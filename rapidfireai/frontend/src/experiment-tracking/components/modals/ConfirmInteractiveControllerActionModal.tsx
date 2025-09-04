import React from 'react';
import { ConfirmModalV2 } from './ConfirmModalV2';

export type ConfirmActionType = 'stop' | 'resume' | 'delete';

interface ConfirmActionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  actionType: ConfirmActionType;
  runName: string;
}

const actionMessages = {
  stop: 'stop',
  resume: 'resume',
  delete: 'delete'
};

export const ConfirmActionModal: React.FC<ConfirmActionModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  actionType,
  runName,
}) => {
  const handleSubmit = () => {
    return onConfirm();
  };

  return (
    <ConfirmModalV2
      isOpen={isOpen}
      onClose={onClose}
      handleSubmit={handleSubmit}
      title={`${actionType.charAt(0).toUpperCase() + actionType.slice(1)} Experiment Run`}
      helpText={
        <div>
          <p>
            <b>Are you sure you want to {actionMessages[actionType]} the experiment run "{runName}"?</b>
          </p>
        </div>
      }
      confirmButtonText={actionType.charAt(0).toUpperCase() + actionType.slice(1)}
    />
  );
};
