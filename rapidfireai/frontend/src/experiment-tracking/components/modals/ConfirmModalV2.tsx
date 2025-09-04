import React, { useState, useCallback, useEffect } from 'react';
import { Modal, Button } from '@databricks/design-system';

interface ConfirmModalProps {
  isOpen: boolean;
  handleSubmit: () => void;  
  onClose: () => void;
  title: React.ReactNode;
  helpText: React.ReactNode;
  confirmButtonText: React.ReactNode;
}

export const ConfirmModalV2: React.FC<ConfirmModalProps> = ({
  isOpen,
  handleSubmit,
  onClose,
  title,
  helpText,
  confirmButtonText,
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
  }, [isOpen]);

  const onRequestCloseHandler = useCallback(() => {
    if (!isSubmitting) {
      onClose();
    }
  }, [isSubmitting, onClose]);

//   const handleSubmitWrapper = useCallback(async () => {
//     console.log('ConfirmModal: handleSubmitWrapper called');
//     setIsSubmitting(true);
//     setError(null);
//     try {
//       await handleSubmit();
//       console.log('ConfirmModal: handleSubmit finished successfully');
//       onClose();
//     } catch (error) {
//       console.error('ConfirmModal: handleSubmit error', error);
//       setError('An error occurred while processing your request.');
//     } finally {
//       setIsSubmitting(false);
//     }
//   }, [handleSubmit, onClose]);

//   const handleOk = useCallback((e: React.MouseEvent) => {
//     console.log('ConfirmModal: handleOk called');
//     e.preventDefault();
//     handleSubmitWrapper();
//   }, [handleSubmitWrapper]);

const handleSubmitWrapper = async () => {
    setIsSubmitting(true);
    try {
      await handleSubmit();
    } catch (error) {
      console.error('ConfirmModalV2: handleSubmit error', error);
    } finally {
      setIsSubmitting(false);
      onClose();
    }
  };

  const handleOk = async (e: React.MouseEvent) => {
    e.preventDefault();
    await handleSubmitWrapper();
  };

  return (
    <Modal
      data-testid="confirm-modal"
      title={title}
      visible={isOpen}
      onCancel={onRequestCloseHandler}
      footer={[
        <Button key="cancel" onClick={onRequestCloseHandler} componentId={''}>
          Cancel
        </Button>,
        <Button
            key="submit"
            type="primary"
            loading={isSubmitting}
            onClick={handleOk} componentId={''}        
        >
          {confirmButtonText}
        </Button>,
      ]}
    //   centered
    >
      <div className="modal-explanatory-text">{helpText}</div>
      {error && (
        <div style={{ color: 'red', marginTop: '10px' }}>{error}</div>
      )}
    </Modal>
  );
};
