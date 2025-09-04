import { useCallback } from 'react';
import { useIntl } from 'react-intl';
import type { NotificationInstance } from '@databricks/design-system';

const CONTROLLER_NOTIFICATION_DURATION = 120; // Seconds
const CONTROLLER_NOTIFICATION_KEY = 'CONTROLLER_NOTIFICATION_KEY';

export type ControllerAction = 'resume' | 'clone_modify' | 'stop' | 'delete';

export const useControllerNotification = (notification: NotificationInstance) => {
  const { formatMessage } = useIntl();

  const getMessage = useCallback(
    (action: ControllerAction, status: 'success' | 'error') => {
      const messages = {
        resume: {
          success: formatMessage({
            defaultMessage: 'Run resumed successfully',
            description: 'Controller notification > Resume run success',
          }),
          error: formatMessage({
            defaultMessage: 'Failed to resume run',
            description: 'Controller notification > Resume run error',
          }),
        },
        clone_modify: {
          success: formatMessage({
            defaultMessage: 'Run cloned and modified successfully',
            description: 'Controller notification > Clone and modify run success',
          }),
          error: formatMessage({
            defaultMessage: 'Failed to clone and modify run',
            description: 'Controller notification > Clone and modify run error',
          }),
        },
        stop: {
          success: formatMessage({
            defaultMessage: 'Run stopped successfully',
            description: 'Controller notification > Stop run success',
          }),
          error: formatMessage({
            defaultMessage: 'Failed to stop run',
            description: 'Controller notification > Stop run error',
          }),
        },
        delete: {
          success: formatMessage({
            defaultMessage: 'Run deleted successfully, page should refresh',
            description: 'Controller notification > Delete run success',
          }),
          error: formatMessage({
            defaultMessage: 'Failed to delete run',
            description: 'Controller notification > Delete run error',
          }),
        },
      };

      return messages[action][status];
    },
    [formatMessage]
  );

  return useCallback(
    (action: ControllerAction, status: 'success' | 'error', errorMessage?: string) => {
      // If there is a similar notification visible already, close it first
      notification.close(CONTROLLER_NOTIFICATION_KEY);

      // Display the notification
      notification[status]({
        message: errorMessage || getMessage(action, status),
        duration: CONTROLLER_NOTIFICATION_DURATION,
        placement: 'topRight',
        key: CONTROLLER_NOTIFICATION_KEY,
      });
    },
    [notification, getMessage]
  );
};
