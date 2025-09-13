import { useDesignSystemTheme } from '@databricks/design-system';

type StatusType = 'error' | 'success' | 'warning';

interface StatusIndicatorProps {
  status: StatusType;
  size?: number;
}

export const StatusIndicator = ({ status, size = 8 }: StatusIndicatorProps) => {
  const { theme } = useDesignSystemTheme();

  const getStatusColor = (status: StatusType) => {
    switch (status) {
      case 'error':
        return theme.colors.actionDangerPrimaryText;
      case 'success':
        return theme.colors.textPrimary;
      case 'warning':
        return theme.colors.textPrimary;
      default:
        return theme.colors.textPrimary;
    }
  };

  return (
    <div
      css={{
        width: size,
        height: size,
        borderRadius: '50%',
        backgroundColor: getStatusColor(status),
        display: 'inline-block',
      }}
    />
  );
};