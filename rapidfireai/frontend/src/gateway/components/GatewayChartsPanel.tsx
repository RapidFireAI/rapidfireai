import { useDesignSystemTheme, Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

interface GatewayChartsPanelProps {
  experimentIds: string[];
  showTokenStats?: boolean;
  additionalControls?: React.ReactNode;
  hideTooltipLinks?: boolean;
  tooltipLinkUrlBuilder?: (experimentId: string, timestampMs: number, timeIntervalSeconds: number) => string;
  tooltipLinkText?: React.ReactNode;
  filters?: string[];
}

export const GatewayChartsPanel = ({ additionalControls }: GatewayChartsPanelProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        alignItems: 'center',
        color: theme.colors.textSecondary,
      }}
    >
      {additionalControls}
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Usage charts are not yet available in the RapidFire dashboard."
          description="Placeholder message for gateway usage charts"
        />
      </Typography.Text>
    </div>
  );
};
