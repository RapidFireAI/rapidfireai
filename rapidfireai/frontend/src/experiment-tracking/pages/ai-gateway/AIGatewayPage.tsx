import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { Header, Spacer, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { useMemo } from 'react';

const AIGatewayPage = () => {
  const { theme } = useDesignSystemTheme();

  const gatewayUrl = useMemo(() => {
    const mlflowUrl = (window as any).__MLflowGatewayUrl || `http://${window.location.hostname}:8852`;
    return `${mlflowUrl}/#/gateway`;
  }, []);

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={
          <FormattedMessage defaultMessage="AI Gateway" description="Header title for the AI Gateway management page" />
        }
      />
      <Spacer shrinks={false} />
      <div
        css={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          borderRadius: theme.borders.borderRadiusMd,
        }}
      >
        <iframe
          src={gatewayUrl}
          title="AI Gateway"
          css={{
            width: '100%',
            height: '100%',
            border: 'none',
            flex: 1,
          }}
        />
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, AIGatewayPage);
