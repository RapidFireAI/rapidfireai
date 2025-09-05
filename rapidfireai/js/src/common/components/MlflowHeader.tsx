import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { Link } from '../utils/RoutingUtils';
import { RFDocsUrl, Version } from '../constants';
import { DarkThemeSwitch } from '@mlflow/mlflow/src/common/components/DarkThemeSwitch';
import { Button, MenuIcon, useDesignSystemTheme } from '@databricks/design-system';
import logo from '../../common/static/RapidFire_Square_Bug.png';

export const MlflowHeader = ({
  isDarkTheme = false,
  setIsDarkTheme = (val: boolean) => {},
  sidebarOpen,
  toggleSidebar,
}: {
  isDarkTheme?: boolean;
  setIsDarkTheme?: (isDarkTheme: boolean) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <header
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        color: theme.colors.textSecondary,
        display: 'flex',
        paddingLeft: theme.spacing.sm,
        paddingRight: theme.spacing.md,
        paddingTop: theme.spacing.sm + theme.spacing.xs,
        paddingBottom: theme.spacing.xs,
        a: {
          color: theme.colors.textSecondary,
        },
        alignItems: 'center',
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <Button
          type="tertiary"
          componentId="mlflow_header.toggle_sidebar_button"
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
          aria-pressed={sidebarOpen}
          icon={<MenuIcon />}
        />
        <Link to={ExperimentTrackingRoutes.rootRoute}>
          <img
              css={{
                display: 'block',
                height: theme.spacing.md * 1.75,
                color: theme.colors.textPrimary,
              }}
              alt="RapidFireAI"
              src={logo}
            />
        </Link>
        <span
          css={{
            fontSize: theme.typography.fontSizeLg,
            color: theme.colors.textPrimary,
            marginLeft: theme.spacing.sm,
          }}
        >
          RapidFire
        </span>
        <span
          css={{
            fontSize: theme.typography.fontSizeSm,
          }}
        >
          {Version}
        </span>
      </div>
      <div css={{ flex: 1 }} />
      <div css={{ display: 'flex', gap: theme.spacing.lg, alignItems: 'center' }}>
        <DarkThemeSwitch isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
        <a href="https://github.com/RapidFireAI/rapidfireai">GitHub</a>
        <a href={RFDocsUrl}>Docs</a>
      </div>
    </header>
  );
};
