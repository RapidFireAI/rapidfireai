import {
  CheckCircleIcon,
  ClockIcon,
  StopCircleIcon,
  XCircleIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';

const ErrorIcon = () => {
  const { theme } = useDesignSystemTheme();
  return <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />;
};

const FinishedIcon = () => {
  const { theme } = useDesignSystemTheme();
  return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />;
};

// KILLED is the MLflow native enum value the RapidFire fork uses for an
// intentionally stopped run (user IC-stop, Optuna prune, etc.). It is
// semantically distinct from FAILED -- the run did not error, it was
// asked to halt -- so we render it with the warning (amber) color and a
// stop-circle icon rather than the red error icon used for FAILED.
const StoppedIcon = () => {
  const { theme } = useDesignSystemTheme();
  return <StopCircleIcon css={{ color: theme.colors.textValidationWarning }} />;
};

export const RunStatusIcon = ({ status }: { status: string }) => {
  switch (status) {
    case 'FAILED':
      return <ErrorIcon />;
    case 'KILLED':
      return <StoppedIcon />;
    case 'FINISHED':
      return <FinishedIcon />;
    case 'SCHEDULED':
    case 'RUNNING':
      return <ClockIcon />; // This one is the same color as the link
    default:
      return null;
  }
};
