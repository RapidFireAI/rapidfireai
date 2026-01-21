import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { DispatcherService } from '../sdk/DispatcherService';

export interface RunningExperimentResponse {
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  experiment_name?: string;
}

/**
 * Hook to check if there's currently a running experiment.
 * Returns the running experiment data and status.
 */
export const useRunningExperiment = (enabled = true) => {
  const query = useQuery<RunningExperimentResponse>(
    ['running-experiment-for-icops'],
    async () => {
      const response = await DispatcherService.getRunningExperiment();
      return response as RunningExperimentResponse;
    },
    {
      enabled,
      staleTime: 10 * 1000, // 10 seconds - check experiment status frequently
      cacheTime: 30 * 1000, // 30 seconds
      retry: 1,
      refetchOnWindowFocus: false,
      refetchInterval: 10 * 1000, // Poll every 10 seconds
    }
  );

  const runningExperimentName = query.data?.status === 'running' ? query.data?.experiment_name : null;
  // eslint-disable-next-line no-console
  console.log('[useRunningExperiment] data:', query.data, 'runningExperimentName:', runningExperimentName);

  return {
    ...query,
    runningExperimentName,
  };
};

/**
 * Gets the appropriate stale time based on whether the experiment is currently running
 * - Active experiments (RUNNING status): 30 seconds (logs update frequently)
 * - Inactive experiments (COMPLETED/FAILED/CANCELLED): 5 minutes (logs are static)
 */
const getStaleTime = (isExperimentRunning: boolean): number => {
  return isExperimentRunning ? 30 * 1000 : 5 * 60 * 1000;
};

/**
 * Hook to fetch experiment logs. These logs are relevant for all experiments
 * (running, completed, failed, cancelled) and update frequently.
 */
export const useExperimentLogs = (experimentName: string, enabled = true) => {
  return useQuery(
    ['experiment-logs', experimentName],
    async () => {
      const logs = await DispatcherService.getLogs({ experiment_name: experimentName });
      return Array.isArray(logs) ? logs : [];
    },
    {
      enabled: enabled && !!experimentName,
      staleTime: 30 * 1000, // 30 seconds - logs update frequently for all experiments
      cacheTime: 10 * 60 * 1000, // 10 minutes
      retry: 2,
      refetchOnWindowFocus: false,
    }
  );
};

/**
 * Hook to fetch interactive controller logs. These logs are only relevant when
 * there's an active running experiment, so we check the experiment status first
 * and adjust caching strategy accordingly.
 */
export const useExperimentICLogs = (experimentName: string, enabled = true) => {
  // IC logs are only relevant when there's an active running experiment
  // First, check if the experiment is currently running
  const { data: runningExperiment } = useQuery<RunningExperimentResponse>(
    ['running-experiment'],
    async () => {
      const response = await DispatcherService.getRunningExperiment();
      return response as RunningExperimentResponse;
    },
    {
      enabled: enabled && !!experimentName,
      staleTime: 10 * 1000, // 10 seconds - check experiment status frequently
      cacheTime: 30 * 1000, // 30 seconds
      retry: 1,
      refetchOnWindowFocus: false,
    }
  );

  const isExperimentRunning = runningExperiment?.status === 'running';

  return useQuery(
    ['experiment-ic-logs', experimentName],
    async () => {
      const logs = await DispatcherService.getICLogs({ experiment_name: experimentName });
      return Array.isArray(logs) ? logs : [];
    },
    {
      enabled: enabled && !!experimentName,
      staleTime: getStaleTime(isExperimentRunning),
      cacheTime: 10 * 60 * 1000, // 10 minutes
      retry: 2,
      refetchOnWindowFocus: false,
    }
  );
};
