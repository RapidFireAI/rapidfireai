import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { DispatcherService } from '../sdk/DispatcherService';

export type RapidfireExperimentMode = 'fit' | 'evals';

/**
 * Resolves the RapidFire mode ("fit" | "evals") of an experiment from the dispatcher
 * DB's experiments table — its experiment_mode column is the golden source of
 * experiment type. Returns an undefined mode when it can't be resolved (e.g. the
 * experiment is unknown to the dispatcher), so callers fall back to mode-agnostic
 * behavior. `isResolved` turns true once the lookup has settled (success or error).
 */
export const useExperimentMode = (experimentName?: string) => {
  const { data, isFetched } = useQuery(
    ['experiment-mode', experimentName],
    async () => {
      const response = (await DispatcherService.getExperimentByName(experimentName ?? '')) as {
        experiment_mode?: string;
      };
      return response?.experiment_mode ?? null;
    },
    {
      enabled: !!experimentName,
      staleTime: Infinity, // the mode of a given experiment never changes
      cacheTime: 60 * 60 * 1000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  );

  const mode: RapidfireExperimentMode | undefined = data === 'fit' || data === 'evals' ? data : undefined;
  return { mode, isResolved: isFetched };
};
