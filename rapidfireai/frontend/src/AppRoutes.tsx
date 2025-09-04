import { createMLflowRoutePath } from './common/utils/RoutingUtils';

export const AppRoutes = {
  // Root routes
  root: '/',
  
  // Experiment routes with new prefix
  experiments: {
    root: createMLflowRoutePath('/experiments'),
    list: createMLflowRoutePath('/experiments/list'),
    details: createMLflowRoutePath('/experiments/:experimentId'),
    run: createMLflowRoutePath('/experiments/:experimentId/runs/:runUuid'),
    compare: createMLflowRoutePath('/experiments/compare'),
  },
};
