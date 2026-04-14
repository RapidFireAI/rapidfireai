import type { DocumentTitleHandle } from '../common/utils/RoutingUtils';
import { createLazyRouteElement, createMLflowRoutePath } from '../common/utils/RoutingUtils';
import { GatewayPageId, GatewayRoutePaths } from './routes';

export const getGatewayRouteDefs = () => {
  return [
    // Endpoint create/details render outside the GatewayPage shell.
    // Registered first so they match before the catch-all /gateway/* parent route.
    {
      path: GatewayRoutePaths.createEndpointPage,
      element: createLazyRouteElement(() => import('./pages/CreateEndpointPage')),
      pageId: GatewayPageId.createEndpointPage,
      handle: { getPageTitle: () => 'Create Endpoint' } satisfies DocumentTitleHandle,
    },
    {
      path: GatewayRoutePaths.endpointDetailsPage,
      element: createLazyRouteElement(() => import('./pages/EndpointPage')),
      pageId: GatewayPageId.endpointDetailsPage,
      handle: {
        getPageTitle: (params?: Record<string, string>) => `Endpoint ${params?.['endpointId']}`,
      } satisfies DocumentTitleHandle,
    },
    // Catch-all: /gateway, /gateway/api-keys, /gateway/usage, /gateway/budgets
    // all render the GatewayPage shell, which switches content based on the URL.
    {
      path: createMLflowRoutePath('/gateway/*'),
      element: createLazyRouteElement(() => import('./pages/GatewayPage')),
      pageId: GatewayPageId.gatewayPage,
      handle: { getPageTitle: () => 'AI Gateway' } satisfies DocumentTitleHandle,
    },
  ];
};
