import { createLazyRouteElement } from '../common/utils/RoutingUtils';

export const getRapidFireUIRouteDefs = () => [
  {
    path: '/rapidfire-ui-demo',
    element: createLazyRouteElement(() => import('./demo/DemoPage')),
    pageId: 'rapidfire.ui.demo',
  },
]; 