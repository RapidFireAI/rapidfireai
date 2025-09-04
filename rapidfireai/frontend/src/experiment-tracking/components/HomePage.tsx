/* eslint-disable react-hooks/rules-of-hooks */
import { useEffect, useRef, useState, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { ReduxState, type ThunkDispatch } from '../../redux-types';
import { values } from 'lodash';

import { 
  getExperimentApi, 
  searchExperimentsApi, 
  setCompareExperiments, 
  setExperimentTagApi, 
  updateMLflowServerStatus, 
  updateDispatcherStatus
} from '../actions';

import { getUUID } from '../../common/utils/ActionUtils';
import { useExperimentIds } from './experiment-page/hooks/useExperimentIds';
import Utils from '../../common/utils/Utils';
import { ExperimentEntity } from '../types';
import Routes from '../routes';
import { ThunkAction } from 'redux-thunk';
import { AnyAction } from 'redux';
import { checkMLflowServer, checkDispatcher } from 'experiment-tracking/utils/ProxyCheckUtils';

import { Link, Navigate } from '../../common/utils/RoutingUtils';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import ExperimentListView from './ExperimentListView';
import { Spinner, useDesignSystemTheme, ParagraphSkeleton, TitleSkeleton, LegacySkeleton, useLegacyNotification, Button, NotificationInstance } from '@databricks/design-system';
import { GetExperimentsContextProvider } from './experiment-page/contexts/GetExperimentsContext';
import { ExperimentView } from './experiment-page/ExperimentView';
import { NoExperimentView } from './NoExperimentView';

const getExperimentActions = {
  setExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

const getFirstActiveExperiment = (experiments: ExperimentEntity[]) => {
  const sorted = [...experiments].sort(Utils.compareExperiments);
  return sorted.find(({ lifecycleStage }) => lifecycleStage === 'active');
};

export const setupAndFetchExperiments = (
  searchRequestId: string
): ThunkAction<Promise<void>, ReduxState, unknown, AnyAction> => {
  return async (dispatch) => {
    
    const isMLflowServerValid = await checkMLflowServer();
    const isDispatcherValid = await checkDispatcher();

    if (isMLflowServerValid) {
      await dispatch(searchExperimentsApi(searchRequestId));
    } else {
      dispatch(updateMLflowServerStatus({ isValid: false, uri: "" }));
    }

    if (isDispatcherValid) {
      // await dispatch(searchExperimentsApi(searchRequestId));
      // flash notification that dispatcher server is down
    } else {
      dispatch(updateDispatcherStatus({ isValid: false, uri: "" }));
    }
  };
};

const ExperimentsPageSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  
  return (
    <div css={{ display: 'flex', height: 'calc(100% - 60px)' }}>
      {/* Left sidebar skeleton */}
      <div css={{ 
        width: '250px', 
        height: '100%', 
        paddingTop: 24, 
        borderRight: `1px solid ${theme.colors.borderDecorative}`,
        padding: theme.spacing.md 
      }}>
        <TitleSkeleton />
        {/* Experiment list items */}
        {[...Array(5)].map((_, i) => (
          <div key={i} css={{ marginBottom: theme.spacing.md }}>
            <ParagraphSkeleton seed={`exp-${i}`} />
          </div>
        ))}
      </div>

      {/* Main content skeleton */}
      <div css={{ 
        flex: 1, 
        padding: theme.spacing.md, 
        paddingTop: theme.spacing.lg 
      }}>
        {/* Header area */}
        <div css={{ marginBottom: theme.spacing.lg }}>
          <TitleSkeleton />
          <ParagraphSkeleton seed="header-desc" />
        </div>

        {/* Tabs/filters area */}
        <div css={{ 
          display: 'flex', 
          gap: theme.spacing.md,
          marginBottom: theme.spacing.lg 
        }}>
          {[...Array(3)].map((_, i) => (
            <LegacySkeleton key={i} css={{ width: '100px', height: '32px' }} />
          ))}
        </div>

        {/* Table content */}
        {[...Array(5)].map((_, i) => (
          <div key={i} css={{ marginBottom: theme.spacing.lg }}>
            <ParagraphSkeleton seed={`row-${i}`} />
          </div>
        ))}
      </div>
    </div>
  );
};

const NO_CLUSTER_NOTIFICATION_KEY = 'NO_CLUSTER_NOTIFICATION_KEY';
const NO_CLUSTER_NOTIFICATION_DURATION = null; // null for permanent notification

const HomePage = () => {
  const dispatch = useDispatch<ThunkDispatch>();
  const { theme } = useDesignSystemTheme();
  const searchRequestId = useRef(getUUID());
  const [isLoading, setIsLoading] = useState(true);
  const [notificationFn, notificationContainer] = useLegacyNotification();
  
  const experimentIds = useExperimentIds();
  const experiments = useSelector((state: ReduxState) => values(state.entities.experimentsById));
  const mlflowServerStatus = useSelector((state: ReduxState) => state.clusters.mlflowServer.status);
  const hasExperiments = experiments.length > 0;

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        await dispatch(setupAndFetchExperiments(searchRequestId.current));
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [dispatch]);

  if (!mlflowServerStatus.isValid) {
    return (
      <div css={{ height: '100%' }}>
        <ExperimentsPageSkeleton />
        {notificationContainer}
      </div>
    );
  }

  if (!experimentIds.length) {
    const firstExp = getFirstActiveExperiment(experiments);
    if (firstExp) {
      return <Navigate to={Routes.getExperimentPageRoute(firstExp.experimentId)} replace />;
    }
  }

  const loadingState = (
    <div css={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <Spinner size="large" />
    </div>
  );

  return (
    <RequestStateWrapper requestIds={[searchRequestId.current]} customSpinner={loadingState}>
      <div css={{ display: 'flex', height: 'calc(100% - 60px)' }}>
        <div css={{ height: '100%', paddingTop: 24, display: 'flex' }}>
          <ExperimentListView activeExperimentIds={experimentIds || []} experiments={experiments} />
        </div>

        <div css={{ height: '100%', flex: 1, padding: theme.spacing.md, paddingTop: theme.spacing.lg }}>
          <GetExperimentsContextProvider actions={getExperimentActions}>
            {hasExperiments ? <ExperimentView /> : <NoExperimentView />}
          </GetExperimentsContextProvider>
        </div>
      </div>
      {notificationContainer}
    </RequestStateWrapper>
  );
};

export default HomePage;
