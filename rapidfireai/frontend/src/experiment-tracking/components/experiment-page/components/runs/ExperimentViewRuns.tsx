import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useLegacyNotification } from '@databricks/design-system';
import {
  DatasetSummary,
  ExperimentEntity,
  LIFECYCLE_FILTER,
  MODEL_VERSION_FILTER,
  RunDatasetWithTags,
  UpdateExperimentViewStateFn,
} from '../../../../types';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunsControls } from './ExperimentViewRunsControls';
import { ExperimentViewRunsTable } from './ExperimentViewRunsTable';
import { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import Utils from '../../../../../common/utils/Utils';
import { ATTRIBUTE_COLUMN_SORT_KEY, MLFLOW_LOGGED_IMAGE_ARTIFACTS_PATH } from '../../../../constants';
import { RunRowType } from '../../utils/experimentPage.row-types';
import { useExperimentRunRows } from '../../utils/experimentPage.row-utils';
import { useFetchedRunsNotification } from '../../hooks/useFetchedRunsNotification';
import { DatasetWithRunType, ExperimentViewDatasetDrawer } from './ExperimentViewDatasetDrawer';
import { useExperimentViewLocalStore } from '../../hooks/useExperimentViewLocalStore';
import { EvaluationArtifactCompareView } from '../../../evaluation-artifacts-compare/EvaluationArtifactCompareView';
import {
  shouldEnableExperimentPageAutoRefresh,
  shouldUseGetLoggedModelsBatchAPI,
} from '../../../../../common/utils/FeatureUtils';
import { CreateNewRunContextProvider } from '../../hooks/useCreateNewRun';
import { useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';
import { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { RunsCompare } from '../../../runs-compare/RunsCompare';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { ReduxState, ThunkDispatch } from '../../../../../redux-types';
import { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { useIsTabActive } from '../../../../../common/hooks/useIsTabActive';
import { ExperimentViewRunsTableResizer } from './ExperimentViewRunsTableResizer';
import { RunsChartsSetHighlightContextProvider } from '../../../runs-charts/hooks/useRunsChartTraceHighlight';
import { useLoggedModelsForExperimentRunsTable } from '../../hooks/useLoggedModelsForExperimentRunsTable';
import { ExperimentViewRunsRequestError } from '../ExperimentViewRunsRequestError';
import { useLoggedModelsForExperimentRunsTableV2 } from '../../hooks/useLoggedModelsForExperimentRunsTableV2';
import { useResizableMaxWidth } from '@mlflow/mlflow/src/shared/web-shared/hooks/useResizableMaxWidth';
import { useControllerNotification } from '../../hooks/useInteractiveControllerNotification';
import InteractiveControllerComponent from '../../../run-page/InteractiveController';
import RightSlidingDrawer from '../../../../../rapidfire-ui/components/RightSlidingDrawer';
import TerminalLogViewer from '../../../TerminalLogViewer';
import { useExperimentLogs, useExperimentICLogs } from '../../../../hooks/useExperimentLogs';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';
import { DispatcherService } from '../../../../sdk/DispatcherService';

export interface ExperimentViewRunsOwnProps {
  isLoading: boolean;
  experiments: ExperimentEntity[];
  modelVersionFilter?: MODEL_VERSION_FILTER;
  lifecycleFilter?: LIFECYCLE_FILTER;
  datasetsFilter?: DatasetSummary[];
  onMaximizedChange?: (newIsMaximized: boolean) => void;

  searchFacetsState: ExperimentPageSearchFacetsState;
  uiState: ExperimentPageUIState;
}

export interface ExperimentViewRunsProps extends ExperimentViewRunsOwnProps {
  runsData: ExperimentRunsSelectorResult;
  isLoadingRuns: boolean;
  loadMoreRuns: () => Promise<any>;
  moreRunsAvailable: boolean;
  requestError: ErrorWrapper | Error | null;
  refreshRuns: () => void;
}

/**
 * Creates time with milliseconds set to zero, usable in calculating
 * relative time
 */
const createCurrentTime = () => {
  const mountTime = new Date();
  mountTime.setMilliseconds(0);
  return mountTime;
};

const INITIAL_RUN_COLUMN_SIZE = 295;
const CHARTS_MIN_WIDTH = 350;

export const ExperimentViewRuns = React.memo((props: ExperimentViewRunsProps) => {
  const [compareRunsMode] = useExperimentPageViewMode();
  const { theme } = useDesignSystemTheme();
  const updateUIState = useUpdateExperimentViewUIState();
  
  const {
    experiments,
    runsData,
    uiState,
    searchFacetsState,
    isLoadingRuns,
    loadMoreRuns,
    moreRunsAvailable,
    requestError,
    refreshRuns,
  } = props;

  const isComparingExperiments = experiments.length > 1;

  // Non-persistable view model state is being created locally
  const [viewState, setViewState] = useState(new ExperimentPageViewState());

  const { experimentId } = experiments[0];
  const expandRowsStore = useExperimentViewLocalStore(experimentId);
  const [expandRows, updateExpandRows] = useState<boolean>(expandRowsStore.getItem('expandRows') === 'true');

  useEffect(() => {
    expandRowsStore.setItem('expandRows', expandRows);
  }, [expandRows, expandRowsStore]);

  const {
    paramKeyList,
    metricKeyList,
    tagsList,
    paramsList,
    metricsList,
    runInfos,
    runUuidsMatchingFilter,
    datasetsList,
    inputsOutputsList,
  } = runsData;

  // Fetch logs using React Query with intelligent caching based on experiment status
  const experimentName = experiments[0]?.name;
  const { data: logs = [], isLoading: isLoadingLogs, error: logsError } = useExperimentLogs(
    experimentName,
    compareRunsMode === 'LOGS'
  );
  const { data: icLogs = [], isLoading: isLoadingICLogs, error: icLogsError } = useExperimentICLogs(
    experimentName,
    compareRunsMode === 'IC_LOGS'
  );

  const [hasRunningExperiment, setHasRunningExperiment] = useState<boolean>(true);

  useEffect(() => {
    const checkRunningExperiment = async () => {
      /* eslint-disable no-console */
      try {
        const response = await DispatcherService.getRunningExperiment();
        console.log('getRunningExperiment response:', response);
        const isRunning = Boolean(response && typeof response === 'object' && Object.keys(response).length > 0);
        console.log('isRunning:', isRunning);
        setHasRunningExperiment(isRunning);
      } catch (error) {
        console.log('getRunningExperiment error:', error);
        setHasRunningExperiment(false);
      }
      /* eslint-enable no-console */
    };
  
    checkRunningExperiment();
    const interval = setInterval(checkRunningExperiment, 5000);
    return () => clearInterval(interval);
  }, []);

  // Check if the experiment has ended by looking at run statuses
  // Experiment has ended if there are runs and none of them are currently running
  const isExperimentEnded = !hasRunningExperiment;

  const modelVersionsByRunUuid = useSelector(({ entities }: ReduxState) => entities.modelVersionsByRunUuid);

  /**
   * Create a list of run infos with assigned metrics, params and tags
   */
  const runData = useMemo(
    () =>
      runInfos.map((runInfo, index) => ({
        runInfo,
        params: paramsList[index],
        metrics: metricsList[index],
        tags: tagsList[index],
        datasets: datasetsList[index],
        inputs: inputsOutputsList?.[index]?.inputs || {},
        outputs: inputsOutputsList?.[index]?.outputs || {},
      })),
    [datasetsList, metricsList, paramsList, runInfos, tagsList, inputsOutputsList],
  );

  const { orderByKey, searchFilter } = searchFacetsState;
  // In new view state model, runs state is in the uiState instead of the searchFacetsState.
  const { runsPinned, runsExpanded, runsHidden, runListHidden } = uiState;

  const isComparingRuns = compareRunsMode !== 'TABLE';

  const updateViewState = useCallback<UpdateExperimentViewStateFn>(
    (newPartialViewState) => setViewState((currentViewState) => ({ ...currentViewState, ...newPartialViewState })),
    [],
  );

  const addColumnClicked = useCallback(() => {
    updateViewState({ columnSelectorVisible: true });
  }, [updateViewState]);

  const shouldNestChildrenAndFetchParents = useMemo(
    () => (!orderByKey && !searchFilter) || orderByKey === ATTRIBUTE_COLUMN_SORT_KEY.DATE,
    [orderByKey, searchFilter],
  );

  // Value used a reference for the "date" column
  const [referenceTime, setReferenceTime] = useState(createCurrentTime);

  // We're setting new reference date only when new runs data package has arrived
  useEffect(() => {
    setReferenceTime(createCurrentTime);
  }, [runInfos]);

  const filteredTagKeys = useMemo(() => Utils.getVisibleTagKeyList(tagsList), [tagsList]);

  const [isDatasetDrawerOpen, setIsDatasetDrawerOpen] = useState<boolean>(false);
  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType>();
  
  // Drawer state
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const [selectedRun, setSelectedRun] = useState<{ runUuid: string; runName: string } | null>(null);

  const experimentIds = useMemo(() => experiments.map(({ experimentId }) => experimentId), [experiments]);

  // Check if we should use new GetLoggedModels API.
  // If true, logged (and registered) models will be fetched based on runs inputs/outputs.
  const isUsingGetLoggedModelsAPI = shouldUseGetLoggedModelsBatchAPI();

  // Conditionally use legacy hook for fetching all logged models in the experiment
  const loggedModelsV3ByRunUuidFromExperiment = useLoggedModelsForExperimentRunsTable({
    experimentIds,
    enabled: !isUsingGetLoggedModelsAPI,
  });

  // Conditionally use new hook for fetching logged models based on runs inputs/outputs
  const loggedModelsV3ByRunUuidFromRunInputsOutputs = useLoggedModelsForExperimentRunsTableV2({
    runData,
    enabled: isUsingGetLoggedModelsAPI,
  });

  // Select the appropriate logged models based on the feature flag
  const loggedModelsV3ByRunUuid = isUsingGetLoggedModelsAPI
    ? loggedModelsV3ByRunUuidFromRunInputsOutputs
    : loggedModelsV3ByRunUuidFromExperiment;

  // Use new, memoized version of the row creation function.
  // Internally disabled if the flag is not set.
  const visibleRuns = useExperimentRunRows({
    experiments,
    paramKeyList,
    metricKeyList,
    modelVersionsByRunUuid,
    runsExpanded,
    tagKeyList: filteredTagKeys,
    nestChildren: shouldNestChildrenAndFetchParents,
    referenceTime,
    runData,
    runUuidsMatchingFilter,
    runsPinned,
    runsHidden,
    groupBy: uiState.groupBy,
    groupsExpanded: uiState.groupsExpanded,
    runsHiddenMode: uiState.runsHiddenMode,
    runsVisibilityMap: uiState.runsVisibilityMap,
    useGroupedValuesInCharts: uiState.useGroupedValuesInCharts,
    searchFacetsState,
    loggedModelsV3ByRunUuid,
  });

  const [notificationsFn, notificationContainer] = useLegacyNotification();
  const showFetchedRunsNotifications = useFetchedRunsNotification(notificationsFn);
  const showControllerNotification = useControllerNotification(notificationsFn);

  const [tableAreaWidth, setTableAreaWidth] = useState(INITIAL_RUN_COLUMN_SIZE);

  const loadMoreRunsCallback = useCallback(() => {
    if (moreRunsAvailable && !isLoadingRuns) {
      // Don't do this if we're loading runs
      // to prevent too many requests from being
      // sent out
      loadMoreRuns().then((runs) => {
        // Display notification about freshly loaded runs
        showFetchedRunsNotifications(runs, runInfos);
      });
    }
  }, [moreRunsAvailable, isLoadingRuns, loadMoreRuns, runInfos, showFetchedRunsNotifications]);

  const datasetSelected = useCallback((dataset: RunDatasetWithTags, run: RunRowType) => {
    setSelectedDatasetWithRun({ datasetWithTags: dataset, runData: run });
    setIsDatasetDrawerOpen(true);
  }, []);

  // InteractiveController handlers
  const handleOpenController = useCallback((runUuid: string, runName: string) => {
    setSelectedRun({ runUuid, runName });
    setIsDrawerOpen(true);
  }, []);

   // Function to hide a run
   const handleHideRun = (runUuid: string) => {
    updateUIState((existingState: ExperimentPageUIState) => ({
      ...existingState,
      runsHidden: !existingState.runsHidden.includes(runUuid)
        ? [...existingState.runsHidden, runUuid]
        : existingState.runsHidden.filter((r) => r !== runUuid),
    }));
    // Optionally refresh the runs list
    refreshRuns?.();
  };

  // Function to close the drawer
  const handleCloseDrawer = useCallback(() => {
    setIsDrawerOpen(false);
    setSelectedRun(null);
  }, []);

  const isTabActive = useIsTabActive();
  const autoRefreshEnabled = uiState.autoRefreshEnabled && shouldEnableExperimentPageAutoRefresh() && isTabActive;
  const usingGroupedValuesInCharts = uiState.useGroupedValuesInCharts ?? true;

  const tableElement =
    requestError instanceof Error && !isLoadingRuns ? (
      <ExperimentViewRunsRequestError error={requestError} />
    ) : (
      <ExperimentViewRunsTable
        experiments={experiments}
        runsData={runsData}
        searchFacetsState={searchFacetsState}
        viewState={viewState}
        isLoading={isLoadingRuns}
        updateViewState={updateViewState}
        onAddColumnClicked={addColumnClicked}
        rowsData={visibleRuns}
        loadMoreRunsFunc={loadMoreRunsCallback}
        moreRunsAvailable={moreRunsAvailable}
        onOpenController={handleOpenController}
        onDatasetSelected={datasetSelected}
        expandRows={expandRows}
        uiState={uiState}
        compareRunsMode={compareRunsMode}
        showControllerNotification={showControllerNotification}
        isExperimentEnded={isExperimentEnded}
      />
    );

  // Generate a unique storage key based on the experiment IDs
  const configStorageKey = useMemo(
    () =>
      experiments
        .map((e) => e.experimentId)
        .sort()
        .join(','),
    [experiments],
  );

  const { resizableMaxWidth, ref } = useResizableMaxWidth(CHARTS_MIN_WIDTH);

  return (
    <CreateNewRunContextProvider visibleRuns={visibleRuns} refreshRuns={refreshRuns}>
      <RunsChartsSetHighlightContextProvider>
        <ExperimentViewRunsControls
          viewState={viewState}
          updateViewState={updateViewState}
          runsData={runsData}
          searchFacetsState={searchFacetsState}
          experimentId={experimentId}
          requestError={requestError}
          expandRows={expandRows}
          updateExpandRows={updateExpandRows}
          refreshRuns={refreshRuns}
          uiState={uiState}
          isLoading={isLoadingRuns}
          isComparingExperiments={isComparingExperiments}
        />
        <div
          ref={ref}
          css={{
            minHeight: 225, // This is the exact height for displaying a minimum five rows and table header
            height: '100%',
            position: 'relative',
            display: (compareRunsMode === 'LOGS' || compareRunsMode === 'IC_LOGS') ? 'block' : 'flex',
            width: (compareRunsMode === 'LOGS' || compareRunsMode === 'IC_LOGS') ? '100%' : 'auto',
          }}
        >
          {compareRunsMode === 'LOGS' ? (
            <div css={{ width: '100%', height: '100%' }}>
              <TerminalLogViewer 
                logs={isLoadingLogs ? ['Loading experiment logs...'] : logsError ? ['Error fetching experiment logs...'] : logs}
                emptyStateMessage="No experiment logs available yet..."
              />
            </div>
          ) : compareRunsMode === 'IC_LOGS' ? (
            <div css={{ width: '100%', height: '100%' }}>
              <TerminalLogViewer 
                logs={isLoadingICLogs ? ['Loading interactive control logs...'] : icLogsError ? ['Error fetching interactive control logs...'] : icLogs}
                emptyStateMessage="No interactive control logs available yet..."
              />
            </div>
          ) : isComparingRuns ? (
            <ExperimentViewRunsTableResizer
              onResize={setTableAreaWidth}
              runListHidden={runListHidden}
              width={tableAreaWidth}
              maxWidth={resizableMaxWidth}
            >
              {tableElement}
            </ExperimentViewRunsTableResizer>
          ) : (
            tableElement
          )}
          {compareRunsMode === 'CHART' && (
            <RunsCompare
              isLoading={isLoadingRuns}
              comparedRuns={visibleRuns}
              metricKeyList={runsData.metricKeyList}
              paramKeyList={runsData.paramKeyList}
              experimentTags={runsData.experimentTags}
              compareRunCharts={uiState.compareRunCharts}
              compareRunSections={uiState.compareRunSections}
              groupBy={usingGroupedValuesInCharts ? uiState.groupBy : null}
              autoRefreshEnabled={autoRefreshEnabled}
              hideEmptyCharts={uiState.hideEmptyCharts}
              globalLineChartConfig={uiState.globalLineChartConfig}
              chartsSearchFilter={uiState.chartsSearchFilter}
              storageKey={configStorageKey}
              minWidth={CHARTS_MIN_WIDTH}
            />
          )}
          {compareRunsMode === 'ARTIFACT' && (
            <EvaluationArtifactCompareView
              comparedRuns={visibleRuns}
              viewState={viewState}
              updateViewState={updateViewState}
              onDatasetSelected={datasetSelected}
              disabled={Boolean(uiState.groupBy)}
            />
          )}
          {notificationContainer}
          {selectedDatasetWithRun && (
            <ExperimentViewDatasetDrawer
              isOpen={isDatasetDrawerOpen}
              setIsOpen={setIsDatasetDrawerOpen}
              selectedDatasetWithRun={selectedDatasetWithRun}
              setSelectedDatasetWithRun={setSelectedDatasetWithRun}
            />
          )}
          {/* Interactive Controller Drawer */}
          <RightSlidingDrawer
            isOpen={isDrawerOpen && selectedRun !== null}
            onClose={handleCloseDrawer}
            width={700}
            showBackdrop
            closeOnBackdropClick
            closeOnEscape
            customHeader={
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                width: '100%',
                padding: '0 20px'
              }}>
                <div style={{ 
                  fontSize: '18px', 
                  fontWeight: '600', 
                  color: theme.colors.textPrimary
                }}>
                  Interactive Controller
                </div>
                <button
                  onClick={handleCloseDrawer}
                  style={{
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    padding: '8px',
                    borderRadius: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: theme.colors.textSecondary,
                    fontSize: '16px'
                  }}
                >
                  ×
                </button>
              </div>
            }
          >
            {selectedRun && (
              <InteractiveControllerComponent
                runUuid={selectedRun.runUuid}
                runName={selectedRun.runName}
                onClose={handleCloseDrawer}
                showControllerNotification={showControllerNotification}
                onHideRun={handleHideRun}
                refreshRuns={refreshRuns}
              />
            )}
          </RightSlidingDrawer>
        </div>
      </RunsChartsSetHighlightContextProvider>
    </CreateNewRunContextProvider>
  );
});
