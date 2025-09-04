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
  shouldEnableRunsTableRunNameColumnResize,
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
import TerminalLogViewer from '../../../TerminalLogViewer';
import { DispatcherService } from 'experiment-tracking/sdk/DispatcherService';
import { useControllerNotification } from '../../hooks/useInteractiveControllerNotification';
import { set } from 'lodash';
import RightSlidingDrawer from 'rapidfire-ui/components/RightSlidingDrawer';
import InteractiveControllerComponent from '../../../run-page/InteractiveController';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';
import { RUNS_VISIBILITY_MODE } from '../../models/ExperimentPageUIState';

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
  requestError: ErrorWrapper | null;
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

export const INITIAL_RUN_COLUMN_SIZE = 295;

const getTableLayoutStyles = (isComparingRuns = false, runListHidden = false) =>
  shouldEnableRunsTableRunNameColumnResize()
    ? {
        display: 'flex',
      }
    : {
        display: 'grid',
        gridTemplateColumns: isComparingRuns
          ? runListHidden
            ? '10px 1fr'
            : `${INITIAL_RUN_COLUMN_SIZE}px 1fr`
          : '1fr',
      };

function cleanLogStrings(logs: string[]): string[] {
  return logs.map(log => {
    // Use a regular expression to match and remove "experiment.py:xxx", "ml_controller.py:xxx", and " - INFO"
    return log.replace(/(experiment\.py|ml_controller\.py):\d+ -|- INFO/g, '');
  });
}

export const ExperimentViewRuns = React.memo((props: ExperimentViewRunsProps) => {
  const [compareRunsMode] = useExperimentPageViewMode();
  const [logs, setLogs] = useState<string[]>(['No experiment logs available yet...']);
  const [icLogs, setICLogs] = useState<string[]>(['No interactive control logs available yet...']);
  const { theme } = useDesignSystemTheme();
  const updateUIState = useUpdateExperimentViewUIState();
  
  // Drawer state
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedRun, setSelectedRun] = useState<{ runUuid: string; runName: string } | null>(null);
  
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
  } = runsData;

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
      })),
    [datasetsList, metricsList, paramsList, runInfos, tagsList],
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

  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType>();

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
  });

  // Fetch logs when in 'LOGS' mode
  useEffect(() => {
    if (compareRunsMode === 'LOGS') {
      const fetchLogs = async () => {
        try {
          const fetchedLogs = await DispatcherService.getLogs({ experiment_name: experiments[0].name });
          if (fetchedLogs && Array.isArray(fetchedLogs) && (fetchedLogs.length === 0 || fetchedLogs[0] === '')) {
            setLogs(['No experiment logs available yet...']);
            return;
          }

          const filteredLogs = cleanLogStrings(fetchedLogs as string[]);
          setLogs(filteredLogs as string[]);
        } catch (error) {
          console.error('Failed to fetch logs:', error);
          // Optionally, you can set an error state here and display it to the user
        }
      };

      if (experiments.length > 1) {
        setLogs(['Please select a single experiment to view experiment logs.']);
        return;
      }

      fetchLogs();
    }

    if (compareRunsMode === 'IC_LOGS') {
      const fetchICLogs = async () => {
        try {
          const fetchedICLogs = await DispatcherService.getICLogs({ experiment_name: experiments[0].name });
          if (fetchedICLogs && Array.isArray(fetchedICLogs) && (fetchedICLogs.length === 0 || fetchedICLogs[0] === '')) {
            setICLogs(['No interactive control logs available yet...']);
            return;
          }
          const filteredICLogs = cleanLogStrings(fetchedICLogs as string[]);
          setICLogs(filteredICLogs as string[]);
        } catch (error) {
          console.error('Failed to fetch logs:', error);
          // Optionally, you can set an error state here and display it to the user
        }
      };

      if (experiments.length > 1) {
        setICLogs(['Please select a single experiment to view interactive control logs.']);
        return;
      }

      fetchICLogs();
    }
  }, [compareRunsMode]);

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
    setIsDrawerOpen(true);
  }, []);

  const isTabActive = useIsTabActive();
  const autoRefreshEnabled = uiState.autoRefreshEnabled && shouldEnableExperimentPageAutoRefresh() && isTabActive;

  // Function to open the drawer with a specific run
  const handleOpenController = (runUuid: string, runName: string) => {
    setSelectedRun({ runUuid, runName });
    setIsDrawerOpen(true);
  };

  // Function to close the drawer
  const handleCloseDrawer = () => {
    setIsDrawerOpen(false);
    setSelectedRun(null);
  };

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

  const tableElement = (
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
      onDatasetSelected={datasetSelected}
      expandRows={expandRows}
      uiState={uiState}
      compareRunsMode={compareRunsMode}
      showControllerNotification={showControllerNotification}
      onOpenController={handleOpenController}
    />
  );

  return (
    <CreateNewRunContextProvider visibleRuns={visibleRuns} refreshRuns={refreshRuns}>
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
      />
      <div
        css={[
          {
            minHeight: 225, // This is the exact height for displaying a minimum five rows and table header
            height: '100%',
            position: 'relative',
          },
          getTableLayoutStyles(isComparingRuns, runListHidden),
        ]}
      >
        {isComparingRuns && shouldEnableRunsTableRunNameColumnResize() ? (
          <ExperimentViewRunsTableResizer
            onResize={setTableAreaWidth}
            runListHidden={runListHidden}
            width={tableAreaWidth}
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
            groupBy={uiState.groupBy}
            autoRefreshEnabled={autoRefreshEnabled}
            showControllerNotification={showControllerNotification}
            refreshRuns={refreshRuns}
          />
        )}
        {/* {compareRunsMode === 'ARTIFACT' && (
          <EvaluationArtifactCompareView
            comparedRuns={visibleRuns}
            viewState={viewState}
            updateViewState={updateViewState}
            onDatasetSelected={datasetSelected}
            disabled={Boolean(uiState.groupBy)}
          />
        )} */}
        {compareRunsMode === 'LOGS' && (
          <TerminalLogViewer logs={logs} />
        )}
         {compareRunsMode === 'IC_LOGS' && (
          <TerminalLogViewer logs={icLogs} />
        )}
        {notificationContainer}
        {selectedDatasetWithRun && (
          <ExperimentViewDatasetDrawer
            isOpen={isDrawerOpen}
            setIsOpen={setIsDrawerOpen}
            selectedDatasetWithRun={selectedDatasetWithRun}
            setSelectedDatasetWithRun={setSelectedDatasetWithRun}
          />
        )}
      </div>
      
      {/* Interactive Controller Drawer */}
      <RightSlidingDrawer
        isOpen={isDrawerOpen}
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
                transition: 'all 0.2s ease'
              }}
              aria-label="Close drawer"
            >
              ✕
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
    </CreateNewRunContextProvider>
  );
});
