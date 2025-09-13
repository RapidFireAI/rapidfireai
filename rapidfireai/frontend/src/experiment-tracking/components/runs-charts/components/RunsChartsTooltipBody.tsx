import React, { useState, useEffect } from 'react';
import { isNil } from 'lodash';
import { Button, CloseIcon, CopyIcon, Input, PinIcon, PinFillIcon, PlayIcon, Tooltip, StopIcon, VisibleIcon, Typography, TrashIcon, Checkbox } from '@databricks/design-system';
import { css, Theme } from '@emotion/react';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { useExperimentIds } from '../../experiment-page/hooks/useExperimentIds';
import { RunsChartsRunData, RunsChartsLineChartXAxisType } from './RunsCharts.common';
import {
  RunsChartsTooltipBodyProps,
  RunsChartsTooltipMode,
  containsMultipleRunsTooltipData,
} from '../hooks/useRunsChartsTooltip';
import {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartType,
  RunsChartsScatterCardConfig,
  RunsChartsContourCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsParallelCardConfig,
} from '../runs-charts.types';
import {
  type RunsCompareMultipleTracesTooltipData,
  type RunsMetricsSingleTraceTooltipData,
} from './RunsMetricsLinePlot';
import { RunsMultipleTracesTooltipBody } from './RunsMultipleTracesTooltipBody';
import { shouldEnableRelativeTimeDateAxis } from 'common/utils/FeatureUtils';
import { useDispatch } from 'react-redux';
import { DispatcherService } from 'experiment-tracking/sdk/DispatcherService';
import { deleteRunApi } from 'experiment-tracking/actions';
import { ConfirmActionModal, ConfirmActionType } from 'experiment-tracking/components/modals/ConfirmInteractiveControllerActionModal';

interface CloneModifyResponse {
  result: boolean;
  err_msg?: string;
  error?: string;
}

interface RunsChartsContextMenuContentDataType {
  runs: RunsChartsRunData[];
  onTogglePin?: (runUuid: string) => void;
  onHideRun?: (runUuid: string) => void;
}

type RunsChartContextMenuHoverDataType = RunsChartsCardConfig;

const createBarChartValuesBox = (cardConfig: RunsChartsBarCardConfig, activeRun: RunsChartsRunData) => {
  const { metricKey } = cardConfig;
  const metric = activeRun?.metrics[metricKey];

  if (!metric) {
    return null;
  }

  return (
    <div css={styles.value}>
      <strong>{metric.key}:</strong> {metric.value}
    </div>
  );
};

const createScatterChartValuesBox = (cardConfig: RunsChartsScatterCardConfig, activeRun: RunsChartsRunData) => {
  const { xaxis, yaxis } = cardConfig;
  const xKey = xaxis.key;
  const yKey = yaxis.key;

  const xValue = xaxis.type === 'METRIC' ? activeRun.metrics[xKey]?.value : activeRun.params[xKey]?.value;

  const yValue = yaxis.type === 'METRIC' ? activeRun.metrics[yKey]?.value : activeRun.params[yKey]?.value;

  return (
    <>
      {xValue && (
        <div css={styles.value}>
          <strong>X ({xKey}):</strong> {xValue}
        </div>
      )}
      {yValue && (
        <div css={styles.value}>
          <strong>Y ({yKey}):</strong> {yValue}
        </div>
      )}
    </>
  );
};

const createContourChartValuesBox = (cardConfig: RunsChartsContourCardConfig, activeRun: RunsChartsRunData) => {
  const { xaxis, yaxis, zaxis } = cardConfig;
  const xKey = xaxis.key;
  const yKey = yaxis.key;
  const zKey = zaxis.key;

  const xValue = xaxis.type === 'METRIC' ? activeRun.metrics[xKey]?.value : activeRun.params[xKey]?.value;

  const yValue = yaxis.type === 'METRIC' ? activeRun.metrics[yKey]?.value : activeRun.params[yKey]?.value;

  const zValue = zaxis.type === 'METRIC' ? activeRun.metrics[zKey]?.value : activeRun.params[zKey]?.value;

  return (
    <>
      {xValue && (
        <div css={styles.value}>
          <strong>X ({xKey}):</strong> {xValue}
        </div>
      )}
      {yValue && (
        <div css={styles.value}>
          <strong>Y ({yKey}):</strong> {yValue}
        </div>
      )}
      {zValue && (
        <div css={styles.value}>
          <strong>Z ({zKey}):</strong> {zValue}
        </div>
      )}
    </>
  );
};

const normalizeRelativeTimeChartTooltipValue = (value: string | number) => {
  if (typeof value === 'number') {
    return value;
  }
  return value.split(' ')[1] || '00:00:00';
};

const getTooltipXValue = (
  hoverData: RunsMetricsSingleTraceTooltipData | undefined,
  xAxisKey: RunsChartsLineChartXAxisType,
) => {
  if (xAxisKey === RunsChartsLineChartXAxisType.METRIC) {
    return hoverData?.xValue ?? '';
  }

  if (shouldEnableRelativeTimeDateAxis() && xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE) {
    return normalizeRelativeTimeChartTooltipValue(hoverData?.xValue ?? '');
  }

  // Default return for other cases
  return hoverData?.xValue;
};

const createLineChartValuesBox = (
  cardConfig: RunsChartsLineCardConfig,
  activeRun: RunsChartsRunData,
  hoverData?: RunsMetricsSingleTraceTooltipData,
) => {
  const { metricKey: metricKeyFromConfig, xAxisKey } = cardConfig;
  const metricKey = hoverData?.metricEntity?.key || metricKeyFromConfig;

  // If there's available value from x axis (step or time), extract entry from
  // metric history instead of latest metric.
  const metricValue = hoverData?.yValue ?? activeRun?.metrics[metricKey].value;

  if (isNil(metricValue)) {
    return null;
  }

  const xValue = getTooltipXValue(hoverData, xAxisKey);

  return (
    <>
      {hoverData && (
        <div css={styles.value}>
          <strong>{hoverData.label}:</strong> {xValue}
        </div>
      )}
      <div css={styles.value}>
        <strong>{metricKey}:</strong> {metricValue}
      </div>
    </>
  );
};

const createParallelChartValuesBox = (
  cardConfig: RunsChartsParallelCardConfig,
  activeRun: RunsChartsRunData,
  isHovering?: boolean,
) => {
  const { selectedParams, selectedMetrics } = cardConfig as RunsChartsParallelCardConfig;
  const paramsList = selectedParams.map((paramKey) => {
    const param = activeRun?.params[paramKey];
    if (param) {
      return (
        <div key={paramKey}>
          <strong>{param.key}:</strong> {param.value}
        </div>
      );
    }
    return true;
  });
  const metricsList = selectedMetrics.map((metricKey) => {
    const metric = activeRun?.metrics[metricKey];
    if (metric) {
      return (
        <div key={metricKey}>
          <strong>{metric.key}:</strong> {metric.value}
        </div>
      );
    }
    return true;
  });

  // show only first 3 params and primary metric if hovering, else show all
  if (isHovering) {
    return (
      <>
        {paramsList.slice(0, 3)}
        {(paramsList.length > 3 || metricsList.length > 1) && <div>...</div>}
        {metricsList[metricsList.length - 1]}
      </>
    );
  } else {
    return (
      <>
        {paramsList}
        {metricsList}
      </>
    );
  }
};

/**
 * Internal component that displays metrics/params - its final design
 * is a subject to change
 */
const ValuesBox = ({
  activeRun,
  cardConfig,
  isHovering,
  hoverData,
}: {
  activeRun: RunsChartsRunData;
  cardConfig: RunsChartsCardConfig;
  isHovering?: boolean;
  hoverData?: RunsMetricsSingleTraceTooltipData;
}) => {
  if (cardConfig.type === RunsChartType.BAR) {
    return createBarChartValuesBox(cardConfig as RunsChartsBarCardConfig, activeRun);
  }

  if (cardConfig.type === RunsChartType.SCATTER) {
    return createScatterChartValuesBox(cardConfig as RunsChartsScatterCardConfig, activeRun);
  }

  if (cardConfig.type === RunsChartType.CONTOUR) {
    return createContourChartValuesBox(cardConfig as RunsChartsContourCardConfig, activeRun);
  }

  if (cardConfig.type === RunsChartType.LINE) {
    return createLineChartValuesBox(cardConfig as RunsChartsLineCardConfig, activeRun, hoverData);
  }

  if (cardConfig.type === RunsChartType.PARALLEL) {
    return createParallelChartValuesBox(cardConfig as RunsChartsParallelCardConfig, activeRun, isHovering);
  }

  return null;
};

export const RunsChartsTooltipBody = ({
  closeContextMenu,
  contextData,
  hoverData,
  chartData,
  runUuid,
  isHovering,
  mode,
  showControllerNotification,
  refreshRuns, 
}: RunsChartsTooltipBodyProps<
  RunsChartsContextMenuContentDataType,
  RunsChartContextMenuHoverDataType,
  RunsMetricsSingleTraceTooltipData | RunsCompareMultipleTracesTooltipData
>) => {
  const dispatch = useDispatch();
  const { runs, onTogglePin, onHideRun } = contextData;
  const [isEditable, setIsEditable] = useState(false);
  const [runStatus, setRunStatus] = useState('');
  const [textareaContent, setTextareaContent] = useState('{}');
  const [originalTextareaContent, setOriginalTextareaContent] = useState('');
  const [warmStart, setWarmStart] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [currentAction, setCurrentAction] = useState<ConfirmActionType | null>(null);
  const [chunkNumber, setChunkNumber] = useState(0);

  const [experimentId] = useExperimentIds();
  const activeRun = runs?.find((run) => run.uuid === runUuid);

  useEffect(() => {
    const fetchRunConfiguration = async () => {
      try {
        const run_id = Number(runs.find((run) => run.uuid === runUuid)?.displayName);
        if (!run_id) {
          console.error('Run not found');
          return;
        }
        
        // TODO: update this so that we dispatch and update Redux State
        // await dispatch(getDispatcherRunApi({ run_id: run_id }));
        const response = await DispatcherService.getRunUi({ run_id: run_id });
        // Type guard to check if response has a 'config' property
        if (response && typeof response === 'object' && 'config' in response) {
          const config = response.config;
          setTextareaContent(JSON.stringify(config, null, 2));
          setOriginalTextareaContent(JSON.stringify(config, null, 2));
          if ('status' in response && typeof response.status === 'string') {
            setRunStatus(response.status);
          }
          if ('num_chunks_visited' in response && typeof response.num_chunks_visited === 'number') {
            setChunkNumber(response.num_chunks_visited);
          }
        } else {
          console.error('Response does not contain a config object');

          // TODO: Give notification to user that the run configuration could not be fetched
        }
          
      } catch (error) {
        console.error('Error fetching run configuration:', error);
      }
    };
  
    if (runUuid) {
      fetchRunConfiguration();
    }
  }, [dispatch, runUuid, runs]);

  const handleActionClick = (action: ConfirmActionType) => {
    setCurrentAction(action);
    setIsModalOpen(true);
  };

  const handleConfirm = async () => {
    if (!currentAction || !activeRun) return;

    const run_id = Number(activeRun.displayName);
    try {
      let response;
      switch (currentAction) {
        case 'stop':
          response = await DispatcherService.stopRun({ run_id });
          break;
        case 'resume':
          response = await DispatcherService.resumeRun({ run_id });
          break;
        case 'delete':
          response = await DispatcherService.deleteRun({ run_id });
          if (response && typeof response === 'object' && 'error' in response) {
           break;
          } else {
            await dispatch(deleteRunApi(runUuid));
          }
          break;
      }

      const error_message = response && typeof response === 'object' && 'error' in response ? response.error : null;

      if (error_message && showControllerNotification) {
        // throw new Error(error_message);
        showControllerNotification(currentAction, 'error');
      }

      if (showControllerNotification) {
        showControllerNotification(currentAction, 'success');
        if (typeof refreshRuns === 'function') {
          refreshRuns();
         }
      }
      closeContextMenu();
    } catch (error) {
      console.error(`Error ${currentAction}ing run:`, error);
      if (showControllerNotification) {
        showControllerNotification(currentAction, 'error');
      }
    } finally {
      setIsModalOpen(false);
      setCurrentAction(null);
    }
  };
  
  const handleResumeRun = async () => {
    try {
      const run_id = Number(runs.find((run) => run.uuid === runUuid)?.displayName);
      const response = await DispatcherService.resumeRun({ run_id: run_id });
      const error_message = response && typeof response === 'object' && 'error' in response ? response.error : null;
      if (error_message) {
        showControllerNotification?.('resume', 'error');
        return;
      }
      showControllerNotification?.('resume', 'success');
      refreshRuns?.();
    } catch (error) {
      showControllerNotification?.('resume', 'error');
    }
  };

  const handleStopRun = async () => {
    try {
      const run_id = Number(runs.find((run) => run.uuid === runUuid)?.displayName);
      const response = await DispatcherService.stopRun({ run_id: run_id });
      const error_message = response && typeof response === 'object' && 'error' in response ? response.error : null;
      if (error_message) {
        showControllerNotification?.('stop', 'error');
        return;
      }
      showControllerNotification?.('stop', 'success');
      refreshRuns?.();
    } catch (error) {
      showControllerNotification?.('stop', 'error');
    }
  };

  const handleDeleteRun = async () => {
    try {
      const run_id = Number(runs.find((run) => run.uuid === runUuid)?.displayName);
      const response = await DispatcherService.deleteRun({ run_id: run_id });
      const error_message = response && typeof response === 'object' && 'error' in response ? response.error : null;
      if (error_message) {
        showControllerNotification?.('delete', 'error');
        return;
      }

      // We call onHideRun to update the UI, backend will handle deleting the run
      if (onHideRun && typeof onHideRun === 'function') {
        onHideRun(runUuid);
      }
      showControllerNotification?.('delete', 'success');
      refreshRuns?.();
    } catch (error) {
      console.error('Error deleting run:', error);
      showControllerNotification?.('delete', 'error');
    }
  };

  const handleCloneRun = async () => {
    setIsEditable(false);
    
    try {
      // We've set up the ml_config param to expect a string
      // TODO: Update dispatcher to accept a JSON object for easier handling
      const updatedConfig = textareaContent;
      const run_id = Number(runs.find((run) => run.uuid === runUuid)?.displayName);
      // Parse the textareaContent as JSON to convert from string to object
      let parsedConfig;
      try {
        parsedConfig = JSON.parse(textareaContent);
      } catch (parseError) {
        console.error('Error parsing config JSON:', parseError);
        showControllerNotification?.('clone_modify', 'error', 'Invalid JSON configuration');
        return;
      }

      const response = await DispatcherService.cloneModifyRun({ 
        run_id: run_id,
        config: parsedConfig,
        warm_start: warmStart,  // Send boolean directly instead of converting to string
      });

      // Handle both object and stringified JSON responses
      let errorMessage = null;
      if (typeof response === 'string') {
        try {
          const parsedResponse = JSON.parse(response) as CloneModifyResponse;
          if (parsedResponse.result === false && parsedResponse.err_msg) {
            errorMessage = parsedResponse.err_msg;
          }
        } catch (e) {
          // If parsing fails, treat the whole string as an error message
          errorMessage = response;
        }
      } else if (response && typeof response === 'object') {
        const typedResponse = response as CloneModifyResponse;
        if (typedResponse.result === false && typedResponse.err_msg) {
          errorMessage = typedResponse.err_msg;
        } else if (typedResponse.error) {
          errorMessage = typedResponse.error;
        }
      }
      
      if (errorMessage) {
        console.error('Error clone-modifying run:', errorMessage);
        if (typeof showControllerNotification === 'function') {
          showControllerNotification('clone_modify', 'error', errorMessage);
        }
        return;
      }

      if (typeof showControllerNotification === 'function') {
        showControllerNotification('clone_modify', 'success');
      }
    } catch (error) {
      console.error('Error clone-modifying run:', error);
      if (typeof showControllerNotification === 'function') {
        showControllerNotification('clone_modify', 'error');
      }
    }
  };

  if (
    containsMultipleRunsTooltipData(hoverData) &&
    mode === RunsChartsTooltipMode.MultipleTracesWithScanline &&
    isHovering
  ) {
    return <RunsMultipleTracesTooltipBody hoverData={hoverData} />;
  }

  const singleTraceHoverData = containsMultipleRunsTooltipData(hoverData) ? hoverData.hoveredDataPoint : hoverData;

  if (!activeRun) {
    return null;
  }

  const runName = activeRun.displayName || activeRun.uuid;
  const metricSuffix = singleTraceHoverData?.metricEntity ? ` (${singleTraceHoverData.metricEntity.key})` : '';

  return (
    <div css={styles.container}>
        <ConfirmActionModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onConfirm={handleConfirm}
          actionType={currentAction || 'stop'} // Provide a default to satisfy TypeScript
          runName={activeRun.displayName}
        />
      {!isHovering && (
        <div css={styles.leftPanel}>
          <Tooltip title="Resume run" placement="left">
            <Button
              componentId="resume-run-button"
              size="middle"
              icon={<PlayIcon color='success' />}
              css={styles.iconButton}
              onClick={handleResumeRun}
              // onClick={() => handleActionClick('resume')}
            />
          </Tooltip>
          <Tooltip title="Stop run" placement="left">
            <Button
              componentId="stop-run-button"
              size="middle"
              icon={<StopIcon color='danger' />}
              css={styles.iconButton}
              onClick={handleStopRun}
              // onClick={() => handleActionClick('stop')}
            />
          </Tooltip>
          <Tooltip title="Clone run" placement="left">
            <Button
              componentId="clone-run-button"
              size="middle"
              icon={<CopyIcon />}
              onClick={() => {
                setIsEditable(true);
                setWarmStart(false);
              }}
            />
          </Tooltip>
          <Tooltip title="Delete run" placement="left">
            <Button
              componentId="delete-run-button"
              size="middle"
              icon={<TrashIcon />}
              onClick={handleDeleteRun}
              // onClick={() => handleActionClick('delete')}
            />
          </Tooltip>
        </div>
      )}
      <div css={styles.contentWrapper}>
        <div css={styles.header}>
          <div css={styles.colorPill} style={{ backgroundColor: activeRun.color }} />
          {activeRun.groupParentInfo ? (
            <Typography.Text>{runName + metricSuffix}</Typography.Text>
          ) : (
            <Link
              to={Routes.getRunPageRoute(experimentId, runUuid)}
              target="_blank"
              css={styles.runLink}
              onClick={closeContextMenu}
            >
              {runName + metricSuffix}
            </Link>
          )}
        </div>
        <div css={styles.valueBoxWrapper}> 
          <ValuesBox
            isHovering={isHovering}
            activeRun={activeRun}
            cardConfig={chartData}
            hoverData={singleTraceHoverData}
          />
          <Typography.Text><strong>Status:</strong> {runStatus}</Typography.Text>
          <Typography.Text><strong>Chunk Number:</strong> {chunkNumber}</Typography.Text>
        </div>

        {!isHovering && (
          <> 
            <Tooltip
              title={isEditable ? "You can now edit the config" : "Click 'Clone' to edit this config"}
              placement="top"
            >
              <Input.TextArea 
                value={textareaContent}
                onChange={(e) => setTextareaContent(e.target.value)}
                readOnly={!isEditable}
                autoSize={false}  // Disable autoSize to enforce our height limits
                rows={8}         // Set initial number of rows
              />
            </Tooltip>
          </>
        )}
        {!isHovering && isEditable && (
          <>
            <Checkbox
              isChecked={warmStart}
              onChange={() => setWarmStart(!warmStart)}
            >
              Warm start?
            </Checkbox>  
            <Button
              componentId="save-run-button"
              size="middle"
              type='primary'
              onClick={() => {
                setIsEditable(false);
                handleCloneRun();
              }}
            >
              Submit
            </Button>
            <Button
              componentId="cancel-run-button"
              size="middle"
              onClick={() => {
                setIsEditable(false);
                setTextareaContent(originalTextareaContent);
                setWarmStart(false);
              }}
              >
                Cancel
            </Button>
          </>
        )}
      </div>
      <div css={styles.actionsWrapper}>
        {!isHovering && (
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_runscomparetooltipbody.tsx_259"
            size="small"
            onClick={closeContextMenu}
            icon={<CloseIcon />}
          />
        )}
        {activeRun.pinnable && onTogglePin && (
          <Tooltip
            title={
              activeRun.pinned ? (
                <FormattedMessage
                  defaultMessage="Unpin run"
                  description="A tooltip for the pin icon button in the runs table next to the pinned run"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Click to pin the run"
                  description="A tooltip for the pin icon button in the runs chart tooltip next to the not pinned run"
                />
              )
            }
            placement="bottom"
          >
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_runscomparetooltipbody.tsx_282"
              size="small"
              onClick={() => {
                onTogglePin(runUuid);
                closeContextMenu();
              }}
              icon={activeRun.pinned ? <PinFillIcon /> : <PinIcon />}
            />
          </Tooltip>
        )}
        {onHideRun && (
          <Tooltip
            title={
              <FormattedMessage
                defaultMessage="Click to hide the run"
                description='A tooltip for the "hide" icon button in the runs chart tooltip'
              />
            }
            placement="bottom"
          >
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_runscomparetooltipbody.tsx_302"
              data-testid="experiment-view-compare-runs-tooltip-visibility-button"
              size="small"
              onClick={() => {
                onHideRun(runUuid);
                closeContextMenu();
              }}
              icon={<VisibleIcon />}
            />
          </Tooltip>
        )}
      </div>
    </div>
  );
};


const styles = {
  container: {
    display: 'flex',
    maxHeight: '100%',    // Change to 100% to fill parent
    overflow: 'hidden',   // Change to hidden
    width: '100%',        // Add width 100%
  },
  leftPanel: (theme: Theme) => ({
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 8,
    marginRight: 16,
    padding: 8,
    backgroundColor: theme.colors.backgroundSecondary, // Use a theme color for consistency
    borderRadius: theme.general.borderRadiusBase,
  }),
  runLink: (theme: Theme) => ({
    color: theme.colors.primary,
    '&:hover': {},
  }),
  actionsWrapper: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 8,
    alignItems: 'flex-start',
    marginLeft: 16,
  },
  header: {
    display: 'flex',
    gap: 8,
    alignItems: 'center',
  },
  value: {
    maxWidth: 450,
    whiteSpace: 'nowrap' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  valueBoxWrapper: {
    display: 'flex',
    // flexDirection: 'column' as const,
    gap: 8,
  },
  contentWrapper: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 12,
    flex: 1,
    overflow: 'auto',     // Add overflow auto here
    maxHeight: '100%',    // Add max height
  },
  colorPill: { width: 12, height: 12, borderRadius: '100%' },
  iconButton: css`
  &.ant-btn {
    border: 1px solid rgba(0, 0, 0, 0.1);
    background: rgba(255, 255, 255, 0.1);
    box-shadow: none;
    
    &:hover, &:focus {
      background: rgba(255, 255, 255, 0.2);
    }
  }
`,
};
