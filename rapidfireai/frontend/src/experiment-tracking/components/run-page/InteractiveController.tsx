import React, { useState, useEffect } from 'react';
import { Button, CopyIcon, Input, PlayIcon, Tooltip, StopIcon, TrashIcon, Checkbox, Typography } from '@databricks/design-system';
import { css, Theme } from '@emotion/react';
import { useDispatch } from 'react-redux';
import { DispatcherService } from '../../../experiment-tracking/sdk/DispatcherService';

interface CloneModifyResponse {
  result: boolean;
  err_msg?: string;
  error?: string;
}

export interface InteractiveControllerComponentProps {
  runUuid: string;
  runName: string;
  onClose?: () => void;
  onHideRun?: (runUuid: string) => void;
  showControllerNotification?: (action: 'resume' | 'stop' | 'delete' | 'clone_modify', status: 'success' | 'error', message?: string) => void;
  refreshRuns?: () => void;
}

const InteractiveControllerComponent: React.FC<InteractiveControllerComponentProps> = ({
  runUuid,
  runName,
  onClose,
  onHideRun,
  showControllerNotification,
  refreshRuns,
}) => {
  const dispatch = useDispatch();
  const [isEditable, setIsEditable] = useState(false);
  const [textareaContent, setTextareaContent] = useState('{}');
  const [originalTextareaContent, setOriginalTextareaContent] = useState('');
  const [warmStart, setWarmStart] = useState(false);
  const [runStatus, setRunStatus] = useState('');
  const [chunkNumber, setChunkNumber] = useState(0);
  const [isRunAvailable, setIsRunAvailable] = useState(true);

  useEffect(() => {
    const fetchRunConfiguration = async () => {
      try {
        // Try numeric run_id first (fit mode), fallback to runUuid string (evals mode)
        const numericId = Number(runName);
        const run_id = !isNaN(numericId) && numericId > 0 ? numericId : runUuid;

        if (!run_id) {
          console.error('Run not found');
          setIsRunAvailable(false);
          return;
        }

        const response = await DispatcherService.getRunUi({ run_id: run_id });
        if (response && typeof response === 'object' && 'config' in response) {
          setIsRunAvailable(true);
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
          setIsRunAvailable(false);
        }
      } catch (error) {
        console.error('Error fetching run configuration:', error);
      }
    };

    fetchRunConfiguration();
  }, [runName, runUuid]);

  // Helper to get run_id - use numeric runName for fit mode, fallback to runUuid for evals mode
  const getRunId = (): number | string => {
    const numericId = Number(runName);
    return !isNaN(numericId) && numericId > 0 ? numericId : runUuid;
  };

  const handleResumeRun = async () => {
    try {
      const run_id = getRunId();
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
      const run_id = getRunId();
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
      const run_id = getRunId();
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
      onClose?.();
    } catch (error) {
      showControllerNotification?.('delete', 'error');
    }
  };

  const handleCloneRun = async () => {
    try {
      const run_id = getRunId();

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
        warm_start: warmStart,
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
        showControllerNotification?.('clone_modify', 'error', errorMessage);
        console.error('Clone/Modify error:', errorMessage);
        return;
      }
      
      showControllerNotification?.('clone_modify', 'success');
      refreshRuns?.();
      setIsEditable(false);
    } catch (error) {
      showControllerNotification?.('clone_modify', 'error');
      console.error('Clone/Modify error:', error);
    }
  };

  return (
    <div css={styles.container}>
      <div css={styles.leftPanel}>
        <div css={styles.actionsHeader}>Actions</div>
        <Tooltip content="Resume run" componentId="resume-run-tooltip">
          <Button
            componentId="resume-run-button"
            size="middle"
            icon={<PlayIcon color='success' />}
            css={styles.iconButton}
            onClick={handleResumeRun}
            disabled={!isRunAvailable || runStatus?.toLowerCase() === 'completed'}
          >
            Resume
          </Button>
        </Tooltip>
        <Tooltip content="Stop run" componentId="stop-run-tooltip">
          <Button
            componentId="stop-run-button"
            size="middle"
            icon={<StopIcon color='danger' />}
            css={styles.iconButton}
            onClick={handleStopRun}
            disabled={!isRunAvailable || runStatus?.toLowerCase() === 'completed'}
          >
            Stop
          </Button>
        </Tooltip>
        <Tooltip content="Clone run" componentId="clone-run-tooltip">
          <Button
            componentId="clone-run-button"
            size="middle"
            icon={<CopyIcon color='ai' />}
            onClick={() => {
              setIsEditable(true);
              setWarmStart(false);
            }}
            disabled={!isRunAvailable || runStatus?.toLowerCase() === 'completed'}
          >
            Clone
          </Button>
        </Tooltip>
        <Tooltip content="Delete run" componentId="delete-run-tooltip">
          <Button
            componentId="delete-run-button"
            size="middle"
            icon={<TrashIcon color='danger' />}
            onClick={handleDeleteRun}
            disabled={!isRunAvailable || runStatus?.toLowerCase() === 'completed'}
          >
            Delete
          </Button>
        </Tooltip>
      </div>

      <div css={styles.contentWrapper}>
          <div css={styles.header}>
            <Typography.Title level={4} css={styles.headerTitle}>Run Details</Typography.Title>
            <div css={styles.headerSeparator}></div>
            <div css={styles.headerDetails}>
              <div css={styles.detailRow}>
                <span css={styles.detailLabel}>Run Name:</span>
                <span css={styles.detailValue}>{runName}</span>
              </div>
              <div css={styles.detailRow}>
                <span css={styles.detailLabel}>Run Status:</span>
                <span css={styles.detailValue}>{runStatus}</span>
              </div>
              <div css={styles.detailRow}>
                <span css={styles.detailLabel}>Chunk Number:</span>
                <span css={styles.detailValue}>{chunkNumber}</span>
              </div>
            </div>
          </div>

          <div css={styles.sectionSeparator}></div>

          <div css={styles.configSection}>
            <Typography.Title level={5} css={styles.configHeader}>Run Configuration</Typography.Title>
            <Tooltip content={isEditable ? "You can now edit the config" : "Click 'Clone' to edit this config"} componentId="edit-run-tooltip">
              <div css={styles.textAreaWrapper}>
                <Input.TextArea 
                  componentId="config-textarea"
                  value={textareaContent}
                  onChange={(e) => setTextareaContent(e.target.value)}
                  readOnly={!isEditable}
                  css={styles.textArea}
                  autoSize={false}
                />
              </div>
            </Tooltip>
        </div>
        
        {isEditable && (
        <>
            <Checkbox
            componentId="warm-start-checkbox"
            isChecked={warmStart}
            onChange={() => setWarmStart(!warmStart)}
          >
            Warm start?
          </Checkbox>
          
          <div css={{ display: 'flex', gap: 8 }}>
            <Button
              componentId="save-run-button"
              size="middle"
              type='primary'
              onClick={handleCloneRun}
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
          </div>
        </>
        )}
      </div>
    </div>
  );
};

const styles = {
    container: {
      display: 'flex',
      padding: '20px',
      gap: '20px',
      height: '100%',
      minHeight: '100%',
    },
    leftPanel: (theme: Theme) => ({
      display: 'flex',
      flexDirection: 'column' as const,
      gap: 16,
      padding: 12,
      backgroundColor: theme.colors.backgroundSecondary,
      borderRadius: theme.general.borderRadiusBase,
      flexShrink: 0,
      alignItems: 'stretch',
      minWidth: 120,
    }),
    actionsHeader: (theme: Theme) => ({
      fontSize: '14px',
      fontWeight: 600,
      color: theme.colors.textSecondary,
      marginBottom: 8,
      textTransform: 'uppercase' as const,
      letterSpacing: '0.5px',
      textAlign: 'center' as const,
    }),
    contentWrapper: {
      display: 'flex',
      flexDirection: 'column' as const,
      gap: 2,
      flex: 1,
      minHeight: 0,
      overflow: 'hidden',
    },
    header: {
        display: 'flex',
        flexDirection: 'column' as const,
        gap: 4,
        marginBottom: 4,
        paddingBottom: 4,
        flexShrink: 0,
    },
    headerTitle: {
        fontSize: '20px',
        fontWeight: 600,
        margin: 0,
    },
    headerSeparator: (theme: Theme) => ({
        height: '1px',
        backgroundColor: theme.colors.border,
        width: '100%',
        margin: '4px 0',
    }),
    headerDetails: {
        display: 'flex',
        flexDirection: 'column' as const,
        gap: 4,
    },
    detailRow: {
        display: 'flex',
        alignItems: 'center',
        gap: 8,
    },
    detailLabel: (theme: Theme) => ({
        fontSize: '14px',
        color: theme.colors.textSecondary,
        fontWeight: 500,
        minWidth: '100px',
        flexShrink: 0,
    }),
    detailValue: (theme: Theme) => ({
        fontSize: '12px',
        fontFamily: "'Monaco', 'Menlo', 'Ubuntu Mono', monospace",
        lineHeight: 1.4,
        color: theme.colors.textPrimary,
    }),
    sectionSeparator: (theme: Theme) => ({
        height: '1px',
        backgroundColor: theme.colors.border,
        width: '100%',
        margin: '4px 0',
    }),
    configSection: {
        display: 'flex',
        flexDirection: 'column' as const,
        gap: 12,
        flex: 1,
        minHeight: 0,
    },
    configHeader: {
        fontSize: '16px',
        fontWeight: 600,
        margin: 0,
        color: '#666',
    },
    title: {
      fontSize: '16px',
      fontWeight: 600,
    },
    textAreaWrapper: {
      flex: 1,
      minHeight: 0,
      display: 'flex',
      flexDirection: 'column' as const,
    },
    textArea: css`
      height: 100% !important;
      min-height: 0 !important;
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 12px;
      line-height: 1.4;
      resize: none;
    `,
    iconButton: css`
      &.ant-btn {
        border: 1px solid rgba(0, 0, 0, 0.1);
        background: rgba(255, 255, 255, 0.1);
        box-shadow: none;
        width: 100% !important;
        min-width: 0 !important;
        justify-content: flex-start;
        text-align: left;
        
        .ant-btn-icon {
          margin-right: 8px;
        }
        
        &:hover, &:focus {
          background: rgba(255, 255, 255, 0.2);
        }
        
        &:disabled {
          width: 100% !important;
          min-width: 0 !important;
          opacity: 0.6;
        }
      }
    `,
  };

export default InteractiveControllerComponent;
