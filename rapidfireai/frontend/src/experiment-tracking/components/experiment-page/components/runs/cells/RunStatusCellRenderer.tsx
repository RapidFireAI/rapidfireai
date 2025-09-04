import React, { useEffect, useState } from 'react';
import { ICellRendererParams } from '@ag-grid-community/core';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { RunRowType } from '../../../utils/experimentPage.row-types';
import { DispatcherService } from 'experiment-tracking/sdk/DispatcherService';

export const RunStatusCellRenderer: React.FC<ICellRendererParams> = ({ data }) => {
  const { theme } = useDesignSystemTheme();
  const [runStatus, setRunStatus] = useState<string>('UNKNOWN');
  const [isLoading, setIsLoading] = useState(true);
  
  // If this is a group row, don't render anything
  if ((data as RunRowType).groupParentInfo) {
    return null;
  }

  const runName = (data as RunRowType).runName;
  const runUuid = (data as RunRowType).runUuid;

  useEffect(() => {
    const fetchRunStatus = async () => {
      if (!runName) {
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        const run_id = Number(runName);
        if (!run_id) {
          setRunStatus('UNKNOWN');
          setIsLoading(false);
          return;
        }
        
        const response = await DispatcherService.getRunUi({ run_id: run_id });
        if (response && typeof response === 'object' && 'status' in response && typeof response.status === 'string') {
          setRunStatus(response.status);
        } else {
          setRunStatus('UNKNOWN');
        }
      } catch (error) {
        console.error('Error fetching run status:', error);
        setRunStatus('ERROR');
      } finally {
        setIsLoading(false);
      }
    };

    fetchRunStatus();
  }, [runName, runUuid]);

  // Show loading state
  if (isLoading) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '4px 8px',
          borderRadius: '4px',
          backgroundColor: theme.colors.backgroundSecondary,
          border: `1px solid ${theme.colors.border}`,
          minWidth: '60px',
          maxWidth: '100px'
        }}
      >
        <Typography.Text
          size="sm"
          style={{
            color: theme.colors.textSecondary,
            fontWeight: 500,
            textAlign: 'center',
            lineHeight: 1
          }}
        >
          ...
        </Typography.Text>
      </div>
    );
  }
  
  // Define status colors and labels
  const getStatusConfig = (status: string) => {
    switch (status.toUpperCase()) {
      case 'RUNNING':
        return {
          color: 'info',
          label: 'Running',
          backgroundColor: theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100
        };
      case 'ACTIVE':
        return {
          color: 'info',
          label: 'Running',
          backgroundColor: theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100
        };
      case 'FINISHED':
        return {
          color: 'success',
          label: 'Finished',
          backgroundColor: theme.isDarkMode ? theme.colors.green800 : theme.colors.green100
        };
      case 'FAILED':
        return {
          color: 'error',
          label: 'Failed',
          backgroundColor: theme.isDarkMode ? theme.colors.red800 : theme.colors.red100
        };
      case 'SCHEDULED':
        return {
          color: 'info',
          label: 'Scheduled',
          backgroundColor: theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100
        };
      case 'KILLED':
        return {
          color: 'error',
          label: 'Killed',
          backgroundColor: theme.isDarkMode ? theme.colors.red800 : theme.colors.red100
        };
      case 'ERROR':
        return {
          color: 'error',
          label: 'Error',
          backgroundColor: theme.isDarkMode ? theme.colors.red800 : theme.colors.red100
        };
      default:
        return {
          color: 'secondary',
          label: status,
          backgroundColor: theme.colors.backgroundSecondary
        };
    }
  };

  const statusConfig = getStatusConfig(runStatus);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '4px 8px',
        borderRadius: '4px',
        backgroundColor: statusConfig.backgroundColor,
        border: `1px solid ${theme.colors.border}`,
        minWidth: '60px',
        maxWidth: '100px'
      }}
    >
      <Typography.Text
        size="sm"
        color={statusConfig.color}
        style={{
          fontWeight: 500,
          textAlign: 'center',
          lineHeight: 1
        }}
      >
        {statusConfig.label}
      </Typography.Text>
    </div>
  );
}; 