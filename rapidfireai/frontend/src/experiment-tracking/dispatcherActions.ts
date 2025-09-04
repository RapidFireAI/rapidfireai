import { Dispatch } from 'redux';
import { DispatcherService } from './sdk/DispatcherService';
import { getUUID } from '../common/utils/ActionUtils';

// Action types
export const HEALTH_CHECK_API = 'HEALTH_CHECK_API';
export const INIT_DISPATCHER_API = 'INIT_DISPATCHER_API';
export const CLONE_MODIFY_RUN_API = 'CLONE_MODIFY_RUN_API';
export const STOP_RUN_API = 'STOP_RUN_API';
export const RESUME_RUN_API = 'RESUME_RUN_API';
export const DELETE_RUN_API = 'DELETE_RUN_API';
export const GET_ALL_RUNS_UI_API = 'GET_ALL_RUNS_UI_API';
export const GET_RUN_UI_API = 'GET_RUN_UI_API';
export const GET_RUNNING_EXPERIMENT_API = 'GET_RUNNING_EXPERIMENT_API';
export const CREATE_EXPERIMENT_API = 'CREATE_EXPERIMENT_API';
export const GET_ALL_EXPERIMENT_NAMES_API = 'GET_ALL_EXPERIMENT_NAMES_API';
export const GET_EXPERIMENT_ERROR_API = 'GET_EXPERIMENT_ERROR_API';
export const SET_EXPERIMENT_ERROR_API = 'SET_EXPERIMENT_ERROR_API';
export const SET_EXPERIMENT_STATUS_API = 'SET_EXPERIMENT_STATUS_API';
export const GET_ETL_CONTROLLER_TASK_STATUS_API = 'GET_ETL_CONTROLLER_TASK_STATUS_API';
export const SET_ETL_CONTROLLER_TASK_API = 'SET_ETL_CONTROLLER_TASK_API';
export const GET_ETL_CONTROLLER_RUNNING_TASK_API = 'GET_ETL_CONTROLLER_RUNNING_TASK_API';
export const GET_ETL_CONTROLLER_SCHEDULED_TASK_API = 'GET_ETL_CONTROLLER_SCHEDULED_TASK_API';
export const GET_ML_CONTROLLER_RUNNING_TASKS_API = 'GET_ML_CONTROLLER_RUNNING_TASKS_API';
export const GET_ML_CONTROLLER_SCHEDULED_TASKS_API = 'GET_ML_CONTROLLER_SCHEDULED_TASKS_API';
export const GET_ML_CONTROLLER_TASK_STATUS_API = 'GET_ML_CONTROLLER_TASK_STATUS_API';
export const SET_ML_CONTROLLER_TASK_API = 'SET_ML_CONTROLLER_TASK_API';
export const GET_ML_CONTROLLER_CLONE_MODIFY_TASK_API = 'GET_ML_CONTROLLER_CLONE_MODIFY_TASK_API';
export const GET_ML_TRAIN_CONTROLLER_PROGRESS_API = 'GET_ML_TRAIN_CONTROLLER_PROGRESS_API';
export const GET_DISPATCHER_RUN_API = 'GET_DISPATCHER_RUN_API';
export const GET_RUNS_BY_STATUS_API = 'GET_RUNS_BY_STATUS_API';
export const GET_ALL_RUNS_API = 'GET_ALL_RUNS_API';
export const RESET_ALL_TABLES_API = 'RESET_ALL_TABLES_API';
export const CANCEL_CURRENT_API = 'CANCEL_CURRENT_API';

// Action creators
export const healthCheckApi = (id = getUUID()) => ({
  type: HEALTH_CHECK_API,
  payload: DispatcherService.healthCheck(),
  meta: { id },
});

export const initDispatcherApi = (data: any, id = getUUID()) => ({
  type: INIT_DISPATCHER_API,
  payload: DispatcherService.initDispatcher(data),
  meta: { id },
});

export const cloneModifyRunApi = (data: any, id = getUUID()) => ({
  type: CLONE_MODIFY_RUN_API,
  payload: DispatcherService.cloneModifyRun(data),
  meta: { id },
});

export const stopRunApi = (data: any, id = getUUID()) => ({
  type: STOP_RUN_API,
  payload: DispatcherService.stopRun(data),
  meta: { id },
});

export const resumeRunApi = (data: any, id = getUUID()) => ({
  type: RESUME_RUN_API,
  payload: DispatcherService.resumeRun(data),
  meta: { id },
});

export const deleteRunApi = (data: any, id = getUUID()) => ({
  type: DELETE_RUN_API,
  payload: DispatcherService.deleteRun(data),
  meta: { id },
});

export const getAllRunsUiApi = (id = getUUID()) => ({
  type: GET_ALL_RUNS_UI_API,
  payload: DispatcherService.getAllRunsUi(),
  meta: { id },
});

export const getRunUiApi = (data: any, id = getUUID()) => ({
  type: GET_RUN_UI_API,
  payload: DispatcherService.getRunUi(data),
  meta: { id },
});

export const getRunningExperimentApi = (id = getUUID()) => ({
  type: GET_RUNNING_EXPERIMENT_API,
  payload: DispatcherService.getRunningExperiment(),
  meta: { id },
});

export const createExperimentApi = (data: any, id = getUUID()) => ({
  type: CREATE_EXPERIMENT_API,
  payload: DispatcherService.createExperiment(data),
  meta: { id },
});

export const getAllExperimentNamesApi = (id = getUUID()) => ({
  type: GET_ALL_EXPERIMENT_NAMES_API,
  payload: DispatcherService.getAllExperimentNames(),
  meta: { id },
});

export const getExperimentErrorApi = (id = getUUID()) => ({
  type: GET_EXPERIMENT_ERROR_API,
  payload: DispatcherService.getExperimentError(),
  meta: { id },
});

export const setExperimentErrorApi = (data: any, id = getUUID()) => ({
  type: SET_EXPERIMENT_ERROR_API,
  payload: DispatcherService.setExperimentError(data),
  meta: { id },
});

export const setExperimentStatusApi = (data: any, id = getUUID()) => ({
  type: SET_EXPERIMENT_STATUS_API,
  payload: DispatcherService.setExperimentStatus(data),
  meta: { id },
});

export const getEtlControllerTaskStatusApi = (data: any, id = getUUID()) => ({
  type: GET_ETL_CONTROLLER_TASK_STATUS_API,
  payload: DispatcherService.getEtlControllerTaskStatus(data),
  meta: { id },
});

export const setEtlControllerTaskApi = (data: any, id = getUUID()) => ({
  type: SET_ETL_CONTROLLER_TASK_API,
  payload: DispatcherService.setEtlControllerTask(data),
  meta: { id },
});

export const getEtlControllerRunningTaskApi = (id = getUUID()) => ({
  type: GET_ETL_CONTROLLER_RUNNING_TASK_API,
  payload: DispatcherService.getEtlControllerRunningTask(),
  meta: { id },
});

export const getEtlControllerScheduledTaskApi = (id = getUUID()) => ({
  type: GET_ETL_CONTROLLER_SCHEDULED_TASK_API,
  payload: DispatcherService.getEtlControllerScheduledTask(),
  meta: { id },
});

export const getMlControllerRunningTasksApi = (id = getUUID()) => ({
  type: GET_ML_CONTROLLER_RUNNING_TASKS_API,
  payload: DispatcherService.getMlControllerRunningTasks(),
  meta: { id },
});

export const getMlControllerScheduledTasksApi = (id = getUUID()) => ({
  type: GET_ML_CONTROLLER_SCHEDULED_TASKS_API,
  payload: DispatcherService.getMlControllerScheduledTasks(),
  meta: { id },
});

export const getMlControllerTaskStatusApi = (data: any, id = getUUID()) => ({
  type: GET_ML_CONTROLLER_TASK_STATUS_API,
  payload: DispatcherService.getMlControllerTaskStatus(data),
  meta: { id },
});

export const setMlControllerTaskApi = (data: any, id = getUUID()) => ({
  type: SET_ML_CONTROLLER_TASK_API,
  payload: DispatcherService.setMlControllerTask(data),
  meta: { id },
});

export const getMlControllerCloneModifyTaskApi = (id = getUUID()) => ({
  type: GET_ML_CONTROLLER_CLONE_MODIFY_TASK_API,
  payload: DispatcherService.getMlControllerCloneModifyTask(),
  meta: { id },
});

export const getMlTrainControllerProgressApi = (data: any, id = getUUID()) => ({
  type: GET_ML_TRAIN_CONTROLLER_PROGRESS_API,
  payload: DispatcherService.getMlTrainControllerProgress(data),
  meta: { id },
});

export const getDispatcherRunApi = (data: any, id = getUUID()) => ({
  type: GET_DISPATCHER_RUN_API,
  payload: DispatcherService.getRun(data),
  meta: { id },
});

export const getRunsByStatusApi = (data: any, id = getUUID()) => ({
  type: GET_RUNS_BY_STATUS_API,
  payload: DispatcherService.getRunsByStatus(data),
  meta: { id },
});

export const getAllRunsApi = (id = getUUID()) => ({
  type: GET_ALL_RUNS_API,
  payload: DispatcherService.getAllRuns(),
  meta: { id },
});

export const resetAllTablesApi = (data: any, id = getUUID()) => ({
  type: RESET_ALL_TABLES_API,
  payload: DispatcherService.resetAllTables(data),
  meta: { id },
});

export const cancelCurrentApi = (id = getUUID()) => ({
  type: CANCEL_CURRENT_API,
  payload: DispatcherService.cancelCurrent(),
  meta: { id },
});

// Thunk action creator for checking dispatcher status
export const checkDispatcherStatus = () => async (dispatch: Dispatch) => {
  try {
    await dispatch(healthCheckApi());
    dispatch(updateDispatcherStatus({ isValid: true, uri: DispatcherService.getApiUrl('') }));
  } catch (error) {
    console.error('Dispatcher health check failed:', error);
    dispatch(updateDispatcherStatus({ isValid: false, uri: null }));
  }
};

export const UPDATE_DISPATCHER_STATUS = 'UPDATE_DISPATCHER_STATUS';
export const updateDispatcherStatus = (status: { isValid: boolean; uri: string | null }) => ({
  type: UPDATE_DISPATCHER_STATUS,
  payload: status,
});
