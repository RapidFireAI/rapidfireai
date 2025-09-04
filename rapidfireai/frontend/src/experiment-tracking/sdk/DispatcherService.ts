import { getBigIntJson, postJson } from '../../common/utils/FetchUtils';

export class DispatcherService {
  static getApiUrl(endpoint: string) {
    return `/dispatcher/${endpoint}`;
  }

  // Misc routes
  static healthCheck = () => getBigIntJson({ url: this.getApiUrl('health-check') });
  static initDispatcher = (data: any) => postJson({ relativeUrl: this.getApiUrl('init-dispatcher'), data });

  static async cloneModifyRun(data: any) {
    const response = await postJson({ relativeUrl: this.getApiUrl('clone-modify-run'), data });
    return response;
  }

  static async stopRun(data: any) {
    const response = await postJson({ relativeUrl: this.getApiUrl('stop-run'), data });
    return response;
  }

  static async resumeRun(data: any) {
    const response = await postJson({ relativeUrl: this.getApiUrl('resume-run'), data });
    return response;
  }

  static async deleteRun(data: any) {
    const response = await postJson({ relativeUrl: this.getApiUrl('delete-run'), data });
    return response;
  }

  // UI routes
  static getAllRunsUi = () => getBigIntJson({ url: this.getApiUrl('get-all-runs') });
  static getRunUi = (data: any) => postJson({ relativeUrl: this.getApiUrl('get-run'), data });

  // Experiment routes
  static getRunningExperiment = () => getBigIntJson({ url: this.getApiUrl('get-running-experiment') });
  static createExperiment = (data: any) => postJson({ relativeUrl: this.getApiUrl('create-experiment'), data });
  static getAllExperimentNames = () => getBigIntJson({ url: this.getApiUrl('get-all-experiment-names') });
  static getExperimentError = () => getBigIntJson({ url: this.getApiUrl('get-experiment-error') });
  static setExperimentError = (data: any) => postJson({ relativeUrl: this.getApiUrl('set-experiment-error'), data });
  static setExperimentStatus = (data: any) => postJson({ relativeUrl: this.getApiUrl('set-experiment-status'), data });

  // ETL Controller routes
  static getEtlControllerTaskStatus = (data: any) => postJson({ relativeUrl: this.getApiUrl('get-etl-controller-task-status'), data });
  static setEtlControllerTask = (data: any) => postJson({ relativeUrl: this.getApiUrl('set-etl-controller-task'), data });
  static getEtlControllerRunningTask = () => getBigIntJson({ url: this.getApiUrl('get-etl-controller-running-task') });
  static getEtlControllerScheduledTask = () => getBigIntJson({ url: this.getApiUrl('get-etl-controller-scheduled-task') });

  // ML Controller routes
  static getMlControllerRunningTasks = () => getBigIntJson({ url: this.getApiUrl('get-ml-controller-running-tasks') });
  static getMlControllerScheduledTasks = () => getBigIntJson({ url: this.getApiUrl('get-ml-controller-scheduled-tasks') });
  static getMlControllerTaskStatus = (data: any) => postJson({ relativeUrl: this.getApiUrl('get-ml-controller-task-status'), data });
  static setMlControllerTask = (data: any) => postJson({ relativeUrl: this.getApiUrl('set-ml-controller-task'), data });
  static getMlControllerCloneModifyTask = () => getBigIntJson({ url: this.getApiUrl('get-ml-controller-clone-modify-task') });
  static getMlTrainControllerProgress = (data: any) => getBigIntJson({ url: this.getApiUrl('get-ml-train-controller-progress'), data: data });

  // Run routes
  static getRun = (data: any) => postJson({ relativeUrl: this.getApiUrl('get-run'), data });
  static getRunsByStatus = (data: any) => postJson({ relativeUrl: this.getApiUrl('get-runs-by-status'), data });
  static getAllRuns = () => getBigIntJson({ url: this.getApiUrl('get-all-runs') });

  // Cleanup routes
  static resetAllTables = (data: any) => postJson({ relativeUrl: this.getApiUrl('reset-all-tables'), data });
  static cancelCurrent = () => getBigIntJson({ url: this.getApiUrl('cancel-current') });

  // Log routes
  static getLogs = (data: any) => postJson({ relativeUrl: this.getApiUrl('get-experiment-logs'), data });
  static getICLogs = (data: any) => postJson({ relativeUrl: this.getApiUrl('get-ic-logs'), data });
}
