/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { MlflowService } from '../../experiment-tracking/sdk/MlflowService';
import { Services as ModelRegistryService } from '../../model-registry/services';

export const getExperimentNameValidator = (getExistingExperimentNames: any) => {
  return (rule: any, value: any, callback: any) => {
    if (!value) {
      // no need to execute below validations when no value is entered
      // eslint-disable-next-line callback-return
      callback(undefined);
    } else if (getExistingExperimentNames().includes(value)) {
      // getExistingExperimentNames returns the names of all active experiments
      // check whether the passed value is part of the list
      // eslint-disable-next-line callback-return
      callback(`Experiment "${value}" already exists.`);
    } else {
      // on-demand validation whether experiment already exists in deleted state
      MlflowService.getExperimentByName({ experiment_name: value })
        .then((res) =>
          callback(`Experiment "${value}" already exists in deleted state.
                                 You can restore the experiment, or permanently delete the
                                 experiment from the .trash folder (under tracking server's
                                 root folder) in order to use this experiment name again.`),
        )
        .catch((e) => callback(undefined)); // no experiment returned
    }
  };
};

export const modelNameValidator = (rule: any, name: any, callback: any) => {
  if (!name) {
    callback(undefined);
    return;
  }

  ModelRegistryService.getRegisteredModel({ name: name })
    .then(() => callback(`Model "${name}" already exists.`))
    .catch((e) => callback(undefined));
};

export const numClusterWorkersValidator = () => {
  return (rule: any, value: any, callback: any) => {
    const integerValue = parseInt(value, 10);
    if (!value) {
      // no need to execute below validations when no value is entered
      // eslint-disable-next-line callback-return
      callback(undefined);
    } else if (!Number.isInteger(integerValue)) {
      // eslint-disable-next-line callback-return
      callback(`"${value}" is not an integer.`);
    } else if (integerValue < 1) {
      // eslint-disable-next-line callback-return
      callback(`Number of workers must be greater than 0, not "${value}".`);
    } else {
      // eslint-disable-next-line callback-return
      callback(undefined);
    }
  }; 
};

export const clusterNameValidator = () => {
  return (rule: any, value: any, callback: any) => {
    const regex = /^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$/;
    if (!value) {
      // no need to execute below validations when no value is entered
      return callback('Cluster name is required.');
    } else if (value.length > 23) {
      return callback(`Cluster name must be less than or equal to 23 characters.`);
    } else if (!regex.test(value)) {
      return callback(`Cluster name does not match the required format. It should consist of alphanumeric characters, hyphens, underscores, and periods. It must start and end with an alphanumeric character.`);
    } else {
      // If we reach here, the cluster name is valid
      // eslint-disable-next-line callback-return
      return callback(undefined);
    }
  }; 
};
