import { getBigIntJson } from "common/utils/FetchUtils";

export const checkMLflowServer = async (): Promise<boolean> => {
    try {
      const response = await getBigIntJson({
        url: '/ajax-api/2.0/mlflow/experiments/search',
        data: { max_results: 1 } // We only need to check if the API is responsive, so we limit to 1 result
      });
      
      // If we get a response without throwing an error, we consider the server valid
      return true;
    } catch (error) {
      console.error('Error checking MLflow server:', error);
      return false;
    }
};

export const checkDispatcher = async (): Promise<boolean> => {
  try {
    const response = await fetch('/dispatcher/health-check', {
      method: 'GET',
    });
    
    if (!response.ok) {
      throw new Error('Dispatcher health check failed');
    }

    // If we get a response without throwing an error, we consider the server valid
    return true;
  } catch (error) {
    console.error('Error checking Dispatcher:', error);
    return false;
  }
};
