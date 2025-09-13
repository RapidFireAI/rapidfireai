import { FETCH_CLUSTERS_REQUEST, FETCH_CLUSTERS_SUCCESS, FETCH_CLUSTERS_FAILURE, SET_PROXY_TARGET, UPDATE_MLFLOW_SERVER_STATUS } from '../actions';

export interface Cluster {
  name: string;
  id: number;
  num_workers: number;
  worker_instance_type: string;
  aws_region: string;
  status: 'running' | 'stopped' | 'pending' | 'failed' | 'deleted' | 'deleting' | 'degraded';
  jupyter_uri: string | null;
  dispatcher_uri: string | null;
  mlflow_uri: string | null;
  plutono_uri: string | null;
}
  
export interface ClusterState {
  data: Cluster[];
  count: number;
  loading: boolean;
  error: string | null;
  proxyTarget: string | null;
  mlflowServer: {
    status: {
      isValid: boolean;
      uri: string | null;
    }
  };
  dispatcher: {
    status: {
      isValid: boolean;
      uri: string | null;
    }
  };
}

const initialState: ClusterState = {
  data: [],
  loading: false,
  error: null,
  proxyTarget: null,
  count: 0,
  mlflowServer: {
    status: {
      isValid: true,
      uri: null
    }
  },
  dispatcher: {
    status: {
      isValid: true,
      uri: null
    }
  }
};

export const clusterReducer = (state = initialState, action: { type: any; payload: any; error: any; }) => {
  switch (action.type) {
    case FETCH_CLUSTERS_REQUEST:
      return { ...state, loading: true };
    case FETCH_CLUSTERS_SUCCESS:
    return { 
        ...state, 
        loading: false, 
        data: action.payload.data, 
        count: action.payload.count,
        error: null 
      };
    case FETCH_CLUSTERS_FAILURE:
      return { ...state, loading: false, error: action.error };
    case SET_PROXY_TARGET:
      return { ...state, proxyTarget: action.payload };
    case UPDATE_MLFLOW_SERVER_STATUS:
      return {
        ...state,
        mlflowServer: {
          ...state.mlflowServer,
          status: action.payload
        }
      };
    default:
      return state;
  }
};
