import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { DispatcherService } from '../sdk/DispatcherService';

/** Status wording shared with the notebook's index table. */
export type IndexTableStatus = 'Building' | 'Complete' | 'Failed';

/** Vector-store summary of a RAG source, as served by `GET /dispatcher/get-all-contexts`. */
export interface IndexTableVectorStore {
  /** Raw store type ("faiss", "pgvector", "pinecone", ...). */
  type: string | null;
  /** Display label ("FAISS", "PGVector", "Pinecone", raw type or "Unknown"). */
  label: string | null;
  /** pgvector only; password already masked by the dispatcher. */
  connection: string | null;
  /** pgvector only. */
  collection_name: string | null;
  /** Pinecone only. */
  index_name: string | null;
  /** Pinecone only; an empty namespace is rendered as `''` like the notebook. */
  namespace: string | null;
}

/** One RAG source (vector-store index) of the current experiment. */
export interface IndexTableRow {
  context_id: number;
  context_hash: string;
  status: IndexTableStatus;
  /** Final build duration, or elapsed seconds while Building, or null. */
  duration_seconds: number | null;
  started_at: number | null;
  completed_at: number | null;
  device: 'GPU' | 'CPU';
  vector_store: IndexTableVectorStore | null;
  error: string;
}

export const INDEX_TABLE_BUILDING_POLL_MS = 3 * 1000;
export const INDEX_TABLE_IDLE_POLL_MS = 15 * 1000;

/**
 * Polls the dispatcher for the RAG sources (vector-store indexes) of the current
 * experiment. The contexts table is cleared whenever a new experiment is created,
 * so there is nothing to key by. Polls fast while any index is still building and
 * slows down once every row has settled.
 */
export const useExperimentContexts = (enabled = true) =>
  useQuery<IndexTableRow[]>(
    ['experiment-contexts'],
    async () => {
      const rows = await DispatcherService.getAllContexts();
      return Array.isArray(rows) ? (rows as IndexTableRow[]) : [];
    },
    {
      enabled,
      staleTime: 2 * 1000,
      cacheTime: 10 * 60 * 1000,
      retry: 2,
      refetchOnWindowFocus: false,
      refetchInterval: (data) =>
        data?.some((row) => row.status === 'Building') ? INDEX_TABLE_BUILDING_POLL_MS : INDEX_TABLE_IDLE_POLL_MS,
    },
  );
