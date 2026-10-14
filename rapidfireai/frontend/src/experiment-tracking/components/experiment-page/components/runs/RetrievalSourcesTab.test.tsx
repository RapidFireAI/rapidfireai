import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DispatcherService } from '../../../../sdk/DispatcherService';
import { RetrievalSourcesTab } from './RetrievalSourcesTab';
import type { RetrievalSourceRow } from '../../../../hooks/useExperimentContexts';

jest.mock('../../../../sdk/DispatcherService', () => ({
  DispatcherService: {
    getAllContexts: jest.fn(),
  },
}));

const mockedGetAllContexts = DispatcherService.getAllContexts as unknown as jest.Mock;

const faissRow: RetrievalSourceRow = {
  context_id: 1,
  context_hash: 'abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789',
  status: 'Complete',
  duration_seconds: 12.34,
  started_at: 1_700_000_000,
  completed_at: 1_700_000_012,
  device: 'GPU',
  vector_store: {
    type: 'faiss',
    label: 'FAISS',
    connection: null,
    collection_name: null,
    index_name: null,
    namespace: null,
  },
  error: '',
};

const pgvectorRow: RetrievalSourceRow = {
  context_id: 2,
  context_hash: 'ffffffff0123456789abcdef0123456789abcdef0123456789abcdef01234567',
  status: 'Building',
  duration_seconds: 3.0,
  started_at: 1_700_000_000,
  completed_at: null,
  device: 'CPU',
  vector_store: {
    type: 'pgvector',
    label: 'PGVector',
    connection: 'postgresql://rf:***@db:5432/rf',
    collection_name: 'docs',
    index_name: null,
    namespace: null,
  },
  error: '',
};

const renderTab = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });
  return renderWithIntl(
    <QueryClientProvider client={queryClient}>
      <DesignSystemProvider>
        <RetrievalSourcesTab />
      </DesignSystemProvider>
    </QueryClientProvider>,
  );
};

describe('RetrievalSourcesTab', () => {
  beforeEach(() => {
    mockedGetAllContexts.mockReset();
  });

  it('renders the five notebook columns for FAISS-only rows', async () => {
    mockedGetAllContexts.mockResolvedValue([faissRow]);
    renderTab();

    await waitFor(() => expect(screen.getByTestId('retrieval-source-row-1')).toBeInTheDocument());

    for (const header of ['RAG Source ID', 'Status', 'Duration', 'Device', 'Vector Store']) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
    expect(screen.queryByText('Connection')).not.toBeInTheDocument();
    expect(screen.queryByText('Collection Name')).not.toBeInTheDocument();
    expect(screen.queryByText('Index Name')).not.toBeInTheDocument();
    expect(screen.queryByText('Namespace')).not.toBeInTheDocument();

    expect(screen.getByText('Complete')).toBeInTheDocument();
    expect(screen.getByText('12.3s')).toBeInTheDocument();
    expect(screen.getByText('GPU')).toBeInTheDocument();
    expect(screen.getByText('FAISS')).toBeInTheDocument();
    expect(screen.getByText('abcdef01...')).toBeInTheDocument();
  });

  it('adds the pgvector detail columns when a row has them', async () => {
    mockedGetAllContexts.mockResolvedValue([faissRow, pgvectorRow]);
    renderTab();

    await waitFor(() => expect(screen.getByTestId('retrieval-source-row-2')).toBeInTheDocument());

    expect(screen.getByText('Connection')).toBeInTheDocument();
    expect(screen.getByText('Collection Name')).toBeInTheDocument();
    expect(screen.queryByText('Index Name')).not.toBeInTheDocument();
    expect(screen.getByText('postgresql://rf:***@db:5432/rf')).toBeInTheDocument();
    expect(screen.getByText('docs')).toBeInTheDocument();
    expect(screen.getByText('Building')).toBeInTheDocument();
    expect(screen.getByText('3.0s')).toBeInTheDocument();
    // The FAISS row shows a dash in the detail columns it does not have.
    expect(screen.getAllByText('-').length).toBeGreaterThanOrEqual(2);
  });

  it('shows the empty state when there are no RAG sources', async () => {
    mockedGetAllContexts.mockResolvedValue([]);
    renderTab();

    await waitFor(() => expect(screen.getByText(/No retrieval sources built yet/)).toBeInTheDocument());
  });

  it('shows an error alert when the dispatcher call fails', async () => {
    mockedGetAllContexts.mockRejectedValue(new Error('dispatcher unreachable'));
    renderTab();

    await waitFor(() => expect(screen.getByText('Failed to load retrieval sources')).toBeInTheDocument());
    expect(screen.getByText('dispatcher unreachable')).toBeInTheDocument();
  });
});
