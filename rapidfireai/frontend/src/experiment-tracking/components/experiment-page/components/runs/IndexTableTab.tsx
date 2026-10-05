import React, { useMemo } from 'react';
import {
  Alert,
  Empty,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import {
  useExperimentContexts,
  type IndexTableRow,
  type IndexTableStatus,
  type IndexTableVectorStore,
} from '../../../../hooks/useExperimentContexts';

/**
 * IndexTableTab
 *
 * The dashboard twin of the notebook's "Preprocessing RAG Sources" table: one row
 * per RAG source (vector-store index) of the current evals experiment, with the
 * same columns the notebook prints. Data comes from the dispatcher's
 * `get-all-contexts` endpoint via `useExperimentContexts`, which keeps polling
 * while any index is still building, so Status and Duration update live.
 *
 * The four vector-store detail columns (Connection, Collection Name, Index Name,
 * Namespace) only appear when at least one row carries that field, so a
 * FAISS-only experiment shows the same five columns as the notebook.
 */

const STATUS_TAG_COLOR: Record<IndexTableStatus, React.ComponentProps<typeof Tag>['color']> = {
  Building: 'lemon',
  Complete: 'lime',
  Failed: 'coral',
};

type DetailColumnKey = keyof Pick<IndexTableVectorStore, 'connection' | 'collection_name' | 'index_name' | 'namespace'>;

const DETAIL_COLUMNS: { key: DetailColumnKey; header: React.ReactNode }[] = [
  {
    key: 'connection',
    header: (
      <FormattedMessage defaultMessage="Connection" description="Index table column: pgvector connection string" />
    ),
  },
  {
    key: 'collection_name',
    header: (
      <FormattedMessage defaultMessage="Collection Name" description="Index table column: pgvector collection name" />
    ),
  },
  {
    key: 'index_name',
    header: <FormattedMessage defaultMessage="Index Name" description="Index table column: Pinecone index name" />,
  },
  {
    key: 'namespace',
    header: <FormattedMessage defaultMessage="Namespace" description="Index table column: Pinecone namespace" />,
  },
];

const formatDuration = (seconds: number | null | undefined) =>
  seconds === null || seconds === undefined ? '-' : `${seconds.toFixed(1)}s`;

const shortHash = (hash: string) => (hash ? `${hash.slice(0, 8)}...` : '');

const StatusTag = ({ row }: { row: IndexTableRow }) => {
  const tag = (
    <Tag componentId="rapidfire.index-table.status-tag" color={STATUS_TAG_COLOR[row.status] ?? 'default'}>
      {row.status}
    </Tag>
  );
  if (row.status === 'Failed' && row.error) {
    return <span title={row.error}>{tag}</span>;
  }
  return tag;
};

export interface IndexTableTabProps {
  /** Pass false to pause polling (the tab is normally unmounted when hidden). */
  enabled?: boolean;
}

export const IndexTableTab = ({ enabled = true }: IndexTableTabProps) => {
  const { theme } = useDesignSystemTheme();
  const { data, isLoading, error } = useExperimentContexts(enabled);
  const rows: IndexTableRow[] = useMemo(() => data ?? [], [data]);

  const visibleDetailColumns = useMemo(
    () =>
      DETAIL_COLUMNS.filter(({ key }) =>
        rows.some((row) => row.vector_store?.[key] !== null && row.vector_store?.[key] !== undefined),
      ),
    [rows],
  );

  if (error) {
    return (
      <div css={{ padding: theme.spacing.md }}>
        <Alert
          componentId="rapidfire.index-table.error"
          type="error"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Failed to load the index table"
              description="Error shown when the RAG index table cannot be fetched from the dispatcher"
            />
          }
          description={error instanceof Error ? error.message : String(error)}
        />
      </div>
    );
  }

  if (isLoading && !data) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
        <Spinner />
      </div>
    );
  }

  const emptyState = (
    <Empty
      description={
        <FormattedMessage
          defaultMessage="No RAG sources built yet. Rows appear here once run_evals starts building indexes."
          description="Empty state of the RAG index table tab on the experiment page"
        />
      }
    />
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, padding: theme.spacing.sm }}>
      <Typography.Hint>
        <FormattedMessage
          defaultMessage="RAG sources (vector-store indexes) of this experiment. Refreshes automatically while indexes build."
          description="Hint above the RAG index table on the experiment page"
        />
      </Typography.Hint>
      <Table
        scrollable
        empty={rows.length === 0 ? emptyState : null}
        css={{
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.general.borderRadiusBase,
        }}
      >
        <TableRow isHeader>
          <TableHeader componentId="rapidfire.index-table.rag-source-id-header" css={{ flex: 1 }}>
            <FormattedMessage
              defaultMessage="RAG Source ID"
              description="Index table column: RAG source (context) id"
            />
          </TableHeader>
          <TableHeader componentId="rapidfire.index-table.status-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Status" description="Index table column: build status" />
          </TableHeader>
          <TableHeader componentId="rapidfire.index-table.duration-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Duration" description="Index table column: build duration" />
          </TableHeader>
          <TableHeader componentId="rapidfire.index-table.device-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Device" description="Index table column: GPU or CPU" />
          </TableHeader>
          <TableHeader componentId="rapidfire.index-table.vector-store-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Vector Store" description="Index table column: vector store type" />
          </TableHeader>
          {visibleDetailColumns.map(({ key, header }) => (
            <TableHeader key={key} componentId={`rapidfire.index-table.${key}-header`} css={{ flex: 2 }}>
              {header}
            </TableHeader>
          ))}
        </TableRow>
        {rows.map((row) => (
          <TableRow key={row.context_id} data-testid={`index-table-row-${row.context_id}`}>
            <TableCell css={{ flex: 1 }}>
              <div css={{ display: 'flex', flexDirection: 'column' }}>
                <span>{row.context_id}</span>
                <Typography.Hint css={{ fontFamily: 'monospace' }}>
                  {shortHash(row.context_hash)}
                </Typography.Hint>
              </div>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <StatusTag row={row} />
            </TableCell>
            <TableCell css={{ flex: 1 }}>{formatDuration(row.duration_seconds)}</TableCell>
            <TableCell css={{ flex: 1 }}>{row.device}</TableCell>
            <TableCell css={{ flex: 1 }}>{row.vector_store?.label ?? '-'}</TableCell>
            {visibleDetailColumns.map(({ key }) => (
              <TableCell key={key} css={{ flex: 2 }}>
                <span css={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>{row.vector_store?.[key] ?? '-'}</span>
              </TableCell>
            ))}
          </TableRow>
        ))}
      </Table>
    </div>
  );
};

export default IndexTableTab;
