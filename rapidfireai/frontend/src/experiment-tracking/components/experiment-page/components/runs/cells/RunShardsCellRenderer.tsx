import React from 'react';
import { ICellRendererParams } from '@ag-grid-community/core';
import { RunRowType } from '../../../utils/experimentPage.row-types';

/**
 * Cell renderer for the shards-progress column in the experiment runs table.
 *
 * Reads two MLflow tags written by the evals controller (see
 * ``rapidfireai/evals/scheduling/controller.py`` -- ``set_tag`` calls for
 * ``rapidfire.progress.current`` and ``rapidfire.progress.total``) and
 * renders them as ``current/total`` (e.g. ``2/4``).
 *
 * Falls back to ``-`` when either tag is missing -- e.g. fit-mode runs
 * (which don't emit these tags yet) or grouped/aggregate rows.
 */
const TAG_CURRENT = 'rapidfire.progress.current';
const TAG_TOTAL = 'rapidfire.progress.total';

export interface RunShardsCellRendererProps extends ICellRendererParams {
  data: RunRowType;
}

export const RunShardsCellRenderer = React.memo(({ data }: RunShardsCellRendererProps) => {
  if (!data || !data.tags) {
    return <>-</>;
  }
  const current = data.tags[TAG_CURRENT]?.value;
  const total = data.tags[TAG_TOTAL]?.value;
  if (current === undefined || total === undefined) {
    return <>-</>;
  }
  return <>{`${current}/${total}`}</>;
});
