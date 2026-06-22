import React from 'react';
import { ICellRendererParams } from '@ag-grid-community/core';
import { RunRowType } from '../../../utils/experimentPage.row-types';

/**
 * Cell renderer for the shards-progress column in the experiment runs table.
 *
 * Reads two MLflow tags written by both the evals controller (see
 * ``rapidfireai/evals/scheduling/controller.py``) and the fit controller
 * (see ``rapidfireai/fit/backend/controller.py`` --
 * ``_set_progress_tags``) and renders them as ``current/total``
 * (e.g. ``2/4``). For fit-mode runs ``total`` is ``num_chunks`` and
 * ``current`` is chunks completed in the current epoch (the scheduler
 * resets it at epoch boundaries).
 *
 * Falls back to ``-`` when either tag is missing -- e.g. grouped /
 * aggregate rows, or legacy runs created before this tagging was added.
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
