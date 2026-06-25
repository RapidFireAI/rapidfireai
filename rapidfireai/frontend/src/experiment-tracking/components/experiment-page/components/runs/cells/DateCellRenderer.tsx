import { Theme } from '@emotion/react';
import React from 'react';
import Utils from '../../../../../../common/utils/Utils';
import { RunRowDateAndNestInfo } from '../../../utils/experimentPage.row-types';
import { useIntl } from 'react-intl';

export interface DateCellRendererProps {
  value: RunRowDateAndNestInfo;
}

// The status icon used to live here as a prefix, but it has moved into
// its own dedicated Status column (see RunStatusCellRenderer) so that
// users can scan a run's lifecycle state without parsing the "Created"
// timestamp. We intentionally do not render an icon here anymore --
// rendering both would duplicate the visual signal and waste a column's
// worth of horizontal space.
export const DateCellRenderer = React.memo(({ value }: DateCellRendererProps) => {
  const { startTime, referenceTime } = value || {};
  const intl = useIntl();
  if (!startTime) {
    return <>-</>;
  }

  return (
    <span css={styles.cellWrapper} title={Utils.formatTimestamp(startTime, intl)}>
      {Utils.timeSinceStr(startTime, referenceTime)}
    </span>
  );
});

const styles = {
  cellWrapper: (theme: Theme) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
  }),
};
