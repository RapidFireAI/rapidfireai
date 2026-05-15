/**
 * SQL Warehouse context — Databricks-only feature.
 *
 * In the RapidFire OSS fork, SQL warehouses are not used (V4 traces API is
 * disabled). This module provides a no-op stub so chart components that
 * unconditionally call `useSqlWarehouseContextSafe()` work without a
 * provider in the tree.
 */
export interface SqlWarehouseContextValue {
  warehouseId: string | null;
  setWarehouseId: (id: string | null) => void;
}

export const useSqlWarehouseContextSafe = (): SqlWarehouseContextValue | null => null;
