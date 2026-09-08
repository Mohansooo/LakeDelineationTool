"""
Aggregate small sub-basins and stream reaches for TauDEM / cleanGeofabric outputs.

Expected inputs (defaults match cleanGeofabric.py outputs):
  merged_basins/reservoirBasins_final.shp
  merged_basins/reservoirStreams_final.shp

Rule H / Rule I lookups are inverted-index replacements of the original
``basin.loc[basin[col] == key]`` scans. Merge order, predicates, and the
read-after-write sequence inside Rule I are unchanged.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Hashable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd


# ==============================================================================
# COLUMN NAMES — edit these to match your shapefile attribute tables
# ==============================================================================
# Shared topology
BASIN_ID = "DN"             # primary basin object id
RIVER_ID = "LINKNO"             # stream reach ID
NEXT_DOWN_ID = "DSLINKNO"       # downstream link / basin ID (outlet sentinel below)

# Basin areas (km² after conversion; see AREA_SCALE)
UNIT_AREA: Optional[str] = None  # local sub-basin area; None -> polygon area
UP_AREA = "DSContArea"           # cumulative drainage area at pour point

# River hydraulics
SLOPE = "Slope"
LENGTH = "Length"                # reach length; converted with LENGTH_SCALE

# Masking / special units
LAKE_FLAG = "is_lake"            # >0 marks reservoir/lake sub-basins (not aggregated)
GAUGE_FLAG: Optional[str] = None  # numeric 0/1 column; None -> derive from GAUGE_IDS
GAUGE_IDS = "STATION_NU"         # gauge attribute only (not the basin object id)

# Basin attributes carried through to aggregated output (from pour-point basin)
LAKE_ID = "lake_id"
LAKE_AREA = "lake_area"          # km² (shapefile-safe name, ≤10 chars)
FRAC_LAKE_AREA = "frac_lake"     # fraction of basin covered by lake (geom intersection / basin area)

# Extra river attributes carried through to aggregated output
STREAM_ORDER = "strmOrder"
HILLSLOPE: Optional[str] = None  # not present in TauDEM; left out of output if None

# Unit conversions applied after loading shapefiles
AREA_SCALE = 1e-6                # m² -> km² for TauDEM DSContArea / USContArea
LENGTH_SCALE = 1e-3              # m -> km for TauDEM Length


# ==============================================================================
# INPUT / OUTPUT PATHS AND THRESHOLDS
# ==============================================================================
INPUT_BASINS = "merged_basins/reservoirBasins_final.shp"
INPUT_RIVERS = "merged_basins/reservoirStreams_final.shp"
OUTPUT_BASINS = "final_basin/aggregated_basins.shp"
OUTPUT_RIVERS = "final_basin/aggregated_rivers.shp"

MIN_SUB_AREA = 90.0          # km²
MIN_RIV_SLOPE = 0.0000001     # minimum accepted river slope (WATFLOOD manual)
MIN_RIV_LENGTH = 1.0          # km

# Sentinel written to DSLINKNO for the most-downstream basin(s).
# Match this to MESH outlet_value (e.g. -9999).
OUTLET_VALUE = -9999


# ==============================================================================
# HELPERS
# ==============================================================================
def _require_columns(gdf: gpd.GeoDataFrame, columns: list[str], label: str) -> None:
  missing = [col for col in columns if col not in gdf.columns]
  if missing:
    raise ValueError(f"Missing columns in {label}: {missing}")


def _area_km2(series: pd.Series, scale: float) -> pd.Series:
  return pd.to_numeric(series, errors="coerce").fillna(0.0) * scale


def _length_km(series: pd.Series, scale: float) -> pd.Series:
  return pd.to_numeric(series, errors="coerce").fillna(0.0) * scale


def _basin_attr_cols(basin: gpd.GeoDataFrame) -> list[str]:
  """Attribute columns to carry from the pour-point basin into aggregated output."""
  candidates = [
    GAUGE_IDS,
    LAKE_FLAG,
    LAKE_ID,
    LAKE_AREA,
    FRAC_LAKE_AREA,
  ]
  return [c for c in candidates if c in basin.columns]


def _is_outlet_id(down_id: object, outlet_value: int) -> bool:
  """Return True for outlet sentinels (configured value, or legacy <= 0)."""
  try:
    down = int(down_id)
  except (TypeError, ValueError):
    return True
  return down == int(outlet_value) or down <= 0


def _is_sentinel_object_id(obj_id: object, outlet_value: int) -> bool:
  """True if ``obj_id`` cannot be a basin/reach identifier.

  DSLINKNO uses ``_is_outlet_id`` (0 / negative / sentinel all mean outlet).
  LINKNO/DN/agg may legitimately be 0 in TauDEM — only the configured
  sentinel and negative IDs are illegal object ids.
  """
  try:
    val = int(obj_id)
  except (TypeError, ValueError):
    return True
  return val == int(outlet_value) or val < 0


def _to_int64_link_ids(
  series: pd.Series,
  outlet_value: int,
  label: str,
  *,
  fill_outlets: bool = True,
) -> pd.Series:
  """
  Cast topology IDs to int64.

  ``fill_outlets=True`` is only for downstream pointers (DSLINKNO): NaN/inf
  become the outlet sentinel. Never use that fill on object IDs (LINKNO /
  DN) — a sentinel is not a basin id and would dissolve every unattached
  edge unit into one polygon.
  """
  numeric = pd.to_numeric(series, errors="coerce")
  bad = numeric.isna() | np.isinf(numeric.to_numpy())
  n_bad = int(bad.sum())
  if n_bad:
    if not fill_outlets:
      raise ValueError(
        f"{n_bad} non-finite value(s) in {label}; refusing to invent basin IDs."
      )
    print(
      f"Warning: {n_bad} non-finite value(s) in {label}; "
      f"writing outlet sentinel {int(outlet_value)}."
    )
    numeric = numeric.mask(bad, int(outlet_value))
  return numeric.astype("int64")


def _pour_point_down(basin: gpd.GeoDataFrame, id_col: str) -> pd.DataFrame:
  """
  One (agg, aggdown) row per aggregate: prefer the true pour-point
  ``id_col == agg``. Groups that lost that row (absorbed representative)
  fall back to the member with largest ``_uparea``.
  """
  pour = basin.loc[basin[id_col] == basin["agg"], ["agg", "aggdown"]].copy()
  missing_mask = ~basin["agg"].isin(pour["agg"])
  if missing_mask.any():
    fallback = (
      basin.loc[missing_mask, ["agg", "aggdown", "_uparea"]]
      .sort_values("_uparea", ascending=True)
      .groupby("agg", as_index=False)
      .tail(1)[["agg", "aggdown"]]
    )
    pour = pd.concat([pour, fallback], ignore_index=True)
  return pour.drop_duplicates(subset=["agg"], keep="first")


def _remap_aggdown_to_survivors(
  basin: gpd.GeoDataFrame,
  id_col: str,
  agg_col: str = "agg",
  aggdown_col: str = "aggdown",
  outlet_value: int = OUTLET_VALUE,
) -> gpd.GeoDataFrame:
  """
  Rewrite ``aggdown`` so every link targets a surviving aggregate id.

  Small basins are absorbed into a downstream ``agg`` id during aggregation, but
  lakes and other protected units can keep a stale ``aggdown`` that still points
  at the absorbed (now missing) id. Map each original id to its final ``agg`` and
  resolve ``aggdown`` through that map. Terminal / self-draining links become
  ``outlet_value``.
  """
  outlet_value = int(outlet_value)
  id_to_agg = {
    int(orig): int(agg)
    for orig, agg in zip(basin[id_col].to_numpy(), basin[agg_col].to_numpy())
  }
  survivors = set(id_to_agg.values())

  def remap_one(down_id: object, self_agg: int) -> int:
    try:
      down = int(down_id)
    except (TypeError, ValueError):
      return outlet_value
    if _is_outlet_id(down, outlet_value):
      return outlet_value

    seen: set[int] = set()
    while down not in survivors:
      if down not in id_to_agg or down in seen:
        # Leaves the aggregated domain — treat as outlet.
        return outlet_value
      seen.add(down)
      down = id_to_agg[down]

    if down == self_agg:
      return outlet_value
    return down

  out = basin.copy()
  out[aggdown_col] = [
    remap_one(down, int(self_agg))
    for down, self_agg in zip(out[aggdown_col].to_numpy(), out[agg_col].to_numpy())
  ]
  return out


def _validate_topology(
  basins: gpd.GeoDataFrame,
  rivers: gpd.GeoDataFrame,
  id_col: str,
  down_col: str,
  outlet_value: int = OUTLET_VALUE,
) -> None:
  """Raise when any DSLINKNO points at a missing LINKNO / basin id."""
  basin_ids = set(basins[id_col].astype(int))
  river_ids = set(rivers[id_col].astype(int)) if id_col in rivers.columns else set()
  ids = basin_ids | river_ids
  outlet_value = int(outlet_value)

  def dangling(gdf: gpd.GeoDataFrame) -> list[int]:
    downs = set(int(d) for d in gdf[down_col].to_numpy())
    return sorted(
      d for d in downs
      if d not in ids and not _is_outlet_id(d, outlet_value)
    )

  bad_b = dangling(basins)
  bad_r = dangling(rivers)
  if bad_b or bad_r:
    raise ValueError(
      "Aggregated topology has DSLINKNO values with no matching LINKNO: "
      f"basins={bad_b[:10]}{'...' if len(bad_b) > 10 else ''}, "
      f"rivers={bad_r[:10]}{'...' if len(bad_r) > 10 else ''}. "
      "Downstream links were not fully remapped onto surviving aggregates."
    )


def prepare_input_tables(
  input_basin: gpd.GeoDataFrame,
  input_river: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
  """
  Validate, merge stream attributes into basins, and add derived area / flag columns.
  """
  basin = input_basin.copy()
  river = input_river.copy()

  _require_columns(basin, [BASIN_ID, "geometry"], "basin layer")
  _require_columns(
    river,
    [RIVER_ID, NEXT_DOWN_ID, SLOPE, LENGTH, UP_AREA],
    "river layer",
  )

  basin[BASIN_ID] = basin[BASIN_ID].astype(int)
  river[RIVER_ID] = river[RIVER_ID].astype(int)
  river[NEXT_DOWN_ID] = river[NEXT_DOWN_ID].fillna(OUTLET_VALUE).astype(int)

  join_cols = [
    c
    for c in river.columns
    if (c not in basin.columns or c == RIVER_ID) and c != "geometry"
  ]
  basin = basin.merge(
    river[join_cols].rename(columns={RIVER_ID: BASIN_ID}),
    on=BASIN_ID,
    how="left",
    suffixes=("", "_riv"),
  )

  if UNIT_AREA and UNIT_AREA in basin.columns:
    basin["_unitarea"] = _area_km2(basin[UNIT_AREA], AREA_SCALE)
  elif UNIT_AREA and UNIT_AREA in river.columns:
    basin["_unitarea"] = _area_km2(basin[UNIT_AREA], AREA_SCALE)
  else:
    basin["_unitarea"] = basin.geometry.area * AREA_SCALE

  if UP_AREA in basin.columns:
    basin["_uparea"] = _area_km2(basin[UP_AREA], AREA_SCALE)
  else:
    raise ValueError(
      f"Upstream area column '{UP_AREA}' not found after basin/river merge."
    )

  if LAKE_FLAG in basin.columns:
    basin["_lake_cat"] = pd.to_numeric(basin[LAKE_FLAG], errors="coerce").fillna(0)
  else:
    basin["_lake_cat"] = 0
    basin[LAKE_FLAG] = 0

  if LAKE_ID not in basin.columns:
    basin[LAKE_ID] = -1
  else:
    basin[LAKE_ID] = pd.to_numeric(basin[LAKE_ID], errors="coerce").fillna(-1).astype(int)

  if LAKE_AREA not in basin.columns:
    basin[LAKE_AREA] = 0.0
  else:
    basin[LAKE_AREA] = pd.to_numeric(basin[LAKE_AREA], errors="coerce").fillna(0.0)

  if FRAC_LAKE_AREA not in basin.columns:
    basin[FRAC_LAKE_AREA] = 0.0
  else:
    basin[FRAC_LAKE_AREA] = pd.to_numeric(basin[FRAC_LAKE_AREA], errors="coerce").fillna(0.0)

  if GAUGE_FLAG and GAUGE_FLAG in basin.columns:
    basin["_has_gauge"] = pd.to_numeric(basin[GAUGE_FLAG], errors="coerce").fillna(0)
  elif GAUGE_IDS in basin.columns:
    basin["_has_gauge"] = basin[GAUGE_IDS].fillna("").astype(str).str.strip().ne("").astype(int)
  else:
    basin["_has_gauge"] = 0
    basin[GAUGE_IDS] = ""

  if GAUGE_IDS in basin.columns:
    basin[GAUGE_IDS] = basin[GAUGE_IDS].fillna("").astype(str)

  river["_lengthkm"] = _length_km(river[LENGTH], LENGTH_SCALE)
  river["_uparea"] = _area_km2(river[UP_AREA], AREA_SCALE)

  return basin, river


# ==============================================================================
# INDEXED LOOKUPS (observationally equivalent to boolean masks)
# ==============================================================================
def _index_labels_by_value(series: pd.Series) -> dict[Any, list[Hashable]]:
  """Map each value to index *labels* in DataFrame index order."""
  out: dict[Any, list[Hashable]] = defaultdict(list)
  for lab, val in series.items():
    out[val].append(lab)
  return out


def _index_first_label(series: pd.Series) -> dict[Any, Hashable]:
  """Map each value to the first index label (same as ``df[df[col]==v].index[0]``)."""
  out: dict[Any, Hashable] = {}
  for lab, val in series.items():
    if val not in out:
      out[val] = lab
  return out


def _reassign_aggdown(
  basin: gpd.GeoDataFrame,
  aggdown_index: dict[Any, list[Hashable]],
  labels: list[Hashable],
  new_val: object,
) -> None:
  """Set ``aggdown`` on ``labels`` and keep ``aggdown_index`` in sync."""
  if not labels:
    return
  old_vals = basin.loc[labels, "aggdown"]
  for lab, old in old_vals.items():
    bucket = aggdown_index.get(old)
    if not bucket:
      continue
    try:
      bucket.remove(lab)
    except ValueError:
      pass
    if not bucket:
      del aggdown_index[old]
  basin.loc[labels, "aggdown"] = new_val
  aggdown_index[new_val].extend(labels)


def rule_H_indexed(
  basin: gpd.GeoDataFrame,
  xx_df: pd.DataFrame,
  outlet_value: int = OUTLET_VALUE,
) -> gpd.GeoDataFrame:
  """
  Absorb headwater groups listed in ``xx_df``.

  Equivalent to the original loop::

      basin.loc[basin["agg"] == aggold, "aggdown"] = new_aggdown
      basin.loc[basin["agg"] == aggold, "agg"] = new_agg

  Candidates in one Rule H pass are disjoint from each other's merge
  targets, so a snapshot-style move of each ``aggold`` group is exact.
  """
  agg_index = _index_labels_by_value(basin["agg"])
  for i in range(len(xx_df)):
    aggold = xx_df["aggold"].iloc[i]
    new_agg = xx_df["agg"].iloc[i]
    new_aggdown = xx_df["aggdown"].iloc[i]
    # Outlet sentinels are not basin IDs. Absorbing every coastal / missing
    # downstream into -9999 is what glued the map edge into one polygon.
    if pd.isna(new_agg) or _is_sentinel_object_id(new_agg, outlet_value):
      continue
    labels = agg_index.get(aggold)
    if not labels:
      continue
    labels = list(labels)
    basin.loc[labels, "aggdown"] = new_aggdown
    basin.loc[labels, "agg"] = new_agg
    if aggold != new_agg:
      agg_index[new_agg].extend(labels)
      del agg_index[aggold]
  return basin


def rule_I_indexed(
  basin: gpd.GeoDataFrame,
  small_subbasin: pd.DataFrame,
  id_col: str,
  down_col: str,
  min_sub_area: float,
  outlet_value: int = OUTLET_VALUE,
) -> gpd.GeoDataFrame:
  """
  Absorb internal (non-headwater) small groups, preserving the original
  per-statement read-after-write sequence.

  Original body (one candidate per iteration)::

      xx = basin[basin[id_col] == cand].index[0]
      if sum(_unitarea where agg == basin.loc[xx, agg]) < min_sub_area:
          xy = basin[basin[down_col] == basin.loc[xx, id_col]].index
          xz = xy rows with max _uparea
          if Mask[xz] < 2:
              zz = rows where aggdown == basin.loc[xz, agg]   # BEFORE writes
              # line 1: rows where agg == xz.agg  -> agg = xx.agg
              # line 2: rows where agg == xz.agg  -> aggdown = xx.aggdown
              #         (xz.agg is re-read AFTER line 1)
              # zz.aggdown = xx.agg

  ``id_col`` / ``down_col`` never change inside the while loop; ``agg`` and
  ``aggdown`` indexes are updated after every write so later candidates in
  this same ``for`` see the mutated state.
  """
  id_first = _index_first_label(basin[id_col])
  down_index = _index_labels_by_value(basin[down_col])
  agg_index = _index_labels_by_value(basin["agg"])
  aggdown_index = _index_labels_by_value(basin["aggdown"])

  for i in range(len(small_subbasin)):
    cand = small_subbasin["agg"].iloc[i]
    if cand not in id_first:
      raise IndexError(
        f"Rule I: no row with {id_col}=={cand!r} (matches original .index[0] failure)"
      )
    xx = id_first[cand]
    xx_agg = basin.at[xx, "agg"]
    if pd.isna(xx_agg) or _is_sentinel_object_id(xx_agg, outlet_value):
      continue
    group_xx = agg_index.get(xx_agg, [])
    group_area = (
      float(basin.loc[group_xx, "_unitarea"].sort_index().sum()) if group_xx else 0.0
    )
    if group_area >= min_sub_area:
      continue

    xx_id = basin.at[xx, id_col]
    xy = down_index.get(xx_id)
    if not xy:
      continue

    xz = basin.loc[xy, "_uparea"].idxmax()
    if not (basin.at[xz, "Mask"] < 2):
      continue

    xz_agg_before = basin.at[xz, "agg"]
    zz_labels = list(aggdown_index.get(xz_agg_before, ()))

    xx_aggdown = basin.at[xx, "aggdown"]

    # Line 1 — mask uses xz.agg *before* the write; then agg changes.
    pos1 = list(agg_index.get(xz_agg_before, ()))
    if pos1:
      basin.loc[pos1, "agg"] = xx_agg
      if xz_agg_before != xx_agg:
        agg_index[xx_agg].extend(pos1)
        del agg_index[xz_agg_before]

    # Line 2 — mask uses xz.agg *after* line 1 (union of old xz group and
    # whoever already had agg == xx_agg). Same as the original loc chain.
    xz_agg_after = basin.at[xz, "agg"]
    pos2 = list(agg_index.get(xz_agg_after, ()))
    _reassign_aggdown(basin, aggdown_index, pos2, xx_aggdown)

    # zz was snapshotted before either write; only aggdown changes.
    if zz_labels:
      _reassign_aggdown(basin, aggdown_index, zz_labels, xx_agg)

  return basin


def _rebuild_agg_basin(
  basin: gpd.GeoDataFrame,
  id_col: str,
  down_col: str,
  drop_small_outlets,
) -> pd.DataFrame:
  """Rebuild the working table. Same keys/sums as the original groupby."""
  agg_basin = (
    basin[["agg", "aggdown", "_unitarea"]]
    .groupby(["agg", "aggdown"], as_index=False)
    .agg({"_unitarea": "sum"})
  )
  agg_basin = agg_basin.rename(columns={"agg": id_col, "aggdown": down_col})
  agg_basin = agg_basin.merge(basin[[id_col, "_uparea", "Mask"]], on=id_col, how="left")
  agg_basin = agg_basin.rename(columns={id_col: "agg", down_col: "aggdown"})
  return drop_small_outlets(agg_basin)


def _mark_main_stems(
  agg_river: gpd.GeoDataFrame,
  down_col: str,
  riv_id_col: str,
) -> gpd.GeoDataFrame:
  """
  Pick the highest-_uparea path inside each aggregate.

  Equivalent to the original per-agg boolean scan; ``down_col`` is static
  so its inverted index is built once.
  """
  agg_river = agg_river.copy()
  agg_river["mask"] = 0

  down_index = _index_labels_by_value(agg_river[down_col])
  groups: dict[Any, list[Hashable]] = defaultdict(list)
  # ``unique()`` is first-appearance order; groups are independent.
  for lab, agg_id in agg_river["agg"].items():
    if pd.isna(agg_id):
      continue
    groups[agg_id].append(lab)

  for _agg_id, _members in groups.items():
    xx = _members
    visited = set()
    while True:
      yy = agg_river.loc[xx, "_uparea"].idxmax()
      if yy in visited:
          break
      visited.add(yy)
      agg_river.at[yy, "mask"] = 1
      dest = agg_river.at[yy, riv_id_col]
      downstream = down_index.get(dest)
      if not downstream:
          break
      xx = downstream

  return agg_river


# ==============================================================================
# CORE AGGREGATION (logic preserved from 01-pre-process-geospatial-fabric.ipynb)
# ==============================================================================
def basin_aggregation(
  input_basin: gpd.GeoDataFrame,
  input_river: gpd.GeoDataFrame,
  min_sub_area: float,
  min_riv_slope: float,
  min_riv_length: float,
  outlet_value: int = OUTLET_VALUE,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
  """
  Aggregate basins and rivers based on drainage area, slope, and reservoir masking.

  Returns aggregated basin and river GeoDataFrames. Each basin is identified by
  BASIN_ID. Gauge IDs, lake flags/IDs, lake area, and fractional lake area are
  pour-point attributes only.

  Terminal basins receive ``outlet_value`` in ``DSLINKNO`` (default ``OUTLET_VALUE``).
  """
  outlet_value = int(outlet_value)
  basin, river = prepare_input_tables(input_basin, input_river)
  area_guard = float(basin["_unitarea"].sum())

  id_col = BASIN_ID
  down_col = NEXT_DOWN_ID
  riv_id_col = RIVER_ID

  river[SLOPE] = river[SLOPE].clip(lower=min_riv_slope)
  river.loc[river[SLOPE] >= 1.0, SLOPE] = min_riv_slope
  river["_lengthkm"] = river["_lengthkm"].clip(lower=min_riv_length)

  # Normalize any legacy outlet markers (-1, 0, ...) to the configured sentinel.
  river.loc[
    river[down_col].map(lambda d: _is_outlet_id(d, outlet_value)),
    down_col,
  ] = outlet_value

  basin["Mask"] = 0
  basin.loc[basin[down_col].map(lambda d: _is_outlet_id(d, outlet_value)), "Mask"] = 1
  basin.loc[basin["_has_gauge"] > 0, "Mask"] = 2
  basin.loc[basin["_lake_cat"] > 0, "Mask"] = 3
  basin["agg"] = basin[id_col]
  basin["aggdown"] = basin[down_col]
  basin.loc[
    basin["aggdown"].map(lambda d: _is_outlet_id(d, outlet_value)),
    "aggdown",
  ] = outlet_value

  def _drop_small_outlets(df: pd.DataFrame) -> pd.DataFrame:
    is_outlet = df["aggdown"].map(lambda d: _is_outlet_id(d, outlet_value))
    return df[~(is_outlet & (df["_uparea"] < min_sub_area) | (df["Mask"] == 3))]

  agg_basin = basin[["agg", "aggdown", "_unitarea", "_uparea", "Mask"]].copy()
  agg_basin = _drop_small_outlets(agg_basin)
  lake_subs = basin[basin["Mask"] == 3]["agg"]
  no_subbasin = len(basin)

  while True:
    headwaters = (
      ~agg_basin["agg"].isin(agg_basin["aggdown"])
      & (agg_basin["_unitarea"] < min_sub_area)
      & (agg_basin["Mask"] < 2)
    )
    small_subbasin = agg_basin[headwaters]
    small_subbasin = small_subbasin[~small_subbasin["aggdown"].isin(lake_subs)].sort_values(
      by="_uparea", ascending=False
    )
    if not small_subbasin.empty:
      small_subbasin = small_subbasin.rename(columns={"agg": "aggold", "aggdown": "agg"})
      xx = small_subbasin.merge(agg_basin[["agg", "aggdown"]], on="agg", how="left")
      basin = rule_H_indexed(basin, xx, outlet_value=outlet_value)
      agg_basin = _rebuild_agg_basin(basin, id_col, down_col, _drop_small_outlets)

    condition = (
      agg_basin["agg"].isin(agg_basin["aggdown"])
      & (agg_basin["_unitarea"] < min_sub_area)
      & (agg_basin["Mask"] != 3)
    )
    small_subbasin = agg_basin[condition].sort_values(by="_uparea", ascending=False)
    if not small_subbasin.empty:
      basin = rule_I_indexed(
        basin,
        small_subbasin,
        id_col=id_col,
        down_col=down_col,
        min_sub_area=min_sub_area,
        outlet_value=outlet_value,
      )
      agg_basin = _rebuild_agg_basin(basin, id_col, down_col, _drop_small_outlets)

    if len(agg_basin[agg_basin["_unitarea"] < min_sub_area]) == no_subbasin:
      break
    no_subbasin = len(agg_basin[agg_basin["_unitarea"] < min_sub_area])

  # Never dissolve on the outlet sentinel: every coastal unit that drained
  # to -9999 / <=0 would become one polygon. Split those rows back to self.
  sentinel_group = basin["agg"].map(lambda a: _is_sentinel_object_id(a, outlet_value))
  if sentinel_group.any():
    n_split = int(sentinel_group.sum())
    print(
      f"Warning: {n_split} unit(s) had aggregate id equal to the outlet "
      f"sentinel; leaving them unaggregated instead of dissolving as one."
    )
    basin.loc[sentinel_group, "agg"] = basin.loc[sentinel_group, id_col]

  # Lakes / gauges keep their original downstream ids; remap those onto the
  # surviving aggregate that absorbed each missing target.
  basin = _remap_aggdown_to_survivors(
    basin, id_col=id_col, outlet_value=outlet_value
  )
  pour_down = _pour_point_down(basin, id_col)
  basin = basin.drop(columns=["aggdown"]).merge(pour_down, on="agg", how="left")
  basin["aggdown"] = basin["aggdown"].where(
    basin["aggdown"].notna(), outlet_value
  )

  agg_basin = basin.dissolve(by="agg", aggfunc={"_unitarea": "sum"}, as_index=False).rename(
    columns={"agg": id_col}
  )

  dissolved_area = float(agg_basin["_unitarea"].sum())
  if abs(dissolved_area - area_guard) > 1e-6 * max(1.0, abs(area_guard)):
    raise ValueError(
      f"Unit-area sum changed during aggregation: {area_guard} -> {dissolved_area}"
    )

  # Carry pour-point attributes; object id remains BASIN_ID
  keep_cols = [id_col, "aggdown", "_uparea"] + _basin_attr_cols(basin)
  keep_cols = list(dict.fromkeys(keep_cols))
  agg_basin = agg_basin.merge(
    basin[keep_cols].copy(),
    on=id_col,
    how="left",
  ).rename(columns={"aggdown": down_col})

  if id_col == riv_id_col:
    agg_river = river.merge(basin[[id_col, "agg"]].copy(), on=riv_id_col, how="left")
  else:
    agg_river = river.merge(
      basin[[id_col, "agg"]].copy(),
      left_on=riv_id_col,
      right_on=id_col,
      how="left",
    )
  agg_river = _mark_main_stems(agg_river, down_col=down_col, riv_id_col=riv_id_col)
  agg_river = agg_river[agg_river["mask"] == 1].copy()
  agg_river["_slope_weighted"] = agg_river[SLOPE] * agg_river["_lengthkm"]

  agg_river = agg_river.dissolve(
    by="agg",
    aggfunc={"_lengthkm": "sum", "_slope_weighted": "sum"},
    as_index=False,
  ).rename(columns={"agg": riv_id_col})

  agg_river[SLOPE] = agg_river["_slope_weighted"] / agg_river["_lengthkm"].replace(0, np.nan)
  basin_topo = agg_basin[[id_col, down_col, "_uparea"]].copy()
  if id_col != riv_id_col:
    basin_topo = basin_topo.rename(columns={id_col: riv_id_col})
  agg_river = agg_river.merge(basin_topo, on=riv_id_col, how="left")

  extra_river_cols = [riv_id_col]
  if STREAM_ORDER in river.columns:
    extra_river_cols.append(STREAM_ORDER)
  if HILLSLOPE and HILLSLOPE in river.columns:
    extra_river_cols.append(HILLSLOPE)
  agg_river = agg_river.merge(river[extra_river_cols].copy(), on=riv_id_col, how="left")

  unit_out = UNIT_AREA or "unit_a_km2"
  agg_basin = agg_basin.rename(columns={"_unitarea": unit_out, "_uparea": UP_AREA})
  agg_river[LENGTH] = agg_river["_lengthkm"]
  agg_river[UP_AREA] = agg_river["_uparea"]
  drop_riv = ["_lengthkm", "_uparea", "_slope_weighted", "mask"]
  if id_col != riv_id_col and id_col in agg_river.columns:
    drop_riv.append(id_col)
  agg_river = agg_river.drop(columns=drop_riv, errors="ignore")

  # Object IDs must stay real basin/reach IDs. Only DSLINKNO may be the sentinel.
  sentinel_id = agg_basin[id_col].map(lambda a: _is_sentinel_object_id(a, outlet_value))
  if sentinel_id.any():
    print(
      f"Warning: dropping {int(sentinel_id.sum())} dissolved basin(s) "
      "whose LINKNO is an outlet sentinel."
    )
    agg_basin = agg_basin.loc[~sentinel_id].copy()

  agg_basin[id_col] = _to_int64_link_ids(
    agg_basin[id_col], outlet_value, f"basins.{id_col}", fill_outlets=False
  )
  agg_basin[down_col] = _to_int64_link_ids(
    agg_basin[down_col], outlet_value, f"basins.{down_col}", fill_outlets=True
  )
  agg_river[riv_id_col] = _to_int64_link_ids(
    agg_river[riv_id_col], outlet_value, f"rivers.{riv_id_col}", fill_outlets=False
  )
  agg_river[down_col] = _to_int64_link_ids(
    agg_river[down_col], outlet_value, f"rivers.{down_col}", fill_outlets=True
  )

  # Basin object ids (DN) match river LINKNO values after aggregation; expose a
  # single LINKNO key on both layers for MESH / topology tools.
  if id_col != riv_id_col:
    agg_basin = agg_basin.rename(columns={id_col: riv_id_col})
    id_col = riv_id_col

  _validate_topology(
    agg_basin,
    agg_river,
    id_col=id_col,
    down_col=down_col,
    outlet_value=outlet_value,
  )

  return agg_basin, agg_river


def run_aggregation(
  basins_path: str = INPUT_BASINS,
  rivers_path: str = INPUT_RIVERS,
  output_basins_path: str = OUTPUT_BASINS,
  output_rivers_path: str = OUTPUT_RIVERS,
  min_sub_area: float = MIN_SUB_AREA,
  min_riv_slope: float = MIN_RIV_SLOPE,
  min_riv_length: float = MIN_RIV_LENGTH,
  outlet_value: int = OUTLET_VALUE,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
  """Load cleanGeofabric outputs, aggregate, and write shapefiles."""
  print(f"Loading basins: {basins_path}")
  basins = gpd.read_file(basins_path)
  print(f"Loading rivers: {rivers_path}")
  rivers = gpd.read_file(rivers_path)

  agg_basins, agg_rivers = basin_aggregation(
    basins,
    rivers,
    min_sub_area,
    min_riv_slope,
    min_riv_length,
    outlet_value=outlet_value,
  )

  os.makedirs(os.path.dirname(output_basins_path) or ".", exist_ok=True)
  os.makedirs(os.path.dirname(output_rivers_path) or ".", exist_ok=True)

  print(f"Writing aggregated basins ({len(agg_basins)} features): {output_basins_path}")
  agg_basins.to_file(output_basins_path, driver="ESRI Shapefile")
  print(f"Writing aggregated rivers ({len(agg_rivers)} features): {output_rivers_path}")
  agg_rivers.to_file(output_rivers_path, driver="ESRI Shapefile")

  return agg_basins, agg_rivers


if __name__ == "__main__":
  run_aggregation()
