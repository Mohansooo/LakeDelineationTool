"""
Merge lake-split subbasins and internal stream links into unified reservoir units.

Called by tau-dem-delineation-srun.slurm after TauDEM Pass 3 and before
cleanGeofabric.py.

Problem being solved
--------------------
TauDEM Pass 3 delineates one subbasin per stream link (DN = LINKNO). When a
reservoir sits on a river, the network is fragmented: many short links and
small polygons lie inside or across the lake. This script collapses those
internal pieces into a single merged reservoir link and a single reservoir
catchment polygon per lake.

METHOD BREAKDOWN
----------------

A. POUR-POINT TO STREAM LINK MATCHING
   get_validated_node_connection()
     Snaps each inflow/outflow pour point to the nearest stream LINKNO by
     comparing distances to link endpoints inside a search buffer.

B. OUTLET SELECTION (when a lake has multiple outflows)
   build_outflow_candidates()          -> one candidate dict per outflow point
   filter_upstream_duplicate_outflows()  -> drop upstream exits on same branch
   select_winning_outflow_link()         -> override snap OR algorithmic rank
   rank_algorithmic_outflow()            -> gauge -> doutend -> strmOrder -> area

   Uses vector attributes only (no rasters). Ranking priority matches
   rasterFlowpathEdit.py, but strmOrder replaces raster local_accum.

C. INTERNAL LINK IDENTIFICATION ("swallow" set)
   collect_internal_links()
     trace_downstream_links() from inflow pour points
     + geometry tests on stream segments intersecting the lake polygon
   Internal links are dissolved into one link per lake; their subbasins are
   unioned into one reservoir polygon.

D. STREAM TOPOLOGY REWIRING & HYDROMETRIC AGGREGATION
   dissolve() internal links by merged_ID (winning outlet LINKNO)
   compute_lake_path_metrics() traces the longest inflow-to-outlet path through
     each lake and recalculates Length, strmDrop, StraightL, DOUTEND/START/MID,
     and Slope for the merged link
   Rewire DSLINKNO: walk original down_map off the swallowed set; never
     write W→X and X→W (or W→W). Domain exit is -1 only.

E. BASIN FABRIC ASSEMBLY
   Remove swallowed subbasin polygons; append new reservoir catchment polygons
   Merged basin/stream primary ID remains the winning outlet LINKNO (DN / LINKNO)
   export_shapefile() -> merged_basins/reservoirBasins.shp, reservoirStreams.shp

Inputs
------
  delineation-product/final-delineated-watersheds.shp  (TauDEM Pass 3 basins)
  delineation-product/final-delineated-streams.shp      (TauDEM Pass 3 streams)
  lakes/filtered_lakes.shp                              (reservoir polygons)
  taudem-interim-files/final/snapped-outlets.shp        (lake in/outflow points)
  points/gauges_in_basin.shp
  outlet_overrides.csv (optional; shared with rasterFlowpathEdit.py)

Outputs
-------
  merged_basins/reservoirBasins.shp   -> basins with lake units merged in
  merged_basins/reservoirStreams.shp  -> streams with internal lake links dissolved
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from rasterFlowpathEdit import load_overrides
from collections import Counter

# ==============================================================================
# CONFIGURATION
# ==============================================================================
GAUGE_SEARCH_RADIUS = 750       # meters; max distance to count a gauge as "nearby"
MIN_INTERNAL_STREAM_LEN = 180   # meters; stream-lake overlap length that triggers swallow
OVERRIDES_CSV = "outlet_overrides.csv"
OUTPUT_DIR = "merged_basins"

PATHS = {
    "basins": "delineation-product/final-delineated-watersheds.shp",
    "streams": "delineation-product/final-delineated-streams.shp",
    "lakes": "lakes/filtered_lakes.shp",
    "intersection": "taudem-interim-files/final/snapped-outlets.shp",
    "gauges": "points/gauges_in_basin.shp",
}


# ==============================================================================
# A. POUR-POINT TO STREAM LINK MATCHING
# ==============================================================================
def get_validated_node_connection(point_geom, streams_gdf, search_dist, mode="inflow"):
    """
    Snap a pour point to the nearest stream LINKNO at a lake junction.

    At each inflow/outflow point two links meet. Distances are measured to
    coords[-1] (d_start) and coords[0] (d_end). The link on the upstream side
    of the junction (potential_down) is returned for inflow; the link on the
    downstream side (potential_up) is returned for outflow.
    """
    candidates = streams_gdf[streams_gdf.intersects(point_geom.buffer(search_dist))].copy()
    if candidates.empty:
        return None

    def get_node_dists(row):
        coords = list(row.geometry.coords)
        return pd.Series(
            [point_geom.distance(Point(coords[-1])), point_geom.distance(Point(coords[0]))],
            index=["d_start", "d_end"],
        )

    dists = candidates.apply(get_node_dists, axis=1)
    candidates = pd.concat([candidates, dists], axis=1)

    potential_up = candidates.sort_values("d_end").iloc[0]
    potential_down = candidates.sort_values("d_start").iloc[0]

    return int(potential_down["LINKNO"]) if mode == "inflow" else int(potential_up["LINKNO"])


def parse_lake_id_column(intersection):
    """
    Extract numeric Hylak_id from snapped-outlets attribute table.

    pourPointsPass2.py names points 'Lake_<id>'; TauDEM may preserve that in
    the 'name' column. Falls back to an existing lake_id column if present.
    """
    if "name" in intersection.columns:
        return (
            intersection["name"]
            .str.extract(r"Lake_(\d+)")
            .astype(float)
            .fillna(-1)
            .astype(int)
        )
    return pd.to_numeric(intersection["lake_id"], errors="coerce").fillna(-1).astype(int)


def estimate_search_buffer(basins):
    """
    Estimate the pour-point-to-stream search radius from basin polygon geometry.

    Uses the smallest coordinate step on the first basin polygon as a proxy
    for DEM cell size, then returns 75% of that value. This keeps the search
    buffer proportional to grid resolution without reading a raster.
    """
    coords = np.array(basins.geometry.iloc[0].exterior.coords)
    diffs = np.abs(np.diff(coords, axis=0))
    diffs = diffs[diffs > 0]
    dem_res = np.min(diffs) if len(diffs) else 1.0
    return dem_res * 0.75


# ==============================================================================
# B. OUTLET SELECTION
# ==============================================================================
def build_stream_lookup_tables(streams):
    """
    Pre-build dictionaries keyed by LINKNO for fast attribute lookups.

    Returns:
      link_to_dout        - DOUTEND: downstream distance to basin pour point
      link_to_accum       - DSContArea: downstream contributing area of the link
      link_to_downstream  - DSLINKNO: next link downstream in the network
      link_to_strmorder   - strmOrder: TauDEM stream order (local strength proxy)
    """
    by_link = streams.set_index("LINKNO")
    return (
        by_link["DOUTEND"].to_dict(),
        by_link["DSContArea"].to_dict(),
        by_link["DSLINKNO"].to_dict(),
        by_link["strmOrder"].to_dict(),
    )


def is_link_upstream_of(link_a, link_b, link_to_downstream):
    """
    Return True if link_a is upstream of link_b on the same stream branch.

    Walks the DSLINKNO chain from link_a until the basin outlet (-1) is reached.
    Used to eliminate upstream duplicate outflow candidates.
    """
    nxt = link_to_downstream.get(link_a, -1)
    while nxt not in (-1, 0):
        if nxt == link_b:
            return True
        nxt = link_to_downstream.get(nxt, -1)
    return False


def build_outflow_candidates(out_pts, streams, buffer_dist, link_to_dout, link_to_accum, link_to_strmorder, gauges):
    """
    Build one ranking candidate per lake outflow pour point.

    Each candidate is a dict with vector attributes used by the outlet picker:
      link_no     - stream link snapped to this outflow point
      point       - pour point geometry
      doutend     - link DOUTEND (prefer exits closer to basin edge)
      dscontarea  - link DSContArea (prefer larger catchments)
      strm_order  - link strmOrder (prefer higher-order / stronger streams)
      gauge_dist  - distance to nearest gauge within GAUGE_SEARCH_RADIUS
    """
    candidates = []
    for _, row in out_pts.iterrows():
        point = row.geometry
        link_no = get_validated_node_connection(point, streams, buffer_dist, "outflow")
        if link_no is None:
            continue

        gauge_distances = gauges.distance(point)
        valid_gauges = gauge_distances[gauge_distances <= GAUGE_SEARCH_RADIUS]
        candidates.append({
            "link_no": link_no,
            "point": point,
            "doutend": link_to_dout.get(link_no, float("inf")),
            "dscontarea": link_to_accum.get(link_no, 0),
            "strm_order": link_to_strmorder.get(link_no, 0),
            "gauge_dist": valid_gauges.min() if not valid_gauges.empty else float("inf"),
        })
    return candidates


def filter_upstream_duplicate_outflows(candidates, link_to_downstream):
    """
    Remove upstream duplicate exits, keeping the furthest-downstream per branch.

    A candidate is dropped if another candidate:
      - is downstream of it on the same network (link_a flows into link_b), OR
      - shares the same link_no but has higher strm_order (tie-break)

    This mirrors rasterFlowpathEdit.filter_upstream_duplicates(), but uses
    strmOrder instead of raster accumulation for same-link ties.
    """
    surviving = []
    for i, cand_a in enumerate(candidates):
        link_a = cand_a["link_no"]
        is_duplicate = False
        for j, cand_b in enumerate(candidates):
            if i == j:
                continue
            link_b = cand_b["link_no"]
            if link_a == -1 or link_b == -1:
                continue
            if is_link_upstream_of(link_a, link_b, link_to_downstream):
                is_duplicate = True
                break
            if link_a == link_b and cand_b["strm_order"] > cand_a["strm_order"]:
                is_duplicate = True
                break
        if not is_duplicate:
            surviving.append(cand_a)
    return surviving


def rank_algorithmic_outflow(candidates):
    """
    Pick the best outflow candidate by sorted vector ranking.

    Priority (lower sort key wins; negated fields mean higher value wins):
      1. gauge_dist   - prefer outflows nearest a stream gauge
      2. doutend      - prefer exits closest to the basin pour point
      3. strm_order   - prefer higher TauDEM stream order
      4. dscontarea   - prefer larger downstream contributing area
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return sorted(candidates, key=lambda x: (
        x["gauge_dist"],
        x["doutend"],
        -x["strm_order"],
        -x["dscontarea"],
    ))[0]


def select_winning_outflow_link(lake_id, surviving_candidates, overrides_gdf, cell_size):
    """
    Choose the single winning outlet LINKNO for a lake.

    Tier 1 - Manual override (outlet_overrides.csv):
      If the override coordinate falls within 3 DEM cells of a surviving
      candidate pour point, snap to that candidate's link.

    Tier 2 - Algorithmic ranking:
      If no override snap (or no override for this lake), call
      rank_algorithmic_outflow() on the surviving candidates.

    Returns the winning LINKNO, or None if no valid choice exists.
    """
    lake_override = overrides_gdf[overrides_gdf["lake_id"] == str(lake_id)]
    if not lake_override.empty:
        override_pt = lake_override.iloc[0].geometry
        snap_threshold = cell_size * 3.0
        nearby = sorted(
            (c for c in surviving_candidates if c["point"].distance(override_pt) <= snap_threshold),
            key=lambda c: c["point"].distance(override_pt),
        )
        if nearby:
            print(f"Lake {lake_id}: Override snapped to outflow link {nearby[0]['link_no']}.")
            return nearby[0]["link_no"]
        print(f"Lake {lake_id}: Override did not snap to an outflow candidate; using algorithmic selection.")

    chosen = rank_algorithmic_outflow(surviving_candidates)
    return None if chosen is None else chosen["link_no"]


def select_winner_from_lake_geometry(lake_geom, streams):
    """
    When snapped-outlets has only inflow points, pick the stream link that
    leaves the lake polygon. Prefer a downstream tip outside the lake, then
    higher strmOrder, then larger DSContArea.
    """
    hits = streams[streams.intersects(lake_geom)].copy()
    if hits.empty:
        return None

    def exits_lake(row):
        coords = list(row.geometry.coords)
        if len(coords) < 2:
            return False
        return not lake_geom.contains(Point(coords[0]))

    hits = hits.copy()
    hits["_exits"] = hits.apply(exits_lake, axis=1)
    sort_cols = [c for c in ["_exits", "strmOrder", "DSContArea"] if c in hits.columns]
    hits = hits.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    return int(hits.iloc[0]["LINKNO"])


# ==============================================================================
# C. INTERNAL LINK IDENTIFICATION
# ==============================================================================
def trace_downstream_links(start_link, down_map, stop_at):
    """
    Walk downstream from start_link, collecting every link in the chain.

    Stops when:
      - DSLINKNO is -1 (basin outlet)
      - the next link is already in stop_at (typically an outflow candidate link)
      - a cycle is detected

    Returns the set of link IDs traced.
    """
    collected = set()
    curr = start_link
    while curr and curr != -1 and curr not in collected:
        collected.add(curr)
        if curr in stop_at:
            break
        curr = down_map.get(curr, -1)
    return collected


def collect_internal_links(lake_geom, in_pts, raw_outflow_candidates, streams, buffer_dist, down_map, all_downstream_ids):
    """
    Identify all stream links that should be swallowed into the reservoir unit.

    Two complementary methods:

    METHOD 1 - Inflow tracing
      For each inflow pour point, snap to a link and trace downstream until
      an outflow candidate link is reached. All links on that path are internal.

    METHOD 2 - Geometric intersection
      For each stream segment intersecting the lake polygon, apply three tests:
        a) Midpoint of the segment lies inside the lake
        b) Overlap length with lake > MIN_INTERNAL_STREAM_LEN (180 m)
        c) Both segment endpoints lie inside the lake
      If any test passes (and the link is not a terminal network link), trace
      downstream from that segment until an outflow candidate is reached.

    Links in all_downstream_ids (links that are someone's DSLINKNO) are excluded
    from Method 2 to avoid swallowing the main downstream trunk incorrectly.
    """
    internal = set()

    # Method 1: trace from inflow pour points
    for _, pt_row in in_pts.iterrows():
        start = get_validated_node_connection(pt_row.geometry, streams, buffer_dist, "inflow")
        internal |= trace_downstream_links(start, down_map, raw_outflow_candidates)

    # Method 2: geometric intersection with lake polygon
    for _, row in streams[streams.intersects(lake_geom)].iterrows():
        sid = int(row["LINKNO"])
        overlap_len = row.geometry.intersection(lake_geom).length
        mid_inside = lake_geom.contains(row.geometry.interpolate(0.5, normalized=True))
        coords = list(row.geometry.coords)
        both_tips_inside = lake_geom.contains(Point(coords[0])) and lake_geom.contains(Point(coords[-1]))
        should_swallow = mid_inside or (overlap_len > MIN_INTERNAL_STREAM_LEN) or both_tips_inside

        if sid in all_downstream_ids or sid in internal or not should_swallow:
            continue
        internal |= trace_downstream_links(sid, down_map, raw_outflow_candidates)

    return internal


def path_reenters_set(start_id, down_map, internal):
    """True if walking original DSLINKNO from start_id hits ``internal`` before outlet."""
    seen = set()
    x = start_id
    while not _is_network_outlet(x):
        try:
            x = int(x)
        except (TypeError, ValueError):
            return False
        if x in seen:
            return False
        seen.add(x)
        if x in internal:
            return True
        x = down_map.get(x, -1)
    return False


def close_internal_along_winner_path(winner_id, internal, down_map):
    """
    Add sliver links that sit on the original path between the winner and
    other links already swallowed into the same lake.

    Diagnostic pattern: W=478 → 8285 (not swallowed) → 8349 (swallowed into W).
    8285 must join the lake unit so the reservoir polygon actually contains
    the outlet path and W does not point at a leftover sliver that then
    points back at W.
    """
    closed = set(int(x) for x in internal if x is not None)
    try:
        winner_id = int(winner_id)
    except (TypeError, ValueError):
        return closed
    closed.add(winner_id)
    x = down_map.get(winner_id, -1)
    seen = set()
    added = []
    while not _is_network_outlet(x):
        try:
            x = int(x)
        except (TypeError, ValueError):
            break
        if x in seen:
            break
        seen.add(x)
        if x in closed:
            x = down_map.get(x, -1)
            continue
        if path_reenters_set(down_map.get(x, -1), down_map, closed):
            closed.add(x)
            added.append(x)
            x = down_map.get(x, -1)
            continue
        break
    if added:
        print(
            f"Lake winner {winner_id}: swallowed {len(added)} outlet-path sliver(s) "
            f"{added[:12]}{'...' if len(added) > 12 else ''}"
        )
    return closed


# ==============================================================================
# D. STREAM TOPOLOGY REWIRING & HYDROMETRIC AGGREGATION
# ==============================================================================
def compute_lake_path_metrics(links, target_exit, streams_work, down_map):
    """
    Trace the longest inflow-to-outlet path through internal lake links and
    derive hydrometric attributes for the merged reservoir segment.

    For each inflow headwater (USLINKNO1 outside the internal set), walk
    downstream through internal links until the winning outlet or the set
    boundary. The path with the greatest summed Length wins; its metrics
    replace the dissolved link attributes:

      Length    - sum of segment lengths along the path
      strmDrop  - sum of segment strmDrop values
      StraightL - straight-line distance from upstream end (coords[-1]) of the first
                  segment to downstream end (coords[0]) of the last segment
      DOUTEND   - minimum DOUTEND among path segments (most downstream point)
      DOUTSTART - DOUTEND + Length
      DOUTMID   - average of DOUTSTART and DOUTEND
      Slope     - strmDrop / Length (computed after assignment)
    """
    internal_set = set(links)
    inflows = [
        link for link in links
        if streams_work.loc[streams_work["LINKNO"] == link, "USLINKNO1"].values[0] not in internal_set
    ]

    max_path_len = 0.0
    best_path_metrics = {
        "Length": 0.0,
        "strmDrop": 0.0,
        "StraightL": 0.0,
        "DOUTEND": 0.0,
        "DOUTSTART": 0.0,
        "DOUTMID": 0.0,
    }

    for start in inflows:
        curr = start
        path_segments = []

        while curr in internal_set:
            row_data = streams_work.loc[streams_work["LINKNO"] == curr]
            if row_data.empty:
                break
            path_segments.append(row_data.iloc[0])
            if curr == target_exit:
                break
            curr = down_map.get(curr, -1)

        if not path_segments:
            continue

        path_length = sum(seg["Length"] for seg in path_segments)
        path_drop = (
            sum(seg["strmDrop"] for seg in path_segments)
            if "strmDrop" in path_segments[0]
            else 0.0
        )

        start_geom = path_segments[0].geometry
        end_geom = path_segments[-1].geometry
        path_straight = Point(start_geom.coords[-1]).distance(Point(end_geom.coords[0]))

        path_dout_end = (
            min(seg["DOUTEND"] for seg in path_segments)
            if "DOUTEND" in path_segments[0]
            else 0.0
        )
        path_dout_start = path_dout_end + path_length
        path_dout_mid = (path_dout_start + path_dout_end) / 2.0

        if path_length > max_path_len:
            max_path_len = path_length
            best_path_metrics = {
                "Length": path_length,
                "strmDrop": path_drop,
                "StraightL": path_straight,
                "DOUTEND": path_dout_end,
                "DOUTSTART": path_dout_start,
                "DOUTMID": path_dout_mid,
            }

    return best_path_metrics


def build_stream_agg_logic(streams):
    """Default dissolve aggregation: max for all attributes except Length."""
    agg_logic = {
        col: "max"
        for col in streams.columns
        if col not in ["geometry", "merged_ID", "LINKNO", "Length"]
    }
    agg_logic.update({
        "Length": "max",
        "DSContArea": "max",
        "USContArea": "max",
        "Magnitude": "max",
        "strmOrder": "max",
    })
    return agg_logic


def _is_network_outlet(link_id):
    """TauDEM terminal marker: missing, -1, or 0."""
    if link_id is None or (isinstance(link_id, float) and pd.isna(link_id)):
        return True
    try:
        return int(link_id) <= 0
    except (TypeError, ValueError):
        return True


def first_surviving_downstream(start_id, down_map, swallowed_map, self_id):
    """
    Walk the *original* DSLINKNO chain from start_id until a link that still
    exists after dissolve.

    A link exists after dissolve when it was never swallowed, or when it is
    a lake winner (swallowed_map[id] == id). Links swallowed into self_id are
    skipped. Another lake's winner is a valid exit.
    """
    if _is_network_outlet(start_id):
        return -1
    seen = set()
    x = int(start_id)
    while not _is_network_outlet(x):
        if x in seen:
            return -1
        seen.add(x)
        mapped = swallowed_map.get(x)
        if mapped is None:
            return int(x)
        mapped = int(mapped)
        if mapped != int(self_id):
            return mapped
        x = down_map.get(x, -1)
        if _is_network_outlet(x):
            return -1
        try:
            x = int(x)
        except (TypeError, ValueError):
            return -1
    return -1


def rewire_dissolved_topology(streams_dissolved, down_map, swallowed_map, lake_to_winner):
    """
    Set DSLINKNO after lake dissolve without creating self-loops or 2-cycles.

    1. Each winner W points at the first original-downstream link that is not
       swallowed into W (domain exit -> -1).
    2. Any other link that pointed at a swallowed id is retargeted to that
       lake's winner, unless that would make W → X → W; then X points at
       the same exit D as W.
    """
    winner_ids = {int(w) for w in lake_to_winner.values() if w is not None}
    winner_down = {}
    for winner_id in winner_ids:
        orig = down_map.get(int(winner_id), -1)
        winner_down[int(winner_id)] = first_surviving_downstream(
            orig, down_map, swallowed_map, int(winner_id)
        )

    new_down = []
    for link_no, ds_id in zip(
        streams_dissolved["LINKNO"].to_numpy(),
        streams_dissolved["DSLINKNO"].to_numpy(),
    ):
        try:
            link_no = int(link_no)
        except (TypeError, ValueError):
            new_down.append(-1)
            continue
        if link_no in winner_down:
            new_down.append(int(winner_down[link_no]))
            continue
        if _is_network_outlet(ds_id):
            new_down.append(-1)
            continue
        try:
            ds_id = int(ds_id)
        except (TypeError, ValueError):
            new_down.append(-1)
            continue
        if ds_id not in swallowed_map:
            new_down.append(ds_id)
            continue
        target = int(swallowed_map[ds_id])
        if winner_down.get(target) == link_no:
            # X sits between W and W's swallowed trunk. Do not write X→W.
            new_down.append(
                first_surviving_downstream(
                    down_map.get(link_no, -1), down_map, swallowed_map, target
                )
            )
        else:
            new_down.append(target)

    out = streams_dissolved.copy()
    out["DSLINKNO"] = new_down

    # Report leftover cycles instead of inventing outlets.
    succ = dict(zip(out["LINKNO"].astype(int), out["DSLINKNO"].astype(int)))
    leftover = []
    seen_global = set()
    for start in succ:
        if start in seen_global:
            continue
        stack = []
        onstack = {}
        x = start
        while x in succ and not _is_network_outlet(x):
            if x in onstack:
                leftover.append(stack[onstack[x]:])
                break
            if x in seen_global:
                break
            onstack[x] = len(stack)
            stack.append(x)
            seen_global.add(x)
            x = succ[x]
    if leftover:
        cores = []
        keys = set()
        for cyc in leftover:
            key = frozenset(cyc)
            if key not in keys:
                keys.add(key)
                cores.append(cyc)
        raise ValueError(
            "Lake rewire still produced DSLINKNO cycle(s): "
            f"{cores[:10]}. Inspect swallowed links for these IDs."
        )
    return out


def apply_lake_metrics(streams_dissolved, lake_metrics):
    """Assign path-traced hydrometric values and recalculate Slope on merged links."""
    for merged_id, metrics in lake_metrics.items():
        idx = streams_dissolved["LINKNO"] == merged_id
        if streams_dissolved[idx].empty:
            continue

        for metric_name, value in metrics.items():
            if metric_name in streams_dissolved.columns:
                streams_dissolved.loc[idx, metric_name] = value

        if metrics["Length"] > 0:
            streams_dissolved.loc[idx, "Slope"] = metrics["strmDrop"] / metrics["Length"]
        elif "Slope" in streams_dissolved.columns:
            streams_dissolved.loc[idx, "Slope"] = 0.0


# ==============================================================================
# E. EXPORT
# ==============================================================================
def export_shapefile(gdf, filename):
    """
    Write a shapefile with type-aware numeric formatting.

    Float columns are rounded to 3 decimal places (Slope is left unrounded).
    Integer columns are preserved as ints. Wide float fields use float:24.3
    in the Fiona schema to avoid DBF truncation. Falls back to string columns
    if the schema export fails.
    """
    export_gdf = gdf.copy()

    float_cols = export_gdf.select_dtypes(include=["float64", "float32"]).columns
    for col in float_cols:
        if col != "Slope":
            export_gdf[col] = pd.to_numeric(export_gdf[col], errors="coerce").fillna(0.0).round(3)

    if "Slope" in export_gdf.columns:
        export_gdf["Slope"] = pd.to_numeric(export_gdf["Slope"], errors="coerce").fillna(0.0)

    int_cols = export_gdf.select_dtypes(include=["int64", "int32"]).columns
    for col in int_cols:
        export_gdf[col] = export_gdf[col].fillna(-1).astype(int)

    schema = gpd.io.file.infer_schema(export_gdf)
    for col in float_cols:
        if col not in schema["properties"]:
            continue
        if col == "Slope":
            continue
        if col == "lake_area":
            schema["properties"][col] = "float:24.1"
        else:
            schema["properties"][col] = "float:24.3"

    try:
        export_gdf.to_file(filename, driver="ESRI Shapefile", schema=schema, engine="fiona")
    except Exception as exc:
        print(f"Fiona export failed for {filename}, attempting fallback. Error: {exc}")
        for col in list(float_cols) + list(int_cols):
            if col in export_gdf.columns:
                export_gdf[col] = export_gdf[col].astype(str)
        export_gdf.to_file(filename, driver="ESRI Shapefile")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def process_reservoir_basins():
    """
    Orchestrate the full lake-merge workflow.

    Per-lake loop:
      1. Get outflow/inflow pour points from snapped-outlets.shp
      2. Build and filter outflow candidates; pick winning outlet link
      3. Collect internal links to swallow
      4. Union swallowed subbasin polygons into one reservoir catchment

    Post-loop:
      5. Dissolve swallowed stream links; rewire DSLINKNO topology
      6. Export merged basins and streams
    """
    print("Loading files...")
    basins = gpd.read_file(PATHS["basins"])
    lakes = gpd.read_file(PATHS["lakes"])
    streams = gpd.read_file(PATHS["streams"])
    intersection = gpd.read_file(PATHS["intersection"])
    gauges = gpd.read_file(PATHS["gauges"])

    intersection["lake_id"] = parse_lake_id_column(intersection)

    # Align all layers to the stream network CRS
    target_crs = streams.crs
    lakes = lakes.to_crs(target_crs)
    intersection = intersection.to_crs(target_crs)
    basins = basins.to_crs(target_crs)
    gauges = gauges.to_crs(target_crs)

    overrides_gdf = load_overrides(OVERRIDES_CSV, target_crs.to_wkt())
    link_to_dout, link_to_accum, link_to_downstream, link_to_strmorder = build_stream_lookup_tables(streams)

    buffer_dist = estimate_search_buffer(basins)
    cell_size = buffer_dist / 0.75  # reverse the 0.75 factor to get DEM resolution

    lakes["Hylak_id"] = pd.to_numeric(lakes["Hylak_id"], errors="coerce").fillna(-2).astype(int)
    basins["DN"] = basins["DN"].astype(int)
    streams["LINKNO"] = streams["LINKNO"].astype(int)

    down_map = streams.set_index("LINKNO")["DSLINKNO"].to_dict()

    # Links that appear as someone else's DSLINKNO (network junctions / trunks).
    # Excluded from geometric swallow to avoid absorbing the main downstream stem.
    terminal_link_ids = set(streams["DSLINKNO"].unique())

    lake_to_links = {}      # lake_id -> list of internal link IDs to dissolve
    lake_to_outlet = {}     # lake_id -> DSLINKNO of winning outlet (downstream link)
    lake_to_winner = {}     # lake_id -> winning outlet LINKNO itself
    all_swallowed_ids = set()
    catchment_results = []

    unique_lakes = intersection["lake_id"].unique()
    print(f"Processing {len(unique_lakes)} lakes...")
    skipped_lakes = []

    # --- Per-lake processing ---
    for l_id in unique_lakes:
        if l_id < 0:
            skipped_lakes.append((int(l_id), "lake_id < 0 in snapped-outlets"))
            continue

        out_pts = intersection[(intersection["lake_id"] == l_id) & (intersection["point_type"] == "outflow")]
        in_pts = intersection[(intersection["lake_id"] == l_id) & (intersection["point_type"] == "inflow")]
        lake_polys = lakes[lakes["Hylak_id"] == l_id]
        if lake_polys.empty:
            skipped_lakes.append((int(l_id), "no polygon in filtered_lakes.shp"))
            continue

        lake_geom = lake_polys.geometry.iloc[0]
        used_geom_outlet = False

        if out_pts.empty:
            winner_id = select_winner_from_lake_geometry(lake_geom, streams)
            if winner_id is None:
                skipped_lakes.append((int(l_id), "inflow only and no stream intersects the lake"))
                continue
            raw_outflow_candidates = {int(winner_id)}
            used_geom_outlet = True
            print(
                f"Lake {int(l_id)}: no outflow point; using intersecting stream "
                f"{winner_id} as outlet."
            )
        else:
            outflow_candidates = build_outflow_candidates(
                out_pts, streams, buffer_dist, link_to_dout, link_to_accum, link_to_strmorder, gauges,
            )
            if not outflow_candidates:
                skipped_lakes.append((int(l_id), "outflow point did not snap to a stream LINKNO"))
                continue
            raw_outflow_candidates = {c["link_no"] for c in outflow_candidates}
            surviving_candidates = filter_upstream_duplicate_outflows(
                outflow_candidates, link_to_downstream
            )
            if not surviving_candidates:
                skipped_lakes.append((int(l_id), "all outflow candidates dropped as upstream duplicates"))
                continue
            winner_id = select_winning_outflow_link(
                l_id, surviving_candidates, overrides_gdf, cell_size
            )
            if winner_id is None:
                skipped_lakes.append((int(l_id), "no winning outlet LINKNO"))
                continue

        lake_to_outlet[l_id] = down_map.get(winner_id, -1)  # link downstream of the outlet
        lake_to_winner[l_id] = winner_id

        # C. Internal link identification
        internal_links = collect_internal_links(
            lake_geom, in_pts, raw_outflow_candidates, streams,
            buffer_dist, down_map, terminal_link_ids,
        )
        # Always swallow the winning outlet link so the merged unit keeps that DN
        # and the original winner polygon is not left behind.
        internal_links.add(winner_id)
        internal_links = close_internal_along_winner_path(
            winner_id, internal_links, down_map
        )

        # Union subbasins whose DN matches swallowed internal links
        swallowed_basins = basins[basins["DN"].isin(internal_links)]
        if swallowed_basins.empty:
            skipped_lakes.append((int(l_id), f"no basin polygon with DN in internal links (winner={winner_id})"))
            continue

        lake_to_links[l_id] = list(internal_links)
        all_swallowed_ids.update(internal_links)

        merged_geom = swallowed_basins.geometry.union_all()

        # HydroLAKES Lake_area is km²
        lake_area_km2 = float(pd.to_numeric(lake_polys.iloc[0]["Lake_area"], errors="coerce") or 0.0)
        basin_area_km2 = float(merged_geom.area) / 1e6
        intersection_geom = merged_geom.intersection(lake_geom)
        lake_in_basin_km2 = float(intersection_geom.area) / 1e6

        if basin_area_km2 > 0:
            fractional_lake_in_basin = lake_in_basin_km2 / basin_area_km2
        else:
            fractional_lake_in_basin = 0.0
        fractional_lake_in_basin = max(0.0, min(1.0, fractional_lake_in_basin))

        # Keep the winning outlet LINKNO as the basin DN so the primary ID
        # stays the stream-network ID (not a lake-offset or gauge ID).
        catchment_results.append({
            "DN": int(winner_id),
            "lake_id": l_id,
            "geometry": merged_geom,
            "is_lake": 1,
            # Shapefile DBF fields are limited to 10 characters
            "lake_area": lake_area_km2,  # km²
            "frac_lake": fractional_lake_in_basin,
            "outlet_src": "geom" if used_geom_outlet else "point",
        })

    # --- D. Stream dissolve, hydrometric aggregation, and topology rewire ---
    if skipped_lakes:
        print(f"Skipped {len(skipped_lakes)} lake(s) that will NOT have a reservoir unit:")
        for lid, reason in skipped_lakes:
            print(f"  Hylak_id={lid}: {reason}")
    
    from collections import Counter
    winner_counts = Counter(lake_to_winner.values())
    dup_winners = {w: c for w, c in winner_counts.items() if c > 1}
    if dup_winners:
        print("=== 警告：以下winner LINKNO被多个湖共用 ===")
        for w in dup_winners:
            sharing = [l for l, ww in lake_to_winner.items() if ww == w]
            print(f"  winner {w} 被这些湖共用: {sharing}")

    all_links_flat = [link for links in lake_to_links.values() for link in links]
    dup_links = {lnk: c for lnk, c in Counter(all_links_flat).items() if c > 1}
    if dup_links:
        print("=== 警告：以下link被多个湖同时认领 ===")
        for lnk in dup_links:
            owners = [l for l, links in lake_to_links.items() if lnk in links]
            print(f"  link {lnk} 被这些湖同时认领: {owners}")
    
    print("Merging segments and recalculating hydrometric statistics...")
    streams_work = streams.copy()
    lake_metrics = {}
    swallowed_map = {}  # old LINKNO -> winning outlet LINKNO (merged lake ID)

    for l_id, links in lake_to_links.items():
        if not links:
            continue
        winner_id = lake_to_winner.get(l_id)
        if winner_id is None:
            continue
        new_id = int(winner_id)
        for link in links:
            swallowed_map[link] = new_id
        lake_metrics[new_id] = compute_lake_path_metrics(
            links, winner_id, streams_work, down_map,
        )

    # Assign merged_ID: swallowed links get the winning outlet LINKNO; others keep LINKNO
    streams_work["merged_ID"] = streams_work["LINKNO"].replace(swallowed_map)

    streams_dissolved = (
        streams_work
        .dissolve(by="merged_ID", aggfunc=build_stream_agg_logic(streams))
        .reset_index()
        .rename(columns={"merged_ID": "LINKNO"})
    )

    apply_lake_metrics(streams_dissolved, lake_metrics)

    try:
        streams_dissolved = rewire_dissolved_topology(
            streams_dissolved, down_map, swallowed_map, lake_to_winner
    )
    except ValueError:
        print("=== 环路诊断信息（崩溃前打印）===")
        suspect_pairs = [(478, 8285), (6255, 6319), (13107, 13171),
                          (64, 7104), (9856, 9920), (10095, 10223), (2560, 8640)]
        for a, b in suspect_pairs:
            a_winner_for = [l for l, w in lake_to_winner.items() if w == a]
            b_winner_for = [l for l, w in lake_to_winner.items() if w == b]
            print(f"--- 配对 ({a}, {b}) ---")
            print(f"  {a}: down_map={down_map.get(a)}; swallowed_map={swallowed_map.get(a)}; 是winner的湖={a_winner_for}")
            print(f"  {b}: down_map={down_map.get(b)}; swallowed_map={swallowed_map.get(b)}; 是winner的湖={b_winner_for}")
            for lid in a_winner_for:
                print(f"  湖{lid}(赢家{a})的internal_links是否包含{b}: {b in lake_to_links.get(lid, [])}")
            for lid in b_winner_for:
                print(f"  湖{lid}(赢家{b})的internal_links是否包含{a}: {a in lake_to_links.get(lid, [])}")
            x = a
            for step in range(4):
                nxt = down_map.get(x)
                print(f"  第{step}步: {x} -> {nxt} (swallowed_map[{nxt}]={swallowed_map.get(nxt)})")
                if nxt is None:
                    break
                x = nxt
        raise
    streams_dissolved = streams_dissolved.drop(columns=["USLINKNO1", "USLINKNO2"], errors="ignore")

    # --- E. Assemble final basin fabric ---
    # Drop every swallowed subbasin, then append one merged reservoir polygon per lake.
    final_geofabric = pd.concat(
        [
            basins[~basins["DN"].isin(all_swallowed_ids)],
            gpd.GeoDataFrame(catchment_results, crs=target_crs),
        ],
        ignore_index=True,
    )
    final_geofabric["is_lake"] = final_geofabric["is_lake"].fillna(0).astype(int)
    final_geofabric["lake_id"] = final_geofabric["lake_id"].fillna(-1).astype(int)
    final_geofabric["lake_area"] = final_geofabric["lake_area"].fillna(0.0)
    final_geofabric["frac_lake"] = final_geofabric["frac_lake"].fillna(0.0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    export_shapefile(final_geofabric, f"{OUTPUT_DIR}/reservoirBasins.shp")
    export_shapefile(streams_dissolved, f"{OUTPUT_DIR}/reservoirStreams.shp")
    print("Processing complete.")


if __name__ == "__main__":
    process_reservoir_basins()
