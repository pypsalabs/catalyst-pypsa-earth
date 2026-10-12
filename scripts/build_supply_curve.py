# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Builds the supply curve of a site-specific technology per cluster region.

Some technologies (enhanced geothermal, for example) have a potential and a cost
that depend on the site. A per-cell table of such sites is turned into a few
tranches per cluster region: cells are binned by their specific investment, the
bins are merged until at most ``tranches`` remain, and each tranche keeps its
capacity-weighted costs, its total capacity and, where the output depends on the
air temperature, an hourly availability series from the cutout's temperature.

Relevant Settings
-----------------

.. code:: yaml

    supply_curves:
        <technology>:
            potential:
            cutout:
            tranches:
            capex_bins:
            developable_share:
            components:
            filter:
            ambient_response:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`supply_curves_cf`

Inputs
------

- ``potential``: csv with one row per site (cell), columns ``lon``, ``lat``, ``capacity_mw``
  (developable capacity of the cell), ``fom_usd_per_kw_yr`` (fixed O&M per kW and year,
  in the currency of the cost table), ``lifetime_years``, ``capacity_factor`` (availability),
  one investment column per cost component named in ``components`` (per kW), and optionally
  the design temperature and relative-output columns of ``ambient_response``.
- ``cutouts/{cutout}.nc``: the atlite cutout; only its ``temperature`` is used.
- ``resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson``: the cluster regions.

Outputs
-------

- ``resources/supply_curves/supply_curve_{technology}_elec_s{simpl}_{clusters}.nc``: dataset with
  dimensions ``bus`` and ``tranche``: ``p_nom_max`` (MW), ``capex_<component>`` (per kW),
  ``capex`` (their sum), ``fom`` (per kW and year), ``capacity_factor``, ``lifetime``, ``n_cells``,
  the capacity-weighted means of every other numeric column of the table, and the hourly
  ``profile`` (per unit of ``p_nom``, capacity-weighted over the tranche's cells).

Description
-----------

Cells are assigned to the cluster region containing their centre. Their capacity is
scaled by ``developable_share`` (the share of the potential that can be developed on
top of the land exclusions already in the table). Per region the cells are binned by
their total specific investment with the edges ``capex_bins`` (cells above the last
edge are dropped), and adjacent bins with the smallest combined capacity are merged
until at most ``tranches`` bins remain, so that the cheapest sites keep a tranche of
their own instead of being averaged away.

With ``ambient_response`` the table holds, per cell, the output relative to the design
point at a few air-temperature offsets (K); the hourly multiplier is the piecewise-linear
interpolation at the cutout's temperature deviation from the cell's design temperature,
normalised to an annual mean of one so that the annual capacity factor stays the
table's. The profile may exceed one in cold hours.
"""

import logging

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from _helpers import configure_logging, create_logger

logger = create_logger(__name__)

DEFAULT_AMBIENT = {
    "design_temperature": "t_design_c",
    "offsets": {
        "ambient_m20": -20,
        "ambient_m10": -10,
        "ambient_p10": 10,
        "ambient_p20": 20,
    },
}
MEAN_COLUMNS_SKIP = {"lon", "lat", "capacity_mw"}


def assign_regions(cells: pd.DataFrame, regions: gpd.GeoDataFrame) -> pd.Series:
    """Bus (region name) containing each cell centre; NaN outside every region."""
    pts = gpd.GeoDataFrame(
        cells[["lon", "lat"]],
        geometry=gpd.points_from_xy(cells.lon, cells.lat),
        crs="EPSG:4326",
    )
    right = regions[["geometry"]].to_crs("EPSG:4326")
    right["bus"] = right.index.astype(str)
    joined = gpd.sjoin(pts, right, how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]
    return joined["bus"].reindex(cells.index)


def merge_bins(bins: list, tranches: int) -> list:
    """Merge adjacent bins (lists of cell indices with a capacity) until at most ``tranches`` remain."""
    while len(bins) > tranches:
        caps = [b["capacity"] for b in bins]
        pair = int(np.argmin([caps[i] + caps[i + 1] for i in range(len(bins) - 1)]))
        a, b = bins[pair], bins[pair + 1]
        bins[pair : pair + 2] = [
            {
                "cells": a["cells"] + b["cells"],
                "capacity": a["capacity"] + b["capacity"],
            }
        ]
    return bins


def tranche_cells(cells: pd.DataFrame, edges: list, tranches: int) -> list:
    """Cells of one region -> list of index lists, one per tranche, cheapest first."""
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = cells.index[(cells.capex >= lo) & (cells.capex < hi)]
        cap = cells.loc[sel, "capacity"].sum()
        if len(sel) and cap > 0:
            bins.append({"cells": list(sel), "capacity": cap})
    return [b["cells"] for b in merge_bins(bins, tranches)]


def ambient_multiplier(
    cells: pd.DataFrame, temperature: xr.DataArray, ambient: dict
) -> np.ndarray:
    """Hourly output multiplier per cell (time x cells), mean one over the period."""
    offsets = ambient["offsets"]
    x = np.array(sorted(offsets.values(), key=float) + [0.0], dtype=float)
    x.sort()
    cols = {float(v): k for k, v in offsets.items()}
    t = temperature.sel(
        x=xr.DataArray(cells.lon.values, dims="cell"),
        y=xr.DataArray(cells.lat.values, dims="cell"),
        method="nearest",
    ).values  # time x cell, K
    design = cells[ambient["design_temperature"]].values + 273.15
    out = np.empty_like(t)
    for i in range(t.shape[1]):
        y = np.array(
            [1.0 if dx == 0 else cells.iloc[i][cols[dx]] for dx in x], dtype=float
        )
        m = np.interp(t[:, i] - design[i], x, y)
        out[:, i] = m / m.mean()
    return out


def build_supply_curve(cells, regions, cfg, cutout=None):
    cfg = dict(cfg)
    components = cfg["components"]
    edges = list(cfg.get("capex_bins", [0, np.inf]))
    if not np.isinf(edges[-1]):
        logger.info(f"cells with an investment above {edges[-1]} are dropped")
    n_tranches = int(cfg.get("tranches", 5))
    share = float(cfg.get("developable_share", 1.0))

    cells = cells.copy()
    if cfg.get("filter"):
        cells = cells.query(cfg["filter"]).copy()
    cells["capex"] = sum(cells[col] for col in components.values())
    cells["capacity"] = cells["capacity_mw"] * share
    cells["bus"] = assign_regions(cells, regions)
    cells = cells[cells.bus.notna() & (cells.capacity > 0)]
    logger.info(
        f"{len(cells)} cells inside the {len(regions)} regions, {cells.capacity.sum() / 1e3:.1f} GW developable"
    )

    ambient = cfg.get("ambient_response")
    if ambient is True or ambient is None and cutout is not None:
        ambient = DEFAULT_AMBIENT
    if ambient and not set(ambient["offsets"]) <= set(cells.columns):
        logger.warning("no ambient response columns in the table: flat profiles")
        ambient = None
    temperature = cutout.data["temperature"] if ambient else None

    numeric = [
        c
        for c in cells.columns
        if c not in MEAN_COLUMNS_SKIP and pd.api.types.is_numeric_dtype(cells[c])
    ]
    records, profiles = [], {}
    for bus, group in cells.groupby("bus"):
        tr = tranche_cells(group, edges, n_tranches)
        if temperature is not None and tr:
            mult = ambient_multiplier(group, temperature, ambient)
        for k, idx in enumerate(tr):
            g = group.loc[idx]
            w = g.capacity / g.capacity.sum()
            rec = {c: float((g[c] * w).sum()) for c in numeric}
            rec.update(
                bus=bus, tranche=k, p_nom_max=float(g.capacity.sum()), n_cells=len(g)
            )
            records.append(rec)
            cf = rec["capacity_factor"]
            if temperature is not None:
                pos = group.index.get_indexer(idx)
                profiles[(bus, k)] = cf * (mult[:, pos] * w.values).sum(axis=1)
            else:
                profiles[(bus, k)] = np.full(
                    len(cutout.data.time) if cutout is not None else 1, cf
                )
        logger.info(
            f"{bus}: {len(tr)} tranches, "
            + ", ".join(
                f"{r['p_nom_max'] / 1e3:.1f} GW at {r['capex']:,.0f}/kW"
                for r in records
                if r["bus"] == bus
            )
        )

    df = pd.DataFrame(records).set_index(["bus", "tranche"])
    ds = df.to_xarray()
    for comp, col in components.items():
        ds[f"capex_{comp}"] = ds[col]
    ds = ds.rename({"fom_usd_per_kw_yr": "fom", "lifetime_years": "lifetime"})
    ds["bus"] = ds["bus"].astype(str)
    ds = ds.reindex(bus=regions.index.astype(str))
    time = cutout.data.time if cutout is not None else pd.Index([0], name="time")
    prof = xr.DataArray(
        np.full((len(time), ds.sizes["bus"], ds.sizes["tranche"]), np.nan),
        coords={"time": time, "bus": ds.bus, "tranche": ds.tranche},
        dims=("time", "bus", "tranche"),
    )
    for (bus, k), p in profiles.items():
        prof.loc[dict(bus=bus, tranche=k)] = p
    ds["profile"] = prof
    ds.attrs.update(
        components=list(components), developable_share=share, tranches=n_tranches
    )
    return ds


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_supply_curve", technology="egs", simpl="", clusters="1"
        )
    configure_logging(snakemake)

    cfg = snakemake.params.supply_curves[snakemake.wildcards.technology]
    cells = pd.read_csv(snakemake.input.potential, low_memory=False)
    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    cutout = atlite.Cutout(snakemake.input.cutout)

    ds = build_supply_curve(cells, regions, cfg, cutout)
    ds.to_netcdf(snakemake.output[0])
