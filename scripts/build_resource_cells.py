# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Hourly capacity factor of every eligible cutout cell, the input of ``build_resource_classes``.

Built once per base run next to the renewable profiles (``renewable_profiles/cells_<technology>.nc``) and
only for technologies with ``renewable: <technology>: resource_classes`` > 1. The eligible MW per cell are the
``potential (y, x)`` of ``build_renewable_profiles`` (land exclusions, ``capacity_per_sqkm``), so the exclusion
rasters are not computed again; this rule makes one atlite conversion pass over the same cutout window with the
same ``resource`` and ``correction_factor``, aggregated to single cells (a sparse selector matrix): the cells
with potential and the cells that hold plants of the technology (their mean capacity factor places the plants
in a class).

Output: ``cf_t (time, cell)`` float32 (correction factor applied, as in the profiles), ``cf (cell)`` its mean,
``mw (cell)`` eligible MW (0 for plant-only cells), ``x``, ``y`` (cell centres); attrs ``dx``, ``dy``,
``correction_factor``.
"""

import copy

import atlite
import numpy as np
import pandas as pd
import xarray as xr
from _helpers import configure_logging, create_logger
from build_renewable_profiles import drop_nan_duplicate_coords
from build_resource_classes import plants_of
from scipy.sparse import csr_matrix

logger = create_logger(__name__)


def window_like(cutout, potential, tol=1e-3):
    """Restrict the cutout to the grid of ``potential`` (the window of build_renewable_profiles) and check it."""
    px, py = potential.x.values, potential.y.values
    cutout.data = cutout.data.sel(
        x=slice(px.min() - tol, px.max() + tol), y=slice(py.min() - tol, py.max() + tol)
    )
    cx, cy = cutout.data.x.values, cutout.data.y.values
    if cx.shape != px.shape or cy.shape != py.shape or not (
        np.allclose(cx, px, atol=tol) and np.allclose(cy, py, atol=tol)
    ):
        raise ValueError(
            f"cutout grid ({len(cy)} x {len(cx)}) does not match the potential grid of the profile file "
            f"({len(py)} x {len(px)})"
        )
    return cutout


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_resource_cells", technology="onwind")
    configure_logging(snakemake)

    tech = snakemake.wildcards.technology
    cfg = snakemake.params.renewable[tech]
    resource = copy.deepcopy(cfg["resource"])
    correction_factor = cfg.get("correction_factor", 1.0)

    with xr.open_dataset(snakemake.input.profile) as ds:
        potential = ds["potential"].load()

    cutout = atlite.Cutout(snakemake.input.cutout)
    cutout.data = drop_nan_duplicate_coords(cutout.data)
    cutout = window_like(cutout, potential)
    ny, nx = len(cutout.data.y), len(cutout.data.x)
    dx, dy = float(cutout.dx), float(cutout.dy)
    x0, y0 = float(cutout.data.x.min()), float(cutout.data.y.min())

    # cells: with potential, or holding plants of the technology; flat index in the order of cutout.grid (y, x)
    mw_grid = potential.transpose("y", "x").values
    flat = set(np.flatnonzero(mw_grid.ravel() > 0))
    plants = plants_of(pd.read_csv(snakemake.input.powerplants, index_col=0), tech)
    ix = np.round((plants.lon.values - x0) / dx).astype(int)
    iy = np.round((plants.lat.values - y0) / dy).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if (~ok).any():
        logger.warning(f"{int((~ok).sum())} {tech} plants outside the cutout window")
    flat |= set(iy[ok] * nx + ix[ok])
    flat = np.array(sorted(flat), dtype=int)
    logger.info(
        f"{tech}: {int((mw_grid > 0).sum())} cells with potential ({mw_grid.sum() / 1e3:.1f} GW), "
        f"{len(flat)} cells in total incl. plant cells, window {ny} x {nx}"
    )

    selector = csr_matrix((np.ones(len(flat)), (np.arange(len(flat)), flat)), shape=(len(flat), ny * nx))
    func = getattr(cutout, resource.pop("method"))
    resource["dask_kwargs"] = {"scheduler": None}
    cf_t = func(matrix=selector, index=pd.RangeIndex(len(flat), name="cell"), per_unit=False, **resource)
    cf_t = (correction_factor * cf_t).transpose("time", "cell")
    values = cf_t.values
    bad = ~np.isfinite(values).all(axis=0)
    yy, xx = np.divmod(flat, nx)
    mw = mw_grid.ravel()[flat]
    if bad.any():
        logger.warning(
            f"{int(bad.sum())} cells with NaN weather data ({mw[bad].sum() / 1e3:.2f} GW eligible): set to 0"
        )
        values[:, bad] = 0.0

    out = xr.Dataset(
        {
            "cf_t": (("time", "cell"), values.astype(np.float32)),
            "cf": ("cell", values.mean(axis=0)),
            "mw": ("cell", mw),
            "x": ("cell", cutout.data.x.values[xx]),
            "y": ("cell", cutout.data.y.values[yy]),
        },
        coords={"time": cf_t.time.values, "cell": np.arange(len(flat))},
        attrs=dict(dx=dx, dy=dy, correction_factor=correction_factor, technology=tech),
    )
    logger.info(
        f"{tech}: mean capacity factor of the cells with potential {np.average(out.cf[mw > 0], weights=mw[mw > 0]):.3f} "
        f"(MW-weighted), range {float(out.cf[mw > 0].min()):.3f}-{float(out.cf[mw > 0].max()):.3f}"
    )
    out.to_netcdf(snakemake.output[0], encoding={"cf_t": {"zlib": True, "complevel": 1}})
