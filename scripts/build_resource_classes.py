# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Resource classes: split the wind or solar potential of every clustered region into capacity-factor classes.

Configuration: ``renewable: <technology>: resource_classes: n`` (default 1 = off; ``onwind`` and ``solar``),
the key of PyPSA-Eur (>= v2025.04), where the same split is made in ``build_renewable_profiles``. This fork
builds the renewable profiles on the base network, before clustering, and simplify / cluster merge all
generators of one carrier at a bus, so here the classes are built on the run's own *clustered* regions and
``add_extra_components`` splits the one generator per bus and carrier into one generator per class
(``"<bus> <carrier> c<k>"``), like the site-specific supply curves (``build_supply_curve``).

Inputs: the cell file of ``build_resource_cells`` (hourly capacity factor and eligible MW of every eligible
cutout cell, computed once per base run), the clustered onshore regions, ``powerplants.csv``.

Method, per clustered region:

1. Every cell's eligible MW is split between the regions it overlaps by area share (normalised per cell, so
   the assigned potential adds up to the potential of ``build_renewable_profiles``).
2. Class edges: ``n`` equal-width bins between the lowest and highest mean capacity factor of the region's
   eligible cells (PyPSA-Eur bins over every cell touching the region, also ineligible ones, which can leave
   the top classes empty; here only cells with potential count). Class 0 is the worst.
3. Per class: potential = eligible MW of its cells; profile = mean of its cells' hourly capacity factor
   weighted by eligible MW x mean capacity factor, the weighting of the atlite layout in
   ``build_renewable_profiles``, so one class reproduces the profile of that rule; ``clip_p_max_pu`` last.
4. Fleet: plants of ``powerplants.csv`` of the technology (wind: every non-offshore plant; solar: every
   non-CSP plant; in operation by 2024) are placed by their coordinates in a region (the nearest one when
   outside all) and, by the mean capacity factor of their cell against the region's edges, in a class; a
   class without potential passes its plants to the nearest class that has some. ``fleet_mw`` is used by
   ``add_extra_components`` only as the shares of the bus's (calibrated) fleet.

Outputs (netCDF): ``profile (time, bus, bin)``, ``p_nom_max (bus, bin)`` [MW], ``cf (bus, bin)`` (mean of the
profile), ``fleet_mw (bus, bin)``, ``edges (bus, edge)``, and the cell table over ``pair`` (``pair_bus``,
``pair_x``, ``pair_y``, ``pair_share``, ``pair_mw``, ``pair_cf``, ``pair_bin``) for maps and checks.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from _helpers import configure_logging, create_logger
from scipy import sparse

logger = create_logger(__name__)

TECHNOLOGIES = ("onwind", "solar")


def plants_of(ppl: pd.DataFrame, technology: str, year: int = 2024) -> pd.DataFrame:
    """Plants of ``powerplants.csv`` (powerplantmatching columns) that make up the fleet of ``technology``."""
    tech = ppl.Technology.fillna("").str.lower()
    if technology == "onwind":
        sel = (ppl.Fueltype == "Wind") & (tech != "offshore")
    elif technology == "solar":
        sel = (ppl.Fueltype == "Solar") & (tech != "csp")
    else:
        raise ValueError(f"resource classes: no plant mapping for {technology}")
    sel &= ppl.Capacity.fillna(0) > 0
    if "DateIn" in ppl:
        sel &= ~(ppl.DateIn > year)
    out = ppl.loc[sel & ppl.lat.notna() & ppl.lon.notna(), ["Capacity", "lat", "lon"]]
    n_drop = int((sel & (ppl.lat.isna() | ppl.lon.isna())).sum())
    if n_drop:
        logger.warning(f"{technology}: {n_drop} plants without coordinates left out of the fleet shares")
    return out


def cell_boxes(x, y, dx, dy):
    return shapely.box(x - dx / 2, y - dy / 2, x + dx / 2, y + dy / 2)


def region_shares(cells: pd.DataFrame, regions: gpd.GeoSeries, dx: float, dy: float) -> pd.DataFrame:
    """
    Area share of every cell in every clustered region it overlaps, normalised per cell.

    Cells (``x``, ``y``, ``mw``) that overlap no region go to the nearest one (logged). Returns one row per
    (cell, bus) pair: ``cell``, ``bus``, ``share``.
    """
    boxes = gpd.GeoDataFrame(
        {"cell": cells.index}, geometry=cell_boxes(cells.x.values, cells.y.values, dx, dy), crs=4326
    )
    boxes.index = cells.index.rename(None)
    reg = gpd.GeoDataFrame({"bus": regions.index}, geometry=regions.values, crs=4326)
    pairs = gpd.sjoin(boxes, reg, predicate="intersects", how="inner")[["cell", "bus"]]
    geo_cell = boxes.geometry.loc[pairs.cell.values].values
    geo_reg = reg.set_index("bus").geometry.loc[pairs.bus.values].values
    # area in degrees: the pieces of one cell lie at the same latitude
    pairs["area"] = shapely.area(shapely.intersection(geo_cell, geo_reg))
    pairs = pairs[pairs.area > 0]
    missing = cells.index.difference(pairs.cell.unique())
    if len(missing):
        near = gpd.sjoin_nearest(
            boxes.loc[boxes.cell.isin(missing)].to_crs(3857), reg.to_crs(3857), how="inner"
        ).drop_duplicates("cell")
        logger.warning(
            f"{len(missing)} cells with {cells.loc[missing, 'mw'].sum() / 1e3:.2f} GW eligible overlap no "
            "clustered region: assigned to the nearest"
        )
        pairs = pd.concat([pairs, near[["cell", "bus"]].assign(area=1.0)], ignore_index=True)
    pairs["share"] = pairs.area / pairs.groupby("cell").area.transform("sum")
    return pairs[["cell", "bus", "share"]].reset_index(drop=True)


def class_edges(cf: pd.Series, mw: pd.Series, nbins: int) -> np.ndarray:
    """Equal-width edges between the lowest and highest mean capacity factor of cells with potential."""
    v = cf[mw > 0]
    if v.empty:
        return np.full(nbins + 1, np.nan)
    eps = 1e-6
    return np.linspace(v.min() - eps, v.max() + eps, nbins + 1)


def bin_of(cf, edges):
    """Class of a capacity factor for the given edges (clipped to the outer classes)."""
    nbins = len(edges) - 1
    return np.clip(np.searchsorted(edges, cf, side="right") - 1, 0, nbins - 1)


def build_classes(
    cells: xr.Dataset,
    regions: gpd.GeoSeries,
    plants: pd.DataFrame,
    nbins: int,
    clip_p_max_pu: float = 0.0,
) -> xr.Dataset:
    """
    Class profiles, potentials and fleet per clustered region (see the module docstring).

    ``cells``: dataset of ``build_resource_cells`` (``cf_t (time, cell)``, ``cf``, ``mw``, ``x``, ``y`` over
    ``cell``; attrs ``dx``, ``dy``). ``regions``: geometries indexed by bus. ``plants``: ``Capacity``, ``lat``,
    ``lon``.
    """
    dx, dy = float(cells.attrs["dx"]), float(cells.attrs["dy"])
    cell = pd.DataFrame(
        {k: cells[k].values for k in ["x", "y", "mw", "cf"]},
        index=pd.RangeIndex(cells.sizes["cell"], name="cell"),
    )
    elig = cell[cell.mw > 0]
    pairs = region_shares(elig, regions, dx, dy)
    # pairs.cell are labels of `cell` = positions in the cell file (cf_t columns)
    pairs["mw"] = cell.mw.loc[pairs.cell].values * pairs.share.values
    pairs["cf"] = cell.cf.loc[pairs.cell].values
    pairs["x"] = cell.x.loc[pairs.cell].values
    pairs["y"] = cell.y.loc[pairs.cell].values
    pairs = pairs[pairs.mw > 0]

    buses = pd.Index(regions.index, name="bus")
    edges = pd.DataFrame(np.nan, index=buses, columns=range(nbins + 1))
    pairs["bin"] = -1
    for bus, grp in pairs.groupby("bus"):
        e = class_edges(grp.cf, grp.mw, nbins)
        edges.loc[bus] = e
        pairs.loc[grp.index, "bin"] = bin_of(grp.cf.values, e)

    # profiles: sparse weights (bus, bin) x cell, weight = MW x mean CF (atlite layout), one matmul
    pairs["key"] = pairs.bus.map({b: i for i, b in enumerate(buses)}) * nbins + pairs["bin"]
    w = pairs.mw * pairs.cf
    cf_t = cells["cf_t"].transpose("time", "cell").values  # float32 (time, cell)
    nkey = len(buses) * nbins
    W = sparse.csr_matrix(
        (w.values, (pairs.key.values, pairs.cell.values)), shape=(nkey, cf_t.shape[1])
    )  # duplicates are summed
    wsum = np.asarray(W.sum(axis=1)).ravel()
    prof = np.asarray(W @ cf_t.T).T / np.where(wsum > 0, wsum, 1.0)
    prof[:, wsum == 0] = 0.0
    if clip_p_max_pu:
        prof = np.where(prof >= clip_p_max_pu, prof, 0.0)
    prof = prof.reshape(len(cells.time), len(buses), nbins)

    p_nom_max = (
        pairs.groupby(["bus", "bin"]).mw.sum().reindex(pd.MultiIndex.from_product([buses, range(nbins)]))
    ).fillna(0.0).values.reshape(len(buses), nbins)

    # fleet: plant -> region (nearest outside), plant cell -> mean CF -> class of that region
    fleet = np.zeros((len(buses), nbins))
    n_far = 0
    if not plants.empty:
        pts = gpd.GeoDataFrame(plants, geometry=gpd.points_from_xy(plants.lon, plants.lat), crs=4326)
        reg = gpd.GeoDataFrame({"bus": regions.index}, geometry=regions.values, crs=4326)
        inside = gpd.sjoin(pts, reg, predicate="within", how="left")
        inside = inside[~inside.index.duplicated()]
        far = inside.bus.isna()
        if far.any():
            near = gpd.sjoin_nearest(pts.loc[far].to_crs(3857), reg.to_crs(3857), how="left")
            near = near[~near.index.duplicated()]
            inside.loc[far, "bus"] = near.bus
            n_far = int(far.sum())
        # mean CF of the plant's cell; cells without a value (outside the cell file) take the region's
        # capacity-weighted mean
        ix = np.round((plants.lon.values - cell.x.min()) / dx).astype(int)
        iy = np.round((plants.lat.values - cell.y.min()) / dy).astype(int)
        lookup = pd.Series(
            cell.cf.values,
            index=pd.MultiIndex.from_arrays(
                [np.round((cell.x - cell.x.min()) / dx).astype(int), np.round((cell.y - cell.y.min()) / dy).astype(int)]
            ),
        )
        lookup = lookup[~lookup.index.duplicated()]
        pcf = lookup.reindex(pd.MultiIndex.from_arrays([ix, iy])).values
        for i, (bus, cap, c) in enumerate(zip(inside.bus.values, plants.Capacity.values, pcf)):
            if bus not in edges.index or np.isnan(edges.loc[bus, 0]):
                continue
            bi = buses.get_loc(bus)
            if np.isnan(c):
                cfs = pairs.loc[pairs.bus == bus]
                c = float((cfs.cf * cfs.mw).sum() / cfs.mw.sum())
            k = int(bin_of(np.array([c]), edges.loc[bus].values)[0])
            has = np.flatnonzero(p_nom_max[bi] > 0)
            k = int(has[np.argmin(np.abs(has - k))])
            fleet[bi, k] += cap
    if n_far:
        logger.warning(f"{n_far} plants outside every clustered region: assigned to the nearest region")

    ds = xr.Dataset(
        {
            "profile": (("time", "bus", "bin"), prof.astype(np.float32)),
            "p_nom_max": (("bus", "bin"), p_nom_max),
            "cf": (("bus", "bin"), prof.mean(axis=0)),
            "fleet_mw": (("bus", "bin"), fleet),
            "edges": (("bus", "edge"), edges.values),
            "pair_bus": ("pair", pairs.bus.values.astype(str)),
            "pair_x": ("pair", pairs.x.values),
            "pair_y": ("pair", pairs.y.values),
            "pair_share": ("pair", pairs.share.values),
            "pair_mw": ("pair", pairs.mw.values),
            "pair_cf": ("pair", pairs.cf.values),
            "pair_bin": ("pair", pairs["bin"].values.astype(int)),
        },
        coords={"time": cells.time.values, "bus": buses.values, "bin": np.arange(nbins), "edge": np.arange(nbins + 1)},
        attrs=dict(nbins=nbins, clip_p_max_pu=clip_p_max_pu, correction_factor=cells.attrs.get("correction_factor", 1.0)),
    )

    # invariants
    total = float(cells.mw.sum())
    assigned = float(ds.p_nom_max.sum())
    assert abs(assigned - total) <= 1e-6 * max(total, 1.0), f"assigned potential {assigned} != cell potential {total}"
    assert np.isfinite(prof).all() and (prof >= 0).all(), "class profiles not finite / negative"
    return ds


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_resource_classes", technology="onwind", simpl="", clusters="1")
    configure_logging(snakemake)

    tech = snakemake.wildcards.technology
    cfg = snakemake.params.renewable[tech]
    nbins = int(cfg.get("resource_classes", 1))
    regions = gpd.read_file(snakemake.input.regions).set_index("name").geometry
    ppl = pd.read_csv(snakemake.input.powerplants, index_col=0)
    plants = plants_of(ppl, tech)
    with xr.open_dataset(snakemake.input.cells) as cells:
        cells = cells.load()
    ds = build_classes(cells, regions, plants, nbins, cfg.get("clip_p_max_pu", 0.0))
    for bus in ds.bus.values:
        t = pd.DataFrame(
            {
                "GW": ds.p_nom_max.sel(bus=bus).values / 1e3,
                "cf": ds.cf.sel(bus=bus).values,
                "fleet_GW": ds.fleet_mw.sel(bus=bus).values / 1e3,
            }
        )
        logger.info(f"{tech} {bus} classes:\n{t.round(3).to_string()}")
    logger.info(
        f"{tech}: fleet located in classes {float(ds.fleet_mw.sum()) / 1e3:.2f} GW of "
        f"{plants.Capacity.sum() / 1e3:.2f} GW in the plant list"
    )
    ds.to_netcdf(snakemake.output[0])
