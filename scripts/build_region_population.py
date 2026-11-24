# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Population of every bus region inside every administrative unit.

:mod:`build_demand_profiles` distributes the load of an administrative (GADM) unit
to the bus regions it overlaps. With ``load_options: population_within_units: true``
it does so with the population of the overlaps, which this rule counts, instead of
their area.

Relevant Settings
-----------------

.. code:: yaml

    load_options:
        population_within_units:

    build_shape_options:
        worldpop_method:
        year:

Inputs
------

- ``resources/bus_regions/regions_onshore.geojson``: confer :mod:`build_bus_regions`
- ``resources/shapes/gadm_shapes.geojson``: confer :mod:`build_shapes`
- ``data/WorldPop/<country>_ppp_<year>_UNadj_constrained.tif``: the population
  rasters :mod:`build_shapes` has downloaded (retrieved here if missing)

Outputs
-------

- ``resources/bus_regions/population_onshore.csv``: columns ``name`` (bus),
  ``GADM_ID``, ``country``, ``pop``; one row per bus region and unit with people in
  their overlap. Countries with a single bus region have no rows (their load needs
  no distribution).

Description
-----------

The WorldPop rasters have a resolution of 3 arc seconds (a country the size of the
United States has 27 billion cells), so a raster is read in strips of rows and
summed onto a grid of 30 arc seconds, about 1 km. Bus regions and administrative
units are rasterised onto the same grid (a cell belongs to the shape that holds its
centre) and the population is summed per pair.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from _helpers import configure_logging, create_logger
from build_shapes import download_WorldPop
from rasterio import features
from rasterio.windows import Window, from_bounds
from shapely.validation import make_valid

logger = create_logger(__name__)

RESOLUTION = 1.0 / 120.0  # degrees of the grid the population is summed onto
STRIP_CELLS = 2.5e7  # raster cells read at a time (100 MB as float32)


def population_grid(raster, bounds, resolution=RESOLUTION):
    """
    Population of ``raster`` inside ``bounds`` (west, south, east, north), summed
    onto a grid of about ``resolution`` degrees.

    Returns the grid and its affine transform. The raster is read in strips of
    rows, so the memory need is that of the coarse grid.
    """
    with rasterio.open(raster) as src:
        factor = max(1, int(round(resolution / abs(src.res[0]))))
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, src.width, src.height))
        col, row = int(window.col_off), int(window.row_off)
        width, height = int(window.width), int(window.height)
        ncols, nrows = -(-width // factor), -(-height // factor)
        grid = np.zeros((nrows, ncols))
        strip = max(1, int(STRIP_CELLS // (width * factor))) * factor
        for first in range(0, height, strip):
            rows = min(strip, height - first)
            values = src.read(1, window=Window(col, row + first, width, rows))
            values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
            padded = np.zeros((-(-rows // factor) * factor, ncols * factor), "f4")
            padded[:rows, :width] = values
            grid[first // factor : first // factor + padded.shape[0] // factor] = (
                padded.reshape(-1, factor, ncols, factor).sum(axis=(1, 3))
            )
        transform = src.window_transform(
            Window(col, row, width, height)
        ) * rasterio.Affine.scale(factor)
    return grid, transform


def population_of_overlaps(regions, units, grid, transform):
    """
    Population per pair of a region and a unit (two GeoSeries), from the
    population ``grid``: a DataFrame with the columns ``name``, ``GADM_ID``, ``pop``.
    """

    def raster_ids(shapes):
        return features.rasterize(
            ((geom, i + 1) for i, geom in enumerate(shapes) if not geom.is_empty),
            out_shape=grid.shape,
            transform=transform,
            fill=0,
            dtype="int32",
        )

    region_id, unit_id = raster_ids(regions), raster_ids(units)
    inside = (grid > 0) & (region_id > 0) & (unit_id > 0)
    pair = (region_id[inside] - 1).astype("int64") * len(units) + unit_id[inside] - 1
    pop = np.bincount(pair, weights=grid[inside], minlength=len(regions) * len(units))
    found = np.flatnonzero(pop)
    return pd.DataFrame(
        {
            "name": regions.index[found // len(units)],
            "GADM_ID": units.index[found % len(units)],
            "pop": pop[found],
        }
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_region_population")

    configure_logging(snakemake)

    options = snakemake.params.build_shape_options
    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    units = gpd.read_file(snakemake.input.gadm_shapes).set_index("GADM_ID")
    units["geometry"] = units.geometry.apply(make_valid)

    tables = []
    for country, group in regions.geometry.groupby(regions.country):
        if len(group) == 1:
            continue
        raster, _ = download_WorldPop(
            country, options["worldpop_method"], options["year"], False, False
        )
        grid, transform = population_grid(raster, group.total_bounds)
        table = population_of_overlaps(
            group, units.geometry[units.country == country], grid, transform
        )
        logger.info(
            f"{country}: {table['pop'].sum() / 1e6:.2f} million people in "
            f"{len(table)} overlaps of {len(group)} bus regions and "
            f"{(units.country == country).sum()} administrative units "
            f"({grid.sum() / 1e6:.2f} million inside the bounds of the regions)"
        )
        tables.append(table.assign(country=country))

    columns = ["name", "GADM_ID", "country", "pop"]
    table = pd.concat(tables) if tables else pd.DataFrame(columns=columns)
    table[columns].to_csv(snakemake.output[0], index=False)
