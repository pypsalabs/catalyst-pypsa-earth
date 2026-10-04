# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Prepares brownfield data from previous planning horizon.

The script serves the sector-coupled rule ``add_brownfield`` and the
electricity-only rule ``add_brownfield_elec``. In the electricity-only
workflow it runs for every planning horizon including the first one, where
there is no previous network and only the build year of the new assets is
set and retired power plants are removed.

Relevant Settings
-----------------

```yaml

    sector:
        hydrogen:
            network:
            H2_retrofit_capacity_per_CH4:
            network_limit:
            network_routes:
            gas_network_repurposing:
            underground_storage:
            hydrogen_colors:
            set_color_shares:
            blue_share:
            pink_share:
            production_technologies:

    existing_capacities
        grouping_years_power:
        grouping_years_heat:
        threshold_capacity:
        default_heating_lifetime:
        conventional_carriers:
        retire_existing:

    snapshots:
        start:
        end:
        inclusive:

    electricity:
        renewable_carriers:
```
Inputs
------
- ``resources/{RDIR}/bus_regions/busmap_elec_s{simpl}.csv``: Busmap after simplifying the network
- ``resources/{RDIR}/bus_regions/busmap_elec_s{simpl}_{clusters}.csv``: Busmap after clustering the network
- ``{RESDIR}/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_export.nc``: prenetwork file obtained prior to solving
- ``solved_previous_horizon``: Network solved at previous time step
- ``resources/{RDIR}/costs_{planning_horizons}_sec.csv``: Technology costs data
- ``resources/{SECDIR}/cops/cop_soil_total_elec_s{simpl}_{clusters}_{planning_horizons}.nc``: Ground/soil source heat pump COP time series aligned to the network snapshots
- ``resources/{SECDIR}/cops/cop_air_total_elec_s{simpl}_{clusters}_{planning_horizons}.nc``: Air source heat pump COP time series aligned to the network snapshots

Electricity-only (``add_brownfield_elec``):

- ``networks/{RDIR}/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: prepared network, the same for all planning horizons
- ``results/{RDIR}/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{planning_horizons}.nc``: network solved at the previous planning horizon (none for the first one)
- ``resources/{RDIR}/costs_{planning_horizons}_elec.csv``: technology costs of the planning horizon, applied to the extendable assets
- ``resources/{RDIR}/costs_{year}_elec.csv``: technology costs the network was built with (``costs: year``)

Output
------
- ``{RESDIR}/prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_export.nc``: Brownfield prenetwork file
- ``networks/{RDIR}/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{planning_horizons}.nc``: Brownfield network of the electricity-only workflow

Description
-----------
To prepare network for brownfield expansion

"""

import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import (
    configure_logging,
    read_csv_nafix,
    sanitize_carriers,
    sanitize_locations,
)
from add_existing_baseyear import (
    add_build_year_to_new_assets,
    rename_clashing_vintages,
)

# from pypsa.clustering.spatial import normed_or_uniform

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


def remove_retired_assets(n: pypsa.Network, year: int) -> None:
    """
    Removes existing assets which have reached the end of their lifetime.

    Only assets with a fixed capacity are considered, i.e. the existing power
    plants with a build year and a lifetime from the power plant data. Assets
    built by the optimisation in previous planning horizons are retired in
    ``add_brownfield``.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network prepared for the planning horizon.
    year : int
        The planning horizon year.

    Returns
    -------
    None
    """
    for c in n.iterate_components(["Link", "Generator", "Store", "StorageUnit"]):
        attr = "e" if c.name == "Store" else "p"
        retired = c.df.index[
            ~c.df[f"{attr}_nom_extendable"]
            & (c.df.build_year != 0)
            & (c.df.build_year + c.df.lifetime < year)
        ]
        if retired.empty:
            continue
        capacity = c.df.loc[retired].groupby("carrier")[f"{attr}_nom"].sum()
        logger.info(
            f"Retiring {len(retired)} {c.name} assets before {year}, "
            f"capacity by carrier:\n{capacity.round(1).to_string()}"
        )
        n.mremove(c.name, retired)


def update_capital_costs(
    n: pypsa.Network,
    costs: pd.DataFrame,
    costs_base: pd.DataFrame,
    storage_techs: dict,
) -> None:
    """
    Moves the capital costs of the extendable assets to the planning horizon.

    The electricity network is built once with the costs of ``costs: year``.
    For every technology the difference between the cost tables of the planning
    horizon and of that year is added to the capital cost of the extendable
    generators, storage units, stores and storage links, so that adders from
    earlier steps (e.g. grid connection costs) are kept. Existing assets and
    transmission are not touched; the distance-dependent connection cost of
    offshore wind stays at ``costs: year``.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network prepared for the planning horizon.
    costs : pd.DataFrame
        Cost table of the planning horizon (``costs_{planning_horizons}_elec.csv``).
    costs_base : pd.DataFrame
        Cost table the network was built with.
    storage_techs : dict
        The ``storage_techs`` config block: storage carrier -> cost table entries.

    Returns
    -------
    None
    """

    def link_cost(table, entry, discharge=False):
        if entry not in table.index:
            return np.nan
        cost = table.at[entry, "capital_cost"]
        if discharge and entry == "fuel cell":
            # NB: fuel cell investment cost is per MWel
            cost *= table.at[entry, "efficiency"]
        return cost

    def difference(entries, **kwargs):
        return sum(
            link_cost(costs, e, **kwargs) - link_cost(costs_base, e, **kwargs)
            for e in entries
        )

    changes = {}

    def shift(c, assets, delta, label):
        if assets.empty or np.isnan(delta) or delta == 0.0:
            return
        df = n.df(c)
        df.loc[assets, "capital_cost"] = (df.loc[assets, "capital_cost"] + delta).clip(
            lower=0.0
        )
        changes[label] = delta

    gens = n.generators[n.generators.p_nom_extendable]
    for carrier in gens.carrier.unique():
        entries = (
            ["offwind", carrier + "-station"]
            if carrier.startswith("offwind")
            else [carrier]
        )
        shift("Generator", gens.index[gens.carrier == carrier], difference(entries), carrier)

    sus = n.storage_units[n.storage_units.p_nom_extendable]
    for carrier in sus.carrier.unique():
        shift("StorageUnit", sus.index[sus.carrier == carrier], difference([carrier]), carrier)

    stores = n.stores[n.stores.e_nom_extendable]
    links = n.links[n.links.p_nom_extendable]
    for carrier, lookup in storage_techs.items():
        shift(
            "Store",
            stores.index[stores.carrier == carrier],
            difference([lookup["store"]]),
            f"{carrier} store",
        )
        charge = lookup.get("bicharger", lookup.get("charger"))
        charge_name = "electrolysis" if charge == "electrolysis" else "charger"
        shift(
            "Link",
            links.index[links.carrier == f"{carrier} {charge_name}"],
            difference([charge]),
            f"{carrier} {charge_name}",
        )
        if "bicharger" not in lookup:
            discharge = lookup["discharger"]
            discharge_name = "fuel cell" if discharge == "fuel cell" else "discharger"
            shift(
                "Link",
                links.index[links.carrier == f"{carrier} {discharge_name}"],
                difference([discharge], discharge=True),
                f"{carrier} {discharge_name}",
            )

    if changes:
        logger.info(
            "Capital cost of extendable assets moved to the planning horizon, change per unit:\n"
            + pd.Series(changes).round(1).to_string()
        )


def add_brownfield(
    n: pypsa.Network, n_p: pypsa.Network, year: int, sector_coupled: bool = True
) -> None:
    """
    Adds brownfield assets from the previous planning horizon to the network.

    Parameters
    ----------
    n : pypsa.Network
        The new PyPSA network to which brownfield assets will be added.
    n_p : pypsa.Network
        The previous PyPSA network from which brownfield assets will be sourced.
    year : int
        The planning horizon year for which brownfield assets are being prepared.
    sector_coupled : bool
        Whether the networks are sector-coupled; the gas and hydrogen pipeline
        handling is skipped otherwise.

    Returns
    -------
    None
    """
    logger.info(f"Preparing brownfield for the year {year}")

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    year_p = planning_horizons[planning_horizons.index(year) - 1]

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]

    # Reset p_nom_min on extendable generators that was set from IRENA stats
    # in add_electricity to prevent double-counting.
    extendable_gens = n.generators.index[n.generators.p_nom_extendable]
    if not extendable_gens.empty:
        n.generators.loc[extendable_gens, "p_nom_min"] = 0.0
        n.generators.loc[extendable_gens, "p_nom"] = 0.0

    for c in n_p.iterate_components(["Link", "Generator", "Store", "StorageUnit"]):
        attr = "e" if c.name == "Store" else "p"

        # Remove generators, links and stores that track global values since they exist in n
        n_p.mremove(c.name, c.df.index[c.df.lifetime == np.inf])

        # Remove assets whose build_year + lifetime < year
        n_p.mremove(c.name, c.df.index[c.df.build_year + c.df.lifetime < year])

        # Remove existing assets, which are in n already: by name, and those without a
        # build year in the input data, which are named after the planning horizon.
        # Assets with a fixed capacity that remain were built in an earlier planning
        # horizon and are kept.
        n_p.mremove(c.name, c.df.index.intersection(getattr(n, c.list_name).index))
        n_p.mremove(
            c.name,
            c.df.index[~c.df[f"{attr}_nom_extendable"] & (c.df.build_year == year_p)],
        )

        # Remove assets if their optimized nominal capacity is lower than a threshold
        threshold = snakemake.params.threshold_capacity
        n_p.mremove(
            c.name,
            c.df.index[
                (c.df[f"{attr}_nom_extendable"] & (c.df[f"{attr}_nom_opt"] < threshold))
            ],
        )

        # Copy optimized assets from previous horizon to current and fix their capacity
        c.df[f"{attr}_nom"] = c.df[f"{attr}_nom_opt"]
        c.df[f"{attr}_nom_extendable"] = False

        n.import_components_from_dataframe(c.df, c.name)

        # Copy time-dependent parameters of the optimized assets from previous horizon to current
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr], c.name, tattr)

        if not sector_coupled:
            continue

        # deal with gas network
        pipe_carrier = ["gas pipeline"]
        if snakemake.params.H2_retrofit:
            # drop capacities of previous year to avoid duplicating
            to_drop = n.links.carrier.isin(pipe_carrier) & (n.links.build_year != year)
            n.mremove("Link", n.links.loc[to_drop].index)

            # subtract the already retrofitted from today's gas grid capacity
            h2_retrofitted_fixed_i = n.links[
                (n.links.carrier == "H2 pipeline retrofitted")
                & (n.links.build_year != year)
            ].index
            gas_pipes_i = n.links[n.links.carrier.isin(pipe_carrier)].index
            CH4_per_H2 = 1 / snakemake.params.H2_retrofit_capacity_per_CH4
            fr = "H2 pipeline retrofitted"
            to = "gas pipeline"
            # today's pipe capacity
            pipe_capacity = n.links.loc[gas_pipes_i, "p_nom"]
            # already retrofitted capacity from gas -> H2
            already_retrofitted = (
                n.links.loc[h2_retrofitted_fixed_i, "p_nom"]
                .rename(lambda x: x.split("-2")[0].replace(fr, to))
                .groupby(level=0)
                .sum()
            )
            remaining_capacity = (
                pipe_capacity
                - CH4_per_H2
                * already_retrofitted.reindex(index=pipe_capacity.index).fillna(0)
            )
            n.links.loc[gas_pipes_i, "p_nom"] = remaining_capacity
        else:
            new_pipes = n.links.carrier.isin(pipe_carrier) & (
                n.links.build_year == year
            )
            n.links.loc[new_pipes, "p_nom"] = 0.0
            n.links.loc[new_pipes, "p_nom_min"] = 0.0


def disable_grid_expansion_if_limit_hit(n: pypsa.Network) -> None:
    """
    Check if transmission expansion limit is already reached; then turn off.

    In particular, this function checks if the total transmission
    capital cost or volume implied by s_nom_min and p_nom_min are
    numerically close to the respective global limit set in
    n.global_constraints. If so, the nominal capacities are set to the
    minimum and extendable is turned off; the corresponding global
    constraint is then dropped.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to check and adjust for transmission expansion limits.

    Returns
    -------
    None
    """
    cols = {"cost": "capital_cost", "volume": "length"}
    for limit_type in ["cost", "volume"]:
        glcs = n.global_constraints.query(
            f"type == 'transmission_expansion_{limit_type}_limit'"
        )

        for name, glc in glcs.iterrows():
            total_expansion = (
                (
                    n.lines.query("s_nom_extendable")
                    .eval(f"s_nom_min * {cols[limit_type]}")
                    .sum()
                )
                + (
                    n.links.query("carrier == 'DC' and p_nom_extendable")
                    .eval(f"p_nom_min * {cols[limit_type]}")
                    .sum()
                )
            ).sum()

            # Allow small numerical differences
            if np.abs(glc.constant - total_expansion) / glc.constant < 1e-6:
                logger.info(
                    f"Transmission expansion {limit_type} is already reached, disabling expansion and limit"
                )
                extendable_acs = n.lines.query("s_nom_extendable").index
                n.lines.loc[extendable_acs, "s_nom_extendable"] = False
                n.lines.loc[extendable_acs, "s_nom"] = n.lines.loc[
                    extendable_acs, "s_nom_min"
                ]

                extendable_dcs = n.links.query(
                    "carrier == 'DC' and p_nom_extendable"
                ).index
                n.links.loc[extendable_dcs, "p_nom_extendable"] = False
                n.links.loc[extendable_dcs, "p_nom"] = n.links.loc[
                    extendable_dcs, "p_nom_min"
                ]

                n.global_constraints.drop(name, inplace=True)


# def adjust_renewable_profiles(n, input_profiles, params, year):
#     """
#     Adjusts renewable profiles according to the renewable technology specified,
#     using the latest year below or equal to the selected year.
#     """

#     # spatial clustering
#     cluster_busmap = read_csv_nafix(snakemake.input.cluster_busmap, index_col=0).squeeze()
#     simplify_busmap = read_csv_nafix(
#         snakemake.input.simplify_busmap, index_col=0
#     ).squeeze()
#     clustermaps = simplify_busmap.map(cluster_busmap)
#     clustermaps.index = clustermaps.index.astype(str)

#     # temporal clustering
#     dr = pd.date_range(**params["snapshots"], freq="h")
#     snapshotmaps = (
#         pd.Series(dr, index=dr).where(lambda x: x.isin(n.snapshots), pd.NA).ffill()
#     )

#     for carrier in params["carriers"]:
#         if carrier == "hydro":
#             continue
#         with xr.open_dataset(getattr(input_profiles, "profile_" + carrier)) as ds:
#             if ds.indexes["bus"].empty or "year" not in ds.indexes:
#                 continue

#             closest_year = max(
#                 (y for y in ds.year.values if y <= year), default=min(ds.year.values)
#             )

#             p_max_pu = (
#                 ds["profile"]
#                 .sel(year=closest_year)
#                 .transpose("time", "bus")
#                 .to_pandas()
#             )

#             # spatial clustering
#             weight = ds["weight"].sel(year=closest_year).to_pandas()
#             weight = weight.groupby(clustermaps).transform(normed_or_uniform)
#             p_max_pu = (p_max_pu * weight).T.groupby(clustermaps).sum().T
#             p_max_pu.columns = p_max_pu.columns + f" {carrier}"

#             # temporal_clustering
#             p_max_pu = p_max_pu.groupby(snapshotmaps).mean()

#             # replace renewable time series
#             n.generators_t.p_max_pu.loc[:, p_max_pu.columns] = p_max_pu


if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_brownfield",
            simpl="",
            clusters="4",
            ll="c1",
            opts="Co2L-4H",
            planning_horizons="2030",
            sopts="144H",
            discountrate=0.071,
        )

    configure_logging(snakemake)

    is_sector_coupled = "sopts" in snakemake.wildcards.keys()

    year = int(snakemake.wildcards.planning_horizons)

    n = pypsa.Network(snakemake.input.network)

    # TODO
    # adjust_renewable_profiles(n, snakemake.input, snakemake.params, year)

    if "costs_base" in snakemake.input.keys():
        update_capital_costs(
            n,
            pd.read_csv(snakemake.input.costs, index_col=0),
            pd.read_csv(snakemake.input.costs_base, index_col=0),
            snakemake.params.storage_techs,
        )

    # tables of the planning horizon (demand, fuel and CO2 prices), see calibrate_network.py
    calibration_meta = (n.meta or {}).get("calibration")
    calibration_tables = {
        k[len("calibration_") :]: v
        for k, v in snakemake.input.items()
        if k.startswith("calibration_")
    }
    if calibration_tables:
        from calibrate_network import calibrate_horizon

        calibrate_horizon(
            n, calibration_tables, pd.read_csv(snakemake.input.costs, index_col=0)
        )

    rename_clashing_vintages(n, snakemake.config["scenario"]["planning_horizons"])

    add_build_year_to_new_assets(n, year)

    if snakemake.config["existing_capacities"].get("retire_existing", True):
        remove_retired_assets(n, year)

    # the first planning horizon of the electricity-only workflow has no previous network
    network_p = snakemake.input.get("network_p")
    if network_p:
        logger.info(f"Preparing brownfield from the file {network_p}")

        n_p = pypsa.Network(network_p)

        add_brownfield(n, n_p, year, sector_coupled=is_sector_coupled)

        disable_grid_expansion_if_limit_hit(n)

    sanitize_carriers(n, snakemake.config)
    sanitize_locations(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    if calibration_meta is not None:
        # targets of the solve-time calibration constraints set in prepare_network
        n.meta["calibration"] = calibration_meta
    n.export_to_netcdf(snakemake.output[0])
