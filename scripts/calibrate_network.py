# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Calibration steps that act on the clustered network: the representation of the
existing fleet and the solve-time constraints that hold the dispatch to observed
statistics.

The calibration *data* (demand, capacities, capacity factors, hydro energy,
efficiencies, operating envelopes, fuel and CO2 prices) enter where the data
vectors enter the workflow, through generic options of the rules that build them:
``load_options.annual_totals`` (build_demand_profiles),
``electricity.powerplants_statistics`` (build_powerplants),
``electricity.estimate_renewable_capacities``, ``electricity.capacity_factor_targets``,
``electricity.powerplants_efficiency``, ``costs.fuel_prices``, ``conventional:``,
``renewable.hydro.hydro_max_hours`` (add_electricity),
``renewable.hydro.normalization`` (build_renewable_profiles) and
``co2.emission_prices_by_country`` (prepare_network). What remains here is called by
``prepare_network`` when ``calibration.enable`` is set:

Relevant Settings
-----------------

.. code:: yaml

    calibration:
        enable:
        brownfield_only:        # freeze the fleet (dispatch of the existing system)
        efficiency_spread:      # > 0: split fuel-burning generators into tranches
        tranches:
        net_import_tolerance:
        energy_bands:           # group -> [lower, upper] factors on the observed TWh
        energy_band_penalty:
        tables:
            generation:         # country, group, twh, cf (energy bands)
            demand:             # country, demand_twh (exporters have no band floor)
            net_imports:        # country, net_imports_twh

- ``efficiency_spread`` > 0 splits every fixed fuel-burning generator into
  ``tranches`` equal parts whose efficiencies are spread symmetrically around the
  fleet value, so that a country's merit order is a staircase rather than one step
  per carrier (the fleet is heterogeneous in vintage). The fuel part of the
  marginal cost follows each tranche's efficiency.
- ``net_imports``: the solve gets a band of +/- ``net_import_tolerance`` (relative,
  with a floor of 1 TWh) around the annual net imports of every country with
  cross-border lines or links (``add_net_import_constraints``, solve_network).
- ``energy_bands``: the annual generation of a group's carriers is kept within the
  band around the observed value (``add_energy_band_constraints``, solve_network):
  fleets that policy, contracts or fuel supply hold away from their merit-order
  position. Lower bounds are skipped for exporters (generation > 1.05 x demand).

``calibrate`` returns the targets of the solve-time constraints, which
``prepare_network`` stores in ``n.meta["calibration"]`` for ``solve_network``.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

# fuel group -> (carriers in the group, carrier used when the group has to be added)
GROUPS = {
    "coal": (["coal", "lignite"], "coal"),
    "gas": (["CCGT", "OCGT"], "CCGT"),
    "oil": (["oil"], "oil"),
    "nuclear": (["nuclear"], "nuclear"),
    "biomass": (["biomass"], "biomass"),
    "geothermal": (["geothermal"], "geothermal"),
    "wind": (["onwind", "offwind-ac", "offwind-dc"], "onwind"),
    "solar": (["solar"], "solar"),
    "hydro": (["ror"], None),  # plus the StorageUnits of carrier hydro (reservoirs); never added
}

# fuel-burning carriers whose fixed generators are split into tranches
FUELS = {
    "coal": "coal",
    "lignite": "lignite",
    "CCGT": "gas",
    "OCGT": "gas",
    "oil": "oil",
    "nuclear": "uranium",
    "biomass": "biomass",
}


def _read(path, index=None):
    df = pd.read_csv(path, comment="#", dtype={"country": str})
    df["country"] = df["country"].str.strip()
    return df.set_index(index) if index else df


def _lookup(table, key, column):
    """Value of ``column`` for ``key`` (country or (country, x)) with ``*`` default."""
    if key in table.index:
        return table.at[key, column]
    default = ("*",) + tuple(key[1:]) if isinstance(key, tuple) else "*"
    if default in table.index:
        return table.at[default, column]
    return np.nan


def _country_of(n, component):
    df = n.df(component)
    bus = df.bus if "bus" in df else df.bus0
    return bus.map(n.buses.country)


def _weights(n):
    return n.snapshot_weightings.generators


def split_tranches(n, spread, tranches, costs=None):
    """
    Split every fixed fuel-burning generator into tranches with spread efficiencies;
    the fuel part of the marginal cost (marginal cost - VOM of the cost table) scales
    with 1 / efficiency.
    """
    if tranches < 2 or spread <= 0:
        return
    g = n.generators
    idx = g.index[g.carrier.isin(FUELS) & ~g.p_nom_extendable & (g.p_nom > 0)]
    if idx.empty:
        return
    factors = 1 + spread * np.linspace(-1, 1, tranches)
    base = g.loc[idx]
    parts = []
    for i, f in enumerate(factors):
        part = base.copy()
        part.index = base.index + f" t{i + 1}"
        for col in ["p_nom", "p_nom_min", "p_nom_max"]:
            part[col] = base[col].values / tranches
        part["efficiency"] = base.efficiency.values * f
        vom = base.carrier.map(costs["VOM"]).fillna(0.0).values if costs is not None and "VOM" in costs else 0.0
        part["marginal_cost"] = vom + (base.marginal_cost.values - vom) / f
        parts.append(part)
    new = pd.concat(parts)
    n.mremove("Generator", idx)
    n.import_components_from_dataframe(new.drop(columns=["p_nom_opt"], errors="ignore"), "Generator")
    logger.info(
        f"split {len(idx)} fuel-burning generators into {tranches} tranches with efficiencies "
        f"x{factors.min():.2f}..x{factors.max():.2f}"
    )


def energy_band_targets(n, generation, bands, demand=None):
    """{country: {group: [lower, upper] MWh}} for the groups of ``bands`` (factors on the observed TWh)."""
    country = _country_of(n, "Generator")
    g = n.generators
    countries = sorted(country.dropna().unique())
    out = {}
    for c in countries:
        total = generation.xs(c, level=0).twh.sum() if c in generation.index.get_level_values(0) else np.nan
        dem = _lookup(demand, c, "demand_twh") if demand is not None else np.nan
        exporter = np.isfinite(total) and np.isfinite(dem) and total > 1.05 * dem
        for group, (lo, hi) in bands.items():
            twh = _lookup(generation, (c, group), "twh")
            if np.isnan(twh) or twh <= 0.01:
                continue
            idx = g.index[(country == c) & g.carrier.isin(GROUPS[group][0]) & (g.p_nom > 0)]
            if idx.empty:
                continue
            lower = None if (lo is None or exporter) else float(lo) * twh * 1e6
            available = (g.loc[idx, "p_nom"] * g.loc[idx, "p_max_pu"]).sum() * 8760
            if lower is not None and lower > 0.95 * available:
                logger.warning(f"energy band {c} {group}: floor {lower/1e6:.1f} TWh above 95 % of the available "
                               f"{available/1e6:.1f} TWh; floor dropped")
                lower = None
            out.setdefault(c, {})[group] = [lower, None if hi is None else float(hi) * twh * 1e6]
    logger.info("energy bands for " + ", ".join(f"{c} ({', '.join(v)})" for c, v in out.items()))
    return out


def add_energy_band_constraints(n, snapshots, bands, penalty=None):
    """
    Annual generation of (country, group) within [lower, upper] MWh (None = unbounded).
    With ``penalty`` [currency/MWh] the band is soft: a slack variable per bound is
    priced into the objective, so the band holds unless the system cannot serve its
    load otherwise (hard bands turn any shortfall elsewhere into load shedding).
    """
    import xarray as xr

    w = xr.DataArray(
        n.snapshot_weightings.generators.loc[snapshots].values,
        coords={"snapshot": snapshots},
        dims=["snapshot"],
    )
    country = _country_of(n, "Generator")
    g = n.generators
    p = n.model["Generator-p"].sel(snapshot=snapshots)
    pairs = [(c, group) for c, groups in bands.items() for group in groups]
    slack = None
    if penalty and pairs:
        # one 3-d variable (pypsa's assign_solution skips arrays it cannot turn into a frame)
        slack = n.model.add_variables(
            lower=0,
            coords=[pd.Index([f"{c}-{group}" for c, group in pairs], name="band"),
                    pd.Index(["min", "max"], name="sense"), pd.Index(["slack"], name="one")],
            name="energy_band_slack",
        )
        n.model.objective += (penalty * slack).sum()
    k = 0
    for c, groups in bands.items():
        for group, (lower, upper) in groups.items():
            idx = g.index[(country == c) & g.carrier.isin(GROUPS[group][0]) & (g.p_nom > 0)]
            if idx.empty:
                continue
            lhs = (p.sel(Generator=idx.values) * w).sum()
            for bound, sense in [(lower, "min"), (upper, "max")]:
                if bound is None:
                    continue
                name = f"energy_{sense}-{c}-{group}"
                if slack is not None:
                    sl = slack.sel(band=f"{c}-{group}", sense=sense, one="slack")
                    expr = lhs + sl if sense == "min" else lhs - sl
                else:
                    expr = lhs
                n.model.add_constraints(expr >= bound if sense == "min" else expr <= bound, name=name)
            k += 1
    logger.info(f"energy band constraints added for {k} (country, group) pairs"
                + (f" (soft, {penalty} per MWh outside the band)" if penalty else ""))


def net_import_targets(n, table):
    """Net-import targets [MWh] for the countries of the table that have cross-border lines or links."""
    country = n.buses.country
    crossing = set()
    for comp in ["Line", "Link"]:
        df = n.df(comp)
        c0, c1 = df.bus0.map(country), df.bus1.map(country)
        cross = df.index[(c0 != c1) & c0.notna() & c1.notna() & (c0 != "") & (c1 != "")]
        crossing |= set(c0[cross]) | set(c1[cross])
    targets = {}
    for c in sorted(crossing):
        v = _lookup(table, c, "net_imports_twh")
        if not np.isnan(v):
            targets[c] = float(v) * 1e6
    logger.info("net-import targets [TWh]: " + ", ".join(f"{c} {v/1e6:.1f}" for c, v in targets.items()))
    return targets


def add_net_import_constraints(n, snapshots, targets, tolerance=0.25, floor=1e6):
    """Annual net imports of each country within +/- max(tolerance x |target|, floor) [MWh] of the target."""
    import xarray as xr

    w = xr.DataArray(
        n.snapshot_weightings.generators.loc[snapshots].values,
        coords={"snapshot": snapshots},
        dims=["snapshot"],
    )
    country = n.buses.country
    for c, target in targets.items():
        lhs = None
        for comp, var in [("Line", "Line-s"), ("Link", "Link-p")]:
            df = n.df(comp)
            if df.empty or var not in n.model.variables:
                continue
            c0, c1 = df.bus0.map(country), df.bus1.map(country)
            imp = df.index[(c1 == c) & (c0 != c)]
            exp = df.index[(c0 == c) & (c1 != c)]
            v = n.model[var].sel(snapshot=snapshots)
            if len(imp):
                eff = xr.DataArray(df.efficiency[imp].values, coords={comp: imp.values}, dims=[comp]) if comp == "Link" else 1.0
                term = (v.sel({comp: imp.values}) * eff * w).sum()
                lhs = term if lhs is None else lhs + term
            if len(exp):
                term = -(v.sel({comp: exp.values}) * w).sum()
                lhs = term if lhs is None else lhs + term
        if lhs is None:
            continue
        tol = max(tolerance * abs(target), floor)
        n.model.add_constraints(lhs >= target - tol, name=f"net_import_min-{c}")
        n.model.add_constraints(lhs <= target + tol, name=f"net_import_max-{c}")
    logger.info(f"net-import band constraints added for {', '.join(targets)}")


def attach_isolated_buses(n):
    """
    Attach every AC bus without any line or link to the nearest connected AC bus of its
    country by a lossless bidirectional link. Isolated buses are OSM artefacts (islands of
    the raw topology that survived clustering); with expansion they get their own
    peakers, in a pure dispatch they can only shed.
    """
    ac = n.buses.index[n.buses.carrier == "AC"]
    connected = set(n.lines.bus0) | set(n.lines.bus1) | set(n.links.bus0) | set(n.links.bus1)
    isolated = [b for b in ac if b not in connected]
    if not isolated or len(ac) - len(isolated) == 0:
        return
    added = 0
    for b in isolated:
        same = [c for c in ac if c in connected and n.buses.country[c] == n.buses.country[b]]
        if not same:
            continue
        d = (n.buses.x[same] - n.buses.x[b]) ** 2 + (n.buses.y[same] - n.buses.y[b]) ** 2
        target = d.idxmin()
        # carrier "attach" (not DC) so set_transmission_limit leaves it alone; no cost, no length
        n.add(
            "Link",
            f"{b} attach",
            bus0=b,
            bus1=target,
            carrier="attach",
            p_nom=1e5,
            p_nom_min=1e5,
            p_nom_max=1e5,
            p_min_pu=-1.0,
            p_max_pu=1.0,
            efficiency=1.0,
            p_nom_extendable=False,
            capital_cost=0.0,
            length=0.0,
            underwater_fraction=0.0,
        )
        added += 1
    if added and "attach" not in n.carriers.index:
        n.add("Carrier", "attach", co2_emissions=0.0, nice_name="isolated-bus attachment")
    for col in n.links.columns:
        if n.links[col].dtype.kind == "f" and n.links[col].isna().any():
            n.links.loc[n.links.index.str.endswith(" attach"), col] = n.links.loc[
                n.links.index.str.endswith(" attach"), col].fillna(0.0)
    logger.info(f"attached {added} isolated buses to the nearest connected bus of their country")


def fix_capacities(n):
    """
    Freeze every generator and remove the extendable storage added by add_extra_components;
    attach isolated buses. Transmission is otherwise left to the ``ll`` wildcard (``v1.0``
    keeps the existing grid; ``copt`` lets the solve reinforce lines where the OSM ratings
    leave a cluster short of its load).
    """
    vre = n.generators.p_nom_extendable
    n.generators.loc[vre, "p_nom"] = n.generators.loc[vre, "p_nom_min"]
    n.generators["p_nom_extendable"] = False
    n.storage_units["p_nom_extendable"] = False
    stores = n.stores.index[n.stores.e_nom_extendable]
    buses = n.stores.loc[stores, "bus"].unique()
    links = n.links.index[n.links.bus0.isin(buses) | n.links.bus1.isin(buses)]
    n.mremove("Link", links)
    n.mremove("Store", stores)
    n.mremove("Bus", buses)
    logger.info(
        f"brownfield only: generators frozen; removed {len(stores)} extendable stores, "
        f"{len(links)} links and {len(buses)} buses"
    )
    attach_isolated_buses(n)


def calibrate(n, costs, config, inputs):
    """
    Apply the fleet representation and compute the solve-time targets.

    Parameters
    ----------
    n : pypsa.Network
    costs : pd.DataFrame
        cost table (index technology; columns VOM, ...)
    config : dict
        the ``calibration`` config block
    inputs : dict
        table name -> path, from the ``calibration.tables`` block

    Returns
    -------
    dict
        ``net_import_targets`` (country -> MWh) and ``energy_bands`` (country ->
        group -> [lower, upper] MWh) for the solve-time constraints; stored in
        ``n.meta["calibration"]`` by ``prepare_network``
    """
    split_tranches(n, config.get("efficiency_spread", 0.0), int(config.get("tranches", 1)), costs)
    if config.get("brownfield_only", False):
        fix_capacities(n)
    targets = {"net_import_targets": {}, "energy_bands": {}}
    if "net_imports" in inputs:
        targets["net_import_targets"] = net_import_targets(n, _read(inputs["net_imports"], "country"))
    if "generation" in inputs and config.get("energy_bands"):
        targets["energy_bands"] = energy_band_targets(
            n,
            _read(inputs["generation"], ["country", "group"]),
            config["energy_bands"],
            _read(inputs["demand"], "country") if "demand" in inputs else None,
        )
    return targets
