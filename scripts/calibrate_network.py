# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Calibrate a clustered network to per-country statistics of a reference year.

The functions here are called by ``prepare_network`` when ``calibration.enable``
is set. They act on the clustered network (``elec_s{simpl}_{clusters}_ec.nc``),
so a calibration change costs one ``prepare_network`` + ``solve_network`` run
and not the base-network build. Every step is optional: a step runs when its
table is configured under ``calibration.tables`` and is skipped otherwise.

Relevant Settings
-----------------

.. code:: yaml

    calibration:
        enable:
        brownfield_only:
        tables:
            demand:
            capacity:
            hydro:
            fuel_prices:
            co2_prices:
            envelope:

Inputs
------

All tables are csv files with a ``country`` column (ISO 3166-1 alpha-2). A row
with ``country`` ``*`` is the default for countries without an entry.

- ``demand``: ``country, demand_twh`` -- annual electricity demand; the load
  time series of each country are scaled so that they sum to it.
- ``capacity``: ``country, group, gw`` -- installed capacity per fuel group
  (``coal, gas, oil, nuclear, biomass, geothermal, wind, solar, hydro``; see
  ``GROUPS``). Existing generators of the group are scaled to it (``hydro``
  = run-of-river generators plus reservoir storage units, without pumped
  storage). A group with a target but no generator is added at the country's
  largest-load bus with the attributes of the group's default carrier from
  the cost table (not for wind, solar and hydro, which need a profile).
- ``hydro``: ``country, inflow_twh, max_hours`` -- annual hydro generation,
  spread over the country's reservoirs and run-of-river plants in proportion
  to capacity (the basin inflow of a plant often exceeds what it can pass),
  and the energy-to-power ratio of the reservoirs (optional column).
- ``generation``: ``country, group, twh, cf`` -- observed generation and
  capacity factors. The ``p_max_pu`` profiles of wind and solar are scaled per
  country and group so that their annual capacity factor matches (a
  per-country loss and weather-year correction of the resource model, factor
  clipped to ``VRE_CF_FACTOR``). With ``energy_bands`` (group -> [lower,
  upper] factor) the annual generation of the group's carriers is kept within
  that band around the observed value by a solve-time constraint
  (``add_energy_band_constraints``): price-dispatched fleets that policy,
  contracts or fuel supply hold below (or above) their merit-order position.
  Lower bounds are skipped for countries whose observed generation exceeds
  their demand by more than 5 % (exporters; single-node islanded runs cannot
  reproduce them) and for groups without a plant in the network.
- ``fuel_prices``: ``country, fuel, price`` -- fuel price in currency/MWh_th
  for ``gas, coal, lignite, oil, uranium, biomass``.
- ``co2_prices``: ``country, price`` -- CO2 price in currency/tCO2 applied to
  the marginal cost of every emitting generator of the country.
- ``envelope``: ``country, carrier, p_min_pu, p_max_pu, efficiency`` -- static
  must-run floor, availability cap and fleet-average efficiency per carrier
  (generators with a time-series ``p_max_pu`` are left alone). Empty cells
  leave the attribute unchanged.
- ``net_imports``: ``country, net_imports_twh`` -- annual net imports; the
  solve gets a band constraint of +/- ``net_import_tolerance`` (relative,
  with a floor of 1 TWh) around it for every country with cross-border lines
  or links (``add_net_import_constraints``, called from ``solve_network``).

With ``efficiency_spread`` > 0 every fuel-burning generator is split into
``tranches`` equal parts whose efficiencies are spread symmetrically around
the fleet average, so that the merit order of a country is a staircase
rather than one step per carrier (the fleet is heterogeneous in vintage).

Outputs
-------

The network is modified in place; ``calibrate`` returns the targets of the
solve-time constraints (net imports, energy bands), which ``prepare_network``
stores in ``n.meta["calibration"]`` for ``solve_network``.
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
VRE_GROUPS = ["wind", "solar"]

# generator carrier -> fuel row of the cost table
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


def calibrate_demand(n, table):
    """Scale each country's load time series to the annual demand of the table."""
    country = _country_of(n, "Load")
    w = _weights(n)
    current = n.loads_t.p_set.mul(w, axis=0).sum().groupby(country).sum() / 1e6
    for c, now in current.items():
        target = _lookup(table, c, "demand_twh")
        if np.isnan(target) or now <= 0:
            continue
        loads = country.index[country == c]
        n.loads_t.p_set[loads] *= target / now
        logger.info(f"demand {c}: {now:.1f} -> {target:.1f} TWh")


def _largest_load_bus(n, country):
    country_of_load = _country_of(n, "Load")
    loads = country_of_load.index[country_of_load == country]
    if loads.empty:
        return n.buses.index[n.buses.country == country][0]
    return n.loads.bus[n.loads_t.p_set[loads].sum().idxmax()]


def _scale_hydro_capacity(n, c, target):
    """Scale run-of-river generators and hydro reservoirs of country c to target MW."""
    g, su = n.generators, n.storage_units
    ror = g.index[(_country_of(n, "Generator") == c) & (g.carrier == "ror")]
    res = su.index[(_country_of(n, "StorageUnit") == c) & (su.carrier == "hydro")]
    now = g.loc[ror, "p_nom"].sum() + su.loc[res, "p_nom"].sum()
    if now <= 0:
        if target > 0:
            logger.warning(f"capacity {c} hydro: target {target/1e3:.2f} GW but no hydro plant; skipped")
        return
    f = target / now
    for comp, idx in [("Generator", ror), ("StorageUnit", res)]:
        for col in ["p_nom", "p_nom_min", "p_nom_max"]:
            n.df(comp).loc[idx, col] *= f
    logger.info(f"capacity {c} hydro: {now/1e3:.2f} -> {target/1e3:.2f} GW")


def calibrate_capacity(n, table, costs):
    """Scale the installed capacity per (country, fuel group) to the table."""
    countries = sorted(_country_of(n, "Generator").dropna().unique())
    for c in countries:
        for group, (carriers, default) in GROUPS.items():
            target = _lookup(table, (c, group), "gw")
            if np.isnan(target):
                continue
            target *= 1e3
            if group == "hydro":
                _scale_hydro_capacity(n, c, target)
                continue
            g = n.generators
            country = _country_of(n, "Generator")
            idx = g.index[(country == c) & g.carrier.isin(carriers)]
            now = g.loc[idx, "p_nom"].sum()
            if target <= 0:
                if now > 0:
                    logger.info(f"capacity {c} {group}: {now/1e3:.2f} -> 0 GW (removed)")
                    n.generators.loc[idx, ["p_nom", "p_nom_min", "p_nom_max"]] = 0.0
                continue
            if now > 0:
                share = g.loc[idx, "p_nom"] / now
            elif group in VRE_GROUPS and g.loc[idx, "p_nom_max"].sum() > 0:
                share = g.loc[idx, "p_nom_max"] / g.loc[idx, "p_nom_max"].sum()
            elif group in VRE_GROUPS:
                logger.warning(
                    f"capacity {c} {group}: target {target/1e3:.2f} GW but no potential; skipped"
                )
                continue
            else:
                bus = _largest_load_bus(n, c)
                name = f"{bus} {default}"
                n.add(
                    "Generator",
                    name,
                    bus=bus,
                    carrier=default,
                    p_nom=target,
                    p_nom_min=target,
                    p_nom_max=target,
                    p_nom_extendable=False,
                    efficiency=costs.at[default, "efficiency"],
                    marginal_cost=costs.at[default, "marginal_cost"],
                    capital_cost=costs.at[default, "capital_cost"],
                    lifetime=costs.at[default, "lifetime"],
                )
                logger.info(f"capacity {c} {group}: added {target/1e3:.2f} GW {default} at {bus}")
                continue
            new = share * target
            n.generators.loc[idx, "p_nom"] = new
            n.generators.loc[idx, "p_nom_min"] = new
            n.generators.loc[idx, "p_nom_max"] = np.maximum(g.loc[idx, "p_nom_max"], new)
            logger.info(f"capacity {c} {group}: {now/1e3:.2f} -> {target/1e3:.2f} GW")


VRE_CF_FACTOR = (0.6, 1.4)
MUST_RUN_LOAD_SHARE = 0.9


def calibrate_vre_cf(n, table):
    """Scale wind and solar profiles per (country, group) to the observed annual capacity factor."""
    w = _weights(n)
    country = _country_of(n, "Generator")
    g = n.generators
    hours = w.sum()
    for c in sorted(country.dropna().unique()):
        for group in VRE_GROUPS:
            target = _lookup(table, (c, group), "cf")
            if np.isnan(target) or target <= 0:
                continue
            idx = g.index[(country == c) & g.carrier.isin(GROUPS[group][0]) & (g.p_nom > 0)]
            idx = idx.intersection(n.generators_t.p_max_pu.columns)
            if idx.empty:
                continue
            p_nom = g.loc[idx, "p_nom"]
            energy = n.generators_t.p_max_pu[idx].mul(p_nom, axis=1).mul(w, axis=0).sum().sum()
            now = energy / (p_nom.sum() * hours)
            if now <= 0:
                continue
            f = float(np.clip(target / now, *VRE_CF_FACTOR))
            n.generators_t.p_max_pu[idx] = (n.generators_t.p_max_pu[idx] * f).clip(upper=1.0)
            logger.info(f"vre cf {c} {group}: {now:.3f} -> {target:.3f} (x{f:.2f})")


def calibrate_hydro(n, table):
    """
    Set the annual hydro energy of each country to the table and spread it over the
    country's plants in proportion to capacity (a uniform capacity factor): the
    atlite inflow of a plant is that of its basin and often exceeds what the plant
    can pass even with seasonal storage, which shows up as spill. Time profiles are
    kept per plant. Optionally set the reservoir energy-to-power ratio.
    """
    w = _weights(n)
    hours = w.sum()
    su = n.storage_units
    su_country = _country_of(n, "StorageUnit")
    gen_country = _country_of(n, "Generator")
    hydro = su.index[(su.carrier == "hydro") & (su.p_nom > 0)]
    hydro = hydro.intersection(n.storage_units_t.inflow.columns)
    ror = n.generators.index[(n.generators.carrier == "ror") & (n.generators.p_nom > 0)]
    ror = ror.intersection(n.generators_t.p_max_pu.columns)
    countries = set(su_country[hydro].dropna()) | set(gen_country[ror].dropna())
    for c in sorted(countries):
        h = hydro[su_country[hydro] == c]
        r = ror[gen_country[ror] == c]
        p_nom_h, p_nom_r = su.p_nom[h], n.generators.p_nom[r]
        inflow = n.storage_units_t.inflow[h].mul(w, axis=0).sum()
        ror_energy = n.generators_t.p_max_pu[r].mul(p_nom_r, axis=1).mul(w, axis=0).sum()
        now = (inflow.sum() + ror_energy.sum()) / 1e6
        target = _lookup(table, c, "inflow_twh")
        if not np.isnan(target) and now > 0:
            cf = target * 1e6 / ((p_nom_h.sum() + p_nom_r.sum()) * hours)
            if len(h):
                factor = (cf * p_nom_h * hours / inflow).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                n.storage_units_t.inflow[h] = n.storage_units_t.inflow[h] * factor
            if len(r):
                factor = (cf * p_nom_r * hours / ror_energy).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                scaled = n.generators_t.p_max_pu[r] * factor
                lost = (scaled - scaled.clip(upper=1.0)).mul(p_nom_r, axis=1).mul(w, axis=0).sum().sum() / 1e6
                n.generators_t.p_max_pu[r] = scaled.clip(upper=1.0)
                if lost > 0.01 * target and len(h):
                    n.storage_units_t.inflow[h] *= 1 + lost * 1e6 / (cf * p_nom_h.sum() * hours)
                elif lost > 0.01 * target:
                    logger.warning(f"hydro {c}: {lost:.2f} TWh of run-of-river above p_nom lost")
            logger.info(f"hydro {c}: {now:.1f} -> {target:.1f} TWh at a uniform capacity factor {cf:.2f}")
        if "max_hours" in table:
            mh = _lookup(table, c, "max_hours")
            if not np.isnan(mh) and len(h):
                n.storage_units.loc[h, "max_hours"] = mh
                logger.info(f"hydro {c}: reservoir max_hours {mh:.0f} h")
    phs = su.index[(su.carrier == "PHS") & (su.max_hours <= 0)]
    if len(phs):
        n.storage_units.loc[phs, "max_hours"] = 6.0
        logger.info(f"PHS: max_hours 0 -> 6 h for {len(phs)} units")


def calibrate_prices(n, costs, fuel_prices=None, co2_prices=None):
    """Recompute marginal costs from per-country fuel and CO2 prices."""
    country = _country_of(n, "Generator")
    g = n.generators
    emissions = n.carriers.co2_emissions.reindex(g.carrier).fillna(0.0).values
    for i, (name, row) in enumerate(g.iterrows()):
        c = country[name]
        if row.carrier not in FUELS:
            continue
        fuel = FUELS[row.carrier]
        price = costs.at[row.carrier, "fuel"] if "fuel" in costs else 0.0
        if fuel_prices is not None:
            p = _lookup(fuel_prices, (c, fuel), "price")
            if not np.isnan(p):
                price = p
        co2 = 0.0
        if co2_prices is not None:
            p = _lookup(co2_prices, c, "price")
            if not np.isnan(p):
                co2 = p
        vom = costs.at[row.carrier, "VOM"] if "VOM" in costs else 0.0
        eff = row.efficiency if row.efficiency > 0 else 1.0
        n.generators.at[name, "marginal_cost"] = vom + (price + co2 * emissions[i]) / eff
    summary = (
        n.generators.assign(country=country)
        .query("carrier in @FUELS")
        .groupby(["country", "carrier"])
        .marginal_cost.mean()
        .unstack()
        .round(1)
    )
    logger.info(f"marginal costs by country [currency/MWh_el]:\n{summary.to_string()}")


def calibrate_envelope(n, table):
    """Static p_min_pu / p_max_pu / efficiency per (country, carrier) for generators without a profile."""
    country = _country_of(n, "Generator")
    g = n.generators
    profiled = g.index.isin(n.generators_t.p_max_pu.columns)
    for name in g.index[~profiled]:
        key = (country[name], g.at[name, "carrier"])
        for attr in ["p_min_pu", "p_max_pu", "efficiency"]:
            if attr not in table:
                continue
            v = _lookup(table, key, attr)
            if not np.isnan(v):
                n.generators.at[name, attr] = v
        if n.generators.at[name, "p_min_pu"] > n.generators.at[name, "p_max_pu"]:
            n.generators.at[name, "p_min_pu"] = n.generators.at[name, "p_max_pu"]
    # a country's must-run total may not exceed MUST_RUN_LOAD_SHARE of its lowest load
    # (an islanded country cannot export it, and the LP would be infeasible)
    load_country = _country_of(n, "Load")
    min_load = n.loads_t.p_set.T.groupby(load_country).sum().T.min()
    g = n.generators
    must_run = (g.p_min_pu * g.p_nom).groupby(country).sum()
    for c, v in must_run.items():
        cap = MUST_RUN_LOAD_SHARE * min_load.get(c, np.inf)
        if v > cap > 0:
            idx = g.index[(country == c) & (g.p_min_pu > 0)]
            n.generators.loc[idx, "p_min_pu"] *= cap / v
            logger.warning(f"envelope {c}: must-run {v/1e3:.2f} GW above {MUST_RUN_LOAD_SHARE:.0%} of the minimum load "
                           f"{min_load[c]/1e3:.2f} GW; floors scaled by {cap / v:.2f}")
    summary = (
        n.generators.loc[~profiled]
        .assign(country=country)
        .groupby(["country", "carrier"])[["p_min_pu", "p_max_pu", "efficiency"]]
        .mean()
    )
    changed = summary[(summary.p_min_pu > 0) | (summary.p_max_pu < 1)]
    if not changed.empty:
        logger.info(f"envelopes:\n{changed.round(2).to_string()}")


def split_tranches(n, spread, tranches):
    """Split every fixed fuel-burning generator into tranches with spread efficiencies."""
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
    Apply every configured calibration step to ``n``.

    Parameters
    ----------
    n : pypsa.Network
    costs : pd.DataFrame
        cost table (index technology; columns fuel, VOM, efficiency, ...)
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
    if "demand" in inputs:
        calibrate_demand(n, _read(inputs["demand"], "country"))
    if "capacity" in inputs:
        calibrate_capacity(n, _read(inputs["capacity"], ["country", "group"]), costs)
    if "hydro" in inputs:
        calibrate_hydro(n, _read(inputs["hydro"], "country"))
    if "generation" in inputs:
        calibrate_vre_cf(n, _read(inputs["generation"], ["country", "group"]))
    if "envelope" in inputs:
        calibrate_envelope(n, _read(inputs["envelope"], ["country", "carrier"]))
    split_tranches(n, config.get("efficiency_spread", 0.0), int(config.get("tranches", 1)))
    if "fuel_prices" in inputs or "co2_prices" in inputs:
        calibrate_prices(
            n,
            costs,
            _read(inputs["fuel_prices"], ["country", "fuel"]) if "fuel_prices" in inputs else None,
            _read(inputs["co2_prices"], "country") if "co2_prices" in inputs else None,
        )
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
