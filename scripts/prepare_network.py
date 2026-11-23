# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such
as.

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- limiting long lines by their **loadability** (``lines: loadability``),
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours
  or segmenting time series into chunks of varying lengths using ``tsam``.

Relevant Settings
-----------------

.. code:: yaml

    links:

    lines:

    co2:
        limit:
        base:
        emission_price:
        automatic_emission:
        budget:

    electricity:
        max_hours:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`co2_cf`, :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv.``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Complete PyPSA network that will be handed to the ``solve_network`` rule.

Description
-----------

.. tip::
    The rule :mod:`prepare_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`prepare_network`.
"""
import re

import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    create_logger,
    sanitize_carriers,
    sanitize_locations,
)
from add_electricity import update_transmission_costs

idx = pd.IndexSlice

logger = create_logger(__name__)


def emission_extractor(
    emission_csv: str, emission_year: int | str, country_names: np.ndarray | list
) -> pd.Series:
    """
    Extracts CO2 emission values for given country codes from the pre-processed
    global emission CSV file.

    Parameters
    ----------
    emission_csv : str
        Path to the CSV file produced by build_co2_emissions. The file has
        columns country_code_a3 (three-letter ISO code), country_code_a2
        (two-letter ISO code), country_name (full country name), and year columns
        named Y_YYYY.
    emission_year : int or str
        Year of CO2 emissions.
    country_names : numpy.ndarray
        Two letter country codes of analysed countries.

    Returns
    -------
    pd.Series
        CO2 emission values (in kt CO2) of studied countries for the given year.
    """
    emission_year = int(emission_year)
    df = pd.read_csv(emission_csv).set_index("country_code_a2")

    # track the available years
    available_years = sorted(
        df.columns.difference(
            ["country_code_a3", "country_code_a2", "country_name"]
        ).map(int)
    )
    # find the closest available year: if present, distance is 0, otherwise the closest year is used
    closest_year = min(available_years, key=lambda y: abs(y - emission_year))
    if emission_year not in available_years:
        logger.warning(
            f"Emission year {emission_year} not found in data. "
            f"Using closest available year {closest_year} instead."
        )

    country_codes = np.atleast_1d(country_names).tolist()
    available_countries = df.index.intersection(country_codes)
    emission_by_country = df.loc[available_countries, str(emission_year)]
    if emission_by_country.index.has_duplicates:
        duplicate_countries = emission_by_country.index[
            emission_by_country.index.duplicated()
        ].unique()
        logger.warning(
            "Multiple emission values found for the following countries: "
            f"{list(duplicate_countries)}. Summing duplicate entries."
        )
        emission_by_country = emission_by_country.groupby(level=0).sum()

    missing_ccs = np.setdiff1d(country_codes, available_countries)
    if missing_ccs.size:
        logger.warning(
            f"The emission value for the following countries has not been found: {missing_ccs}"
        )
    if emission_by_country.empty:
        raise ValueError(
            f"No CO2 emission data could be extracted from '{emission_csv}' for year "
            f"{emission_year} and countries {country_codes}. "
            "The automatic CO2 limit cannot be derived from an empty result. "
            "Please check the emission data file, the requested base year, or the "
            "configured countries."
        )
    return emission_by_country


def add_co2limit(n, annual_emissions, Nyears=1.0):
    n.add(
        "GlobalConstraint",
        "CO2Limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=annual_emissions * Nyears,
    )


def add_gaslimit(n, gaslimit, Nyears=1.0):
    sel = n.carriers.index.intersection(["OCGT", "CCGT", "CHP"])
    n.carriers.loc[sel, "gas_usage"] = 1.0

    n.add(
        "GlobalConstraint",
        "GasLimit",
        carrier_attribute="gas_usage",
        sense="<=",
        constant=gaslimit * Nyears,
    )


def add_emission_prices(n, co2_price=0.0):
    ep = (
        pd.Series({"co2_emissions": co2_price}) * n.carriers.filter(like="_emissions")
    ).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep


def add_emission_prices_by_country(n, fn):
    """
    Add a CO2 price per country (``co2: emission_prices_by_country``: csv with the
    columns ``country`` and ``price`` in currency/tCO2, a row ``*`` is the default) to
    the marginal costs of the emitting generators and storage units of the country.
    """
    table = pd.read_csv(fn, comment="#").set_index("country")["price"]
    emissions = n.carriers.filter(like="_emissions").sum(axis=1)
    for c, eff in [("Generator", "efficiency"), ("StorageUnit", "efficiency_dispatch")]:
        df = n.df(c)
        price = df.bus.map(n.buses.country).map(table).fillna(table.get("*", 0.0))
        intensity = df.carrier.map(emissions).fillna(0.0)
        emitting = intensity > 0
        df.loc[emitting, "marginal_cost"] += (
            price * intensity / df[eff]
        )[emitting]
    logger.info(f"CO2 prices by country from {fn}: " + ", ".join(
        f"{k} {v:.1f}" for k, v in table.items()))


def set_line_s_max_pu(n, s_max_pu):
    n.lines["s_max_pu"] = s_max_pu
    logger.info(f"N-1 security margin of lines set to {s_max_pu}")


# St. Clair's loadability curve of overhead lines without compensation: line length
# in miles, load in multiples of the surge impedance loading (SIL). 3.0 SIL at 50
# miles and 1.0 SIL at 300 miles are its two benchmark points (St. Clair 1953, AIEE
# Transactions 72, 1152-1157); the points between and beyond are read off the curve
# as reproduced in Sekhon 1981 (MSc thesis, University of Manitoba, Fig. 3.1.1).
ST_CLAIR = [
    (50, 3.0),
    (75, 2.45),
    (100, 2.05),
    (150, 1.6),
    (200, 1.35),
    (250, 1.15),
    (300, 1.0),
    (400, 0.8),
    (500, 0.7),
    (600, 0.62),
]


def st_clair(length_km, curve=ST_CLAIR):
    """
    Loadability in multiples of SIL of a line of the given length: the curve
    interpolated between its points on logarithmic axes and continued with its
    first slope below its first point and with its last slope beyond its last.
    """
    miles, k = np.log(np.array(curve, dtype=float)).T
    x = np.log(np.maximum(np.asarray(length_km, dtype=float) / 1.609344, 1e-9))
    below = k[0] + (k[1] - k[0]) / (miles[1] - miles[0]) * (x - miles[0])
    beyond = k[-1] + (k[-1] - k[-2]) / (miles[-1] - miles[-2]) * (x - miles[-1])
    inside = np.interp(x, miles, k)
    k = np.where(x < miles[0], below, np.where(x > miles[-1], beyond, inside))
    return np.exp(k)


def set_line_loadability(n, base, busmaps, curve=ST_CLAIR):
    """
    Limit long AC circuits by their loadability instead of their thermal rating.

    A line's capacity ``s_nom`` is the thermal rating of its conductors, whatever
    its length. A long line is limited earlier, by voltage drop and by angle
    stability. Each AC circuit of the base network, where every circuit still has
    its own length, voltage and line type, is limited to
    ``min(thermal, k(L) * SIL)``:

    - ``thermal`` is its ``s_nom`` and ``SIL = v_nom^2 / sqrt(x / (omega c))``
      per circuit its surge impedance loading;
    - ``k(L)`` is the loadability curve (:func:`st_clair`) at its length.

    The usable share ``s_max_pu`` of a clustered line is multiplied by the sum of
    these limits over the circuits between its two clusters (``busmaps``: base
    bus to simplified bus to cluster), divided by the sum of their thermal
    ratings. The length of the clustered line itself, the distance between two
    cluster centres, does not enter, so the result does not depend on the number
    of clusters. The security margin set before (``lines: s_max_pu``) so applies
    to whichever limit binds. Capacity added to a line by the optimisation is
    limited alike. HVDC links are not touched. Returns one row per clustered AC
    line.
    """
    ac = n.lines.index[n.lines.carrier != "DC"]
    if ac.empty:
        return pd.DataFrame()

    types = base.line_types
    surge_impedance = np.sqrt(
        types.x_per_length / (2 * np.pi * types.f_nom * types.c_per_length * 1e-9)
    )
    circuits = base.lines[(base.lines.carrier != "DC") & (base.lines.s_nom > 0)]
    sil = (
        circuits.v_nom**2
        / circuits["type"].map(surge_impedance)
        * circuits.num_parallel
    )
    loadable = np.minimum(circuits.s_nom, st_clair(circuits.length, curve) * sil)
    cluster = pd.Series(base.buses.index, base.buses.index)
    for busmap in busmaps:
        cluster = cluster.map(busmap.astype(str).rename(index=str))
    ends = pd.DataFrame(
        {"bus0": circuits.bus0.map(cluster), "bus1": circuits.bus1.map(cluster)}
    ).dropna()
    ends = ends[ends.bus0 != ends.bus1]
    pair = ends.min(axis=1) + " " + ends.max(axis=1)
    sums = (
        pd.DataFrame(
            {
                "thermal": circuits.s_nom,
                "sil": sil,
                "loadable": loadable,
                "km": circuits.length * circuits.s_nom,
            }
        )
        .groupby(pair)
        .sum()
    )

    df = pd.DataFrame({"length": n.lines.length[ac], "s_nom": n.lines.s_nom[ac]})
    pair = n.lines.loc[ac, ["bus0", "bus1"]]
    pair = pair.min(axis=1) + " " + pair.max(axis=1)
    for col in ["thermal", "sil", "loadable"]:
        df[col] = pair.map(sums[col])
    df["circuit_length"] = pair.map(sums.km / sums.thermal)
    # a line without circuits between its clusters in the base network (a
    # transformer kept as a line) takes the share of all circuits between clusters
    df["factor"] = (df.loadable / df.thermal).fillna(
        sums.loadable.sum() / sums.thermal.sum() if len(sums) else 1.0
    )
    before = n.lines.s_max_pu[ac]
    n.lines.loc[ac, "s_max_pu"] = before * df.factor
    df["s_max_pu"] = n.lines.s_max_pu[ac]

    logger.info(
        f"Loadability of {len(ac)} AC lines: {(df.factor < 1).sum()} with circuits "
        f"limited below their thermal rating, {df.thermal.notna().sum()} with "
        f"circuits between their clusters in the base network; s_max_pu "
        f"{df.s_max_pu.min():.2f} to {df.s_max_pu.max():.2f} (median "
        f"{df.s_max_pu.median():.2f}); sum of line limits "
        f"{(before * df.s_nom).sum() / 1e3:.0f} GW -> "
        f"{(df.s_max_pu * df.s_nom).sum() / 1e3:.0f} GW"
    )
    return df


def set_transmission_limit(n, ll_type, factor, costs, lines, links):
    links_dc_b = n.links.carrier == "DC" if not n.links.empty else pd.Series()

    _lines_s_nom = (
        np.sqrt(3)
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.num_parallel
        * n.lines.bus0.map(n.buses.v_nom)
    )
    lines_s_nom = n.lines.s_nom.where(n.lines.type == "", _lines_s_nom)

    col = "capital_cost" if ll_type == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[links_dc_b, "p_nom"] @ n.links.loc[links_dc_b, col]
    )

    update_transmission_costs(n, costs)

    if factor == "opt" or float(factor) > 1.0:
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True

        n.links.loc[links_dc_b, "p_nom_min"] = n.links.loc[links_dc_b, "p_nom"]
        n.links.loc[links_dc_b, "p_nom_extendable"] = True

    if ll_type == "l":
        n.lines["s_nom_max"] = n.lines["s_nom"] * float(factor)
        n.links.loc[links_dc_b, "p_nom_max"] = n.links.loc[links_dc_b, "p_nom"] * float(
            factor
        )
    elif factor != "opt":  # implicitly also ll_type != "l"
        con_type = "expansion_cost" if ll_type == "c" else "volume_expansion"
        rhs = float(factor) * ref
        n.add(
            "GlobalConstraint",
            f"l{ll_type}_limit",
            type=f"transmission_{con_type}_limit",
            sense="<=",
            constant=rhs,
            carrier_attribute="AC, DC",
        )

    set_line_nom_max(n, lines, links)

    return n


def average_every_nhours(n, offset):

    # Please note that a casefold is applied to the offset during the resampling.
    # This is because the usage of offset where the time interval is in capital
    # letters is deprecated in future versions of pandas.
    # For example 24H is deprecated. Instead, 24h is allowed.

    logger.info(f"Resampling the network to {offset}")
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset.casefold()).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                pnl[k] = df.resample(offset.casefold()).mean()

    return m


def apply_time_segmentation(n, segments, solver_name):
    logger.info(f"Aggregating time series to {segments} segments.")
    try:
        import tsam.timeseriesaggregation as tsam
    except:
        raise ModuleNotFoundError(
            "Optional dependency 'tsam' not found." "Install via 'pip install tsam'"
        )

    p_max_pu_norm = n.generators_t.p_max_pu.abs().max().clip(lower=1e-6)
    p_max_pu = n.generators_t.p_max_pu / p_max_pu_norm

    load_norm = n.loads_t.p_set.abs().max().clip(lower=1e-6)
    load = n.loads_t.p_set / load_norm

    inflow_norm = n.storage_units_t.inflow.abs().max().clip(lower=1e-6)
    inflow = n.storage_units_t.inflow / inflow_norm

    raw = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)

    agg = tsam.TimeSeriesAggregation(
        raw,
        hoursPerPeriod=len(raw),
        noTypicalPeriods=1,
        noSegments=int(segments),
        segmentation=True,
        solver=solver_name,
    )

    segmented = agg.createTypicalPeriods()

    weightings = segmented.index.get_level_values("Segment Duration")
    offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
    snapshots = [n.snapshots[0] + pd.Timedelta(f"{offset}h") for offset in offsets]

    n.set_snapshots(pd.DatetimeIndex(snapshots, name="name"))
    n.snapshot_weightings = pd.Series(
        weightings, index=snapshots, name="weightings", dtype="float64"
    )

    segmented.index = snapshots
    n.generators_t.p_max_pu = segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm
    n.loads_t.p_set = segmented[n.loads_t.p_set.columns] * load_norm
    n.storage_units_t.inflow = segmented[n.storage_units_t.inflow.columns] * inflow_norm

    return n


def enforce_autarky(n, only_crossborder=False):
    if only_crossborder:
        # storage buses of add_extra_components (e.g. "<bus> battery") carry no country yet, only a location:
        # take the country of the location bus, otherwise their charger / discharger links count as cross-border
        country = n.buses.country.where(
            n.buses.country.fillna("") != "",
            n.buses.get("location", n.buses.country).map(n.buses.country),
        )
        lines_rm = n.lines.loc[
            n.lines.bus0.map(country) != n.lines.bus1.map(country)
        ].index
        links_rm = n.links.loc[
            n.links.bus0.map(country) != n.links.bus1.map(country)
        ].index
    else:
        lines_rm = n.lines.index
        links_rm = n.links.loc[n.links.carrier == "DC"].index
    n.mremove("Line", lines_rm)
    n.mremove("Link", links_rm)


def set_line_nom_max(n, lines, links):
    s_max, s_min = lines.get("s_nom_max"), lines.get("s_nom_max_min")
    p_max, p_min = links.get("p_nom_max"), links.get("p_nom_max_min")
    n.lines.s_nom_max = n.lines.s_nom_max.clip(lower=s_min, upper=s_max)
    n.links.p_nom_max = n.links.p_nom_max.clip(lower=p_min, upper=p_max)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_network",
            simpl="",
            clusters="6",
            ll="copt",
            opts="Co2L-4H",
            # configfile="test/config.sector.yaml",
        )

    configure_logging(snakemake)

    opts = snakemake.wildcards.opts.split("-")

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = pd.read_csv(snakemake.input.tech_costs, index_col=0)
    s_max_pu = snakemake.params.lines["s_max_pu"]

    calibration = snakemake.params.get("calibration", {}) or {}
    if calibration.get("enable", False):
        from calibrate_network import calibrate

        tables = {
            k[len("calibration_") :]: v
            for k, v in snakemake.input.items()
            if k.startswith("calibration_")
        }
        calibration_targets = calibrate(n, costs, calibration, tables)

    set_line_s_max_pu(n, s_max_pu)
    if snakemake.input.get("base_network"):
        set_line_loadability(
            n,
            pypsa.Network(snakemake.input.base_network),
            [
                pd.read_csv(fn, index_col=0, dtype=str).squeeze("columns")
                for fn in snakemake.input.busmaps
            ],
        )

    for o in opts:
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break

    for o in opts:
        m = re.match(r"^\d+seg$", o, re.IGNORECASE)
        if m is not None:
            solver_name = snakemake.config["solving"]["solver"]["name"]
            n = apply_time_segmentation(n, m.group(0)[:-3], solver_name)
            break

    for o in opts:
        if "Co2L" in o:
            m = re.findall(r"[0-9]*\.?[0-9]+$", o)
            if snakemake.params.co2["automatic_emission"]["enable"]:
                country_names = n.buses.country.unique()
                emission_year = snakemake.params.co2["automatic_emission"]["base_year"]
                co2limit = emission_extractor(
                    snakemake.input.emissions, emission_year, country_names
                ).sum()
                if len(m) > 0:
                    co2limit = co2limit * float(m[0])
                logger.info("Setting CO2 limit according to emission base year.")
            elif len(m) > 0:
                co2limit = float(m[0]) * float(snakemake.params.co2["base"])
                logger.info("Setting CO2 limit according to wildcard value.")
            else:
                co2limit = float(snakemake.params.co2["limit"])
                logger.info("Setting CO2 limit according to config value.")
            add_co2limit(n, co2limit, Nyears)
            break

    for o in opts:
        if "CH4L" in o:
            m = re.findall(r"[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                limit = float(m[0]) * 1e6
                add_gaslimit(n, limit, Nyears)
                logger.info("Setting gas usage limit according to wildcard value.")
            else:
                add_gaslimit(n, snakemake.params.electricity.get("gaslimit"), Nyears)
                logger.info("Setting gas usage limit according to config value.")

    for o in opts:
        oo = o.split("+")
        suptechs = map(lambda c: c.split("-", 2)[0], n.carriers.index)
        if oo[0].startswith(tuple(suptechs)):
            carrier = oo[0]
            # handles only p_nom_max as stores and lines have no potentials
            attr_lookup = {
                "p": "p_nom_max",
                "c": "capital_cost",
                "m": "marginal_cost",
            }
            attr = attr_lookup[oo[1][0]]
            factor = float(oo[1][1:])
            if carrier == "AC":  # lines do not have carrier
                n.lines[attr] *= factor
            else:
                comps = {"Generator", "Link", "StorageUnit", "Store"}
                for c in n.iterate_components(comps):
                    sel = c.df.carrier.str.contains(carrier)
                    c.df.loc[sel, attr] *= factor

    if snakemake.input.get("co2_prices"):
        if any("Ep" in o for o in opts):
            raise ValueError("co2: emission_prices_by_country and the Ep option both price CO2")
        add_emission_prices_by_country(n, snakemake.input.co2_prices)

    for o in opts:
        if "Ep" in o:
            m = re.findall(r"[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                logger.info("Setting emission prices according to wildcard value.")
                add_emission_prices(n, float(m[0]))
            else:
                logger.info("Setting emission prices according to config value.")
                add_emission_prices(n, snakemake.params.co2["emission_price"])
            break

    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    lines = snakemake.params.lines
    links = snakemake.params.links
    set_transmission_limit(n, ll_type, factor, costs, lines, links)

    if "ATK" in opts:
        enforce_autarky(n)
    elif "ATKc" in opts:
        enforce_autarky(n, only_crossborder=True)

    sanitize_carriers(n, snakemake.config)
    sanitize_locations(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    if calibration.get("enable", False):
        n.meta["calibration"] = dict(calibration, **calibration_targets)
    n.export_to_netcdf(snakemake.output[0])
