# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Adds extra extendable components to the clustered and simplified network.

Relevant Settings
-----------------

.. code:: yaml

    sector:
        transmission_efficiency:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`sector_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec.nc``:


Description
-----------

The rule :mod:`add_extra_components` attaches additional extendable components to the clustered and simplified network. These can be configured in the ``config.yaml`` at ``electricity: extendable_carriers:``. It processes ``networks/elec_s{simpl}_{clusters}.nc`` to build ``networks/elec_s{simpl}_{clusters}_ec.nc``, which in contrast to the former (depending on the configuration) contain with **zero** initial capacity

- ``StorageUnits`` of carrier 'H2' and/or 'battery'. If this option is chosen, every bus is given an extendable ``StorageUnit`` of the corresponding carrier. The energy and power capacities are linked through a parameter that specifies the energy capacity as maximum hours at full dispatch power and is configured in ``electricity: max_hours:``. This linkage leads to one investment variable per storage unit. The default ``max_hours`` lead to long-term hydrogen and short-term battery storage units.

- ``Stores`` of carrier 'H2' and/or 'battery' in combination with ``Links``. If this option is chosen, the script adds extra buses with corresponding carrier where energy ``Stores`` are attached and which are connected to the corresponding power buses via two links, one each for charging and discharging. This leads to three investment variables for the energy capacity, charging and discharging capacity of the storage unit.
"""

import os

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import (
    configure_logging,
    create_logger,
    lossy_bidirectional_links,
    set_length_based_efficiency,
)
from add_electricity import add_nice_carrier_names
from process_cost_data import annuity

idx = pd.IndexSlice

logger = create_logger(__name__)


def get_available_storage_carriers(
    n: pypsa.Network, carriers: list, storage_techs: dict
) -> list:
    """
    Filter and register available storage carriers from a given list.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object to which valid storage carriers will be added.
    carriers : list of str
        A list of carrier names to filter and potentially register as storage technologies.
    storage_techs : dict
        A dictionary mapping storage carriers to their respective technology parameters.

    Returns
    -------
    list of str
        A list of storage carriers that are both implemented and available in the network.
    """
    implemented = set(storage_techs.keys())
    input_carriers = set(carriers)

    not_implemented = input_carriers - implemented
    if not_implemented:
        logger.warning(
            f"The following carriers are not implemented as storage technologies in PyPSA-Earth:\n - "
            + "\n - ".join(not_implemented)
            + "\nPlease include these carriers and their respective store, bicharger or (charger and discharger) names in the technology data."
        )

    available_carriers = sorted(input_carriers & implemented)

    return available_carriers


def attach_stores(
    n: pypsa.Network,
    costs: pd.DataFrame,
    buses_i: list,
    carriers: list,
    storage_techs: dict,
) -> None:
    """
    Attach stores to the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to attach the stores to.
    costs : pd.DataFrame
        DataFrame containing the cost data.
    buses_i : list
        List of high voltage electricity buses.
    carriers : list
        List of extendable energy carriers.
    storage_techs : dict
        A dictionary mapping storage carriers to their respective technology parameters.
    """
    available_carriers = get_available_storage_carriers(n, carriers, storage_techs)

    for carrier in available_carriers:
        roundtrip_correction = 0.5 if carrier in ["battery", "li-ion"] else 1

        lookup = storage_techs[carrier]
        lookup_store = lookup["store"]
        if "bicharger" in lookup:
            lookup_charge = lookup_discharge = lookup["bicharger"]
        else:
            lookup_charge = lookup["charger"]
            lookup_discharge = lookup["discharger"]

        bus_names = buses_i + f" {carrier}"
        charge_name = "Electrolysis" if lookup_charge == "electrolysis" else "charger"
        discharge_name = (
            "Fuel Cell" if lookup_discharge == "fuel cell" else "discharger"
        )

        discharge_capital_cost = (
            0.0 if "bicharger" in lookup else costs.at[lookup_discharge, "capital_cost"]
        )
        if lookup_discharge == "fuel cell":
            # NB: fuel cell investment cost is per MWel
            discharge_capital_cost *= costs.at[lookup_discharge, "efficiency"]

        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)

        n.madd(
            "Bus",
            bus_names,
            location=buses_i,
            carrier=carrier,
            x=n.buses.loc[list(buses_i)].x.values,
            y=n.buses.loc[list(buses_i)].y.values,
        )

        n.madd(
            "Store",
            bus_names,
            bus=bus_names,
            e_cyclic=True,
            e_nom_extendable=True,
            carrier=carrier,
            capital_cost=costs.at[lookup_store, "capital_cost"],
            lifetime=costs.at[lookup_store, "lifetime"],
        )

        n.madd("Carrier", [f"{carrier} {charge_name}", f"{carrier} {discharge_name}"])

        n.madd(
            "Link",
            bus_names,
            suffix=f" {charge_name}",
            bus0=buses_i,
            bus1=bus_names,
            carrier=f"{carrier} {charge_name.lower()}",
            efficiency=costs.at[lookup_charge, "efficiency"] ** roundtrip_correction,
            capital_cost=costs.at[lookup_charge, "capital_cost"],
            marginal_cost=costs.at[lookup_charge, "marginal_cost"],
            p_nom_extendable=True,
            lifetime=costs.at[lookup_charge, "lifetime"],
        )

        n.madd(
            "Link",
            bus_names,
            suffix=f" {discharge_name}",
            bus0=bus_names,
            bus1=buses_i,
            carrier=f"{carrier} {discharge_name.lower()}",
            efficiency=costs.at[lookup_discharge, "efficiency"] ** roundtrip_correction,
            capital_cost=discharge_capital_cost,
            marginal_cost=costs.at[lookup_discharge, "marginal_cost"],
            p_nom_extendable=True,
            lifetime=costs.at[lookup_discharge, "lifetime"],
        )

    logger.info(
        "Add the following technologies as stores and links:\n - "
        + "\n - ".join(available_carriers)
    )


def attach_storageunits(
    n: pypsa.Network,
    costs: pd.DataFrame,
    buses_i: list,
    carriers: list,
    max_hours: dict,
    storage_techs: dict,
) -> None:
    """
    Attach storage units to the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to attach the storage units to.
    costs : pd.DataFrame
        DataFrame containing the cost data.
    buses_i : list
        List of high voltage electricity buses.
    carriers : list
        List of extendable energy carriers.
    max_hours : dict
        Dictionary of maximum hours for storage units.
    storage_techs : dict
        A dictionary mapping storage carriers to their respective technology parameters.
    """
    available_carriers = get_available_storage_carriers(n, carriers, storage_techs)

    for carrier in available_carriers:
        max_hour = max_hours.get(carrier)
        if max_hour is None:
            logger.warning(f"No max_hours defined for carrier '{carrier}'. Skipping.")
            continue

        roundtrip_correction = 0.5 if carrier in ["battery", "li-ion"] else 1

        lookup = storage_techs[carrier]
        if "bicharger" in lookup:
            lookup_charge = lookup_discharge = lookup["bicharger"]
        else:
            lookup_charge = lookup["charger"]
            lookup_discharge = lookup["discharger"]

        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)

        n.madd(
            "StorageUnit",
            buses_i + f" {carrier}",
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency_store=costs.at[lookup_charge, "efficiency"]
            ** roundtrip_correction,
            efficiency_dispatch=costs.at[lookup_discharge, "efficiency"]
            ** roundtrip_correction,
            max_hours=max_hour,
            cyclic_state_of_charge=True,
            lifetime=costs.at[carrier, "lifetime"],
        )

    logger.info(
        "Add the following technologies as storage units:\n - "
        + "\n - ".join(available_carriers)
    )


def attach_advance_csp(
    n: pypsa.Network,
    costs: pd.DataFrame,
) -> None:
    """
    Attach advance CHP to the network.

    Advance CHP is a hybrid of generation and storage, needing its own workflow

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to attach the storage units to.
    costs : pd.DataFrame
        DataFrame containing the cost data.
    cost_name : str
        Name of the column in the costs DataFrame that represents the capital cost.
    """
    # add separate buses for csp
    main_buses = n.generators.query("carrier == 'csp'").bus
    csp_buses_i = n.madd(
        "Bus",
        main_buses + " csp",
        carrier="csp",
        x=n.buses.loc[main_buses, "x"].values,
        y=n.buses.loc[main_buses, "y"].values,
        country=n.buses.loc[main_buses, "country"].values,
    )
    n.generators.loc[main_buses.index, "bus"] = csp_buses_i

    # add stores for csp
    n.madd(
        "Store",
        csp_buses_i,
        bus=csp_buses_i,
        carrier="csp",
        e_cyclic=True,
        e_nom_extendable=True,
        capital_cost=costs.at["csp-tower TES", "capital_cost"],
        marginal_cost=costs.at["csp-tower TES", "marginal_cost"],
    )

    # add links for csp
    n.madd(
        "Link",
        csp_buses_i,
        bus0=csp_buses_i,
        bus1=main_buses,
        carrier="csp",
        efficiency=costs.at["csp-tower", "efficiency"],
        capital_cost=costs.at["csp-tower", "capital_cost"],
        p_nom_extendable=True,
        marginal_cost=costs.at["csp-tower", "marginal_cost"],
    )


def _class_shares(fleet: np.ndarray, cf: np.ndarray, pot: np.ndarray) -> np.ndarray:
    """Fleet shares per class: the located plants, else CF x potential (PyPSA-Eur's rule for statistics)."""
    w = fleet if fleet.sum() > 0 else cf * pot
    w = np.where(pot > 0, w, 0.0)
    return w / w.sum() if w.sum() > 0 else np.where(pot > 0, pot, 0.0) / max(pot.sum(), 1e-12)


def _fold_small(p: np.ndarray, threshold: float) -> np.ndarray:
    """Move fleet parts below ``threshold`` into the largest part, so none of it is dropped at the next horizon."""
    p = p.copy()
    if p.sum() <= 0:
        return p
    small = (p > 0) & (p < threshold)
    while small.any() and (p >= threshold).any():
        k = int(np.argmax(p))
        p[k] += p[small].sum()
        p[small] = 0.0
        small = (p > 0) & (p < threshold)
    if small.sum() > 1:  # the whole fleet is below the threshold: one part
        k = int(np.argmax(p))
        p[k] = p.sum()
        p[np.arange(len(p)) != k] = 0.0
    return p


def attach_resource_classes(
    n: pypsa.Network, ds: xr.Dataset, carrier: str, threshold: float, iterations: int = 5
) -> pd.DataFrame:
    """
    Split the one generator ``"<bus> <carrier>"`` per clustered bus into resource classes.

    One generator ``"<bus> <carrier> c<k>"`` (column ``resource_class`` = k) per class of ``build_resource_classes``
    with potential. The split is a disaggregation of the generator G it replaces, all other attributes copied:

    - fleet (``p_nom = p_nom_min``): G's calibrated fleet in the shares of the located plants per class (without
      located plants: CF x potential), parts below 1.05 x ``threshold`` folded into the largest one;
    - potential: ``p_nom_max_k = p_nom_k + headroom_k``, G's headroom (``p_nom_max - p_nom``) shared in proportion
      to the room left in each class, so the classes add up to G's potential also where a class holds more
      fleet than its own potential;
    - profiles: the class profiles times one factor per country, chosen so that the fleet produces G's annual
      energy (the capacity-factor calibration of ``add_electricity`` stays as it is; ``iterations`` fixed-point
      steps for the clip at 1).

    Buses without classes keep G. Returns a table per bus for the log.
    """
    g = n.generators
    w = n.snapshot_weightings.generators
    if not ds.indexes["time"].equals(n.snapshots):
        missing = n.snapshots.difference(ds.indexes["time"])
        if len(missing):
            raise ValueError(f"resource classes {carrier}: {len(missing)} snapshots not in the class profiles")
    profile_all = ds["profile"].sel(time=n.snapshots)
    sel = g.index[(g.carrier == carrier) & g.bus.isin(ds.indexes["bus"])]
    dup = g.loc[sel].bus.duplicated()
    if dup.any() or (g.loc[sel, "build_year"] != 0).any():
        raise ValueError(
            f"resource classes {carrier}: expected one generator per bus without build year, got "
            f"{g.loc[sel].bus[dup].tolist()} duplicated / build years {sorted(g.loc[sel, 'build_year'].unique())}"
        )
    no_g = ds.indexes["bus"].difference(g.loc[sel].bus)
    if len(no_g):
        logger.info(f"resource classes {carrier}: no generator at {len(no_g)} buses with classes: {list(no_g)[:5]}")

    rows, series, report, parent = [], {}, [], {}
    pmpu = n.generators_t.p_max_pu
    for G in sel:
        G_row = g.loc[G]
        bus = G_row.bus
        pot = ds.p_nom_max.sel(bus=bus).values
        if pot.sum() <= 0:
            continue
        ks = np.flatnonzero(pot > 0)
        shares = _class_shares(ds.fleet_mw.sel(bus=bus).values, ds.cf.sel(bus=bus).values, pot)
        fleet = _fold_small(G_row.p_nom * shares, 1.05 * threshold)
        fleet_min = fleet * (G_row.p_nom_min / G_row.p_nom) if G_row.p_nom > 0 else np.zeros_like(fleet)
        room = np.clip(pot - fleet, 0.0, None)
        head = max(G_row.p_nom_max - G_row.p_nom, 0.0) if np.isfinite(G_row.p_nom_max) else np.inf
        if np.isfinite(head):
            pnm = fleet + (head * room / room.sum() if room.sum() > 0 else head * pot / pot.sum())
        else:
            pnm = np.full_like(pot, np.inf)
        prof = profile_all.sel(bus=bus).to_pandas()  # time x bin
        for k in ks:
            name = f"{G} c{k}"
            r = G_row.copy()
            r["p_nom"], r["p_nom_min"], r["p_nom_max"] = fleet[k], fleet_min[k], pnm[k]
            r["weight"] = pot[k]
            r["resource_class"] = float(k)
            r.name = name
            rows.append(r)
            parent[name] = G
            series[name] = prof[k].values
        report.append(
            dict(generator=G, bus=bus, country=n.buses.at[bus, "country"], p_nom=G_row.p_nom,
                 p_nom_max=G_row.p_nom_max, class_potential=pot.sum(),
                 E_before=float((pmpu[G] * w).sum() * G_row.p_nom) if G in pmpu else np.nan,
                 classes=len(ks))
        )
    if not rows:
        return pd.DataFrame()
    new = pd.DataFrame(rows)
    prof = pd.DataFrame(series, index=n.snapshots)

    # energy-neutral profile factor per country
    rep = pd.DataFrame(report).set_index("generator")
    country = new.bus.map(n.buses.country)
    factors = {}
    for c, idx in new.groupby(country).groups.items():
        e_before = rep.loc[rep.country == c, "E_before"].sum()
        raw = prof[idx]
        p = new.loc[idx, "p_nom"]
        if e_before <= 0 or p.sum() <= 0:
            factors[c] = 1.0
            continue
        f = 1.0
        for _ in range(iterations):
            e = float((raw.mul(f).clip(upper=1.0).mul(w, axis=0).sum() * p).sum())
            f *= e_before / e
        prof[idx] = raw.mul(f).clip(upper=1.0)
        factors[c] = f
        e = float((prof[idx].mul(w, axis=0).sum() * p).sum())
        if abs(e - e_before) > 1e-4 * e_before:
            logger.warning(f"resource classes {carrier} {c}: fleet energy {e:.0f} vs {e_before:.0f} MWh after the clip")

    # other time series of G (e.g. p_min_pu) carried to its classes
    G_of = pd.Series(parent)
    other = {
        attr: pd.DataFrame({k: frame[G_of[k]] for k in new.index if G_of[k] in frame}, index=n.snapshots)
        for attr, frame in n.generators_t.items()
        if attr != "p_max_pu" and not frame.empty and frame.columns.intersection(G_of.unique()).size
    }

    n.mremove("Generator", rep.index)
    n.import_components_from_dataframe(new.drop(columns=["p_nom_opt"], errors="ignore"), "Generator")
    n.import_series_from_dataframe(prof, "Generator", "p_max_pu")
    for attr, frame in other.items():
        n.import_series_from_dataframe(frame, "Generator", attr)

    # invariants: the split conserves fleet and potential per bus
    g = n.generators
    after = g.loc[new.index].groupby("bus")[["p_nom", "p_nom_min", "p_nom_max"]].sum()
    before = rep.set_index("bus")[["p_nom", "p_nom_max"]]
    for col in ["p_nom", "p_nom_max"]:
        b, a = before[col], after.loc[before.index, col]
        fin = np.isfinite(b)
        assert np.allclose(a[fin], b[fin], rtol=1e-9, atol=1e-6), f"{carrier}: {col} not conserved by the split"
    assert (g.loc[new.index, "p_nom_max"] >= g.loc[new.index, "p_nom"] - 1e-6).all()

    rep["factor"] = rep.country.map(factors)
    rep["dev_potential"] = rep.class_potential / rep.p_nom_max - 1
    logger.info(
        f"resource classes {carrier}: {len(rep)} generators split into {len(new)}; per bus:\n"
        + rep[["bus", "p_nom", "p_nom_max", "class_potential", "dev_potential", "classes", "factor"]]
        .assign(p_nom=lambda d: d.p_nom / 1e3, p_nom_max=lambda d: d.p_nom_max / 1e3,
                class_potential=lambda d: d.class_potential / 1e3)
        .round(3).to_string()
    )
    return rep


def attach_supply_curves(
    n: pypsa.Network,
    costs: pd.DataFrame,
    ds: xr.Dataset,
    carrier: str,
    Nyears: float = 1.0,
) -> None:
    """
    Attach the tranches of a site-specific supply curve as extendable generators.

    One generator ``"<bus> <carrier> t<k>"`` per bus and tranche of the dataset
    built by ``build_supply_curve`` (``supply_curves`` config): its potential is
    the tranche's capacity, its availability the tranche's hourly profile and its
    capital cost the annuity of the tranche's investment plus its fixed O&M. The
    investment per cost component is kept on the generator (columns
    ``capex_<component>``) so that ``add_brownfield`` can move it to the costs of
    a later planning horizon.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to attach the generators to.
    costs : pd.DataFrame
        The cost table; the row ``carrier`` gives marginal cost, lifetime, discount rate and emissions.
    ds : xr.Dataset
        The supply curve dataset (dimensions bus, tranche; ``p_nom_max``, ``capex_*``, ``fom``, ``lifetime``, ``profile``).
    carrier : str
        The carrier of the generators.
    Nyears : float
        Share of a year represented by the snapshots.
    """
    static = [v for v in ds.data_vars if "time" not in ds[v].dims]
    df = ds[static].to_dataframe().dropna(subset=["p_nom_max"])
    df = df[df.p_nom_max > 0]
    if df.empty:
        logger.info(f"No {carrier} potential in any region: no generators added.")
        return
    components = [c for c in ds.attrs.get("components", [])]
    names = [f"{bus} {carrier} t{int(k)}" for bus, k in df.index]
    rate = costs.at[carrier, "discount rate"] if "discount rate" in costs else 0.07
    capital_cost = (
        Nyears
        * 1e3
        * (df.capex.values * annuity(df.lifetime.values, rate) + df.fom.values)
    )
    profile = pd.DataFrame(
        {
            name: ds.profile.sel(bus=bus, tranche=k).values
            for name, (bus, k) in zip(names, df.index)
        },
        index=pd.DatetimeIndex(ds.time.values),
    ).reindex(n.snapshots)
    if carrier not in n.carriers.index:
        n.add(
            "Carrier",
            carrier,
            co2_emissions=(
                float(costs.at[carrier, "co2_emissions"])
                if carrier in costs.index
                else 0.0
            ),
        )
    n.madd(
        "Generator",
        names,
        bus=df.index.get_level_values("bus"),
        carrier=carrier,
        p_nom_extendable=True,
        p_nom_max=df.p_nom_max.values,
        p_max_pu=profile,
        capital_cost=capital_cost,
        marginal_cost=costs.at[carrier, "marginal_cost"],
        lifetime=costs.at[carrier, "lifetime"],
        efficiency=1.0,
        **{f"capex_{c}": df[f"capex_{c}"].values for c in components},
    )
    logger.info(
        f"Added {len(names)} {carrier} supply-curve generators, potential by bus [GW]:\n"
        + df.p_nom_max.groupby(level="bus").sum().div(1e3).round(2).to_string()
    )


def attach_hydrogen_pipelines(
    n: pypsa.Network,
    costs: pd.DataFrame,
    electricity: dict,
    transmission_efficiency: float,
) -> None:
    """
    Attach hydrogen pipelines to the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to attach the hydrogen pipelines to.
    costs : pd.DataFrame
        DataFrame containing the cost data.
    electricity : dict
        Dictionary containing the electricity parameters.
    transmission_efficiency : float
        The efficiency of the hydrogen transmission.
    """
    elec_opts = electricity
    ext_carriers = elec_opts["extendable_carriers"]
    as_stores = ext_carriers.get("Store", [])

    if "H2 pipeline" not in ext_carriers.get("Link", []):
        return

    assert "H2" in as_stores, (
        "Attaching hydrogen pipelines requires hydrogen "
        "storage to be modelled as Store-Link-Bus combination. See "
        "`config.yaml` at `electricity: extendable_carriers: Store:`."
    )

    # determine bus pairs
    attrs = ["bus0", "bus1", "length"]
    candidates = pd.concat(
        [n.lines[attrs], n.links.query('carrier=="DC"')[attrs]]
    ).reset_index(drop=True)

    # remove bus pair duplicates regardless of order of bus0 and bus1
    h2_links = candidates[
        ~pd.DataFrame(np.sort(candidates[["bus0", "bus1"]])).duplicated()
    ]
    h2_links.index = h2_links.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)

    # add pipelines
    n.madd(
        "Link",
        h2_links.index,
        bus0=h2_links.bus0.values + " H2",
        bus1=h2_links.bus1.values + " H2",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=h2_links.length.values,
        capital_cost=costs.at["H2 pipeline", "capital_cost"] * h2_links.length,
        carrier="H2 pipeline",
    )

    # split the pipeline into two unidirectional links to properly apply transmission losses in both directions.
    lossy_bidirectional_links(n, "H2 pipeline")

    # set the pipelines efficiency and the electricity required by the pipeline for compression
    set_length_based_efficiency(n, "H2 pipeline", " H2", transmission_efficiency)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_extra_components", simpl="", clusters=10)

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    transmission_efficiency = snakemake.params.transmission_efficiency
    electricity = snakemake.params.electricity
    storage_techs = snakemake.params.storage_techs

    costs = pd.read_csv(snakemake.input.tech_costs, index_col=0)

    # resource classes first: the generators to split are the ones add_electricity / clustering made
    for tech in snakemake.params.get("resource_classes", {}) or {}:
        with xr.open_dataset(snakemake.input[f"resource_classes_{tech}"]) as ds:
            attach_resource_classes(n, ds.load(), tech, snakemake.params.threshold_capacity)

    buses_i = n.buses.index
    attach_stores(
        n,
        costs,
        buses_i,
        electricity["extendable_carriers"]["Store"],
        storage_techs,
    )

    attach_storageunits(
        n,
        costs,
        buses_i,
        electricity["extendable_carriers"]["StorageUnit"],
        electricity["max_hours"],
        storage_techs,
    )

    if ("csp" in electricity["renewable_carriers"]) and (
        snakemake.params.csp_model == "advanced"
    ):
        attach_advance_csp(n, costs)

    attach_hydrogen_pipelines(n, costs, electricity, transmission_efficiency)

    for tech in snakemake.params.get("supply_curves", {}) or {}:
        attach_supply_curves(
            n,
            costs,
            xr.open_dataset(snakemake.input[f"supply_curve_{tech}"]),
            tech,
            Nyears,
        )

    add_nice_carrier_names(n, config=snakemake.config)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
