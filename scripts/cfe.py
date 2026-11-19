# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
24/7 carbon-free electricity (CFE) buyers for the electricity-only myopic workflow (catalyst).

After the 24/7 CFE model of Riepin, Jenkins, Swezey & Brown (Joule 2025, github.com/PyPSA/247-cfe:
``add_ci``, ``cfe_constraints``, ``excess_constraints``, ``calculate_grid_cfe``), one buyer per AC bus with load:

- a buyer bus ``<bus> CFE`` (location = the bus) with a flat load of ``share`` x the bus's mean load of the
  horizon, taken out of the bus's own load;
- links ``<bus> CFE import`` (grid -> buyer) and ``<bus> CFE export`` (buyer -> grid), fixed and large; the
  import link is closed (``p_nom`` 0) unless ``grid_imports: true``: by default the buyer never buys from the
  grid, every hour of its load is met by its own assets (catalyst decision 2026-11-10);
- extendable copies of the bus's extendable clean assets (generators and storage units whose carrier is in
  ``clean_carriers``, and the battery store with its charger / discharger), same carriers and costs, so that
  procurement is an endogenous choice of the optimisation and the learning market counts the buyer's builds;
- the matching constraint, summed over the year as in 247-cfe:
  sum_t w_t [clean supply of the buyer's own assets + import_t * grid_cfe_t - export_t] >= target * sum_t w_t load_t;
  with the buyer bus's hourly balance this is sum_t w_t import_t (1 - grid_cfe_t) <= (1 - target) sum_t w_t load_t,
  which is the form added here (only with ``grid_imports: true``; without imports the buyer bus's hourly balance
  alone gives a score of 1, so ``target`` has no effect);
- the excess limit sum_t w_t export_t <= excess_share * sum_t w_t load_t;
- shared potentials: a buyer copy and its grid twin together stay within the twin's potential (``p_nom_max``).

``matching: zone`` (the default) is how 24/7 CFE accounting is done: within a market zone. Every buyer still sits
at its bus and procures there, but its import link is open and the buyers of a zone are matched together, in every
hour sum_buyers export_t >= sum_buyers import_t, so the wind a buyer contracts at one bus may cover the load of a
buyer at another. The power itself goes through the grid and its line limits like any other. The excess limit then
holds for the zone's net exports. On a one-bus zone this is the same as ``node`` (every buyer covers its load at
its own bus, import link closed), which is stricter than the accounting wherever a market has several buses.

The zone of a bus (``zones``): its country (``country``, the default), or the market zone listed for it in the
table statistics/market_zones.csv (``statistics``; columns bus, zone; one row per clustered bus), for markets that
are smaller than a country (the US ISOs, Brazil's submarkets, China's provinces) or larger (Germany and Luxembourg).

grid_cfe_t is the clean share of the generation at the bus (the buyer's assets and pure storage left out),
computed outside the LP from a solved network: the previous horizon's network seeds the first solve, every
further solve takes the grid CFE of the one before (``iterations``, 247-cfe: 2). Without grid imports the grid
CFE does not enter the problem and each horizon is solved once; it is still recorded for the summary.

Config (``catalyst: cfe:``)::

    share: 0.10            # buyer load as a share of each bus's load; or a pathway {year: share}, interpolated
                           # linearly between the given years (constant outside), e.g. {2030: 0.01, 2050: 0.10}
    grid_imports: false    # true: the buyer may buy from the grid, credited with the grid's hourly clean share
                           # (247-cfe); false (default): no purchases from the grid, one solve per horizon
    target: 1.0            # CFE score (only binds with grid_imports: true)
    excess_share: 0.20     # exports as a share of the buyer's load
    matching: zone         # zone (default): the buyers of a market zone are matched together hour by hour (needs
                           # grid_imports: false); node: every buyer covers its load at its own bus
    zones: country         # the zone of a bus: its country, or `statistics` = the table market_zones.csv
    start_year: 2030       # first planning horizon with a buyer (the dispatch-only horizons cannot procure)
    iterations: 2          # solves per horizon with grid_imports: true
    clean_carriers: [...]  # what the buyer may build; also the clean carriers of grid_cfe
    clean_threshold: 0.01  # tCO2/MWh_th: other grid carriers at or below count as clean in grid_cfe

The buyers and the grid CFE of the solve are kept in ``n.meta["cfe"]``.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

SUFFIX = " CFE"
STORAGE_CARRIERS = {"PHS", "battery", "iron-air", "H2", "vrfb", "co2-battery"}   # shift energy, not clean or dirty


def config_of(config):
    cfg = ((config or {}).get("catalyst") or {}).get("cfe") or {}
    return cfg if cfg.get("share") else {}


def share_of(cfg, year):
    """Buyer share of the planning horizon: a number, or {year: share} interpolated linearly between the given
    years and held constant outside them."""
    s = cfg["share"]
    if isinstance(s, dict):
        pts = sorted((int(k), float(v)) for k, v in s.items())
        return float(np.interp(int(year), [k for k, _ in pts], [v for _, v in pts]))
    return float(s)


def buyers(n):
    """{buyer bus: grid bus} of the network (empty without CFE)."""
    return dict(((n.meta or {}).get("cfe") or {}).get("buyers", {}))


def _rename(index, bus):
    return pd.Index([f"{bus}{SUFFIX}{i[len(bus):]}" if i.startswith(bus) else f"{bus}{SUFFIX} {i}" for i in index])


def read_zones(path):
    """{clustered bus: market zone} of the table statistics/market_zones.csv."""
    d = pd.read_csv(path, dtype=str)
    return dict(zip(d["bus"], d["zone"]))


def add_cfe_buyer(n, cfg, year, zones=None):
    """Add a buyer to every AC bus with load; returns the meta dict (or None before ``start_year``).

    ``zones``: {bus: market zone} (``zones: statistics``); without it the zone of a bus is its country."""
    if not cfg or year < int(cfg.get("start_year", 0)):
        return None
    clean = set(cfg.get("clean_carriers", []))
    share = share_of(cfg, year)
    w = n.snapshot_weightings.generators
    load_bus = n.loads.bus
    nodes = [b for b in n.buses.index[n.buses.carrier == "AC"] if (load_bus == b).any()]
    for c in ["CFE import", "CFE export"]:
        if c not in n.carriers.index:
            n.add("Carrier", c, co2_emissions=0.0)
    grid_imports = bool(cfg.get("grid_imports", False))
    matching = str(cfg.get("matching", "zone"))
    if matching not in ("node", "zone") or (matching == "zone" and grid_imports):
        raise ValueError("catalyst: cfe: matching is node or zone, and zone needs grid_imports: false")
    meta = dict(buyers={}, share=share, target=float(cfg.get("target", 1.0)),
                excess_share=float(cfg.get("excess_share", 0.0)), year=int(year), grid_imports=grid_imports,
                matching=matching, zones={})
    if not grid_imports and meta["target"] < 1:
        logger.warning(f"CFE target {meta['target']} has no effect without grid imports (catalyst: cfe: "
                       "grid_imports: false): the buyer's own assets cover every hour")
    if str(cfg.get("zones", "country")) == "statistics":
        if zones is None:
            raise ValueError("catalyst: cfe: zones: statistics needs the table statistics/market_zones.csv")
        missing = [b for b in nodes if b not in zones]
        if missing:
            raise ValueError(f"market_zones.csv has no zone for the buses {missing} (table of another clustering?)")
    else:
        zones = None
    for b in nodes:
        buyer = f"{b}{SUFFIX}"
        loads = load_bus.index[load_bus == b]   # the bus's own loads (n.loads grows by one buyer load per node)
        p = n.loads_t.p_set[loads]
        mean_load = float((p.sum(axis=1) * w).sum() / w.sum())
        flat = share * mean_load
        # flat buyer load out of the bus's own loads, in their proportion
        n.loads_t.p_set[loads] = p - p.div(p.sum(axis=1), axis=0).mul(flat, axis=0)
        bus = n.buses.loc[b]
        n.add("Bus", buyer, carrier="AC", x=bus.x, y=bus.y, country=bus.get("country", ""), location=b)
        n.add("Load", buyer, bus=buyer, p_set=flat)
        # closed without grid imports; the link stays so that the summary and older scripts find it
        n.add("Link", f"{buyer} import", bus0=b, bus1=buyer, carrier="CFE import",
              p_nom=1e6 if grid_imports or matching == "zone" else 0.0, marginal_cost=0.001)
        n.add("Link", f"{buyer} export", bus0=buyer, bus1=b, carrier="CFE export", p_nom=1e6, marginal_cost=0.1)

        # generators and storage units: extendable clean twins at the buyer bus
        for cname, attr in [("Generator", "p"), ("StorageUnit", "p")]:
            df = n.df(cname)
            sel = df.index[(df.bus == b) & df[f"{attr}_nom_extendable"] & df.carrier.isin(clean)]
            if sel.empty:
                continue
            new = df.loc[sel].copy()
            new.index = _rename(sel, b)
            new["bus"] = buyer
            new[f"{attr}_nom"] = 0.0
            new[f"{attr}_nom_min"] = 0.0
            n.import_components_from_dataframe(new, cname)
            pnl = n.pnl(cname)
            for tattr, frame in pnl.items():
                cols = frame.columns.intersection(sel)
                if len(cols):
                    n.import_series_from_dataframe(frame[cols].set_axis(_rename(cols, b), axis=1), cname, tattr)

        # battery: own battery bus, store and charger / discharger (positional pairing in add_battery_constraints)
        if "battery" in clean:
            sbus = n.buses.index[(n.buses.carrier == "battery") & (n.buses.get("location", "") == b)]
            if len(sbus) == 1:
                sb = sbus[0]
                bbus = f"{buyer} battery"
                n.add("Bus", bbus, carrier="battery", x=bus.x, y=bus.y, country=bus.get("country", ""), location=b)
                st = n.stores[(n.stores.bus == sb) & n.stores.e_nom_extendable]
                if len(st):
                    new = st.copy()
                    new.index = _rename(st.index, b)
                    new["bus"] = bbus
                    new["e_nom"] = 0.0
                    n.import_components_from_dataframe(new, "Store")
                busmap = {b: buyer, sb: bbus}
                for kind in ["battery charger", "battery discharger"]:
                    lk = n.links[n.links.p_nom_extendable & n.links.index.str.contains(kind)
                                 & (n.links.bus0.isin(busmap) & n.links.bus1.isin(busmap))]
                    if len(lk):
                        new = lk.copy()
                        new.index = _rename(lk.index, b)
                        new["bus0"] = new.bus0.map(busmap)
                        new["bus1"] = new.bus1.map(busmap)
                        new["p_nom"] = 0.0
                        n.import_components_from_dataframe(new, "Link")
        meta["buyers"][buyer] = b
        meta["zones"][buyer] = str((zones or {}).get(b) or bus.get("country", "") or "all")
        logger.info(f"CFE buyer {buyer}: {flat:.0f} MW flat ({share:.0%} of {mean_load:.0f} MW), "
                    f"target {meta['target']}, excess <= {meta['excess_share']:.0%}, "
                    f"grid imports {'allowed' if grid_imports else 'closed'}, matching per {matching}")
    return meta


def grid_cfe(n, buyer_map, cfg):
    """Hourly clean share of the generation at each buyer's grid bus (DataFrame snapshots x grid bus)."""
    clean = set(cfg.get("clean_carriers", []))
    thr = float(cfg.get("clean_threshold", 0.01))
    co2 = n.carriers.co2_emissions.reindex(n.generators.carrier.unique()).fillna(0.0)
    clean_carriers = clean | set(co2.index[co2 <= thr])
    clean_carriers -= {"load shedding"}
    out = {}
    for buyer, b in buyer_map.items():
        g = n.generators[(n.generators.bus == b) & (n.generators.carrier != "load shedding")]
        p = n.generators_t.p.reindex(columns=g.index).fillna(0.0).clip(lower=0)
        total = p.sum(axis=1)
        good = p.loc[:, g.carrier.isin(clean_carriers).values].sum(axis=1)
        # hydro storage units with inflow: clean generation; pure storage left out of both
        su = n.storage_units[(n.storage_units.bus == b) & ~n.storage_units.carrier.isin(STORAGE_CARRIERS)]
        if len(su) and "p_dispatch" in n.storage_units_t and not n.storage_units_t.p_dispatch.empty:
            d = n.storage_units_t.p_dispatch.reindex(columns=su.index).fillna(0.0).sum(axis=1)
            total, good = total + d, good + d
        out[b] = (good / total.where(total > 1e-6)).fillna(0.0).clip(0, 1)
    return pd.DataFrame(out, index=n.snapshots)


def seed_grid_cfe(n, n_p, meta, cfg):
    """grid CFE of the previous horizon's solved network (snapshots are the same weather year), else 0."""
    if n_p is not None and len(n_p.generators_t.p):
        g = grid_cfe(n_p, meta["buyers"], cfg).reindex(n.snapshots).fillna(0.0)
    else:
        g = pd.DataFrame(0.0, index=n.snapshots, columns=list(meta["buyers"].values()))
    meta["grid_cfe"] = {b: [round(float(v), 5) for v in g[b]] for b in g}
    return meta


def add_cfe_constraints(n):
    """Matching, excess and shared-potential constraints of every buyer (called in extra_functionality)."""
    meta = (n.meta or {}).get("cfe") or {}
    if not meta.get("buyers"):
        return
    w = xr.DataArray(n.snapshot_weightings.generators.values, coords={"snapshot": n.snapshots})
    m = n.model
    target, excess = meta["target"], meta["excess_share"]
    zone_matching = meta.get("matching", "node") == "zone"
    demands = {}
    for buyer, b in meta["buyers"].items():
        demand = float((n.loads_t.p_set[buyer] * n.snapshot_weightings.generators).sum()) \
            if buyer in n.loads_t.p_set else float(n.loads.at[buyer, "p_set"] * n.snapshot_weightings.generators.sum())
        cfe = xr.DataArray(np.asarray(meta["grid_cfe"][b], dtype=float), coords={"snapshot": n.snapshots})
        imp = m["Link-p"].sel(Link=f"{buyer} import")
        exp = m["Link-p"].sel(Link=f"{buyer} export")
        if meta.get("grid_imports", True):   # networks from before the option had open imports
            m.add_constraints(((1 - cfe) * w * imp).sum() <= (1 - target) * demand, name=f"CFE-matching-{buyer}")
        demands[buyer] = demand
        if not zone_matching:
            m.add_constraints((w * exp).sum() <= excess * demand, name=f"CFE-excess-{buyer}")
    if zone_matching:   # the buyers of a zone together: no net purchase from the grid in any hour
        zones = {}
        for buyer in meta["buyers"]:
            zones.setdefault(meta["zones"][buyer], []).append(buyer)
        for z, bs in zones.items():
            net = (m["Link-p"].sel(Link=[f"{x} export" for x in bs]).sum("Link")
                   - m["Link-p"].sel(Link=[f"{x} import" for x in bs]).sum("Link"))
            m.add_constraints(net >= 0, name=f"CFE-zone-matching-{z}")
            m.add_constraints((w * net).sum() <= excess * sum(demands[x] for x in bs), name=f"CFE-zone-excess-{z}")

    # shared potentials: buyer copy + grid twin <= the twin's potential (already net of what was built at the
    # location: land use by location, supply-curve tranches by name, so the buyer's own tranche builds come off)
    g = n.generators
    ext = g.index[g.p_nom_extendable]
    supply_curves = set((n.meta or {}).get("supply_curves", {}) or {})
    stem = g.index.to_series().str.replace(r"(-\d{4})+$", "", regex=True)
    built = g.loc[~g.p_nom_extendable, "p_nom"].groupby(stem[~g.p_nom_extendable]).sum()
    pairs, rhs = [], []
    for buyer, b in meta["buyers"].items():
        for gb in ext[g.loc[ext, "bus"] == buyer]:
            twin = b + gb[len(buyer):]
            if twin not in ext or not np.isfinite(g.at[twin, "p_nom_max"]):
                continue
            r = g.at[twin, "p_nom_max"]
            if g.at[gb, "carrier"] in supply_curves:
                r -= built.get(stem[gb], 0.0)
            pairs.append((gb, twin))
            rhs.append(max(r, 0.0))
    pn = m["Generator-p_nom"]
    for (gb, twin), r in zip(pairs, rhs):
        m.add_constraints(pn.sel({"Generator-ext": gb}) + pn.sel({"Generator-ext": twin}) <= r,
                          name=f"CFE-shared-potential-{gb}")
    logger.info(f"CFE constraints for {list(meta['buyers'])}, {len(pairs)} shared potentials")


def summary(n):
    """Per buyer: achieved CFE score, costs per MWh (USD of the cost table), the grid's average price.

    Under zone matching a buyer's imports are another buyer's exports, priced at each one's bus: a single buyer's
    cost then contains what it pays to or receives from the others of its zone, and only ``summary_zones`` means
    something."""
    meta = (n.meta or {}).get("cfe") or {}
    w = n.snapshot_weightings.generators
    out = {}
    for buyer, b in meta.get("buyers", {}).items():
        load = n.loads_t.p_set[buyer] if buyer in n.loads_t.p_set else pd.Series(n.loads.at[buyer, "p_set"], n.snapshots)
        demand = float((load * w).sum())
        price = n.buses_t.marginal_price[b]
        imp = n.links_t.p0[f"{buyer} import"]
        exp = n.links_t.p0[f"{buyer} export"]
        cfe = pd.Series(meta["grid_cfe"][b], index=n.snapshots)
        cost = {}
        own_new = 0.0   # the part of `own` from the assets built in this horizon (still extendable in its network)
        for cname, attr in [("Generator", "p"), ("StorageUnit", "p"), ("Store", "e"), ("Link", "p")]:
            df = n.df(cname)
            if cname == "Link":
                sel = df.index[(df.bus0 == buyer) | (df.bus1 == buyer) | (df.bus0 == f"{buyer} battery")
                               | (df.bus1 == f"{buyer} battery")]
                sel = sel.difference([f"{buyer} import", f"{buyer} export"])
            elif cname == "Store":
                sel = df.index[df.bus == f"{buyer} battery"]
            else:
                sel = df.index[df.bus == buyer]
            if sel.empty:
                continue
            capex = (df.loc[sel, f"{attr}_nom_opt"] * df.loc[sel, "capital_cost"]).groupby(df.loc[sel, "carrier"]).sum()
            for k, v in capex.items():
                cost[k] = cost.get(k, 0.0) + float(v)
            new = sel[df.loc[sel, f"{attr}_nom_extendable"].astype(bool)]
            own_new += float((df.loc[new, f"{attr}_nom_opt"] * df.loc[new, "capital_cost"]).sum())
            if cname == "Generator":
                opex = (n.generators_t.p[sel].mul(w, axis=0) * df.loc[sel, "marginal_cost"]).sum()
                own_new += float(opex[new].sum())
                for k, v in opex.groupby(df.loc[sel, "carrier"]).sum().items():
                    cost[k] = cost.get(k, 0.0) + float(v)
        imports = float((imp * w * price).sum())
        revenue = float((exp * w * price).sum())
        own = sum(cost.values())
        node_load = n.loads_t.p_set[n.loads.index[n.loads.bus == b]].sum(axis=1)
        grid_price = float((node_load * w * price).sum() / (node_load * w).sum())
        score = 1 - float((imp * w * (1 - cfe)).sum()) / demand
        if meta.get("matching", "node") == "zone":
            score = 1.0   # matched hour by hour over the zone; a buyer's imports are another buyer's clean exports
        out[buyer] = dict(
            bus=b, zone=_zone_of(meta, buyer), demand_twh=demand / 1e6, cfe_score=score, grid_cfe_mean=float((cfe * w).sum() / w.sum()),
            import_twh=float((imp * w).sum()) / 1e6, export_twh=float((exp * w).sum()) / 1e6,
            cost_own_per_mwh=own / demand, cost_imports_per_mwh=imports / demand, revenue_per_mwh=revenue / demand,
            cost_per_mwh=(own + imports - revenue) / demand, grid_price_per_mwh=grid_price,
            premium_per_mwh=(own + imports - revenue) / demand - grid_price,
            cost_own_new=own_new,   # USD/a: annuity + running cost of the buyer assets built in this horizon
            cost_by_carrier_per_mwh={k: v / demand for k, v in cost.items()},
        )
    return out


def _zone_of(meta, buyer):
    """Matching boundary of a buyer: its market zone, or its own bus under node matching."""
    if meta.get("matching", "node") == "zone":
        return (meta.get("zones") or {}).get(buyer) or "all"
    return meta["buyers"][buyer]


def summary_zones(n, buyers=None):
    """Per matching boundary (market zone; the buyer's bus under node matching): what the buyers pay together.

    Within a zone only the totals mean something: ``transfer_twh`` is the energy matched across the zone's buses
    (the sum of the buyers' imports), ``export_twh`` the zone's net sales to the grid. The settlement with the grid,
    sum over buyers of (import - export) x the price at the buyer's bus, is split into ``revenue_per_mwh`` (the net
    sales at the zone's price) and ``congestion_per_mwh`` (the rest: what moving power between the zone's buses
    costs at nodal prices). ``grid_price_per_mwh`` is the load-weighted price of the rest of the zone's demand.
    The demand-weighted cost over the zones equals that over the buyers of ``summary``.
    """
    meta = (n.meta or {}).get("cfe") or {}
    if not meta.get("buyers"):
        return {}
    buyers = buyers or meta.get("summary") or summary(n)
    w = n.snapshot_weightings.generators
    zones = {}
    for buyer in meta["buyers"]:
        zones.setdefault(_zone_of(meta, buyer), []).append(buyer)
    out = {}
    for z, bs in zones.items():
        grid = [meta["buyers"][x] for x in bs]
        demand = sum(buyers[x]["demand_twh"] for x in bs) * 1e6
        imp = n.links_t.p0[[f"{x} import" for x in bs]].set_axis(grid, axis=1)
        exp = n.links_t.p0[[f"{x} export" for x in bs]].set_axis(grid, axis=1)
        price = n.buses_t.marginal_price[grid]
        loads = n.loads.index[n.loads.bus.isin(grid)]
        load = n.loads_t.p_set[loads].T.groupby(n.loads.bus[loads]).sum().T.reindex(columns=grid)
        zone_price = (load * price).sum(axis=1) / load.sum(axis=1)
        net = (exp - imp).sum(axis=1)
        settlement = float((((imp - exp) * price).sum(axis=1) * w).sum())   # > 0: the zone pays the grid
        revenue = float((net * zone_price * w).sum())
        own = sum(buyers[x]["cost_own_per_mwh"] * buyers[x]["demand_twh"] * 1e6 for x in bs)
        carriers = {}
        for x in bs:
            for k, v in buyers[x]["cost_by_carrier_per_mwh"].items():
                carriers[k] = carriers.get(k, 0.0) + v * buyers[x]["demand_twh"] * 1e6
        grid_price = float((load.sum(axis=1) * zone_price * w).sum() / (load.sum(axis=1) * w).sum())
        cost = (own + settlement) / demand
        out[z] = dict(
            n_buyers=len(bs), demand_twh=demand / 1e6,
            transfer_twh=float((imp.sum(axis=1) * w).sum()) / 1e6, export_twh=float((net * w).sum()) / 1e6,
            cfe_score=1 - float((net.clip(upper=0).abs() * w).sum()) / demand,   # hours of net purchase from the grid
            cost_own_per_mwh=own / demand, revenue_per_mwh=revenue / demand,
            congestion_per_mwh=(settlement + revenue) / demand, cost_per_mwh=cost,
            grid_price_per_mwh=grid_price, premium_per_mwh=cost - grid_price,
            cost_own_new=sum(buyers[x]["cost_own_new"] for x in bs),
            cost_by_carrier_per_mwh={k: v / demand for k, v in carriers.items()},
        )
    return out
