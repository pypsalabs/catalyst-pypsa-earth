# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Assign listed HVDC projects to the DC links of a network (``links: dc_projects``).

OpenStreetMap draws a project as segments, and may draw its poles or cables as
parallel ways, so a rating per way counts a project many times. A project takes
its links by the OpenStreetMap name of their ways (column ``osm_names`` of the
table, ``|`` separated). Within those ways, its terminals pick one route through
the parallel ones. If the named ways do not connect the terminals, the route
continues through unnamed ways, preferring the named ones; without terminals or
a route the project takes all its named ways, and a project left with under half
its listed length, or with no named way, counts as unmatched. Every corridor (bus
pair) of a route carries the project rating once, shared by its parallel links.
The terminals follow PyPSA-Eur's ``links_p_nom`` table of coordinates.
"""

import ast
import json
import os
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

VOLTAGE_TOLERANCE = 0.1
NAMED_FACTOR = 0.2
MIN_LENGTH_SHARE = 0.5
BRIDGE_COST_KM = 50.0
GENERIC_NAMES = {"HVDC", "DC+", "DC-"}  # say nothing about the project


def _km(x0, y0, x1, y1):
    x0, y0, x1, y1 = map(np.radians, (x0, y0, x1, y1))
    a = (
        np.sin((y1 - y0) / 2) ** 2
        + np.cos(y0) * np.cos(y1) * np.sin((x1 - x0) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _corridor(link):
    return tuple(sorted((str(link.bus0), str(link.bus1))))


def _extra_tags(cell):
    """earth-osm keeps rare tags as a dict in the column ``other_tags``."""
    if not isinstance(cell, str):
        return {}
    try:
        return json.loads(cell)
    except ValueError:
        try:
            return ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return {}


def read_way_names(raw_dir):
    """Name of every HVDC way in the raw OSM tables: ``name:en``, else ``name``."""
    names = {}
    for kind in ("lines", "cables"):
        fn = os.path.join(raw_dir, f"all_raw_{kind}.csv")
        if not os.path.exists(fn) or not os.path.getsize(fn):
            continue
        cols = ("id", "tags.name", "tags.name:en", "tags.frequency", "other_tags")
        raw = pd.read_csv(fn, low_memory=False, usecols=lambda c: c in cols)
        raw = raw[pd.to_numeric(raw["tags.frequency"], errors="coerce").eq(0)]
        for row in raw.reindex(columns=list(cols)).itertuples(index=False):
            extra = _extra_tags(row[4])
            candidates = (
                extra.get("tags.name:en"),
                row[2],
                row[1],
                extra.get("tags.name"),
            )
            name = next((c for c in candidates if isinstance(c, str) and c), None)
            if name:
                names.setdefault(int(row[0]), name.strip())
    return pd.Series(names, dtype=object)


def way_of_line(line_id):
    """OSM way id of a line id such as ``117931185-2_0``; None for another form."""
    m = re.match(r"^(\d+)", str(line_id))
    return int(m.group(1)) if m else None


def _graph(links, buses, bridge_km):
    """Graph of links; weight = length; gaps up to bridge_km are bridged."""
    G = nx.Graph()
    for pair, group in links.groupby(links.apply(_corridor, axis=1)):
        G.add_edge(*pair, weight=float(group.length.min()), bridge=False)
    nodes = list(G.nodes)
    if len(nodes) > 1 and bridge_km > 0:
        xy = buses.loc[nodes, ["x", "y"]].astype(float).values
        for i, j in KDTree(xy).query_pairs(bridge_km / 60.0):
            d = _km(xy[i, 0], xy[i, 1], xy[j, 0], xy[j, 1])
            if d <= bridge_km and not G.has_edge(nodes[i], nodes[j]):
                G.add_edge(nodes[i], nodes[j], weight=float(d), bridge=True)
    return G


def _terminals(G, buses, p, bound):
    """Nearest pair of buses of G to the project's terminals with a route between them."""
    if pd.isna([p.x1, p.y1, p.x2, p.y2]).any() or len(G) < 2:
        return None
    nodes = list(G.nodes)
    tree = KDTree(buses.loc[nodes, ["x", "y"]].astype(float).values)
    ends = []
    for x, y in ((p.x1, p.y1), (p.x2, p.y2)):
        dist, idx = tree.query([x, y], k=min(8, len(nodes)))
        dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
        ends.append([nodes[i] for d, i in zip(dist, idx) if d <= bound])
    pairs = [
        (ra + rb, a, b)
        for ra, a in enumerate(ends[0])
        for rb, b in enumerate(ends[1])
        if a != b and nx.has_path(G, a, b)
    ]
    return min(pairs, key=lambda t: t[0])[1:] if pairs else None


def assign_projects(
    links,
    buses,
    projects,
    reference_year=2024,
    way_names=None,
    distance_upper_bound=1.5,
    bridge_km=20.0,
    min_voltage=200.0,
):
    """
    Match projects to DC links by way name, with terminals to pick the route.

    Parameters
    ----------
    links : pd.DataFrame
        DC links with bus0, bus1, length, v_nom (kV).
    buses : pd.DataFrame
        Buses with x, y, country, indexed by str.
    projects : pd.DataFrame
        name, country1, country2, voltage_kv, p_nom_mw, year, length_km, x1, y1,
        x2, y2, and optionally osm_names.
    reference_year : int
        Projects entering service later are matched and marked not in service.
    way_names : pd.Series, optional
        OSM way name per link (index of ``links``).
    distance_upper_bound : float
        Largest distance (degrees) between a terminal and its DC bus, as in
        ``_find_closest_links``.
    bridge_km : float
        Gaps between DC buses bridged for finding a route only.

    Returns
    -------
    (pd.Series, pd.Series, pd.DataFrame)
        p_nom [MW] per link for the matched links; True per link of a project
        after the reference year; report with one row per project of the region.
    """
    links = links.copy()
    links["bus0"] = links.bus0.astype(str)
    links["bus1"] = links.bus1.astype(str)
    buses = buses.copy()
    buses.index = buses.index.astype(str)
    countries = set(buses.country.dropna())
    corridor = links.apply(_corridor, axis=1)
    per_corridor = corridor.value_counts()
    if way_names is None:
        way_names = pd.Series(index=links.index, dtype=object)
    way_names = way_names.reindex(links.index)
    has_names = "osm_names" in projects

    rows = {}
    ways = {}  # project -> links carrying one of its names
    for key, p in projects.iterrows():
        if not {p.country1, p.country2} <= countries:
            continue
        rows[key] = dict(
            name=p["name"],
            voltage_kv=p.voltage_kv,
            p_nom_mw=p.p_nom_mw,
            year=p.year,
            length_km=p.length_km,
            status="unmatched",
            method="",
            route_km=np.nan,
            segments=0,
            link_ids="",
        )
        if has_names and isinstance(p.osm_names, str):
            found = way_names.isin(p.osm_names.split("|"))
            if found.any():
                ways[key] = links.index[found]
    claimed = (
        pd.Index(np.concatenate([w.values for w in ways.values()]))
        if ways
        else pd.Index([])
    )
    # ways with a name of their own belong to another line, known or not
    named = way_names.notna() & ~way_names.isin(GENERIC_NAMES)
    free = links[~named & ~links.index.isin(claimed)]

    paths = {}
    methods = {}
    for key, row in rows.items():
        p = projects.loc[key]
        if not p.p_nom_mw > 0:
            row["status"] = "no rating"
            continue
        if key not in ways:
            if not p.voltage_kv >= min_voltage:
                row["status"] = "below threshold"
            continue
        found = ways[key]
        G = _graph(links.loc[found], buses, bridge_km)
        ends = _terminals(G, buses, p, distance_upper_bound)
        methods[key] = "name"
        if ends is None:
            # the named ways do not connect the terminals: continue through unnamed ways, preferring the named ones
            v = float(p.voltage_kv)
            near = free[(free.v_nom - v).abs() <= VOLTAGE_TOLERANCE * v]
            G = _graph(pd.concat([links.loc[found], near]), buses, bridge_km)
            own = {_corridor(r) for r in links.loc[found].itertuples()}
            for e in G.edges:
                edge = G.edges[e]
                if edge["bridge"]:
                    edge["weight"] += BRIDGE_COST_KM
                elif tuple(sorted(e)) in own:
                    edge["weight"] *= NAMED_FACTOR
            ends = _terminals(G, buses, p, distance_upper_bound)
            methods[key] = "name and terminals"
        if ends is not None:
            path = nx.shortest_path(G, *ends, weight="weight")
            edges = [tuple(sorted(e)) for e in zip(path[:-1], path[1:])]
            paths[key] = [e for e in edges if not G.edges[e]["bridge"]]
            continue
        del methods[key]
        km = links.loc[found].groupby(corridor[found]).length.min().sum()
        listed = p.length_km if p.length_km > 0 else 0.0
        in_service = pd.notna(p.year) and p.year <= reference_year
        if in_service and km < MIN_LENGTH_SHARE * listed:
            continue  # stays unmatched, for missing_links; its named stubs get no capacity
        paths[key] = sorted(own)
        methods[key] = "name, all ways"

    capacity = {}
    future = {}
    for key, real in paths.items():
        p = projects.loc[key]
        in_service = pd.notna(p.year) and p.year <= reference_year
        for e in real:
            capacity[e] = capacity.get(e, 0.0) + float(p.p_nom_mw)
            future[e] = future.get(e, True) and not in_service
        matched = links[corridor.isin(real)]
        route_km = matched.groupby(corridor[matched.index]).length.min().sum()
        rows[key].update(
            status="matched" if in_service else "matched, not in service",
            method=methods[key],
            route_km=float(route_km),
            segments=len(real),
            link_ids=";".join(matched.index.astype(str)),
        )

    assigned = corridor[corridor.isin(list(capacity))]
    p_nom = pd.Series(
        {i: capacity[c] / per_corridor[c] for i, c in assigned.items()}, dtype=float
    )
    not_in_service = pd.Series({i: future[c] for i, c in assigned.items()}, dtype=bool)
    return p_nom, not_in_service, pd.DataFrame(list(rows.values()))


def missing_links(report, projects, buses, reference_year=2024, bound=0.5):
    """
    Links for projects in service without a route in the network.

    Only where an AC bus of the project's countries lies within ``bound`` degrees
    of both terminals, which leaves out export cables of offshore wind farms.

    Returns
    -------
    pd.DataFrame
        bus0, bus1, p_nom [MW], length [km], v_nom [kV], indexed by project name.
    """
    buses = buses[buses.carrier == "AC"]
    missing = report[
        (report.status == "unmatched") & (report.year <= reference_year)
    ].name
    rows = {}
    for name in missing:
        p = projects[projects.name == name].iloc[0]
        if pd.isna([p.x1, p.y1, p.x2, p.y2]).any():
            continue
        ac = buses[buses.country.isin([p.country1, p.country2])]
        if ac.empty:
            continue
        tree = KDTree(ac[["x", "y"]].astype(float).values)
        d, i = tree.query([[p.x1, p.y1], [p.x2, p.y2]])
        if d.max() > bound or i[0] == i[1]:
            continue
        straight = _km(p.x1, p.y1, p.x2, p.y2)
        rows[name] = dict(
            bus0=str(ac.index[i[0]]),
            bus1=str(ac.index[i[1]]),
            p_nom=float(p.p_nom_mw),
            length=float(p.length_km) if p.length_km > 0 else float(straight),
            v_nom=float(p.voltage_kv),
        )
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["bus0", "bus1", "p_nom", "length", "v_nom"]
    )
