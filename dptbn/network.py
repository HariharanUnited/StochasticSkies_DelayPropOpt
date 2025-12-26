"""
Network definitions and synthetic generator for the DPT-BN PoC.

Phase 2 builds a small but constrained toy schedule with aircraft routings,
pilot pairings, cabin-crew pairings, and passenger connections.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np

from dptbn.config import NetworkDesign


@dataclass
class Flight:
    flight_id: str
    departure_airport: str
    arrival_airport: str
    scheduled_departure: int  # minutes from day start
    scheduled_arrival: int  # minutes from day start
    tail_id: Optional[str] = None
    pilot_id: Optional[str] = None
    crew_id: Optional[str] = None
    inbound_aircraft_flight: Optional[str] = None
    inbound_pilot_flight: Optional[str] = None
    inbound_cabin_flight: Optional[str] = None
    inbound_passenger_flights: List[str] = field(default_factory=list)


@dataclass
class SyntheticNetwork:
    flights: List[Flight]
    aircraft_routes: List[List[str]]
    pilot_pairings: List[List[str]]
    crew_pairings: List[List[str]]
    passenger_connections: Dict[str, List[str]]
    hubs: List[str]
    spokes: List[str]
    design: Optional[NetworkDesign] = None


def network_to_dict(net: SyntheticNetwork) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "flights": [asdict(f) for f in net.flights],
        "aircraft_routes": net.aircraft_routes,
        "pilot_pairings": net.pilot_pairings,
        "crew_pairings": net.crew_pairings,
        "passenger_connections": net.passenger_connections,
        "hubs": net.hubs,
        "spokes": net.spokes,
    }
    if net.design:
        payload["design"] = asdict(net.design)
    return payload


def network_from_dict(data: Dict[str, Any]) -> SyntheticNetwork:
    flights = [Flight(**f) for f in data["flights"]]
    design_data = data.get("design")
    design = NetworkDesign(**design_data) if design_data else None
    return SyntheticNetwork(
        flights=flights,
        aircraft_routes=data["aircraft_routes"],
        pilot_pairings=data["pilot_pairings"],
        crew_pairings=data["crew_pairings"],
        passenger_connections=data["passenger_connections"],
        hubs=data["hubs"],
        spokes=data["spokes"],
        design=design,
    )


def save_network_json(net: SyntheticNetwork, path: Path) -> None:
    path = Path(path)
    payload = network_to_dict(net)
    path.write_text(json.dumps(payload, indent=2))


def load_network_json(path: Path) -> SyntheticNetwork:
    path = Path(path)
    payload = json.loads(path.read_text())
    return network_from_dict(payload)


def _build_airports(design: NetworkDesign) -> Tuple[List[str], List[str]]:
    if design.num_airports > len(design.airport_codes):
        raise ValueError("num_airports cannot exceed provided airport_codes")
    selected_airports = design.airport_codes[: design.num_airports]

    # Choose hubs from the provided hub_codes that are in the selected airport list.
    candidate_hubs = [c for c in design.hub_codes if c in selected_airports]
    if design.num_hubs > len(candidate_hubs):
        raise ValueError("Not enough hubs available within selected airports")

    hubs = candidate_hubs[: design.num_hubs]
    spokes = [a for a in selected_airports if a not in hubs]
    return hubs, spokes


def _sample_departure_times(design: NetworkDesign, rng: np.random.Generator) -> List[int]:
    latest_departure = max(design.day_length - design.max_flight_duration, 0)
    grid = np.arange(0, latest_departure + design.time_step, design.time_step)
    replace = len(grid) < design.num_flights
    dep_times = rng.choice(grid, size=design.num_flights, replace=replace)
    return sorted(int(x) for x in dep_times)


def _sample_duration(design: NetworkDesign, rng: np.random.Generator) -> int:
    durations = np.arange(
        design.min_flight_duration, design.max_flight_duration + design.time_step, design.time_step
    )
    return int(rng.choice(durations))


def _pick_endpoints(hubs: List[str], spokes: List[str], rng: np.random.Generator) -> Tuple[str, str]:
    """Choose (origin, destination) ensuring at least one hub and no spoke-spoke legs."""
    if not spokes:
        origin, dest = rng.choice(hubs, size=2, replace=False)
        return origin, dest

    if rng.random() < 0.5:
        origin = rng.choice(hubs)
        dest_candidates = [a for a in hubs + spokes if a != origin]
        dest = rng.choice(dest_candidates)
    else:
        dest = rng.choice(hubs)
        origin_candidates = [a for a in hubs + spokes if a != dest]
        origin = rng.choice(origin_candidates)
    return origin, dest


def _generate_flights(design: NetworkDesign, hubs: List[str], spokes: List[str], rng: np.random.Generator) -> List[Flight]:
    if design.enforce_roundtrips:
        return _generate_roundtrip_flights(design, hubs, spokes, rng)
    departures = _sample_departure_times(design, rng)
    flights: List[Flight] = []
    for idx, dep_time in enumerate(departures, start=1):
        duration = _sample_duration(design, rng)
        origin, dest = _pick_endpoints(hubs, spokes, rng)
        arr_time = dep_time + duration
        flight = Flight(
            flight_id=f"F{idx:03d}",
            departure_airport=origin,
            arrival_airport=dest,
            scheduled_departure=dep_time,
            scheduled_arrival=arr_time,
        )
        flights.append(flight)
    flights.sort(key=lambda f: f.scheduled_departure)
    return flights


def _generate_roundtrip_flights(
    design: NetworkDesign, hubs: List[str], spokes: List[str], rng: np.random.Generator
) -> List[Flight]:
    flights: List[Flight] = []
    fid = 1
    max_attempts = design.num_flights * 5
    attempts = 0
    while len(flights) < design.num_flights and attempts < max_attempts:
        attempts += 1
        hub = rng.choice(hubs)
        if not spokes:
            break
        spoke = rng.choice(spokes)
        duration_out = _sample_duration(design, rng)
        duration_in = _sample_duration(design, rng)

        latest_dep = design.day_length - duration_out - design.tat_aircraft - duration_in
        if latest_dep < 0:
            continue
        grid_max = latest_dep // design.time_step
        dep_out = int(rng.integers(0, grid_max + 1) * design.time_step)
        arr_out = dep_out + duration_out
        dep_in_min = arr_out + design.tat_aircraft
        dep_in_max = min(dep_out + design.return_within_minutes, design.day_length - duration_in)
        if dep_in_min > dep_in_max:
            continue
        dep_in = int(rng.integers(dep_in_min // design.time_step, dep_in_max // design.time_step + 1) * design.time_step)
        arr_in = dep_in + duration_in

        out_flight = Flight(
            flight_id=f"F{fid:03d}",
            departure_airport=hub,
            arrival_airport=spoke,
            scheduled_departure=dep_out,
            scheduled_arrival=arr_out,
        )
        flights.append(out_flight)
        fid += 1
        if len(flights) >= design.num_flights:
            break
        in_flight = Flight(
            flight_id=f"F{fid:03d}",
            departure_airport=spoke,
            arrival_airport=hub,
            scheduled_departure=dep_in,
            scheduled_arrival=arr_in,
        )
        flights.append(in_flight)
        fid += 1

    flights.sort(key=lambda f: f.scheduled_departure)
    return flights


def _build_chains(
    flights: List[Flight],
    min_ct: int,
    max_ct: int,
    max_len: int,
    rng: np.random.Generator,
) -> List[List[Flight]]:
    """
    Build ordered chains (routes/pairings) such that successive flights are
    time-feasible at the same airport.
    """
    sorted_by_dep = sorted(flights, key=lambda f: f.scheduled_departure)
    unused_ids = {f.flight_id for f in flights}
    chains: List[List[Flight]] = []

    def candidates(current: Flight) -> List[Flight]:
        return [
            f
            for f in sorted_by_dep
            if f.flight_id in unused_ids
            and f.departure_airport == current.arrival_airport
            and min_ct <= f.scheduled_departure - current.scheduled_arrival <= max_ct
        ]

    while unused_ids:
        start = next(f for f in sorted_by_dep if f.flight_id in unused_ids)
        route = [start]
        unused_ids.remove(start.flight_id)
        current = start
        while len(route) < max_len:
            cands = candidates(current)
            if not cands:
                break
            nxt = rng.choice(cands)
            route.append(nxt)
            unused_ids.remove(nxt.flight_id)
            current = nxt
        chains.append(route)
    return chains


def _assign_aircraft(
    flights: List[Flight],
    design: NetworkDesign,
    rng: np.random.Generator,
) -> List[List[str]]:
    chains = _build_chains(
        flights,
        min_ct=design.tat_aircraft,
        max_ct=design.max_ct_aircraft,
        max_len=design.max_aircraft_stints,
        rng=rng,
    )
    if len(chains) > design.num_aircraft:
        raise ValueError("Not enough aircraft to cover all routes")

    for idx, chain in enumerate(chains):
        tail_id = f"AC{idx + 1}"
        for pos, flight in enumerate(chain):
            flight.tail_id = tail_id
            if pos > 0:
                flight.inbound_aircraft_flight = chain[pos - 1].flight_id
    return [[f.flight_id for f in chain] for chain in chains]


def _assign_pairings(
    flights: List[Flight],
    design: NetworkDesign,
    rng: np.random.Generator,
    kind: str,
) -> List[List[str]]:
    if kind == "pilot":
        min_ct, max_ct, max_len, pool = (
            design.mct_pilot,
            design.max_ct_pilot,
            design.max_pilot_stints,
            design.num_pilots,
        )
        prefix = "P"
    elif kind == "crew":
        min_ct, max_ct, max_len, pool = (
            design.mct_crew,
            design.max_ct_crew,
            design.max_crew_stints,
            design.num_crews,
        )
        prefix = "C"
    else:
        raise ValueError("kind must be 'pilot' or 'crew'")

    shuffled = flights[:]
    rng.shuffle(shuffled)
    chains = _build_chains(shuffled, min_ct=min_ct, max_ct=max_ct, max_len=max_len, rng=rng)
    if len(chains) > pool:
        raise ValueError(f"Not enough {kind}s to cover all pairings")

    for idx, chain in enumerate(chains):
        pairing_id = f"{prefix}{idx + 1}"
        for pos, flight in enumerate(chain):
            if kind == "pilot":
                flight.pilot_id = pairing_id
                if pos > 0:
                    flight.inbound_pilot_flight = chain[pos - 1].flight_id
            else:
                flight.crew_id = pairing_id
                if pos > 0:
                    flight.inbound_cabin_flight = chain[pos - 1].flight_id
    return [[f.flight_id for f in chain] for chain in chains]


def _assign_passenger_connections(
    flights: List[Flight],
    design: NetworkDesign,
    rng: np.random.Generator,
) -> Dict[str, List[str]]:
    feasible = []
    for inbound in flights:
        for outbound in flights:
            if inbound.flight_id == outbound.flight_id:
                continue
            if inbound.arrival_airport != outbound.departure_airport:
                continue
            gap = outbound.scheduled_departure - inbound.scheduled_arrival
            if design.min_ct_pax <= gap <= design.max_ct_pax:
                feasible.append((inbound.flight_id, outbound.flight_id))

    if not feasible:
        return {}

    sample_size = max(1, int(len(feasible) * design.passenger_connection_fraction))
    sample_size = min(sample_size, len(feasible))
    chosen = rng.choice(len(feasible), size=sample_size, replace=False)

    connections: Dict[str, List[str]] = {}
    for idx in chosen:
        inbound_id, outbound_id = feasible[idx]
        connections.setdefault(outbound_id, []).append(inbound_id)

    flight_by_id = {f.flight_id: f for f in flights}
    for outbound_id, inbound_ids in connections.items():
        flight_by_id[outbound_id].inbound_passenger_flights.extend(inbound_ids)
    return connections


def generate_synthetic_network(design: NetworkDesign, seed: Optional[int] = None) -> SyntheticNetwork:
    """
    Create a synthetic schedule and passenger connections.

    Routes/pairings are not constructed here; they are expected to be produced
    by the docplex feasibility models in Phase 2.
    """
    rng = np.random.default_rng(seed if seed is not None else design.seed)
    hubs, spokes = _build_airports(design)
    flights = _generate_flights(design, hubs, spokes, rng)
    passenger_connections = _assign_passenger_connections(flights, design, rng)

    return SyntheticNetwork(
        flights=flights,
        aircraft_routes=[],
        pilot_pairings=[],
        crew_pairings=[],
        passenger_connections=passenger_connections,
        hubs=hubs,
        spokes=spokes,
        design=design,
    )


def load_toy_schedule(design: Optional[NetworkDesign] = None) -> SyntheticNetwork:
    """
    Convenience wrapper for generating the toy network with default design params.
    """
    design = design or NetworkDesign()
    return generate_synthetic_network(design=design)
