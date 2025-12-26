"""
Single-day delay simulation with simple propagation.
"""
from typing import Dict, List, Optional

import numpy as np

from dptbn.config import SimulationConfig, NetworkDesign
from dptbn.network import SyntheticNetwork, Flight
from dptbn.simulator_config import SimulatorConfig


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def simulate_day(
    net: SyntheticNetwork,
    sim_config: SimulationConfig,
    sim_params: SimulatorConfig = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Simulate one day of operations for the provided network.

    Returns mapping: flight_id -> {"t","k","q","c","p","g"} continuous minutes.
    """
    flights = sorted(net.flights, key=lambda f: f.scheduled_departure)
    rng = _rng(seed)
    by_id = {f.flight_id: f for f in flights}
    design: NetworkDesign = net.design
    if sim_params is None:
        sim_params = default_simulator_config()

    cat_labels = list(sim_params.base_cat_probs.keys())
    cat_probs = list(sim_params.base_cat_probs.values())

    results: Dict[str, Dict[str, float]] = {}

    def base_noise(scale: float = 1.0) -> float:
        return float(rng.normal(loc=0.0, scale=scale))

    def clamp(x: float) -> float:
        return max(0.0, x) if sim_params.clamp_nonnegative else x

    def slack(parent: Flight, child: Flight, min_ct: int) -> float:
        return (child.scheduled_departure - parent.scheduled_arrival) - min_ct

    for f in flights:
        # Base turnaround delay and category
        b = clamp(rng.normal(loc=sim_params.base_delay_mean, scale=sim_params.base_delay_std))
        base_cat = rng.choice(cat_labels, p=cat_probs) if b > 0 else "None"

        # Aircraft delay: k = arrival delay of inbound; k_eff subtracts slack
        if f.inbound_aircraft_flight and f.inbound_aircraft_flight in results:
            parent = by_id[f.inbound_aircraft_flight]
            k = results[f.inbound_aircraft_flight]["t"]
            k_eff = clamp(k - max(0, slack(parent, f, design.tat_aircraft)))
            if sim_params.probabilistic_cpt:
                k_eff = clamp(k_eff + base_noise(sim_params.noise_ac_std))
        else:
            k = 0.0
            k_eff = 0.0

        # Pilot and cabin delays with special-case following aircraft
        follows_pilot = f.inbound_pilot_flight and f.inbound_pilot_flight == f.inbound_aircraft_flight
        follows_cabin = f.inbound_cabin_flight and f.inbound_cabin_flight == f.inbound_aircraft_flight

        if follows_pilot:
            q = 0.0
        else:
            if f.inbound_pilot_flight and f.inbound_pilot_flight in results:
                parent = by_id[f.inbound_pilot_flight]
                q = clamp(results[f.inbound_pilot_flight]["t"] - max(0, slack(parent, f, design.mct_pilot)))
                if sim_params.probabilistic_cpt:
                    q = clamp(q + base_noise(sim_params.noise_pilot_std))
            else:
                q = 0.0

        if follows_cabin:
            c = 0.0
        else:
            if f.inbound_cabin_flight and f.inbound_cabin_flight in results:
                parent = by_id[f.inbound_cabin_flight]
                c = clamp(results[f.inbound_cabin_flight]["t"] - max(0, slack(parent, f, design.mct_crew)))
                if sim_params.probabilistic_cpt:
                    c = clamp(c + base_noise(sim_params.noise_cabin_std))
            else:
                c = 0.0

        p_vals: List[float] = []
        p_conns: Dict[str, float] = {}
        for fid in f.inbound_passenger_flights:
            if fid in results:
                parent = by_id[fid]
                val = clamp(results[fid]["t"] - max(0, slack(parent, f, design.min_ct_pax)))
                if sim_params.probabilistic_cpt:
                    val = clamp(val + base_noise(sim_params.noise_pax_std))
                p_vals.append(val)
                p_conns[fid] = val
        p = max(p_vals) if p_vals else 0.0

        r = max(k_eff, q, c, p)
        t = max(b, r)

        if t == 0:
            g_label = "None"
        elif b >= r:
            g_label = base_cat
        else:
            g_label = "Reactionary"

        results[f.flight_id] = {
            "t": t,
            "k": k,
            "k_eff": k_eff,
            "q": q,
            "c": c,
            "p": p,
            "p_conns": p_conns,
            "r": r,
            "b": b,
            "g": g_label,
            "base_cat": base_cat,
        }

    return results
