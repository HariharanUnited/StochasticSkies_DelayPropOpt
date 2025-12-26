"""
Driver script to run the arc-based feasibility models in a docplex/cplex environment.

Usage (from repo root, after activating `cplexenv` that has docplex):
    python run_feasibility_models.py --input phase2_network.json --output phase2_network_opt.json

It will:
 - load the saved Phase 2 network (and design)
 - solve aircraft, pilot, crew feasibility models (minimizing number of resources)
 - overwrite the routes/pairings in the network object
 - save to the specified output JSON
"""
import argparse
from pathlib import Path

from dptbn.network import load_network_json, save_network_json
from dptbn.opt_models import (
    solve_aircraft_feasibility_with_log,
    solve_pilot_feasibility_with_log,
    solve_crew_feasibility_with_log,
    validate_routes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run docplex feasibility models for aircraft/pilot/crew.")
    parser.add_argument("--name", required=True, help="Network name (e.g. 'toy_v1'). Reads data/{name}_network.json -> Writes data/{name}_network_opt.json")
    parser.add_argument(
        "--difference",
        choices=["false", "extreme", "stick"],
        default="false",
        help="false: no overlap term; extreme: penalize overlap; stick: reward overlap",
    )
    parser.add_argument(
        "--difference-weight",
        type=float,
        default=0.01,
        help="Weight for overlap term in objective (secondary to minimizing starts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Paths
    data_dir = Path("data")
    if not data_dir.exists():
         # Fallback search if running from wrong dir, though unlikely with module run
         if Path("../data").exists():
             data_dir = Path("../data")
    
    input_path = data_dir / f"{args.name}_network.json"
    output_path = data_dir / f"{args.name}_network_opt.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input network not found at {input_path}")

    net = load_network_json(input_path)
    old_ac = len(net.aircraft_routes)
    old_pilot = len(net.pilot_pairings)
    old_crew = len(net.crew_pairings)

    print("Solving aircraft feasibility (docplex)...")
    routes_ac = solve_aircraft_feasibility_with_log(net, log_output=True)
    print(f"Aircraft routes reduced: {old_ac} -> {len(routes_ac)}")
    validate_routes(net.flights, routes_ac, net.design.tat_aircraft, net.design.max_ct_aircraft, "Aircraft")

    def build_overlap_weights(routes, weight, reward=False):
        weights = {}
        for route in routes:
            for i in range(len(route) - 1):
                key = (route[i], route[i + 1])
                weights[key] = (-weight if reward else weight)
        return weights

    overlap_ac = build_overlap_weights(routes_ac, args.difference_weight, reward=(args.difference == "stick"))

    print("Solving pilot feasibility (docplex)...")
    routes_pilot = solve_pilot_feasibility_with_log(
        net,
        log_output=True,
        overlap_weights=(overlap_ac if args.difference != "false" else None),
    )
    print(f"Pilot pairings reduced: {old_pilot} -> {len(routes_pilot)}")
    validate_routes(net.flights, routes_pilot, net.design.mct_pilot, net.design.max_ct_pilot, "Pilot")

    overlap_pilot = build_overlap_weights(routes_pilot, args.difference_weight, reward=(args.difference == "stick"))

    print("Solving crew feasibility (docplex)...")
    routes_crew = solve_crew_feasibility_with_log(
        net,
        log_output=True,
        overlap_weights=({**overlap_ac, **overlap_pilot} if args.difference != "false" else None),
    )
    print(f"Crew pairings reduced: {old_crew} -> {len(routes_crew)}")
    validate_routes(net.flights, routes_crew, net.design.mct_crew, net.design.max_ct_crew, "Crew")

    net.aircraft_routes = routes_ac
    net.pilot_pairings = routes_pilot
    net.crew_pairings = routes_crew
    # Re-set inbound links on flights based on optimized pairings.
    flights_by_id = {f.flight_id: f for f in net.flights}
    for route in routes_ac:
        for idx, fid in enumerate(route):
            flights_by_id[fid].tail_id = f"ACR{routes_ac.index(route)+1}"
            flights_by_id[fid].inbound_aircraft_flight = route[idx - 1] if idx > 0 else None
    for route in routes_pilot:
        for idx, fid in enumerate(route):
            flights_by_id[fid].pilot_id = f"PLT{routes_pilot.index(route)+1}"
            flights_by_id[fid].inbound_pilot_flight = route[idx - 1] if idx > 0 else None
    for route in routes_crew:
        for idx, fid in enumerate(route):
            flights_by_id[fid].crew_id = f"CRW{routes_crew.index(route)+1}"
            flights_by_id[fid].inbound_cabin_flight = route[idx - 1] if idx > 0 else None

    save_network_json(net, output_path)
    print(f"Optimized network saved to {output_path}")


if __name__ == "__main__":
    main()
