"""
Driver script to run the JOINT robust (coupled) arc-based models in a docplex/cplex environment.

Usage (from repo root, after activating `cplexenv` that has docplex):
    python -m dptbn.run_robust_models --name v2 --start-weight 1.0 --w2 0.05 --w3 0.01

It will:
 - load the saved Phase 2 network (and design)
 - solve a joint aircraft+pilot+crew coupled model:
     * primary: minimize total starts (resources used)
     * secondary: maximize 2-chain coupling (shared arcs)
     * tertiary: maximize 3-chain coupling (shared i->j->k chains)
 - overwrite the routes/pairings in the network object
 - save to data/{name}_network_opt.json (same convention as feasibility runner)
"""
import argparse
from pathlib import Path

from dptbn.network import load_network_json, save_network_json
from dptbn.opt_models import validate_routes
from dptbn.robust_core_models import solve_joint_robust_coupled_with_log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run JOINT robust (coupled) docplex models for aircraft/pilot/crew.")
    p.add_argument(
        "--name",
        required=True,
        help="Network name (e.g. 'v2'). Reads data/{name}_network.json -> Writes data/{name}_network_opt.json",
    )
    p.add_argument(
        "--start-weight",
        type=float,
        default=1.0,
        help="Weight on total starts (resources used). Higher => stronger preference for fewer resources.",
    )
    p.add_argument(
        "--w2",
        type=float,
        default=0.0,
        help="Weight (reward) for 2-chain coupling (shared arcs). Set to 0 to disable 2-chain coupling.",
    )
    p.add_argument(
        "--w3",
        type=float,
        default=0.0,
        help="Weight (reward) for 3-chain coupling (shared i->j->k chains). Set to 0 to disable 3-chain coupling.",
    )
    p.add_argument(
        "--max-triples",
        type=int,
        default=0,
        help="Optional cap on number of triples used for 3-chain coupling (0 = no cap). Useful to keep MILP small.",
    )
    p.add_argument(
        "--log-output",
        action="store_true",
        help="Enable CPLEX log output.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Paths
    data_dir = Path("data")
    if not data_dir.exists():
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

    print("Solving JOINT robust coupled model (docplex)...")
    routes_ac, routes_pilot, routes_crew = solve_joint_robust_coupled_with_log(
        net,
        start_weight=args.start_weight,
        w2=args.w2,
        w3=args.w3,
        max_triples=(args.max_triples if args.max_triples and args.max_triples > 0 else None),
        log_output=args.log_output,
    )

    print(f"Aircraft routes: {old_ac} -> {len(routes_ac)}")
    print(f"Pilot pairings:  {old_pilot} -> {len(routes_pilot)}")
    print(f"Crew pairings:   {old_crew} -> {len(routes_crew)}")

    d = net.design
    validate_routes(net.flights, routes_ac, d.tat_aircraft, d.max_ct_aircraft, "Aircraft")
    validate_routes(net.flights, routes_pilot, d.mct_pilot, d.max_ct_pilot, "Pilot")
    validate_routes(net.flights, routes_crew, d.mct_crew, d.max_ct_crew, "Crew")

    # Persist back into network object (same as feasibility runner)
    net.aircraft_routes = routes_ac
    net.pilot_pairings = routes_pilot
    net.crew_pairings = routes_crew

    # Re-set inbound links on flights based on optimized pairings.
    flights_by_id = {f.flight_id: f for f in net.flights}

    for ridx, route in enumerate(routes_ac, start=1):
        for idx, fid in enumerate(route):
            flights_by_id[fid].tail_id = f"ACR{ridx}"
            flights_by_id[fid].inbound_aircraft_flight = route[idx - 1] if idx > 0 else None

    for ridx, route in enumerate(routes_pilot, start=1):
        for idx, fid in enumerate(route):
            flights_by_id[fid].pilot_id = f"PLT{ridx}"
            flights_by_id[fid].inbound_pilot_flight = route[idx - 1] if idx > 0 else None

    for ridx, route in enumerate(routes_crew, start=1):
        for idx, fid in enumerate(route):
            flights_by_id[fid].crew_id = f"CRW{ridx}"
            flights_by_id[fid].inbound_cabin_flight = route[idx - 1] if idx > 0 else None

    save_network_json(net, output_path)
    print(f"Optimized (robust coupled) network saved to {output_path}")


if __name__ == "__main__":
    main()
