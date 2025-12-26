"""
Phase 3 (steps 1-3): load optimized network, validate, simulate delays, discretize, and save.

Usage (main venv):
    python run_phase3_simulation.py --input phase2_network_opt.json --output phase3_delay_data.json --days 200
"""
import argparse
import json
from pathlib import Path

from dptbn.config import default_config
from dptbn.simulator_config import default_simulator_config
from dptbn.discretise_and_cpts import discretize_many
from dptbn.network import load_network_json
from dptbn.opt_models import validate_routes
from dptbn.sim_many_days import simulate_many_days


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 delay simulation and discretization.")
    parser.add_argument("--name", required=True, help="Network name (e.g. 'toy_v1'). Reads data/{name}_network_opt.json -> Writes data/{name}_delay_data.json")
    parser.add_argument("--days", type=int, default=None, help="Number of simulated days (override config).")
    
    # Simulator Config Overrides
    parser.add_argument("--base-delay-mean", type=float, help="Base turnaround delay mean (mins).")
    parser.add_argument("--base-delay-std", type=float, help="Base turnaround delay std (mins).")
    parser.add_argument("--ground-mean", type=float, help="Ground noise mean.")
    parser.add_argument("--ground-std", type=float, help="Ground noise std.")
    parser.add_argument("--connection-noise-std", type=float, help="Inbound connection noise std.")
    parser.add_argument("--orphan-noise-std", type=float, help="Orphan flight noise std.")
    parser.add_argument("--pax-noise-std", type=float, help="Passenger noise std.")
    parser.add_argument("--departure-noise-std", type=float, help="Departure noise std.")
    parser.add_argument("--probabilistic-cpt", action="store_true", help="Enable probabilistic CPT noise.")
    parser.add_argument("--noise-ac-std", type=float, help="Aircraft noise std (probabilistic only).")
    parser.add_argument("--noise-pilot-std", type=float, help="Pilot noise std (probabilistic only).")
    parser.add_argument("--noise-cabin-std", type=float, help="Cabin noise std (probabilistic only).")
    return parser.parse_args()


def validate_net(net) -> None:
    d = net.design
    if d is None:
        raise ValueError("Network design missing from input JSON.")
    # Validate coverage and CT windows for aircraft/pilot/crew.
    if net.aircraft_routes:
        validate_routes(net.flights, net.aircraft_routes, d.tat_aircraft, d.max_ct_aircraft, "Aircraft")
    if net.pilot_pairings:
        validate_routes(net.flights, net.pilot_pairings, d.mct_pilot, d.max_ct_pilot, "Pilot")
    if net.crew_pairings:
        validate_routes(net.flights, net.crew_pairings, d.mct_crew, d.max_ct_crew, "Crew")


def main() -> None:
    args = parse_args()
    
    # Paths
    data_dir = Path("data")
    input_path = data_dir / f"{args.name}_network_opt.json"
    
    # Fallback: if _network_opt.json doesn't exist, try _network.json? 
    if not input_path.exists():
        fallback = data_dir / f"{args.name}_network.json"
        if fallback.exists():
            print(f"Warning: {input_path} not found, falling back to {fallback}")
            input_path = fallback
        else:
            # Let it fail naturally or print error
            pass

    output_path = data_dir / f"{args.name}_delay_data.json"

    net = load_network_json(input_path)
    validate_net(net)
    sim_cfg = default_config
    
    # Build SimulatorConfig from args + defaults
    base_sim = default_simulator_config()
    sim_params = default_simulator_config()
    # Override if arg provided
    if args.base_delay_mean is not None: sim_params.base_delay_mean = args.base_delay_mean
    if args.base_delay_std is not None: sim_params.base_delay_std = args.base_delay_std
    if args.ground_mean is not None: sim_params.ground_mean = args.ground_mean
    if args.ground_std is not None: sim_params.ground_std = args.ground_std
    if args.connection_noise_std is not None: sim_params.connection_noise_std = args.connection_noise_std
    if args.orphan_noise_std is not None: sim_params.orphan_noise_std = args.orphan_noise_std
    if args.pax_noise_std is not None: sim_params.pax_noise_std = args.pax_noise_std
    if args.departure_noise_std is not None: sim_params.departure_noise_std = args.departure_noise_std
    if args.probabilistic_cpt: sim_params.probabilistic_cpt = True
    if args.noise_ac_std is not None: sim_params.noise_ac_std = args.noise_ac_std
    if args.noise_pilot_std is not None: sim_params.noise_pilot_std = args.noise_pilot_std
    if args.noise_cabin_std is not None: sim_params.noise_cabin_std = args.noise_cabin_std

    num_days = args.days or sim_cfg.num_days
    print(f"Simulating {num_days} days with seed {sim_cfg.seed}...")
    days_continuous = simulate_many_days(net, sim_cfg, num_days=num_days, sim_params=sim_params)
    
    # Optimization: Don't confirm discrete steps here. cpts_creator does it on the fly.
    # numeric_keys = ["t", "k", "k_eff", "q", "c", "p", "r", "b"]
    # days_discrete = discretize_many(days_continuous, sim_cfg.delay_bins, numeric_keys=numeric_keys)
    days_discrete = None

    payload = {
        "metadata": {
            "num_days": num_days,
            "delay_bins": list(sim_cfg.delay_bins),
            "variables": ["t", "k", "k_eff", "q", "c", "p", "r", "b"],
            "categorical_variables": ["g", "base_cat"],
            "design": net.design.__dict__ if net.design else None,
            "sim_params": sim_params.__dict__,
        },
        "days_continuous": days_continuous,
        # "days_discrete": days_discrete, # Redundant
    }
    
    # Switch to Pickle for Speed
    output_pkl = data_dir / f"{args.name}_delay_data.pkl"
    import pickle
    print(f"Pickling simulation data to {output_pkl}...")
    with open(output_pkl, "wb") as f:
        pickle.dump(payload, f)
    print("Done pickling.")
    
    # Legacy JSON (Optional/Commented out for speed)
    # output_path.write_text(json.dumps(payload, indent=2))
    # print(f"Saved Phase 3 simulation to {output_path}")


if __name__ == "__main__":
    main()
