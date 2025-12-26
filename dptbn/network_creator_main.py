import argparse
from pathlib import Path

from dptbn.config import NetworkDesign, default_network_design
from dptbn.network import generate_synthetic_network, save_network_json


def build_design(args: argparse.Namespace) -> NetworkDesign:
    base = default_network_design
    return NetworkDesign(
        num_airports=args.num_airports or base.num_airports,
        num_hubs=args.num_hubs or base.num_hubs,
        num_flights=args.num_flights or base.num_flights,
        num_aircraft=args.num_aircraft or base.num_aircraft,
        num_pilots=args.num_pilots or base.num_pilots,
        num_crews=args.num_crews or base.num_crews,
        min_flight_duration=args.min_flight_duration or base.min_flight_duration,
        max_flight_duration=args.max_flight_duration or base.max_flight_duration,
        mct_crew=args.mct_crew or base.mct_crew,
        mct_pilot=args.mct_pilot or base.mct_pilot,
        max_ct_pilot=args.max_ct_pilot or base.max_ct_pilot,
        max_ct_crew=args.max_ct_crew or base.max_ct_crew,
        max_ct_aircraft=args.max_ct_aircraft or base.max_ct_aircraft,
        tat_aircraft=args.tat_aircraft or base.tat_aircraft,
        min_ct_pax=args.min_ct_pax or base.min_ct_pax,
        max_ct_pax=args.max_ct_pax or base.max_ct_pax,
        max_aircraft_stints=args.max_aircraft_stints or base.max_aircraft_stints,
        max_pilot_stints=args.max_pilot_stints or base.max_pilot_stints,
        max_crew_stints=args.max_crew_stints or base.max_crew_stints,
        passenger_connection_fraction=(
            args.passenger_connection_fraction
            if args.passenger_connection_fraction is not None
            else base.passenger_connection_fraction
        ),
        day_length=args.day_length or base.day_length,
        time_step=base.time_step,
        seed=args.seed if args.seed is not None else base.seed,
        airport_codes=base.airport_codes,
        hub_codes=base.hub_codes,
    )


def summarize(net, path: Path) -> None:
    print(f"Saved network to {path}")
    print(f"Flights: {len(net.flights)}")
    print(f"Hubs: {net.hubs}")
    print(f"Spokes (first 10): {net.spokes[:10]}")
    print(f"Aircraft routes: {len(net.aircraft_routes)}")
    print(f"Pilot pairings: {len(net.pilot_pairings)}")
    print(f"Crew pairings: {len(net.crew_pairings)}")
    print("First 5 flights:")
    for f in net.flights[:5]:
        print(
            f"  {f.flight_id} {f.departure_airport}->{f.arrival_airport} "
            f"{f.scheduled_departure}->{f.scheduled_arrival} "
            f"AC {f.tail_id} P {f.pilot_id} C {f.crew_id}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic DPT-BN network and save to JSON.")
    parser.add_argument("--name", required=True, help="Network name (e.g. 'toy_v1'). Outputs to data/{name}_network.json.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--num-airports", type=int, help="Total airports (<=100).")
    parser.add_argument("--num-hubs", type=int, help="Number of hubs (<= num-airports).")
    parser.add_argument("--num-flights", type=int, help="Number of flights.")
    parser.add_argument("--num-aircraft", type=int, help="Number of aircraft.")
    parser.add_argument("--num-pilots", type=int, help="Number of pilots.")
    parser.add_argument("--num-crews", type=int, help="Number of cabin crews.")
    parser.add_argument("--min-flight-duration", type=int, help="Min block time (mins).")
    parser.add_argument("--max-flight-duration", type=int, help="Max block time (mins).")
    parser.add_argument("--mct-crew", type=int, help="Min connect time crew (mins).")
    parser.add_argument("--mct-pilot", type=int, help="Min connect time pilot (mins).")
    parser.add_argument("--max-ct-pilot", type=int, help="Max connect time pilot (mins).")
    parser.add_argument("--max-ct-crew", type=int, help="Max connect time crew (mins).")
    parser.add_argument("--max-ct-aircraft", type=int, help="Max connect time aircraft (mins).")
    parser.add_argument("--tat-aircraft", type=int, help="Turn time aircraft (mins).")
    parser.add_argument("--min-ct-pax", type=int, help="Min connect time pax (mins).")
    parser.add_argument("--max-ct-pax", type=int, help="Max connect time pax (mins).")
    parser.add_argument("--max-aircraft-stints", type=int, help="Max legs per aircraft route.")
    parser.add_argument("--max-pilot-stints", type=int, help="Max legs per pilot pairing.")
    parser.add_argument("--max-crew-stints", type=int, help="Max legs per crew pairing.")
    parser.add_argument(
        "--passenger-connection-fraction",
        type=float,
        help="Fraction of feasible pax connections to sample (0-1).",
    )
    parser.add_argument("--day-length", type=int, help="Day length in minutes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design = build_design(args)
    net = generate_synthetic_network(design, seed=design.seed)
    
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    out_path = data_dir / f"{args.name}_network.json"
    save_network_json(net, out_path)
    summarize(net, out_path)


if __name__ == "__main__":
    main()
