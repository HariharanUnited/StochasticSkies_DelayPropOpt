"""
Validate Phase 3 simulation outputs against the slack-based propagation rules.

Usage (after generating phase3_delay_data.json):
    python validate_phase3.py --network phase2_network_opt.json --data phase3_delay_data.json
"""
import argparse
import json
from pathlib import Path

from dptbn.network import load_network_json


def slack(parent, child, min_ct):
    return (child.scheduled_departure - parent.scheduled_arrival) - min_ct


def validate_day(day, flights, design):
    by_id = {f.flight_id: f for f in flights}
    for fid, vals in day.items():
        f = by_id[fid]
        k = vals["k"]
        k_eff = vals["k_eff"]
        q = vals["q"]
        c = vals["c"]
        p = vals["p"]
        r = vals["r"]
        b = vals["b"]
        t = vals["t"]
        g = vals["g"]
        base_cat = vals["base_cat"]

        # Aircraft
        if f.inbound_aircraft_flight:
            parent = by_id[f.inbound_aircraft_flight]
            sa = slack(parent, f, design.tat_aircraft)
            expected_k_eff = max(0, k - max(0, sa))
            assert abs(k_eff - expected_k_eff) < 1e-6
        else:
            assert k == 0 == k_eff

        # Pilot
        follows_pilot = f.inbound_pilot_flight and f.inbound_pilot_flight == f.inbound_aircraft_flight
        if follows_pilot:
            assert q == 0
        elif f.inbound_pilot_flight:
            parent = by_id[f.inbound_pilot_flight]
            expected_q = max(0, day[f.inbound_pilot_flight]["t"] - max(0, slack(parent, f, design.mct_pilot)))
            assert abs(q - expected_q) < 1e-6
        else:
            assert q == 0

        # Cabin
        follows_cabin = f.inbound_cabin_flight and f.inbound_cabin_flight == f.inbound_aircraft_flight
        if follows_cabin:
            assert c == 0
        elif f.inbound_cabin_flight:
            parent = by_id[f.inbound_cabin_flight]
            expected_c = max(0, day[f.inbound_cabin_flight]["t"] - max(0, slack(parent, f, design.mct_crew)))
            assert abs(c - expected_c) < 1e-6
        else:
            assert c == 0

        # Pax
        if f.inbound_passenger_flights:
            expected_list = []
            for pfid in f.inbound_passenger_flights:
                if pfid in day:
                    parent = by_id[pfid]
                    expected_list.append(max(0, day[pfid]["t"] - max(0, slack(parent, f, design.min_ct_pax))))
            if expected_list:
                assert abs(p - max(expected_list)) < 1e-6
            else:
                assert p == 0
        else:
            assert p == 0

        # Reactionary and departure delay
        assert abs(r - max(k_eff, q, c, p)) < 1e-6
        assert abs(t - max(b, r)) < 1e-6

        # Cause
        if t == 0:
            assert g == "None"
        elif b >= r:
            assert g == base_cat
        else:
            assert g == "Reactionary"


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 3 delay data.")
    parser.add_argument("--network", default="phase2_network_opt.json", help="Optimized network JSON")
    parser.add_argument("--data", default="phase3_delay_data.json", help="Simulated delay data JSON")
    args = parser.parse_args()

    net = load_network_json(args.network)
    design = net.design
    data = json.loads(Path(args.data).read_text())
    days = data["days_continuous"]
    for idx, day in enumerate(days):
        validate_day(day, net.flights, design)
    print(f"Validation passed for {len(days)} days.")


if __name__ == "__main__":
    main()
