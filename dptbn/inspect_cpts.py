"""
Inspect CPT completeness and uniform rows.

Usage:
    python inspect_cpts.py --cpts phase7_cpts.json
"""
import argparse
import json
from pathlib import Path
from itertools import product


def parse_parent_key(key: str):
    parts = key.split("|")
    # Last part is g (string), others are ints
    *bins, g = parts
    return tuple(int(b) for b in bins), g


def main():
    parser = argparse.ArgumentParser(description="Inspect CPT completeness.")
    parser.add_argument("--cpts", default="phase7_cpts.json", help="CPT JSON file")
    parser.add_argument("--max-report", type=int, default=3, help="How many missing/uniform rows to print per flight")
    args = parser.parse_args()

    data = json.loads(Path(args.cpts).read_text())
    parents = data["metadata"]["parents"]  # e.g., ['k_bin','q_bin','c_bin','p_bin','g']
    cpts = data["cpts"]

    # Infer parent domains from keys
    # Collect all bins (ints) for each position, and g states (strings)
    bins_per_pos = [set() for _ in range(len(parents) - 1)]
    g_states = set()
    sample_table = next(iter(cpts.values()))
    for key in sample_table.keys():
        bins, g = parse_parent_key(key)
        g_states.add(g)
        for i, b in enumerate(bins):
            bins_per_pos[i].add(b)
    g_states = sorted(g_states)

    completion = data["metadata"].get("completion", "observed")
    fill_missing = data["metadata"].get("fill_missing", "prior")
    if completion == "full":
        num_bin_states = len(data["metadata"]["delay_bins"]) + 1
        bins_per_pos = [list(range(num_bin_states)) for _ in range(len(parents) - 1)]
    else:
        bins_per_pos = [sorted(s) for s in bins_per_pos]

    def all_parent_keys():
        for combo in product(*bins_per_pos, g_states):
            bins = combo[:-1]
            g = combo[-1]
            yield "|".join([str(b) for b in bins] + [g])

    all_keys = list(all_parent_keys())

    for fid, table in cpts.items():
        missing = [k for k in all_keys if k not in table]
        uniform = []
        for k, probs in table.items():
            vals = list(probs.values())
            if len(vals) > 0 and all(abs(v - vals[0]) < 1e-9 for v in vals):
                uniform.append(k)
        print(f"Flight {fid}: missing={len(missing)}, uniform={len(uniform)}")
        if missing and args.max_report > 0:
            print("  Missing examples:", missing[: args.max_report])
        if uniform and args.max_report > 0:
            print("  Uniform examples:", uniform[: args.max_report])
        if fill_missing == "none" and missing:
            print("  Note: fill_missing=none; missing combos expected to be filled at inference time.")


if __name__ == "__main__":
    main()
