"""
Build pgmpy model from CPTs (inter-flight) and run a sample query.

Usage:
    python run_pgmpy_inference.py --cpts phase7_cpts.json --flight F001 --evidence k_bin=1 q_bin=1 c_bin=1 p_<fid>_bin=1 g=Ops
"""
import argparse
import pickle

from dptbn.bn_pgmpy import build_model, query_posterior


def parse_evidence(evidence_list):
    ev = {}
    for item in evidence_list:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        if k == "g":
            ev[k] = v
        else:
            ev[k] = int(v)
    return ev


def main():
    parser = argparse.ArgumentParser(description="Run pgmpy inference on a flight CPT.")
    parser.add_argument("--cpts", default="phase7_cpts.json", help="CPT JSON file")
    parser.add_argument("--flight", required=True, help="Flight ID to query (e.g., F001)")
    parser.add_argument(
        "--evidence",
        nargs="*",
        default=[],
        help="Evidence as key=val, e.g., k_bin=1 q_bin=1 c_bin=1 p_<fid>_bin=1 g=Ops",
    )
    parser.add_argument("--cached-model", help="Optional pickle of prebuilt VariableElimination")
    args = parser.parse_args()

    if args.cached_model:
        with open(args.cached_model, "rb") as f:
            infer = pickle.load(f)
    else:
        infer = build_model(args.cpts)
    ev = parse_evidence(args.evidence)
    posterior = query_posterior(infer, args.flight, ev)
    print(f"Posterior for {args.flight} t_bin given evidence {ev}:")
    for k, v in posterior.items():
        print(f"  t_bin={k}: {v:.4f}")


if __name__ == "__main__":
    main()
