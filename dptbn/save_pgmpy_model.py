"""
Build pgmpy model from CPTs and cache to a pickle for reuse.

Usage:
    python save_pgmpy_model.py --cpts phase7_cpts.json --output cached_model.pkl
"""
import argparse
import pickle

from dptbn.bn_pgmpy import build_model


def main():
    parser = argparse.ArgumentParser(description="Build and cache pgmpy model.")
    parser.add_argument("--cpts", default="phase7_cpts.json", help="CPT JSON file")
    parser.add_argument("--output", default="cached_model.pkl", help="Output pickle path")
    args = parser.parse_args()

    infer = build_model(args.cpts)
    with open(args.output, "wb") as f:
        pickle.dump(infer, f)
    print(f"Saved pgmpy VariableElimination to {args.output}")


if __name__ == "__main__":
    main()
