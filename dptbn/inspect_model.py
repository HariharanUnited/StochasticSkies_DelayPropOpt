"""
Load a cached pgmpy model (pickled VariableElimination) and inspect edges/CPDs with clean formatting.

Usage:
    python -m dptbn.inspect_model --name v2 --flight F001
"""
import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="View pgmpy model structure and CPDs.")
    parser.add_argument("--name", required=True, help="Network name (e.g. 'v2'). Loads data/{name}_model.pkl")
    parser.add_argument("--flight", help="Flight ID to show CPD for (e.g., F001)")
    args = parser.parse_args()

    data_dir = Path("data")
    pkl_path = data_dir / f"{args.name}_model.pkl"
    bif_path = data_dir / f"{args.name}_model.bif"

    if pkl_path.exists():
        print(f"Loading model from Pickle: {pkl_path}...")
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        # Handle legacy case where pickle might still be VariableElimination
        if hasattr(model, "model"):
            model = model.model
    elif bif_path.exists():
        print(f"Loading model from BIF: {bif_path}...")
        from pgmpy.readwrite import BIFReader
        reader = BIFReader(str(bif_path))
        model = reader.get_model()
    else:
        json_path = data_dir / f"{args.name}_cpts.json"
        if not json_path.exists():
             print(f"Error: No model found (.bif/.pkl) and source {json_path} missing.")
             return
        
        print(f"No BIF/Pickle found. Building lazily from {json_path}...")
        from dptbn.bn_pgmpy import build_model
        model = build_model(str(json_path))

    print(f"Model loaded. Nodes: {len(model.nodes())}, Edges: {len(model.edges())}")
    
    if args.flight:
        # Renaming ":t_bin" -> "_t" for BIF compatibility
        var = f"{args.flight}_t"
        try:
            cpd = model.get_cpds(var)
            print(f"\nCPD for {var}:")
            print(cpd)
        except ValueError:
            print(f"No CPD found for {var}")
    else:
        print("\nUse --flight [ID] to see specific CPDs. (e.g. F001)")


if __name__ == "__main__":
    main()
