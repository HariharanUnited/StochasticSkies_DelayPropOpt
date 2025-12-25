"""
Standalone script to build and pickle the pgmpy model from existing CPTs.
Usage: python -m dptbn.build_pickle --name v2
"""
import argparse
import time
import pickle
import json
from pathlib import Path
from dptbn.bn_pgmpy import build_model

def main():
    parser = argparse.ArgumentParser(description="Force build pgmpy pickle.")
    parser.add_argument("--name", required=True, help="Network name")
    args = parser.parse_args()

    data_dir = Path("data")
    json_path = data_dir / f"{args.name}_cpts.json"
    pkl_path = data_dir / f"{args.name}_model.pkl"

    if not json_path.exists():
        print(f"Error: {json_path} does not exist. Run cpts_creator first.")
        return

    print(f"Reading {json_path}...")
    start_time = time.time()
    
    # 1. Build Model (Fast now!)
    model = build_model(str(json_path))
    build_time = time.time()
    print(f"Model built in {build_time - start_time:.2f} seconds.")

    # 2. Pickle (Fast now!)
    # 2. Pickle (FAST - Re-enabled)
    print(f"Pickling to {pkl_path}...")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    
    total_time = time.time() - start_time
    print(f"DONE. Total time: {total_time:.2f} seconds.")
    print("Inference will now be instant.")

    # 3. Export to BIF (Optional per user request - Disabled for Speed)
    # bif_path = data_dir / f"{args.name}_model.bif"
    # print(f"Exporting to BIF format: {bif_path}...")
    # try:
    #     from pgmpy.readwrite import BIFWriter
    #     writer = BIFWriter(model)
    #     writer.write_bif(str(bif_path))
    #     print("BIF Export successful.")
    # except Exception as e:
    #     print(f"BIF Export failed: {e}")

if __name__ == "__main__":
    main()
