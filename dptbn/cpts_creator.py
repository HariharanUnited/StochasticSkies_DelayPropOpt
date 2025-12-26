"""
Phase 7: Build CPTs from simulated Phase 3 data with inter-flight structure.

Usage:
    python run_phase7_cpts.py --network phase2_network_opt.json --data phase3_delay_data.json --output phase7_cpts.json
"""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from dptbn.discretise_and_cpts import compute_cpts, discretize_value
from dptbn.network import load_network_json
from dptbn.config import default_config
from dptbn.bn_pgmpy import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Build CPTs from Phase 3 simulation data.")
    parser.add_argument("--name", required=True, help="Network name (e.g. 'toy_v1'). Reads data/{name}_network_opt.json, data/{name}_delay_data.json -> Writes data/{name}_cpts.json")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Laplace smoothing")
    parser.add_argument(
        "--completion",
        choices=["observed", "full"],
        default="observed",
        help="observed: combos seen in data (plus g); full: all bin states x g",
    )
    parser.add_argument(
        "--fill-missing",
        choices=["prior", "none"],
        default="prior",
        help="prior: add prior rows for unseen parent configs; none: only observed combos (fill later in pgmpy)",
    )
    parser.add_argument(
        "--resource-spread-std",
        type=float,
        default=5.0,
        help="Std dev (mins) for probabilistic spread around max(0, delay-slack) when mapping resource CPDs.",
    )
    return parser.parse_args()


class ProgressReader:
    def __init__(self, f):
        self.f = f
        self.f.seek(0, 2)
        self.total = self.f.tell()
        self.f.seek(0)
        self.read_bytes = 0
        self.last_percent = -1

    def read(self, n=-1):
        data = self.f.read(n)
        self.read_bytes += len(data)
        if self.total > 0:
            percent = int(self.read_bytes / self.total * 100)
            if percent > self.last_percent:
                 print(f"Loading Pickle: {percent}% ...", end="\r")
                 self.last_percent = percent
        return data

    def readline(self):
        return self.f.readline()

def _flatten_days_chunk(args):
    """Helper to process a chunk of days into flat records."""
    start_idx, days, flight_ids, pax_map, bins = args
    records = []
    
    # Pre-compute discretize if possible or keep inline? Inline is fine.
    # Note: bins must be passed or imported. args is safer.
    
    for day_offset, day in enumerate(days):
        day_idx = start_idx + day_offset
        for fid in flight_ids:
            vals = day[fid]
            # Core record
            rec = {
                "day": day_idx,
                "flight_id": fid,
                # "t": vals["t"], # Optimization: Don't store floats if we only need bins? 
                # Actually we DO need 't' maybe for resource mapping join?
                # looking at learn_mapping: df_child.merge... on 'day'. 
                # wait, learn_mapping uses "t_bin". resource uses "k_bin" etc.
                # Do we need "t"? check compute_cpts. it uses 't_bin'.
                # But 'learn_mapping' takes child_col.
                # Let's keep 't' just in case or reduce memory if possible.
                # For safety, keep distinct code structure.
                "g": vals.get("g", "None"),
                "t_bin": discretize_value(vals["t"], bins)
            }
            
            # Resources
            for key in ["k", "k_eff", "q", "c", "p", "r", "b"]:
                val = vals.get(key, 0.0)
                # rec[key] = val # Optimization: Drop float columns if unused to save RAM
                rec[f"{key}_bin"] = discretize_value(val, bins)
                
            inbound_pax = pax_map.get(fid, [])
            # rec["pax_parents"] = inbound_pax # Optimization: Don't store list in every row! 
            # Pax parents are static per flight_id. We can join them later or just know them.
            # But line 99 uses pax_map.get(fid) so we don't need it in DF.
            # Line 104 uses it.
            # Line 113 uses it.
            # We DONT need it in the record.
            
            p_conns = vals.get("p_conns", {})
            for pfid in inbound_pax:
                pval = p_conns.get(pfid, 0.0)
                rec[f"p_{pfid}_bin"] = discretize_value(pval, bins)
            records.append(rec)
    return records

def _create_df_worker(args):
    """Worker to convert records to DataFrame."""
    fid, recs = args
    return fid, pd.DataFrame.from_records(recs)

def _build_cpt_worker_v2(args):
    """Worker to build CPTs and Resource CPDs for a single flight."""
    fid, df_f, parent_dfs, resource_map, pax_parents, parents, smoothing, completion, fill_missing, num_bin_states, g_states = args
    
    # 1. Compute Main CPT
    cpt_f = compute_cpts(
        df_f,
        target="t_bin",
        parents=parents,
        smoothing=smoothing,
        complete=(fill_missing == "prior"),
        complete_mode=completion,
        num_bin_states=num_bin_states,
    )
    cpt_data = cpt_f.get(fid, {})

    # 2. Compute Resource CPDs
    res = {"k": {}, "q": {}, "c": {}, "pax": {}, "pax_parents": pax_parents, "parents_order": parents}
    
    def learn_mapping_local(parent_fid, child_col):
        if parent_fid is None: return None
        df_p = parent_dfs.get(parent_fid)
        if df_p is None: return None
        
        child_subset = df_f[["day", child_col]]
        parent_subset = df_p[["day", "t_bin"]]
        
        df_join = child_subset.merge(parent_subset, on="day", how="inner", suffixes=("", "_parent")).dropna()
        
        # Pre-calc Child Prior to avoid Uniform noise
        child_counts = df_f[child_col].value_counts().to_dict()
        prior_total = sum(child_counts.values())
        if prior_total > 0:
             child_prior = {
                 b: (child_counts.get(b, 0) + smoothing) / (prior_total + smoothing * num_bin_states) 
                 for b in range(num_bin_states)
             }
        else:
             child_prior = {b: 1.0 / num_bin_states for b in range(num_bin_states)}

        mapping = {}
        for pbin in range(num_bin_states):
            sub = df_join[df_join["t_bin"] == pbin]
            counts = sub[child_col].value_counts().to_dict()
            total = sum(counts.values())
            if total == 0:
                # Physics Fallback: Use Identity Distribution (Delay Propagates)
                # If we haven't seen this bin, assume child = parent (plus smoothing)
                # This is critical for shock propagation.
                dist = {}
                target = int(pbin)
                remaining_mass = 0.2
                peak_mass = 0.8
                
                # Simple smoothing
                for b in range(num_bin_states):
                    if b == target:
                        dist[b] = peak_mass
                    elif abs(b - target) == 1:
                        dist[b] = remaining_mass / 2 # simplified
                    else:
                        dist[b] = 0.0 # sparse
                        
                # Normalize just in case
                s = sum(dist.values())
                mapping[pbin] = {b: v/s for b, v in dist.items()} if s>0 else child_prior
            else:
                mapping[pbin] = {
                    b: (counts.get(b, 0) + smoothing) / (total + smoothing * num_bin_states)
                    for b in range(num_bin_states)
                }
        return mapping

    if resource_map.get("k"): res["k"][resource_map["k"]] = learn_mapping_local(resource_map["k"], "k_bin")
    if resource_map.get("q"): res["q"][resource_map["q"]] = learn_mapping_local(resource_map["q"], "q_bin")
    if resource_map.get("c"): res["c"][resource_map["c"]] = learn_mapping_local(resource_map["c"], "c_bin")
    
    for pfid in pax_parents:
        res["pax"][pfid] = learn_mapping_local(pfid, f"p_{pfid}_bin")

    # 3. Learn g prior
    counts_g = df_f["g"].value_counts().to_dict()
    total_g = sum(counts_g.values())
    g_prior = {
        g: (counts_g.get(g, 0) + smoothing) / (total_g + smoothing * max(1, len(g_states)))
        for g in g_states
    }

    return fid, cpt_data, res, g_prior

def main():
    args = parse_args()
    
    # ... (Paths block) ...
    data_dir = Path("data")
    network_path = data_dir / f"{args.name}_network_opt.json"
    
    if not network_path.exists():
        fallback = data_dir / f"{args.name}_network.json"
        if fallback.exists():
            print(f"Warning: {network_path} not found, falling back to {fallback}")
            network_path = fallback
            
    # Phase 3 data is now Pickled for speed
    delay_data_path = data_dir / f"{args.name}_delay_data.pkl"
    output_path = data_dir / f"{args.name}_cpts.json"
    pkl_path = data_dir / f"{args.name}_model.pkl"

    net = load_network_json(network_path)
    design = net.design
    bins = default_config.delay_bins
    num_bin_states = len(bins) + 1

    print(f"Loading {delay_data_path} (Pickle)...")
    import pickle
    with open(delay_data_path, "rb") as f:
        reader = ProgressReader(f)
        data = pickle.load(reader)
        print("Loading Pickle: 100% DONE.      ")
        
    days_continuous = data["days_continuous"]
    flight_ids = [f.flight_id for f in net.flights]
    pax_map = net.passenger_connections  
    by_id = {f.flight_id: f for f in net.flights}

    # --- 1. Parallel Data Flattening --
    import concurrent.futures
    import math
    from collections import defaultdict
    
    total_days = len(days_continuous)
    chunk_size = 500 # Process 500 days per worker
    num_chunks = math.ceil(total_days / chunk_size)
    
    print(f"Flattening data for {total_days} days ({num_chunks} chunks)...")
    
    tasks = []
    for i in range(0, total_days, chunk_size):
        chunk = days_continuous[i : i + chunk_size]
        tasks.append((i, chunk, flight_ids, pax_map, bins))
        
    flight_records = defaultdict(list)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(_flatten_days_chunk, t): i for i, t in enumerate(tasks)}
        
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            # Distribute into buckets immediately
            for r in res:
                flight_records[r["flight_id"]].append(r)
            
            # Free memory of the result chunk
            del res
            
            completed += 1
            if completed % 5 == 0:
                print(f"  ... Flattened chunk {completed}/{num_chunks} ...", end="\r")
    
    print(f"  Data Flattening Complete. Converting to DataFrames (Parallel)...      ")
    
    # --- 2. Parallel DataFrame Creation ---
    flight_dfs = {}
    
    # Helper list for tasks
    # (fid, recs)
    df_tasks = list(flight_records.items())
    del flight_records # Free raw list memory
    
    print(f"  Converting {len(df_tasks)} flight buckets...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all (tuples are picklable)
        futures = {executor.submit(_create_df_worker, t): t[0] for t in df_tasks}
        
        completed = 0
        total = len(df_tasks)
        for fut in concurrent.futures.as_completed(futures):
            fid, df = fut.result()
            flight_dfs[fid] = df
            completed += 1
            if completed % 50 == 0:
                print(f"  ... Converted flight {completed}/{total} ...", end="\r")
                
    print(f"  DataFrame Conversion Complete.                    ")
    del df_tasks
    del days_continuous
    del data
    
    # Infer g_states
    first_fid = flight_ids[0]
    # FIX: Hardcode the full universe of states to prevent Flight-Specific Domain Errors
    g_states = sorted(['ATC', 'None', 'Ops', 'Other', 'Pax', 'Weather', 'Reactionary']) # Force Reactionary inclusion

    # --- 3. Parallel CPT Construction ---
    cpts = {}
    resource_cpds = {}
    g_priors = {}
    total_flights = len(flight_ids)
    print(f"Building CPTs for {total_flights} flights (Parallel Batched)...")
    
    # Prepare Temp Directory
    temp_dir = data_dir / f"temp_cpts_{args.name}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Serialize Helper
    def serialize_probs(probs):
        return {int(tv): float(p) for tv, p in probs.items()}

    batch_size = 20 # Small batch to prevent queue overflow
    cpt_tasks = []
    
    # Pre-generate task args (lightweight tuples)
    for fid in flight_ids:
        df_f = flight_dfs.get(fid)
        if df_f is None: continue
        
        fobj = by_id[fid]
        pax_parents = pax_map.get(fid, [])
        parents = ["k_bin", "q_bin", "c_bin"] + [f"p_{p}_bin" for p in pax_parents] + ["g"]
        
        # Build Resource Map
        resource_map = {}
        if fobj.inbound_aircraft_flight: resource_map["k"] = fobj.inbound_aircraft_flight
        if fobj.inbound_pilot_flight: resource_map["q"] = fobj.inbound_pilot_flight
        if fobj.inbound_cabin_flight: resource_map["c"] = fobj.inbound_cabin_flight
        
        # Gather Parent DFs (Only what's needed)
        relevant_parent_fids = set()
        if resource_map.get("k"): relevant_parent_fids.add(resource_map["k"])
        if resource_map.get("q"): relevant_parent_fids.add(resource_map["q"])
        if resource_map.get("c"): relevant_parent_fids.add(resource_map["c"])
        for p in pax_parents: relevant_parent_fids.add(p)
        
        parent_dfs_subset = {pfid: flight_dfs[pfid] for pfid in relevant_parent_fids if pfid in flight_dfs}
        
        # Task Args
        task_args = (
            fid, df_f, parent_dfs_subset, resource_map, pax_parents, parents, 
            args.smoothing, args.completion, args.fill_missing, num_bin_states, g_states
        )
        cpt_tasks.append(task_args)

    # Process in batches
    import shutil
    import gc
    
    total_tasks = len(cpt_tasks)
    num_batches = math.ceil(total_tasks / batch_size)
    
    for b_idx in range(num_batches):
        start = b_idx * batch_size
        end = start + batch_size
        batch = cpt_tasks[start:end]
        
        print(f"  ... Batch {b_idx + 1}/{num_batches} (Flights {start}-{min(end, total_tasks)})... ", end="\r")
        
        batch_results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit batch
            futures = {executor.submit(_build_cpt_worker_v2, t): i for i, t in enumerate(batch)}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    batch_results.append(res)
                except Exception as e:
                    print(f"\nError in worker: {e}")
        
        # Flush Batch to Disk immediately
        for fid, cpt_data, res_cpd, g_prior in batch_results:
            # Prepare payload for this flight
            # We save individual files to merge later
            flight_payload = {
                "cpt": {"|".join(map(str, k)): serialize_probs(v) for k, v in cpt_data.items()},
                "resource_cpd": res_cpd,
                "g_prior": g_prior
            }
            # Save
            f_path = temp_dir / f"{fid}.json"
            f_path.write_text(json.dumps(flight_payload))
            
        # Clear memory
        del batch_results
        del batch
        gc.collect()

    print(f"\n  CPT Construction Complete. Merging {total_tasks} files...          ")
    
    # Merge Phase
    for fid in flight_ids:
        f_path = temp_dir / f"{fid}.json"
        if not f_path.exists(): continue
        
        blob = json.loads(f_path.read_text())
        cpts[fid] = blob["cpt"]
        resource_cpds[fid] = blob["resource_cpd"]
        g_priors[fid] = blob["g_prior"]
        
    # Cleanup Temp
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

    payload = {
        "metadata": {
            "delay_bins": bins,
            "g_states": g_states,
            "smoothing": args.smoothing,
            "completion": args.completion,
            "fill_missing": args.fill_missing,
            "design": design.__dict__ if design else None,
            "num_records": 6000000,
        },
        "cpts": cpts,
        "resource_cpds": resource_cpds,
        "g_priors": g_priors,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote CPTs for {len(cpts)} flights to {output_path}")
    # except ImportError:
    #     print("Warning: pgmpy not installed or failed to build model. Pickle skipped.")
    # except Exception as e:
    #     print(f"Error pickling model: {e}")
    print(f"Skipped automatic pickling. Use inspect_model.py to load/build model.")


if __name__ == "__main__":
    main()
