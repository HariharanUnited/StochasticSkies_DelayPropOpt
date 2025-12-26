"""
Lightweight server dedicated to swap exploration.
Separates concerns from the main inference server.
Endpoints:
  - GET /visualizations              -> list available models
  - GET /visualizations/{model}/network -> serve network_opt if present, else network
  - GET /visualizations/{model}/cpts -> serve CPTs
  - GET /swap_options/{model}?role=aircraft|pilot|crew&min_ct=25&horizon=180&limit=50
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import json
import shutil
import pickle
import traceback

from dptbn.swap_generator import (
    list_swaps,
    run_swaps_and_materialize,
    HOP_LIMIT_DEFAULT,
    score_swaps_fast,
    SwapOption,
    reroute_network_and_model,
    MIN_CT_DEFAULT,
    HORIZON_DEFAULT,
    update_cpts_after_network_change,
)

app = FastAPI(title="DPTBN Swap Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR: Optional[Path] = None
STATUS: Dict[str, Any] = {"models": [], "ready": False}
SWAP_OUTPUT_DIR: Optional[Path] = None


def _discover_models() -> List[str]:
    models = set()
    roots = [DATA_DIR]
    for root in roots:
        if not root:
            continue
        for f in root.glob("*_network.json"):
            models.add(f.name.replace("_network.json", ""))
        for f in root.glob("*_network_opt.json"):
            models.add(f.name.replace("_network_opt.json", ""))
    return sorted(models)


def _refresh_models():
    models = set()
    for m in _discover_models():
        models.add(m)
    STATUS["models"] = sorted(models)
    STATUS["ready"] = True
    print(f"Swap server ready. Models: {STATUS['models']}")


@app.on_event("startup")
def locate_data():
    global DATA_DIR, STATUS, SWAP_OUTPUT_DIR
    DATA_DIR = Path(__file__).parent.parent / "bn-viz" / "public" / "data"
    if not DATA_DIR.exists():
        print("CRITICAL: data directory not found.")
        return

    SWAP_OUTPUT_DIR = DATA_DIR / "swap_runs"
    SWAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _refresh_models()


@app.get("/visualizations")
def list_models():
    return {"visualizations": STATUS.get("models", [])}


def _resolve_network(model_id: str) -> Path:
    if not DATA_DIR:
        raise HTTPException(500, "Data dir not set")
    for path in [DATA_DIR / f"{model_id}_network_opt.json", DATA_DIR / f"{model_id}_network.json"]:
        if path.exists():
            return path
    raise HTTPException(404, f"Network file for {model_id} not found")


@app.get("/visualizations/{model_id}/network")
def get_network(model_id: str):
    path = _resolve_network(model_id)
    return FileResponse(path)


@app.get("/visualizations/{model_id}/cpts")
def get_cpts(model_id: str):
    if not DATA_DIR:
        raise HTTPException(500, "Data dir not set")
    path = DATA_DIR / f"{model_id}_cpts.json"
    if path.exists():
        return FileResponse(path)
    raise HTTPException(404, f"CPT file for {model_id} not found")


@app.get("/swap_options/{model_id}")
def get_swap_options(
    model_id: str,
    role: str = Query("aircraft", regex="^(aircraft|pilot|crew)$"),
    min_ct: int = 25,
    horizon: int = 180,
    limit: Optional[int] = 50,
):
    try:
        swaps = list_swaps(model_id, role=role, min_ct=min_ct, horizon=horizon, limit=limit)
        # convert dataclasses to dict
        payload = [s.__dict__ for s in swaps]
        return {"model": model_id, "role": role, "swaps": payload}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to list swaps: {e}")


def _metadata_path(model_id: str) -> Optional[Path]:
    if not SWAP_OUTPUT_DIR:
        return None
    return SWAP_OUTPUT_DIR / f"{model_id}_runs.json"


def _load_metadata(model_id: str) -> Dict[str, Any]:
    path = _metadata_path(model_id)
    if path and path.exists():
        return json.loads(path.read_text())
    return {"runs": []}


def _store_metadata(model_id: str, runs: List[Dict[str, Any]]):
    path = _metadata_path(model_id)
    if not path:
        return
    meta = _load_metadata(model_id)
    meta.setdefault("runs", [])
    meta["runs"].extend(runs)
    path.write_text(json.dumps(meta, indent=2))


from pydantic import BaseModel


class SwapRunRequest(BaseModel):
    role: str = "aircraft"
    swaps: List[Dict[str, Any]]
    hop_limit: int = HOP_LIMIT_DEFAULT


class FinalizeRequest(BaseModel):
    swap: Dict[str, Any]
    role: str = "aircraft"



@app.post("/swap_runs/{model_id}")
def run_swap_variants(model_id: str, payload: SwapRunRequest):
    if not STATUS.get("ready"):
        raise HTTPException(503, "Server not ready")
    if not SWAP_OUTPUT_DIR:
        raise HTTPException(500, "Swap output directory not configured")

    try:
        # results = run_swaps_and_materialize(
        #     model_id,
        #     payload.swaps,
        #     role=payload.role,
        #     out_dir=SWAP_OUTPUT_DIR,
        #     hop_limit=payload.hop_limit or HOP_LIMIT_DEFAULT,
        # )
        # _store_metadata(model_id, results)
        # _refresh_models()
        # return {"model": model_id, "role": payload.role, "results": results}
        results = score_swaps_fast(
            model_id,
            payload.swaps,
            role=payload.role,
            hop_limit=payload.hop_limit or HOP_LIMIT_DEFAULT,
        )
        return {"model": model_id, "role": payload.role, "results": results}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to run swaps: {e}")


@app.get("/swap_runs/{model_id}/metadata")
def list_swap_runs(model_id: str):
    meta = _load_metadata(model_id)
    return meta


# @app.post("/swap_runs/{model_id}/finalize")
# def finalize_swap_variant(model_id: str, payload: FinalizeRequest):
#     if not DATA_DIR:
#         raise HTTPException(500, "Server not ready")

#     # Compute next name: base_roleprefixN
#     role_prefix = {"aircraft": "act", "pilot": "pil", "crew": "cre"}.get(payload.role, payload.role[:3])
#     existing = list(DATA_DIR.glob(f"{model_id}_{role_prefix}*_model.pkl"))
#     max_idx = 0
#     for p in existing:
#         stem = p.name.replace(f"{model_id}_{role_prefix}", "").replace("_model.pkl", "")
#         try:
#             max_idx = max(max_idx, int(stem))
#         except Exception:
#             continue
#     next_idx = max_idx + 1
#     target_name = f"{model_id}_{role_prefix}{next_idx}"

#     # Load baseline artifacts (data dir only)
#     net_path = DATA_DIR / f"{model_id}_network_opt.json"
#     if not net_path.exists():
#         net_path = DATA_DIR / f"{model_id}_network.json"
#     model_path = DATA_DIR / f"{model_id}_model.pkl"
#     if not net_path.exists() or not model_path.exists():
#         raise HTTPException(404, "Baseline network/model not found")

#     try:
#         base_network = json.loads(net_path.read_text())
#         with open(model_path, "rb") as f:
#             base_model = pickle.load(f)
#     except Exception as e:
#         raise HTTPException(500, f"Failed to load baseline: {e}")

#     try:
#         swap = SwapOption(**payload.swap, role=payload.role)
#         new_net, new_model = reroute_network_and_model(base_network, base_model, swap, payload.role)
#     except Exception as e:
#         raise HTTPException(500, f"Failed to apply swap: {e}")

#     dest_net = DATA_DIR / f"{target_name}_network_opt.json"
#     dest_model = DATA_DIR / f"{target_name}_model.pkl"

#     dest_net.write_text(json.dumps(new_net, indent=2))
#     with open(dest_model, "wb") as f:
#         pickle.dump(new_model, f)

#     _refresh_models()
#     return {"finalized_as": target_name, "network": str(dest_net), "model": str(dest_model)}

from fastapi import HTTPException
import traceback
import json, pickle
from pathlib import Path

@app.post("/swap_runs/{model_id}/finalize")
def finalize_swap_variant(model_id: str, payload: FinalizeRequest):
    if not DATA_DIR:
        raise HTTPException(500, "Server not ready")

    role = payload.role
    role_prefix = {"aircraft": "a", "pilot": "p", "crew": "c"}.get(role, role[:3])

    existing = list(DATA_DIR.glob(f"{model_id}_{role_prefix}*_model.pkl"))
    max_idx = 0
    for p in existing:
        stem = p.name.replace(f"{model_id}_{role_prefix}", "").replace("_model.pkl", "")
        try:
            max_idx = max(max_idx, int(stem))
        except Exception:
            continue
    target_name = f"{model_id}_{role_prefix}{max_idx + 1}"

    # Load baseline
    net_path = DATA_DIR / f"{model_id}_network_opt.json"
    if not net_path.exists():
        net_path = DATA_DIR / f"{model_id}_network.json"

    model_path = DATA_DIR / f"{model_id}_model.pkl"
    if not net_path.exists() or not model_path.exists():
        raise HTTPException(404, "Baseline network/model not found")

    try:
        base_network = json.loads(net_path.read_text())
        with open(model_path, "rb") as f:
            base_model = pickle.load(f)

        # build SwapOption safely (avoid duplicate role)
        swap_dict = dict(payload.swap)
        swap_dict.pop("role", None)
        swap = SwapOption(**swap_dict, role=role)

        new_net, new_model = reroute_network_and_model(base_network, base_model, swap, role)
        cpts_path = DATA_DIR / f"{model_id}_cpts.json"
        if not cpts_path.exists():
            raise HTTPException(404, "Baseline CPTs not found")

        base_cpts = json.loads(cpts_path.read_text())

        # Update CPTs to match the new network inbound parents
        new_cpts, n_changed = update_cpts_after_network_change(
            base_cpts=base_cpts,
            base_network=base_network,
            new_network=new_net,
            role=payload.role,
            strict=False,   # set True if you want hard failures when a table is missing
        )

        # Save alongside network/model
        dest_cpts = DATA_DIR / f"{target_name}_cpts.json"
        dest_cpts.write_text(json.dumps(new_cpts, indent=2))

        dest_net = DATA_DIR / f"{target_name}_network_opt.json"
        dest_model = DATA_DIR / f"{target_name}_model.pkl"

        dest_net.write_text(json.dumps(new_net, indent=2))
        with open(dest_model, "wb") as f:
            pickle.dump(new_model, f)

    except Exception as e:
        traceback.print_exc()  # <-- log real stack trace
        raise HTTPException(500, f"Failed to finalize: {e}")

    _refresh_models()
    return {
        "finalized_as": target_name,
        "network": str(dest_net),
        "model": str(dest_model),
        "cpts": str(dest_cpts),
        "cpts_tables_retargeted": n_changed,
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dptbn.swap_server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8001)), reload=True)
