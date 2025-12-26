
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
import uvicorn
import os
import time
from pathlib import Path
from dptbn.bn_pgmpy import query_posterior, build_model
from dptbn.config import default_config
from dptbn.schemas import (
    GlobalResponse, MetaInfo, FlightState, Visuals, 
    EdgeStyle, NodeStyle, SummaryPanel, FlightMetrics
)
from dptbn.propagation_tool import run_propagation
from dptbn.diagnostics_tool import run_diagnostics
from dptbn.extra_tools import run_multipliers, run_weak_links, run_factors, run_intervention
from dptbn.DM_tool import run_dm_calculations
from dptbn.hub_disruptor_tool import run_hub_disruptor
from dptbn.lateness_contributor_tool import run_lateness_contributor
from dptbn.connection_risk_tool import run_connection_risk
from dptbn.hub_turnaround_tool import run_hub_turnaround

app = FastAPI(title="DPTBN Inference Server v2")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
STATUS = {"models": [], "active": False}
ENGINES: Dict[str, Any] = {}
MODELS: Dict[str, Any] = {} 
DATA_DIR: Optional[Path] = None

@app.on_event("startup")
def load_models():
    global ENGINES, MODELS, STATUS, DATA_DIR
    DATA_DIR = Path(__file__).parent.parent / "bn-viz" / "public" / "data"
    if not DATA_DIR.exists():
        print("CRITICAL WARNING: 'bn-viz/public/data' directory not found.")
        return

    # Auto-discover models: any *_model.pkl or *_cpts.json
    discovered = set()
    for pkl in DATA_DIR.glob("*_model.pkl"):
        discovered.add(pkl.stem.replace("_model", ""))
    for cpt in DATA_DIR.glob("*_cpts.json"):
        discovered.add(cpt.stem.replace("_cpts", ""))

    model_names = sorted(discovered)
    print(f"Loading Models: {model_names} ...")

    for m in model_names:
        # Try loading Pickle first (Fastest)
        pkl_path = DATA_DIR / f"{m}_model.pkl"
        cpts_path = DATA_DIR / f"{m}_cpts.json"
        
        try:
            if pkl_path.exists():
                print(f"  Loading {m} from pickle...")
                import pickle
                with open(pkl_path, "rb") as f:
                    model = pickle.load(f)
            elif cpts_path.exists():
                print(f"  Building {m} from JSON (Slow)...")
                model = build_model(str(cpts_path))
            else:
                print(f"  Model {m} not found in {DATA_DIR}")
                continue

            # Store Engine
            from pgmpy.inference import VariableElimination
            infer = VariableElimination(model)
            # Monkey Patch g_states if missing
            if not hasattr(model, "g_states"):
                model.g_states = [] # Should be loaded
            
            ENGINES[m] = infer
            MODELS[m] = model
            STATUS["models"].append(m)
            
        except Exception as e:
            print(f"  Failed to load {m}: {e}")

    STATUS["active"] = True
    print(f"Server Ready. Models: {STATUS['models']}")


# --- Endpoints ---

from pydantic import BaseModel
from typing import Union
class InferenceRequest(BaseModel):
    model: str
    evidence: Dict[str, Union[int, str]] = {}
    target: List[str] = [] # If empty, query ALL "t" nodes (subject to BFS limit)
    shock_mode: str = "do"   # "do" or "observe"
    hop_limit: int = 5
    top_k_paths: int = 10
    hub: Optional[str] = None
    hub_code: Optional[str] = None
    timebank_start: Optional[int] = None
    timebank_end: Optional[int] = None
    timebank_label: Optional[str] = None


class HubTurnaroundRequest(BaseModel):
    model: str
    hub: str
    pct: float


@app.post("/tools/hub_turnaround")
def tool_hub_turnaround(req: HubTurnaroundRequest):
    try:
        result = run_hub_turnaround(req.model, req.hub, req.pct)
        new_model = result.get("new_model")
        if new_model and new_model not in STATUS["models"]:
            STATUS["models"].append(new_model)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hub turnaround failed: {e}")

class ModelWrapper:
    def __init__(self, model):
        # Unwrap if already wrapped or just raw
        if hasattr(model, "model"):
            self.model = model.model
        else:
            self.model = model

def get_model(name: str):
    if name not in MODELS:
        return None
    return ModelWrapper(MODELS[name])

@app.post("/infer", response_model=GlobalResponse)
def run_tool_1_inference(req: InferenceRequest):
    print(f"\n📨 Inference Request: {req.dict()}")
    # 0. Load Model
    wrapper = get_model(req.model)
    if not wrapper:
        raise HTTPException(status_code=404, detail=f"Model {req.model} not found")

    # Get Engine
    infer_engine = ENGINES.get(req.model)
    if not infer_engine:
        raise HTTPException(status_code=500, detail="Inference engine not ready")
        
    return run_propagation(req, wrapper, infer_engine)

@app.post("/infer_diagnostics", response_model=GlobalResponse)
def run_tool_2_inference(req: InferenceRequest):
    print(f"\n📨 Diagnostic Request: {req.dict()}")
    wrapper = get_model(req.model)
    if not wrapper: raise HTTPException(404, "Model not found")
    infer = ENGINES.get(req.model)
    if not infer: raise HTTPException(500, "Engine not ready")
    return run_diagnostics(req, wrapper, infer)

# --- Visualization & Data Endpoints ---

from fastapi.responses import FileResponse, JSONResponse

@app.get("/visualizations")
def get_visualizations():
    """List available models (visualizations)"""
    return {"visualizations": STATUS["models"]}

@app.get("/visualizations/{model_id}/network")
def get_network_json(model_id: str):
    """Serve specific model network layout"""
    if not DATA_DIR: raise HTTPException(500, "Data Dir not found")
    
    # Prefer Optimized Layout if exists
    path_opt = DATA_DIR / f"{model_id}_network_opt.json"
    if path_opt.exists():
        return FileResponse(path_opt)
        
    path = DATA_DIR / f"{model_id}_network.json"
    if not path.exists():
        raise HTTPException(404, f"Network file for {model_id} not found")
    return FileResponse(path)

@app.get("/visualizations/{model_id}/cpts")
def get_cpts_json(model_id: str):
    """Serve specific model CPTs"""
    if not DATA_DIR: raise HTTPException(500, "Data Dir not found")
    path = DATA_DIR / f"{model_id}_cpts.json"
    if not path.exists():
        raise HTTPException(404, f"CPT file for {model_id} not found")
    return FileResponse(path)

@app.post("/tools/dm_analysis", response_model=GlobalResponse)
def run_tool_3_analysis(req: InferenceRequest):
    wrapper = get_model(req.model)
    if not wrapper:
        raise HTTPException(status_code=404, detail=f"Model {req.model} not found")
        
    # Infer engine is needed for Baseline and DMe/DMx queries
    engine = ENGINES.get(req.model)
    if not engine:
        raise HTTPException(status_code=500, detail="Inference engine not ready")
        
    return run_dm_calculations(req, wrapper, engine)

# New Tools
@app.post("/tools/hub_disruptor", response_model=GlobalResponse)
def tool_hub_disruptor(req: InferenceRequest):
    wrapper = get_model(req.model)
    if not wrapper:
        raise HTTPException(status_code=404, detail=f"Model {req.model} not found")
    infer = ENGINES.get(req.model)
    if not infer:
        raise HTTPException(status_code=500, detail="Inference engine not ready")
    return run_hub_disruptor(req, wrapper, infer)

@app.post("/tools/lateness_contributor", response_model=GlobalResponse)
def tool_lateness_contributor(req: InferenceRequest):
    wrapper = get_model(req.model)
    if not wrapper:
        raise HTTPException(status_code=404, detail=f"Model {req.model} not found")
    infer = ENGINES.get(req.model)
    if not infer:
        raise HTTPException(status_code=500, detail="Inference engine not ready")
    return run_lateness_contributor(req, wrapper, infer)

@app.post("/tools/connection_risk", response_model=GlobalResponse)
def tool_connection_risk(req: InferenceRequest):
    wrapper = get_model(req.model)
    if not wrapper:
        raise HTTPException(status_code=404, detail=f"Model {req.model} not found")
    infer = ENGINES.get(req.model)
    if not infer:
        raise HTTPException(status_code=500, detail="Inference engine not ready")
    return run_connection_risk(req, wrapper, infer)

# Extra Tools (Legacy / Supplemental)
@app.post("/tools/multipliers", response_model=GlobalResponse)
def tool_multipliers(req: InferenceRequest):
    wrapper = get_model(req.model)
    if not wrapper: raise HTTPException(404, "Model not found")
    infer = ENGINES.get(req.model)
    if not infer: raise HTTPException(500, "Engine not ready")
    return run_multipliers(req, wrapper.model, infer)

@app.post("/tools/weak_links", response_model=GlobalResponse)
def tool_weak_links(req: InferenceRequest):
    wrapper = get_model(req.model)
    if not wrapper: raise HTTPException(404, "Model not found")
    infer = ENGINES.get(req.model)
    if not infer: raise HTTPException(500, "Engine not ready")
    return run_weak_links(req, wrapper.model, infer)

@app.post("/tools/factors", response_model=GlobalResponse)
def tool_factors(req: InferenceRequest):
    wrapper = get_model(req.model)
    if not wrapper: raise HTTPException(404, "Model not found")
    infer = ENGINES.get(req.model)
    if not infer: raise HTTPException(500, "Engine not ready")
    return run_factors(req, wrapper.model, infer)

@app.post("/tools/intervention", response_model=GlobalResponse)
def tool_intervention(req: InferenceRequest):
    wrapper = get_model(req.model)
    if not wrapper: raise HTTPException(404, "Model not found")
    infer = ENGINES.get(req.model)
    if not infer: raise HTTPException(500, "Engine not ready")
    return run_intervention(req, wrapper.model, infer)
