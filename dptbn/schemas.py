"""
Unified API Schemas for DPTBN.
Matches tools_spec.md v2.0
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

# --- core parts ---

class FlightMetrics(BaseModel):
    DMc: float = 0.0
    DMe: float = 0.0
    DMx: float = 0.0
    pax_risk: float = 0.0

class DMStats(BaseModel):
    dmc: float
    dme: float
    dmx: float

class TopKItem(BaseModel):
    id: str
    val: float

class Scorecard(BaseModel):
    robustness_score: float
    top_k_dmx: List[TopKItem]
    top_k_dme: List[TopKItem]
    top_k_dmc: List[TopKItem]
    risky_hubs: List[str]
    fragile_tails: List[str]

class FlightState(BaseModel):
    prob_delay: float = Field(..., description="P(Delay > 15m)")
    expected_delay: float = Field(..., description="E[D] in minutes")
    most_likely_bin: int
    most_likely_bin_prob: float = 0.0
    is_affected: bool = False
    cause_breakdown: Optional[Dict[str, float]] = None 
    metrics: FlightMetrics = Field(default_factory=FlightMetrics)
    dm_metrics: Optional[DMStats] = None # Added for Tool 3

class EdgeStyle(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    color: str = "gray"
    thickness: int = 1
    tooltip: Optional[str] = None
    type: Optional[str] = None

class NodeStyle(BaseModel):
    color: str = "white"
    border: Optional[str] = "black"
    tooltip: Optional[str] = None

class SummaryPanel(BaseModel):
    title: str = "Analysis"
    stats: List[Dict[str, Union[str, float, int]]] = []

class Visuals(BaseModel):
    edges: List[EdgeStyle] = []
    nodes: Dict[str, NodeStyle] = {}
    heatmap_mode: str = "prob_delay" 
    summary_panel: Optional[SummaryPanel] = None

class MetaInfo(BaseModel):
    tool_exec: str
    calc_time_ms: float
    model_name: str

# --- The Global Response ---

class GlobalResponse(BaseModel):
    meta_info: MetaInfo
    network_state: Dict[str, FlightState]
    visuals: Visuals
    scorecard: Optional[Scorecard] = None
