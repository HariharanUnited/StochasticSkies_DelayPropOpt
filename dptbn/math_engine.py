"""
Math Engine for DPTBN Tools.
Implements the 'Physics' defined in tools_spec.md (Eq 2, 3, 4).
"""
from typing import Dict, List, Optional
import math

def get_bin_midpoints(bins: List[int]) -> List[float]:
    """
    Convert bin edges [0, 15, 30, 60] into midpoints for Expected Value calc.
    Strategy:
      Bin 0 (<0): Assume -5 min (On Time / Early)
      Bin 1 (0-15): 7.5 min
      Bin 2 (15-30): 22.5 min
      Bin 3 (30-60): 45 min
      Bin 4 (60+): Assume 90 min (Long Delay tail)
    """
    mids = []
    # Handle implicit first bin (< bins[0])
    mids.append(-5.0) 
    
    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i+1]
        mids.append((start + end) / 2.0)
    
    # Handle last bin (> bins[-1])
    mids.append(bins[-1] + 30.0) # Conservative tail assumption
    return mids

def calculate_expected_delay(marginal_dist: Dict[str, float], bins: List[int]) -> float:
    """
    Calculate E[D] based on discrete marginal distribution.
    Eq (3) component: Sum(Prob_i * Value_i)
    
    Args:
        marginal_dist: Dict mapping bin_index (as str/int) to probability (0.0-1.0)
        bins: The configured delay bins (e.g. [0, 15, 30, 60])
    """
    mids = get_bin_midpoints(bins)
    expected_val = 0.0
    
    # Pgmpy returns state names as integers usually? Or implied index.
    # The marginal_dist keys are likely state indices "0", "1", ...
    # We should handle both int and str keys.
    
    # Normalize just in case
    total_p = sum(marginal_dist.values())
    
    for state_key, prob in marginal_dist.items():
        idx = int(state_key)
        if 0 <= idx < len(mids):
            val = mids[idx]
            expected_val += val * (prob / total_p if total_p > 0 else 0)
            
    return round(expected_val, 2)

def calculate_prob_delay(marginal_dist: Dict[str, float], threshold_bin_idx: int = 3) -> float:
    """
    Calculate P(Delay > Threshold).
    Default threshold_bin_idx=3 corresponds to >30mins (Bin 3+).
    """
    prob_sum = 0.0
    for state_key, prob in marginal_dist.items():
        idx = int(state_key)
        if idx >= threshold_bin_idx:
            prob_sum += prob
    return round(prob_sum, 4)

def calculate_dmx(dme_shock: float, dme_baseline: float, root_delay_duration: float) -> float:
    """
    Equation 4 (Exact): DMx = (Dr + Sum(E[D'] - E[D])) / Dr
    Measures the 'Multiplier' effect: Ratio of Total Induced Delay to Root Delay.
    
    Args:
        dme_shock: Total Expected Delay of the network (with shock).
        dme_baseline: Total Expected Delay of the network (baseline).
        root_delay_duration (Dr): The size of the impulse shock in minutes.
        
    Returns:
        float: The multiplier (e.g. 1.5 means 1 min of root delay creates 1.5 mins of total network delay).
    """
    if root_delay_duration <= 0:
        return 0.0 # Avoid division by zero
        
    net_diff = dme_shock - dme_baseline
    
    # Formula: (Dr + Diff) / Dr
    # = 1 + (Diff / Dr)
    val = (root_delay_duration + net_diff) / root_delay_duration
    return round(val, 2)
