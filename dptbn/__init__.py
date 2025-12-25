"""
DPT-BN package scaffold.

Phase 2 adds synthetic network generation utilities.
"""
import numpy as np

# Monkey-patch for docplex compatibility with newer numpy
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128


from dptbn.config import (
    NetworkDesign,
    SimulationConfig,
    default_config,
    default_network_design,
)
from dptbn.network import (
    Flight,
    SyntheticNetwork,
    generate_synthetic_network,
    load_toy_schedule,
    network_to_dict,
    network_from_dict,
    save_network_json,
    load_network_json,
)
from dptbn import opt_models
from dptbn import bn_pgmpy

__all__ = [
    "config",
    "network",
    "sim_one_day",
    "sim_many_days",
    "discretise_and_cpts",
    "bn_inference",
    "NetworkDesign",
    "SimulationConfig",
    "default_config",
    "default_network_design",
    "Flight",
    "SyntheticNetwork",
    "generate_synthetic_network",
    "load_toy_schedule",
    "network_to_dict",
    "network_from_dict",
    "save_network_json",
    "load_network_json",
    "opt_models",
    "bn_pgmpy",
]
