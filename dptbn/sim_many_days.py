"""
Multi-day simulation harness.
"""
from typing import List, Dict, Optional

from dptbn.config import SimulationConfig
from dptbn.network import SyntheticNetwork
from dptbn.simulator_config import SimulatorConfig, default_simulator_config
import concurrent.futures
from dptbn.sim_one_day import simulate_day

def _sim_worker(args):
    """Worker function for parallel processing."""
    net, sim_config, sim_params, seed = args
    return simulate_day(net, sim_config, sim_params=sim_params, seed=seed)


def simulate_many_days(
    net: SyntheticNetwork,
    sim_config: SimulationConfig,
    num_days: Optional[int] = None,
    sim_params: SimulatorConfig = None,
) -> List[Dict[str, Dict[str, float]]]:
    """
    Run multi-day simulation using multiprocessing.
    """
    days = num_days or sim_config.num_days
    if sim_params is None:
        sim_params = default_simulator_config()

    # Prepare arguments for each day
    # Note: SyntheticNetwork must be picklable.
    tasks = [
        (net, sim_config, sim_params, sim_config.seed + d)
        for d in range(days)
    ]
    
    # Use ProcessPoolExecutor to utilize all CPU cores
    # Chunksize can be tuned, but default is usually okay for 20k items.
    results = []
    print(f"Starting parallel simulation of {days} days...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map preserves order
        results_gen = executor.map(_sim_worker, tasks)
        
        for i, res in enumerate(results_gen):
            results.append(res)
            if (i + 1) % 100 == 0:
                print(f"  ... Day {i + 1}/{days} completed", end="\r")
    print("") # Newline after loop
    
    print(f"Completed {len(results)} days of simulation.")
    return results
