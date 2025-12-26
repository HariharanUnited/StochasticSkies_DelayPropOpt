"""
Simulator-specific knobs for Phase 3 delay propagation.
"""
from dataclasses import dataclass
@dataclass
class SimulatorConfig:
    # Base turnaround delay distribution
    base_delay_mean: float = 15.0
    base_delay_std: float = 30.0
    # Base category probabilities (must sum to 1 over keys below)
    base_cat_probs: dict = None
    # Ground/other independent noise
    ground_mean: float = 7.0
    ground_std: float = 5.0
    # Noise added when propagating inbound connections (mean=0, std)
    connection_noise_std: float = 5.0
    # Noise for orphan connections (no inbound)
    orphan_noise_std: float = 2.0
    # Passenger noise when no inbound pax
    pax_noise_std: float = 2.0
    # Departure noise term (mean=0, std)
    departure_noise_std: float = 10.0
    # Clamp negatives?
    clamp_nonnegative: bool = True
    # If True, add stochastic noise to propagated connections (probabilistic CPT-style)
    probabilistic_cpt: bool = False
    noise_ac_std: float = 10.0
    noise_pilot_std: float = 10.0
    noise_cabin_std: float = 10.0
    noise_pax_std: float = 15.0


def _default_probs():
    return {
        "Ops": 0.4,
        "ATC": 0.1,
        "Weather": 0.1,
        "Pax": 0.1,
        "Other": 0.3,
    }


def default_simulator_config() -> SimulatorConfig:
    return SimulatorConfig(base_cat_probs=_default_probs())
