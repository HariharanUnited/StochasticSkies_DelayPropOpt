"""
Central configuration for the synthetic DPT-BN PoC.

Keep Phase 1 values minimal; adjust in Phase 2 when the toy network is defined.
"""
from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class SimulationConfig:
    """Global knobs for simulation and discretization."""

    # Delay bin edges in minutes.
    # Bin 0: negative / on-time ( <0 )
    # Bin 1: 0–<15
    # Bin 2: 15–<30
    # Bin 3: 30–<60
    # Bin 4: 60+
    delay_bins: Sequence[int] = field(default_factory=lambda: [0, 15, 30, 60])
    # Number of synthetic operating days to simulate when building CPTs.
    num_days: int = 700
    # Random seed to make toy runs reproducible.
    seed: int = 42


default_config = SimulationConfig()


# Canonical list of 100 US airports (IATA codes).
AIRPORT_CODES_USA: List[str] = [
    "ATL",
    "LAX",
    "ORD",
    "DFW",
    "JFK",
    "EWR",
    "BOS",
    "PHL",
    "CLT",
    "MIA",
    "IAD",
    "DCA",
    "BWI",
    "TPA",
    "MCO",
    "FLL",
    "MKE",
    "DTW",
    "MSP",
    "SEA",
    "PDX",
    "SFO",
    "SJC",
    "OAK",
    "LAS",
    "PHX",
    "SAN",
    "SLC",
    "IAH",
    "AUS",
    "SAT",
    "MSY",
    "DEN",
    "ANC",
    "HNL",
    "OGG",
    "KOA",
    "LIH",
    "RDU",
    "BNA",
    "STL",
    "CMH",
    "CLE",
    "PIT",
    "IND",
    "BDL",
    "BUF",
    "SYR",
    "ROC",
    "ALB",
    "ORF",
    "RIC",
    "CHS",
    "SAV",
    "JAX",
    "RSW",
    "PBI",
    "DAY",
    "BOI",
    "GEG",
    "DSM",
    "OMA",
    "OKC",
    "TUL",
    "ICT",
    "MCI",
    "SDF",
    "LEX",
    "HSV",
    "BHM",
    "MEM",
    "LIT",
    "ELP",
    "ABQ",
    "CRP",
    "LRD",
    "AMA",
    "LBB",
    "MAF",
    "SPS",
    "GRR",
    "AZO",
    "LAN",
    "MHT",
    "PVD",
    "HPN",
    "SWF",
    "ISP",
    "MDW",
    "ONT",
    "SNA",
    "LGB",
    "BUR",
    "PSP",
    "SBA",
    "SMF",
    "RNO",
    "FAT",
    "EUG",
    "MFR",
]

HUB_CODES_USA: List[str] = [
    "ATL",
    "ORD",
    "DFW",
    "DEN",
    "LAX",
    "SFO",
    "IAH",
    "JFK",
    "CLT",
    "MIA",
]


@dataclass
class NetworkDesign:
    """Design parameters for the synthetic network (Phase 2)."""

    num_airports: int = 15
    num_hubs: int = 3
    num_flights: int = 300
    num_aircraft: int = 60
    num_pilots: int = 60
    num_crews: int = 60
    min_flight_duration: int = 30
    max_flight_duration: int = 180
    mct_crew: int = 25
    mct_pilot: int = 25
    max_ct_pilot: int = 240
    max_ct_crew: int = 240
    max_ct_aircraft: int = 300
    tat_aircraft: int = 25
    min_ct_pax: int = 30
    max_ct_pax: int = 240
    max_aircraft_stints: int = 8
    max_pilot_stints: int = 4
    max_crew_stints: int = 7
    passenger_connection_fraction: float = 0.4
    day_length: int = 24 * 60
    time_step: int = 5
    seed: int = 42
    return_within_minutes: int = 180  # spoke->hub return window
    enforce_roundtrips: bool = True
    airport_codes: List[str] = field(default_factory=lambda: AIRPORT_CODES_USA.copy())
    hub_codes: List[str] = field(default_factory=lambda: HUB_CODES_USA.copy())


default_network_design = NetworkDesign()
