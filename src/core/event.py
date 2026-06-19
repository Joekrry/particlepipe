"""Physics collision event domain model.

An Event represents a single pp collision bunch crossing: its generated and
reconstructed particles, trigger decisions, primary vertex, and metadata.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Optional

from src.core.particles import Particle, ParticleType


class TriggerBit(Flag):
    """LHC-style trigger menu as a bitmask.

    L1 is the hardware trigger (coarse pT thresholds), L2 adds isolation and
    basic reconstruction, L3/HLT runs full offline-quality reconstruction.
    """

    # Level 1 (hardware)
    L1_SINGLE_MU = auto()
    L1_DOUBLE_MU = auto()
    L1_SINGLE_EG = auto()   # single e / gamma
    L1_DOUBLE_EG = auto()
    L1_MET = auto()         # missing transverse energy

    # Level 2 (isolation, basic reconstruction)
    L2_ISOLATED_MU = auto()
    L2_ISOLATED_EG = auto()
    L2_DIMUON_MASS = auto()

    # Level 3 / HLT (full reconstruction)
    L3_Z_TO_MUMU = auto()
    L3_Z_TO_EE = auto()
    L3_H_TO_GAMGAM = auto()
    L3_JPSI_TO_MUMU = auto()

    NONE = 0
    ANY_L1 = L1_SINGLE_MU | L1_DOUBLE_MU | L1_SINGLE_EG | L1_DOUBLE_EG | L1_MET
    ANY_L2 = L2_ISOLATED_MU | L2_ISOLATED_EG | L2_DIMUON_MASS
    ANY_L3 = L3_Z_TO_MUMU | L3_Z_TO_EE | L3_H_TO_GAMGAM | L3_JPSI_TO_MUMU


@dataclass
class PrimaryVertex:
    """Reconstructed primary interaction vertex, coordinates in mm from the IP."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0          # beam direction
    sigma_x: float = 0.0    # position resolution (mm)
    sigma_y: float = 0.0
    sigma_z: float = 0.0
    n_tracks: int = 0
    chi2_ndf: float = 0.0

    @property
    def displacement(self) -> float:
        """3D displacement from the nominal IP (mm)."""
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5


@dataclass
class EventMetadata:
    """Run/event header, mimicking an ATLAS/CMS ntuple."""

    run_number: int = 0
    event_number: int = 0
    lumi_block: int = 0
    bunch_crossing: int = 0
    sqrt_s: float = 13600.0   # centre-of-mass energy (GeV), LHC Run 3
    pileup_mu: float = 60.0   # mean inelastic interactions per crossing
    timestamp: float = field(default_factory=time.time)


@dataclass
class Event:
    """A complete collision event: particles, trigger decisions, and observables."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: EventMetadata = field(default_factory=EventMetadata)
    vertex: PrimaryVertex = field(default_factory=PrimaryVertex)
    particles: list[Particle] = field(default_factory=list)
    trigger: TriggerBit = TriggerBit.NONE

    # Pipeline stage timestamps, wall-clock seconds (time.time), for latency tracking
    t_generated: Optional[float] = None
    t_l1_decision: Optional[float] = None
    t_l2_decision: Optional[float] = None
    t_l3_decision: Optional[float] = None
    t_reconstructed: Optional[float] = None

    # --- Particle selectors ---
    def stable_particles(self) -> list[Particle]:
        return [p for p in self.particles if p.is_stable]

    def charged_particles(self) -> list[Particle]:
        return [p for p in self.particles if p.is_charged and p.is_stable]

    def muons(self) -> list[Particle]:
        return [p for p in self.particles
                if p.ptype in {ParticleType.MUON, ParticleType.ANTIMUON}]

    def electrons(self) -> list[Particle]:
        return [p for p in self.particles
                if p.ptype in {ParticleType.ELECTRON, ParticleType.POSITRON}]

    def photons(self) -> list[Particle]:
        return [p for p in self.particles if p.ptype == ParticleType.PHOTON]

    def by_type(self, ptype: ParticleType) -> list[Particle]:
        return [p for p in self.particles if p.ptype == ptype]

    # --- Event-level observables ---
    @property
    def n_particles(self) -> int:
        return len(self.particles)

    @property
    def n_charged(self) -> int:
        return sum(1 for p in self.particles if p.is_charged and p.is_stable)

    @property
    def scalar_ht(self) -> float:
        """Scalar sum of stable-particle pT (the HT observable), in GeV."""
        return sum(p.pt for p in self.stable_particles())

    @property
    def missing_et(self) -> float:
        """|MET| = |-sum(pT)| over stable visible particles; non-zero because neutrinos escape."""
        invisible = {ParticleType.ELECTRON_NEUTRINO, ParticleType.MUON_NEUTRINO}
        met_x = -sum(p.four_momentum.px for p in self.stable_particles()
                     if p.ptype not in invisible)
        met_y = -sum(p.four_momentum.py for p in self.stable_particles()
                     if p.ptype not in invisible)
        return (met_x ** 2 + met_y ** 2) ** 0.5

    @property
    def processing_latency_ms(self) -> Optional[float]:
        """Generation-to-reconstruction latency (ms), or None if not fully processed."""
        if self.t_generated and self.t_reconstructed:
            return (self.t_reconstructed - self.t_generated) * 1000.0
        return None

    # --- Trigger helpers ---
    def passed_l1(self) -> bool:
        return bool(self.trigger & TriggerBit.ANY_L1)

    def passed_l2(self) -> bool:
        return bool(self.trigger & TriggerBit.ANY_L2)

    def passed_l3(self) -> bool:
        return bool(self.trigger & TriggerBit.ANY_L3)

    def __repr__(self) -> str:
        return (
            f"Event(id={self.event_id[:8]}, "
            f"n_particles={self.n_particles}, "
            f"trigger={self.trigger!r}, "
            f"passed_l3={self.passed_l3()})"
        )
