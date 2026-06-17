"""PDG particle database and the Particle domain model.

PDG-accurate rest masses (GeV/c^2), charges, and selected widths and
lifetimes, plus a mutable Particle carrying kinematics and reconstruction state.

Reference: Particle Data Group, R.L. Workman et al. (2022), PTEP 2022, 083C01.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.core.four_vector import FourVector


class PDG:
    """Selected PDG-2022 properties: masses in GeV/c^2, lifetimes in s, widths in GeV."""

    # Leptons
    ELECTRON_MASS = 0.000510999
    MUON_MASS = 0.105658
    MUON_LIFETIME = 2.1969811e-6
    TAU_MASS = 1.77686

    # Light mesons
    PION_CHARGED_MASS = 0.139570
    PION_NEUTRAL_MASS = 0.134977
    KAON_CHARGED_MASS = 0.493677
    KAON_NEUTRAL_MASS = 0.497611
    RHO_MASS = 0.77526
    RHO_WIDTH = 0.1474

    # Baryons
    PROTON_MASS = 0.938272
    NEUTRON_MASS = 0.939565

    # Gauge bosons
    W_BOSON_MASS = 80.377
    W_BOSON_WIDTH = 2.085
    Z_BOSON_MASS = 91.1876
    Z_BOSON_WIDTH = 2.4952
    HIGGS_MASS = 125.25
    HIGGS_WIDTH = 0.00407

    # Photon / gluon
    PHOTON_MASS = 0.0
    GLUON_MASS = 0.0

    # Charm / beauty
    D0_MASS = 1.86484
    JPSI_MASS = 3.09690
    BPLUS_MASS = 5.27934
    BZERO_MASS = 5.27966
    UPSILON_MASS = 9.46030


class ParticleType(str, enum.Enum):
    """Enumeration of particle species."""

    ELECTRON = "e-"
    POSITRON = "e+"
    MUON = "mu-"
    ANTIMUON = "mu+"
    TAU = "tau-"
    ANTITAU = "tau+"
    ELECTRON_NEUTRINO = "nu_e"
    MUON_NEUTRINO = "nu_mu"
    PI_PLUS = "pi+"
    PI_MINUS = "pi-"
    PI_ZERO = "pi0"
    K_PLUS = "K+"
    K_MINUS = "K-"
    K_ZERO = "K0"
    RHO_PLUS = "rho+"
    RHO_ZERO = "rho0"
    PROTON = "p"
    ANTIPROTON = "pbar"
    NEUTRON = "n"
    PHOTON = "gamma"
    W_PLUS = "W+"
    W_MINUS = "W-"
    Z_BOSON = "Z0"
    HIGGS = "H0"
    JPSI = "J/psi"
    UPSILON = "Upsilon"
    UNKNOWN = "?"


PARTICLE_MASSES: dict[ParticleType, float] = {
    ParticleType.ELECTRON: PDG.ELECTRON_MASS,
    ParticleType.POSITRON: PDG.ELECTRON_MASS,
    ParticleType.MUON: PDG.MUON_MASS,
    ParticleType.ANTIMUON: PDG.MUON_MASS,
    ParticleType.TAU: PDG.TAU_MASS,
    ParticleType.ANTITAU: PDG.TAU_MASS,
    ParticleType.ELECTRON_NEUTRINO: 0.0,
    ParticleType.MUON_NEUTRINO: 0.0,
    ParticleType.PI_PLUS: PDG.PION_CHARGED_MASS,
    ParticleType.PI_MINUS: PDG.PION_CHARGED_MASS,
    ParticleType.PI_ZERO: PDG.PION_NEUTRAL_MASS,
    ParticleType.K_PLUS: PDG.KAON_CHARGED_MASS,
    ParticleType.K_MINUS: PDG.KAON_CHARGED_MASS,
    ParticleType.K_ZERO: PDG.KAON_NEUTRAL_MASS,
    ParticleType.RHO_PLUS: PDG.RHO_MASS,
    ParticleType.RHO_ZERO: PDG.RHO_MASS,
    ParticleType.PROTON: PDG.PROTON_MASS,
    ParticleType.ANTIPROTON: PDG.PROTON_MASS,
    ParticleType.NEUTRON: PDG.NEUTRON_MASS,
    ParticleType.PHOTON: PDG.PHOTON_MASS,
    ParticleType.W_PLUS: PDG.W_BOSON_MASS,
    ParticleType.W_MINUS: PDG.W_BOSON_MASS,
    ParticleType.Z_BOSON: PDG.Z_BOSON_MASS,
    ParticleType.HIGGS: PDG.HIGGS_MASS,
    ParticleType.JPSI: PDG.JPSI_MASS,
    ParticleType.UPSILON: PDG.UPSILON_MASS,
    ParticleType.UNKNOWN: 0.0,
}

PARTICLE_CHARGES: dict[ParticleType, int] = {
    ParticleType.ELECTRON: -1, ParticleType.POSITRON: +1,
    ParticleType.MUON: -1, ParticleType.ANTIMUON: +1,
    ParticleType.TAU: -1, ParticleType.ANTITAU: +1,
    ParticleType.ELECTRON_NEUTRINO: 0, ParticleType.MUON_NEUTRINO: 0,
    ParticleType.PI_PLUS: +1, ParticleType.PI_MINUS: -1, ParticleType.PI_ZERO: 0,
    ParticleType.K_PLUS: +1, ParticleType.K_MINUS: -1, ParticleType.K_ZERO: 0,
    ParticleType.RHO_PLUS: +1, ParticleType.RHO_ZERO: 0,
    ParticleType.PROTON: +1, ParticleType.ANTIPROTON: -1, ParticleType.NEUTRON: 0,
    ParticleType.PHOTON: 0,
    ParticleType.W_PLUS: +1, ParticleType.W_MINUS: -1, ParticleType.Z_BOSON: 0,
    ParticleType.HIGGS: 0, ParticleType.JPSI: 0, ParticleType.UPSILON: 0,
    ParticleType.UNKNOWN: 0,
}


@dataclass
class Particle:
    """A generated or reconstructed particle with kinematics and detector state."""

    particle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ptype: ParticleType = ParticleType.UNKNOWN
    four_momentum: FourVector = field(default_factory=lambda: FourVector(0, 0, 0, 0))
    vertex: tuple[float, float, float] = (0.0, 0.0, 0.0)  # production point (mm)
    status: int = 1  # generator status: 1=stable, 2=decayed
    parent_id: Optional[str] = None
    is_reconstructed: bool = False
    detector_hits: int = 0
    track_chi2: float = 0.0
    isolation: float = 0.0  # ΣpT of other tracks within ΔR < 0.4

    @property
    def mass(self) -> float:
        return PARTICLE_MASSES.get(self.ptype, 0.0)

    @property
    def charge(self) -> int:
        return PARTICLE_CHARGES.get(self.ptype, 0)

    @property
    def pt(self) -> float:
        return self.four_momentum.pt

    @property
    def eta(self) -> float:
        return self.four_momentum.eta

    @property
    def phi(self) -> float:
        return self.four_momentum.phi

    @property
    def energy(self) -> float:
        return self.four_momentum.E

    @property
    def is_stable(self) -> bool:
        return self.status == 1

    @property
    def is_charged(self) -> bool:
        return self.charge != 0

    @property
    def is_lepton(self) -> bool:
        return self.ptype in {
            ParticleType.ELECTRON, ParticleType.POSITRON,
            ParticleType.MUON, ParticleType.ANTIMUON,
            ParticleType.TAU, ParticleType.ANTITAU,
        }

    @property
    def is_muon(self) -> bool:
        return self.ptype in {ParticleType.MUON, ParticleType.ANTIMUON}

    @property
    def is_photon(self) -> bool:
        return self.ptype == ParticleType.PHOTON

    def __repr__(self) -> str:
        # ASCII labels so printing is safe on non-UTF-8 consoles (e.g. Windows cp1252).
        return (
            f"Particle({self.ptype.value}, pT={self.pt:.2f} GeV, "
            f"eta={self.eta:.2f}, phi={self.phi:.2f}, "
            f"q={self.charge:+d})"
        )
