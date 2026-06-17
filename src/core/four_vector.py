"""Lorentz 4-vector algebra for relativistic kinematics.

Covariant 4-momentum (E, px, py, pz) in the (+,-,-,-) metric, natural units (GeV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FourVector:
    """Immutable 4-momentum (E, px, py, pz) in GeV, metric (+,-,-,-)."""

    E: float   # energy (GeV)
    px: float  # GeV/c
    py: float  # GeV/c
    pz: float  # GeV/c

    # ── Constructors ──────────────────────────────────────────────────────
    @classmethod
    def from_mass_and_momentum(
        cls, mass: float, px: float, py: float, pz: float
    ) -> FourVector:
        """Build from rest mass and 3-momentum via E² = m² + |p|²."""
        p2 = px * px + py * py + pz * pz
        E = math.sqrt(max(mass * mass + p2, 0.0))
        return cls(E=E, px=px, py=py, pz=pz)

    @classmethod
    def from_pt_eta_phi_mass(
        cls, pt: float, eta: float, phi: float, mass: float
    ) -> FourVector:
        """Build from collider coordinates (pt, eta, phi, mass)."""
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        return cls.from_mass_and_momentum(mass, px, py, pz)

    # ── Lorentz-invariant quantities ──────────────────────────────────────
    @property
    def invariant_mass(self) -> float:
        """m = √(E² - |p|²); a negative m² from rounding is clamped to 0."""
        m2 = self.E ** 2 - (self.px ** 2 + self.py ** 2 + self.pz ** 2)
        return math.sqrt(max(m2, 0.0))

    @property
    def mass(self) -> float:
        return self.invariant_mass

    # ── Transverse plane (collider frame) ─────────────────────────────────
    @property
    def pt(self) -> float:
        return math.sqrt(self.px ** 2 + self.py ** 2)

    @property
    def phi(self) -> float:
        """Azimuthal angle in (-π, π]."""
        return math.atan2(self.py, self.px)

    @property
    def theta(self) -> float:
        """Polar angle from the beam axis."""
        p3 = math.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        if p3 == 0.0:
            return 0.0
        return math.acos(max(-1.0, min(1.0, self.pz / p3)))

    @property
    def eta(self) -> float:
        """Pseudorapidity η = -ln(tan(θ/2)); +inf along the beam axis."""
        tan_half_theta = math.tan(self.theta / 2.0)
        if tan_half_theta <= 0.0:
            return float("inf")
        return -math.log(tan_half_theta)

    @property
    def rapidity(self) -> float:
        """Rapidity y = ½·ln((E+pz)/(E-pz)); invariant under z-boosts."""
        denom = self.E - self.pz
        if denom <= 0.0:
            return float("inf")
        return 0.5 * math.log((self.E + self.pz) / denom)

    @property
    def p3_mag(self) -> float:
        return math.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)

    @property
    def beta(self) -> float:
        """Velocity β = |p|/E."""
        return self.p3_mag / self.E if self.E > 0 else 0.0

    @property
    def gamma(self) -> float:
        """Lorentz factor γ = E/m; +inf for a massless vector."""
        m = self.invariant_mass
        return self.E / m if m > 0 else float("inf")

    # ── Angular separation ────────────────────────────────────────────────
    def delta_phi(self, other: FourVector) -> float:
        """Δφ wrapped into (-π, π]."""
        dphi = self.phi - other.phi
        while dphi > math.pi:
            dphi -= 2 * math.pi
        while dphi < -math.pi:
            dphi += 2 * math.pi
        return dphi

    def delta_eta(self, other: FourVector) -> float:
        return self.eta - other.eta

    def delta_r(self, other: FourVector) -> float:
        """ΔR = √(Δη² + Δφ²), the standard HEP angular distance."""
        return math.sqrt(self.delta_eta(other) ** 2 + self.delta_phi(other) ** 2)

    # ── Arithmetic ────────────────────────────────────────────────────────
    def __add__(self, other: FourVector) -> FourVector:
        return FourVector(
            E=self.E + other.E,
            px=self.px + other.px,
            py=self.py + other.py,
            pz=self.pz + other.pz,
        )

    def __neg__(self) -> FourVector:
        return FourVector(E=-self.E, px=-self.px, py=-self.py, pz=-self.pz)

    def __sub__(self, other: FourVector) -> FourVector:
        return self + (-other)

    def __mul__(self, scalar: float) -> FourVector:
        return FourVector(
            E=self.E * scalar, px=self.px * scalar,
            py=self.py * scalar, pz=self.pz * scalar,
        )

    def __rmul__(self, scalar: float) -> FourVector:
        return self.__mul__(scalar)

    def dot(self, other: FourVector) -> float:
        """Lorentz inner product with metric (+,-,-,-)."""
        return (self.E * other.E
                - self.px * other.px
                - self.py * other.py
                - self.pz * other.pz)

    # ── Lorentz boost ─────────────────────────────────────────────────────
    def boost_to_rest_frame(self) -> FourVector:
        """Boost into this vector's own rest frame.

        _boost(b) goes to the frame moving with velocity b, so the boost
        velocity is the particle's own velocity p/E.
        """
        beta_x = self.px / self.E
        beta_y = self.py / self.E
        beta_z = self.pz / self.E
        return self._boost(beta_x, beta_y, beta_z)

    def _boost(self, bx: float, by: float, bz: float) -> FourVector:
        """General Lorentz boost with velocity (bx, by, bz)."""
        b2 = bx * bx + by * by + bz * bz
        gamma = 1.0 / math.sqrt(1.0 - b2) if b2 < 1.0 else float("inf")
        bp = bx * self.px + by * self.py + bz * self.pz
        gamma2 = (gamma - 1.0) / b2 if b2 > 0 else 0.0
        new_px = self.px + gamma2 * bp * bx - gamma * bx * self.E
        new_py = self.py + gamma2 * bp * by - gamma * by * self.E
        new_pz = self.pz + gamma2 * bp * bz - gamma * bz * self.E
        new_E = gamma * (self.E - bp)
        return FourVector(E=new_E, px=new_px, py=new_py, pz=new_pz)

    def __repr__(self) -> str:
        return (f"FourVector(E={self.E:.4f}, px={self.px:.4f}, "
                f"py={self.py:.4f}, pz={self.pz:.4f}, "
                f"m={self.invariant_mass:.4f} GeV)")


def invariant_mass(*vectors: FourVector) -> float:
    """Invariant mass of a system of 4-vectors."""
    total = FourVector(E=0.0, px=0.0, py=0.0, pz=0.0)
    for v in vectors:
        total = total + v
    return total.invariant_mass
