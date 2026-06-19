"""Monte Carlo collision event generator.

Simulates LHC pp -> X events with simplified matrix elements and phase-space
sampling: minimum-bias soft QCD, hard-scatter resonances (Z->ll, J/psi->mumu,
H->gamma gamma), an underlying event, pile-up vertex smearing, and power-law pT
spectra. This is not a full shower/hadronisation Monte Carlo, but it is
physically motivated enough for pipeline and analysis benchmarking.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Generator as PythonGenerator

from src.core.event import Event, EventMetadata, PrimaryVertex
from src.core.four_vector import FourVector
from src.core.particles import PDG, PARTICLE_MASSES, Particle, ParticleType


@dataclass
class GeneratorConfig:
    """Steering parameters for the event generator."""

    sqrt_s: float = 13600.0         # centre-of-mass energy (GeV), LHC Run 3
    seed: int = 42
    z_fraction: float = 0.05        # fraction of events containing a Z boson
    jpsi_fraction: float = 0.03
    higgs_fraction: float = 0.001   # tiny cross-section
    min_bias_pt_min: float = 0.3    # min pT for soft tracks (GeV)
    min_bias_n_mean: int = 25       # mean charged multiplicity
    pileup_mu: float = 60.0
    vertex_sigma_z: float = 35.0    # beam sigma_z (mm)
    vertex_sigma_xy: float = 0.015  # beam sigma_xy (mm), about 15 um
    run_number: int = 360026


class CollisionGenerator:
    """Seedable, reproducible pp collision generator (driven by a private RNG)."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self._rng = random.Random(self.config.seed)
        self._event_counter = 0

    # --- Public interface ---
    def generate_one(self) -> Event:
        """Generate a single collision event."""
        self._event_counter += 1
        event = Event(
            metadata=EventMetadata(
                run_number=self.config.run_number,
                event_number=self._event_counter,
                lumi_block=self._event_counter // 1000 + 1,
                bunch_crossing=self._rng.randint(1, 3564),
                sqrt_s=self.config.sqrt_s,
                pileup_mu=self.config.pileup_mu,
                timestamp=time.time(),
            ),
            vertex=self._generate_primary_vertex(),
            t_generated=time.perf_counter(),
        )

        # Pick a hard process by its configured cumulative fraction, else min-bias.
        r = self._rng.random()
        if r < self.config.higgs_fraction:
            self._generate_higgs_diphoton(event)
        elif r < self.config.higgs_fraction + self.config.z_fraction:
            self._generate_z_boson(event)
        elif r < self.config.higgs_fraction + self.config.z_fraction + self.config.jpsi_fraction:
            self._generate_jpsi(event)
        else:
            self._generate_minimum_bias(event)

        self._add_underlying_event(event)
        return event

    def generate(self, n_events: int) -> PythonGenerator[Event, None, None]:
        """Yield n_events collision events."""
        for _ in range(n_events):
            yield self.generate_one()

    # --- Vertex generation ---
    def _generate_primary_vertex(self) -> PrimaryVertex:
        """Smear the primary vertex around the IP with the beam profile."""
        return PrimaryVertex(
            x=self._rng.gauss(0.0, self.config.vertex_sigma_xy),
            y=self._rng.gauss(0.0, self.config.vertex_sigma_xy),
            z=self._rng.gauss(0.0, self.config.vertex_sigma_z),
            sigma_x=self.config.vertex_sigma_xy,
            sigma_y=self.config.vertex_sigma_xy,
            sigma_z=self.config.vertex_sigma_z,
        )

    # --- Phase-space sampling helpers ---
    def _sample_pt(self, pt_min: float, pt_max: float, power: float = -3.5) -> float:
        """Sample pT from a power-law spectrum dN/dpT ~ pT^power via inverse CDF."""
        if abs(power + 1.0) < 1e-6:
            # power = -1 is the log-uniform limit
            return pt_min * math.exp(self._rng.random() * math.log(pt_max / pt_min))
        alpha = power + 1.0
        u = self._rng.random()
        return (pt_min ** alpha + u * (pt_max ** alpha - pt_min ** alpha)) ** (1.0 / alpha)

    def _sample_eta(self, eta_max: float = 2.5) -> float:
        """Sample pseudorapidity uniformly in (-eta_max, eta_max)."""
        return self._rng.uniform(-eta_max, eta_max)

    def _sample_phi(self) -> float:
        """Sample azimuthal angle uniformly in (-pi, pi]."""
        return self._rng.uniform(-math.pi, math.pi)

    def _sample_breit_wigner(self, mass: float, width: float) -> float:
        """Sample a resonance invariant mass near its pole.

        Approximated by a Gaussian of width Gamma/2 about the pole, which avoids
        the heavy Cauchy tails of a true Breit-Wigner that could push the sampled
        mass below the decay kinematic threshold.
        """
        return self._rng.gauss(mass, width / 2.0)

    def _two_body_decay(
        self,
        parent_mass: float,
        daughter1_mass: float,
        daughter2_mass: float,
        parent_pt: float,
        parent_eta: float,
        parent_phi: float,
    ) -> tuple[FourVector, FourVector]:
        """Isotropic two-body decay in the parent rest frame, boosted to the lab.

        Returns the two daughter 4-momenta; their sum reconstructs the parent.
        """
        m = parent_mass
        m1, m2 = daughter1_mass, daughter2_mass
        if m < m1 + m2:
            m = m1 + m2 + 1e-6  # keep above the kinematic threshold

        p_star = math.sqrt(
            max(0.0, (m ** 2 - (m1 + m2) ** 2) * (m ** 2 - (m1 - m2) ** 2)) / (4.0 * m ** 2)
        )

        cos_theta = self._rng.uniform(-1.0, 1.0)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta ** 2))
        phi_decay = self._rng.uniform(-math.pi, math.pi)

        px1 = p_star * sin_theta * math.cos(phi_decay)
        py1 = p_star * sin_theta * math.sin(phi_decay)
        pz1 = p_star * cos_theta
        E1 = math.sqrt(m1 ** 2 + p_star ** 2)
        E2 = math.sqrt(m2 ** 2 + p_star ** 2)

        d1_rest = FourVector(E=E1, px=px1, py=py1, pz=pz1)
        d2_rest = FourVector(E=E2, px=-px1, py=-py1, pz=-pz1)

        parent = FourVector.from_pt_eta_phi_mass(parent_pt, parent_eta, parent_phi, m)

        # Rest -> lab: the lab moves with -v_parent relative to the rest frame.
        bx = -parent.px / parent.E
        by = -parent.py / parent.E
        bz = -parent.pz / parent.E
        return d1_rest._boost(bx, by, bz), d2_rest._boost(bx, by, bz)

    # --- Physics processes ---
    def _generate_z_boson(self, event: Event) -> None:
        """Generate Z -> l+l- (50% mumu, 50% ee)."""
        m_z = self._sample_breit_wigner(PDG.Z_BOSON_MASS, PDG.Z_BOSON_WIDTH)
        pt_z = self._sample_pt(0.5, 120.0, power=-2.5)
        eta_z = self._rng.gauss(0.0, 1.0)
        phi_z = self._sample_phi()

        if self._rng.random() < 0.5:
            l_minus, l_plus, lepton_mass = (
                ParticleType.MUON, ParticleType.ANTIMUON, PDG.MUON_MASS,
            )
        else:
            l_minus, l_plus, lepton_mass = (
                ParticleType.ELECTRON, ParticleType.POSITRON, PDG.ELECTRON_MASS,
            )

        d1_p4, d2_p4 = self._two_body_decay(
            m_z, lepton_mass, lepton_mass, pt_z, eta_z, phi_z
        )

        vtx = (event.vertex.x, event.vertex.y, event.vertex.z)
        z_p4 = FourVector.from_pt_eta_phi_mass(pt_z, eta_z, phi_z, m_z)
        z_particle = Particle(
            ptype=ParticleType.Z_BOSON, four_momentum=z_p4, vertex=vtx, status=2,
        )
        event.particles.append(z_particle)

        for p4, ptype in [(d1_p4, l_minus), (d2_p4, l_plus)]:
            event.particles.append(Particle(
                ptype=ptype, four_momentum=p4, vertex=vtx, status=1,
                parent_id=z_particle.particle_id,
            ))

    def _generate_jpsi(self, event: Event) -> None:
        """Generate J/psi -> mu+mu- (BR about 6%)."""
        m_jpsi = self._sample_breit_wigner(PDG.JPSI_MASS, 0.0000929)
        pt_jpsi = self._sample_pt(3.0, 50.0, power=-4.0)
        eta_jpsi = self._rng.gauss(0.0, 1.5)
        phi_jpsi = self._sample_phi()

        d1_p4, d2_p4 = self._two_body_decay(
            m_jpsi, PDG.MUON_MASS, PDG.MUON_MASS, pt_jpsi, eta_jpsi, phi_jpsi
        )

        vtx = (event.vertex.x, event.vertex.y, event.vertex.z)
        jpsi_p4 = FourVector.from_pt_eta_phi_mass(pt_jpsi, eta_jpsi, phi_jpsi, m_jpsi)
        jpsi_particle = Particle(
            ptype=ParticleType.JPSI, four_momentum=jpsi_p4, vertex=vtx, status=2,
        )
        event.particles.append(jpsi_particle)

        for p4, ptype in [(d1_p4, ParticleType.MUON), (d2_p4, ParticleType.ANTIMUON)]:
            event.particles.append(Particle(
                ptype=ptype, four_momentum=p4, vertex=vtx, status=1,
                parent_id=jpsi_particle.particle_id,
            ))

    def _generate_higgs_diphoton(self, event: Event) -> None:
        """Generate H -> gamma gamma (BR about 0.23%, the discovery golden channel)."""
        m_h = self._sample_breit_wigner(PDG.HIGGS_MASS, PDG.HIGGS_WIDTH)
        pt_h = self._sample_pt(0.1, 150.0, power=-2.0)
        eta_h = self._rng.gauss(0.0, 0.8)
        phi_h = self._sample_phi()

        d1_p4, d2_p4 = self._two_body_decay(
            m_h, PDG.PHOTON_MASS, PDG.PHOTON_MASS, pt_h, eta_h, phi_h
        )

        vtx = (event.vertex.x, event.vertex.y, event.vertex.z)
        h_p4 = FourVector.from_pt_eta_phi_mass(pt_h, eta_h, phi_h, m_h)
        h_particle = Particle(
            ptype=ParticleType.HIGGS, four_momentum=h_p4, vertex=vtx, status=2,
        )
        event.particles.append(h_particle)

        for p4 in (d1_p4, d2_p4):
            event.particles.append(Particle(
                ptype=ParticleType.PHOTON, four_momentum=p4, vertex=vtx, status=1,
                parent_id=h_particle.particle_id,
            ))

    def _generate_minimum_bias(self, event: Event) -> None:
        """Generate a soft-QCD minimum-bias event.

        Charged multiplicity is drawn from a Gaussian about the configured mean
        (a simple stand-in for the negative-binomial seen in data); tracks are a
        mix of pions and kaons with a power-law pT spectrum.
        """
        n_charged = max(1, int(self._rng.gauss(
            self.config.min_bias_n_mean,
            math.sqrt(self.config.min_bias_n_mean * 1.4),
        )))

        track_types = [
            ParticleType.PI_PLUS, ParticleType.PI_MINUS, ParticleType.PI_ZERO,
            ParticleType.K_PLUS, ParticleType.K_MINUS,
        ]

        vtx = (event.vertex.x, event.vertex.y, event.vertex.z)
        for _ in range(n_charged):
            ptype = self._rng.choice(track_types)
            mass = PARTICLE_MASSES[ptype]
            pt = self._sample_pt(self.config.min_bias_pt_min, 30.0, power=-3.5)
            eta = self._sample_eta(eta_max=2.5)
            phi = self._sample_phi()
            p4 = FourVector.from_pt_eta_phi_mass(pt, eta, phi, mass)
            event.particles.append(Particle(
                ptype=ptype, four_momentum=p4, vertex=vtx, status=1,
            ))

    def _add_underlying_event(self, event: Event) -> None:
        """Overlay a few soft low-pT pions from spectator activity."""
        n_ue = self._rng.randint(3, 12)
        vtx = (event.vertex.x, event.vertex.y, event.vertex.z)
        for _ in range(n_ue):
            pt = self._sample_pt(0.2, 3.0, power=-4.0)
            eta = self._sample_eta(eta_max=4.7)
            phi = self._sample_phi()
            ptype = self._rng.choice([ParticleType.PI_PLUS, ParticleType.PI_MINUS])
            p4 = FourVector.from_pt_eta_phi_mass(pt, eta, phi, PDG.PION_CHARGED_MASS)
            event.particles.append(Particle(
                ptype=ptype, four_momentum=p4, vertex=vtx, status=1,
            ))
