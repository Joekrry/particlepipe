"""Multi-level trigger and reconstruction pipeline.

A 3-level trigger analogous to LHC experiments: L1 applies fast pT/ET/MET
threshold cuts, L2 adds lepton isolation, and L3/HLT runs track fitting and
invariant-mass reconstruction. Each stage is a pure function over an Event that
updates the trigger bitmask; TriggerPipeline chains them asynchronously with
bounded concurrency.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from src.core.event import Event, TriggerBit
from src.core.four_vector import invariant_mass
from src.core.particles import Particle, ParticleType

logger = logging.getLogger(__name__)


@dataclass
class TriggerThresholds:
    """Physics thresholds for each trigger level (GeV unless noted)."""

    # L1
    l1_single_mu_pt: float = 20.0
    l1_double_mu_pt: tuple[float, float] = (10.0, 6.0)
    l1_single_eg_et: float = 30.0
    l1_double_eg_et: float = 15.0
    l1_met: float = 40.0

    # L2
    l2_mu_isolation: float = 0.15   # relative isolation sum(pT)/pT
    l2_eg_isolation: float = 0.10
    l2_dimu_mass_lo: float = 60.0   # Z window
    l2_dimu_mass_hi: float = 120.0

    # L3 / HLT mass windows
    l3_z_mumu_mass_lo: float = 60.0
    l3_z_mumu_mass_hi: float = 120.0
    l3_z_ee_mass_lo: float = 60.0
    l3_z_ee_mass_hi: float = 120.0
    l3_hgg_mass_lo: float = 115.0
    l3_hgg_mass_hi: float = 135.0
    l3_jpsi_mass_lo: float = 2.9
    l3_jpsi_mass_hi: float = 3.3

    # Reconstruction quality
    min_tracker_hits: int = 6
    max_track_chi2: float = 5.0


def apply_l1_trigger(event: Event, thresholds: TriggerThresholds) -> Event:
    """L1 hardware trigger: pT/ET/MET threshold scan over stable particles."""
    t_start = time.perf_counter()
    muons = [p for p in event.particles if p.is_muon and p.status == 1]
    egamma = [
        p for p in event.particles
        if p.ptype in {ParticleType.ELECTRON, ParticleType.POSITRON, ParticleType.PHOTON}
        and p.status == 1
    ]
    met = event.missing_et

    if any(m.pt > thresholds.l1_single_mu_pt for m in muons):
        event.trigger |= TriggerBit.L1_SINGLE_MU

    sorted_mu = sorted(muons, key=lambda p: p.pt, reverse=True)
    if (len(sorted_mu) >= 2
            and sorted_mu[0].pt > thresholds.l1_double_mu_pt[0]
            and sorted_mu[1].pt > thresholds.l1_double_mu_pt[1]):
        event.trigger |= TriggerBit.L1_DOUBLE_MU

    if any(p.pt > thresholds.l1_single_eg_et for p in egamma):
        event.trigger |= TriggerBit.L1_SINGLE_EG

    sorted_eg = sorted(egamma, key=lambda p: p.pt, reverse=True)
    if len(sorted_eg) >= 2 and sorted_eg[1].pt > thresholds.l1_double_eg_et:
        event.trigger |= TriggerBit.L1_DOUBLE_EG

    if met > thresholds.l1_met:
        event.trigger |= TriggerBit.L1_MET

    event.t_l1_decision = time.perf_counter()
    logger.debug("L1 trigger: %s (%.2f us)", event.trigger,
                 (event.t_l1_decision - t_start) * 1e6)
    return event


def compute_isolation(
    particle: Particle,
    all_particles: list[Particle],
    cone_dr: float = 0.4,
) -> float:
    """Relative isolation: sum of other charged-track pT within dR, over the candidate pT.

    A small value (< 0.15) marks a lepton that is not embedded in a jet.
    """
    if particle.pt < 1e-6:
        return float("inf")

    sum_pt = 0.0
    for other in all_particles:
        if other.particle_id == particle.particle_id:
            continue
        if not other.is_charged or not other.is_stable:
            continue
        if other.pt < 0.5:
            continue
        if particle.four_momentum.delta_r(other.four_momentum) < cone_dr:
            sum_pt += other.pt
    return sum_pt / particle.pt


def apply_l2_trigger(event: Event, thresholds: TriggerThresholds) -> Event:
    """L2 software trigger: lepton isolation and the di-muon mass window."""
    if not event.passed_l1():
        event.t_l2_decision = time.perf_counter()
        return event

    stable = [p for p in event.particles if p.status == 1]
    for particle in stable:
        if particle.is_lepton:
            particle.isolation = compute_isolation(particle, stable)

    muons = event.muons()
    if any(m.isolation < thresholds.l2_mu_isolation and m.pt > 10.0 for m in muons):
        event.trigger |= TriggerBit.L2_ISOLATED_MU

    if len(muons) >= 2:
        mu_sorted = sorted(muons, key=lambda p: p.pt, reverse=True)
        m_inv = invariant_mass(mu_sorted[0].four_momentum, mu_sorted[1].four_momentum)
        if thresholds.l2_dimu_mass_lo < m_inv < thresholds.l2_dimu_mass_hi:
            event.trigger |= TriggerBit.L2_DIMUON_MASS

    event.t_l2_decision = time.perf_counter()
    return event


def kalman_track_fit(
    particle: Particle,
    b_field_tesla: float = 3.8,
    n_hits: int | None = None,
    detector_resolution_mm: float = 0.010,
    rng: random.Random | None = None,
) -> tuple[float, int]:
    """Simplified track fit returning (chi2_per_ndof, n_hits).

    The expected hit count follows |eta| detector acceptance; the chi2 is built
    from Gaussian hit residuals plus a 1/pT multiple-scattering term. A full fit
    would propagate the 5-parameter helix state and its covariance. Residuals are
    drawn from a per-track RNG seeded by the kinematics, so the fit is both
    reproducible and safe to run concurrently.
    """
    if rng is None:
        rng = random.Random(f"{particle.pt:.6f}:{particle.eta:.6f}:{particle.phi:.6f}")

    if n_hits is None:
        eta_abs = abs(particle.eta)
        if eta_abs < 0.8:
            n_hits = 12
        elif eta_abs < 1.6:
            n_hits = 9
        elif eta_abs < 2.5:
            n_hits = 7
        else:
            return float("inf"), 0

    if n_hits < 3:
        return float("inf"), n_hits

    n_dof = max(1, n_hits - 5)   # five helix parameters
    chi2 = sum(
        (rng.gauss(0.0, detector_resolution_mm) / detector_resolution_mm) ** 2
        for _ in range(n_hits)
    )
    chi2 += (0.5 / max(particle.pt, 0.5)) * n_hits * 0.1   # multiple scattering
    return chi2 / n_dof, n_hits


def _reconstruct_pair(
    candidates: list[Particle],
    mass_lo: float,
    mass_hi: float,
    require_opposite_sign: bool,
) -> bool:
    """True if any candidate pair (opposite-sign when required) has mass in (lo, hi)."""
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if require_opposite_sign and a.charge * b.charge >= 0:
                continue
            if mass_lo < invariant_mass(a.four_momentum, b.four_momentum) < mass_hi:
                return True
    return False


def apply_l3_reconstruction(event: Event, thresholds: TriggerThresholds) -> Event:
    """L3 / HLT: track fitting plus invariant-mass reconstruction of resonances."""
    if not event.passed_l2():
        event.t_l3_decision = time.perf_counter()
        event.t_reconstructed = time.perf_counter()
        return event

    for particle in event.particles:
        if particle.is_charged and particle.is_stable:
            chi2, n_hits = kalman_track_fit(particle)
            particle.track_chi2 = chi2
            particle.detector_hits = n_hits
            particle.is_reconstructed = (
                n_hits >= thresholds.min_tracker_hits
                and chi2 < thresholds.max_track_chi2
            )

    good_muons = [
        m for m in event.muons()
        if m.is_reconstructed and m.pt > 20.0 and abs(m.eta) < 2.4 and m.isolation < 0.15
    ]
    if _reconstruct_pair(good_muons, thresholds.l3_z_mumu_mass_lo,
                         thresholds.l3_z_mumu_mass_hi, True):
        event.trigger |= TriggerBit.L3_Z_TO_MUMU

    good_electrons = [
        e for e in event.electrons()
        if e.is_reconstructed and e.pt > 25.0 and abs(e.eta) < 2.5
    ]
    if _reconstruct_pair(good_electrons, thresholds.l3_z_ee_mass_lo,
                         thresholds.l3_z_ee_mass_hi, True):
        event.trigger |= TriggerBit.L3_Z_TO_EE

    good_photons = [
        g for g in event.photons()
        if g.pt > 25.0 and abs(g.eta) < 2.5
    ]
    if _reconstruct_pair(good_photons, thresholds.l3_hgg_mass_lo,
                         thresholds.l3_hgg_mass_hi, False):
        event.trigger |= TriggerBit.L3_H_TO_GAMGAM

    jpsi_muons = [
        m for m in event.muons()
        if m.is_reconstructed and m.pt > 4.0 and abs(m.eta) < 2.4
    ]
    if _reconstruct_pair(jpsi_muons, thresholds.l3_jpsi_mass_lo,
                         thresholds.l3_jpsi_mass_hi, True):
        event.trigger |= TriggerBit.L3_JPSI_TO_MUMU

    event.t_l3_decision = time.perf_counter()
    event.t_reconstructed = time.perf_counter()
    return event


@dataclass
class PipelineStats:
    """Running statistics for the async trigger pipeline."""

    n_input: int = 0
    n_l1_pass: int = 0
    n_l2_pass: int = 0
    n_l3_pass: int = 0
    total_latency: float = 0.0   # seconds
    start_time: float = field(default_factory=time.time)

    @property
    def l1_rate(self) -> float:
        return self.n_l1_pass / max(self.n_input, 1)

    @property
    def l2_rate(self) -> float:
        return self.n_l2_pass / max(self.n_input, 1)

    @property
    def l3_rate(self) -> float:
        return self.n_l3_pass / max(self.n_input, 1)

    @property
    def throughput_hz(self) -> float:
        elapsed = time.time() - self.start_time
        return self.n_input / max(elapsed, 1e-9)

    @property
    def mean_latency_ms(self) -> float:
        return (self.total_latency / max(self.n_input, 1)) * 1000.0


class TriggerPipeline:
    """Async multi-level trigger pipeline: L1 runs inline, L2 and L3 are offloaded to threads."""

    def __init__(
        self,
        thresholds: TriggerThresholds | None = None,
        n_workers: int = 4,
    ) -> None:
        self.thresholds = thresholds or TriggerThresholds()
        self.n_workers = n_workers
        self.stats = PipelineStats()

    async def process_one(self, event: Event) -> Event:
        """Run one event through L1 inline, then L2 and L3 off the event loop."""
        event = apply_l1_trigger(event, self.thresholds)
        self.stats.n_input += 1

        if event.passed_l1():
            self.stats.n_l1_pass += 1
            event = await asyncio.to_thread(apply_l2_trigger, event, self.thresholds)

            if event.passed_l2():
                self.stats.n_l2_pass += 1
                event = await asyncio.to_thread(
                    apply_l3_reconstruction, event, self.thresholds
                )
                if event.passed_l3():
                    self.stats.n_l3_pass += 1

        if event.t_generated and event.t_reconstructed:
            self.stats.total_latency += event.t_reconstructed - event.t_generated
        return event

    async def process_stream(
        self,
        source: AsyncIterator[Event],
        max_concurrent: int | None = None,
    ) -> AsyncIterator[Event]:
        """Process an async event stream with bounded concurrency, yielding completed events.

        Completion order is not preserved; concurrency defaults to n_workers.
        """
        limit = max_concurrent if max_concurrent is not None else self.n_workers
        semaphore = asyncio.Semaphore(limit)

        async def bounded(ev: Event) -> Event:
            async with semaphore:
                return await self.process_one(ev)

        pending: set[asyncio.Task[Event]] = set()
        async for event in source:
            pending.add(asyncio.create_task(bounded(event)))
            done, pending = await asyncio.wait(
                pending, timeout=0.0, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                yield await task

        if pending:
            for result in await asyncio.gather(*pending):
                yield result

    def log_stats(self) -> None:
        """Log a one-line summary of the pipeline counters."""
        logger.info(
            "Pipeline stats: input=%d, L1=%.1f%%, L2=%.1f%%, L3=%.1f%%, "
            "throughput=%.0f Hz, mean_latency=%.2f ms",
            self.stats.n_input,
            self.stats.l1_rate * 100,
            self.stats.l2_rate * 100,
            self.stats.l3_rate * 100,
            self.stats.throughput_hz,
            self.stats.mean_latency_ms,
        )
