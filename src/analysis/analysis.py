"""Physics analysis algorithms for reconstructed events.

Provides a from-scratch 1D histogram (no external dependencies), a Gaussian
signal plus polynomial background fitter using gradient-descent chi2
minimisation, and a PhysicsAnalysis accumulator that builds invariant-mass
spectra and kinematic distributions and serialises them to JSON.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.core.event import Event
from src.core.four_vector import invariant_mass


@dataclass
class Histogram1D:
    """Fixed-bin 1D histogram with overflow/underflow and Poisson-error tracking."""

    title: str
    n_bins: int
    x_min: float
    x_max: float
    counts: list[float] = field(init=False)
    sum_w2: list[float] = field(init=False)   # sum of weight^2, for Poisson errors
    overflow: float = field(init=False, default=0.0)
    underflow: float = field(init=False, default=0.0)
    n_entries: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.counts = [0.0] * self.n_bins
        self.sum_w2 = [0.0] * self.n_bins

    @property
    def bin_width(self) -> float:
        return (self.x_max - self.x_min) / self.n_bins

    def bin_centre(self, i: int) -> float:
        return self.x_min + (i + 0.5) * self.bin_width

    def bin_edges(self) -> list[float]:
        w = self.bin_width
        return [self.x_min + i * w for i in range(self.n_bins + 1)]

    def fill(self, value: float, weight: float = 1.0) -> None:
        self.n_entries += 1
        if value < self.x_min:
            self.underflow += weight
            return
        if value >= self.x_max:
            self.overflow += weight
            return
        idx = min(int((value - self.x_min) / self.bin_width), self.n_bins - 1)
        self.counts[idx] += weight
        self.sum_w2[idx] += weight * weight

    def stat_errors(self) -> list[float]:
        """Poisson errors: sigma_i = sqrt(sum_w2_i)."""
        return [math.sqrt(s) for s in self.sum_w2]

    def integral(self) -> float:
        return sum(self.counts)

    def normalise_to_unity(self) -> Histogram1D:
        total = self.integral()
        if total <= 0:
            return self
        for i in range(self.n_bins):
            self.counts[i] /= total
            self.sum_w2[i] /= total * total
        return self

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "n_bins": self.n_bins,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "bin_width": self.bin_width,
            "bin_centres": [self.bin_centre(i) for i in range(self.n_bins)],
            "counts": self.counts,
            "errors": self.stat_errors(),
            "overflow": self.overflow,
            "underflow": self.underflow,
            "n_entries": self.n_entries,
            "integral": self.integral(),
        }

    def __iadd__(self, other: Histogram1D) -> Histogram1D:
        """Add another histogram in place (same binning assumed)."""
        for i in range(self.n_bins):
            self.counts[i] += other.counts[i]
            self.sum_w2[i] += other.sum_w2[i]
        self.overflow += other.overflow
        self.underflow += other.underflow
        self.n_entries += other.n_entries
        return self


@dataclass
class GaussianFitResult:
    """Result of a Gaussian signal plus polynomial background fit."""

    mean: float                     # peak position (GeV)
    sigma: float                    # peak width (GeV)
    amplitude: float                # peak height (counts)
    background_coeffs: list[float]  # [a0, a1, a2] for a0 + a1*x + a2*x^2
    chi2: float
    n_dof: int
    n_signal: float                 # integrated signal yield (events)
    n_signal_err: float
    significance: float             # S / sqrt(S + B)

    @property
    def chi2_per_ndf(self) -> float:
        return self.chi2 / max(self.n_dof, 1)

    @property
    def resolution_percent(self) -> float:
        return (self.sigma / max(self.mean, 1e-9)) * 100.0


def _linear_lsq(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares (a0, a1) for y = a0 + a1*x; (mean, 0) when under-determined."""
    n = len(points)
    if n < 2:
        return (points[0][1] if n else 0.0), 0.0
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return sy / n, 0.0
    a1 = (n * sxy - sx * sy) / denom
    a0 = (sy - a1 * sx) / n
    return a0, a1


def fit_gaussian_peak(
    histogram: Histogram1D,
    peak_mass: float,
    fit_range: tuple[float, float],
    signal_window: float = 3.0,
) -> GaussianFitResult:
    """Estimate a Gaussian peak over a linear background in the fit range.

    The background is fit to the sidebands (bins away from the peak) and
    subtracted; the peak mean and width come from the moments of the residual,
    and the yield is the background-subtracted count within signal_window sigma.
    This is robust on sparse spectra where a free gradient-descent fit of a
    Gaussian against a polynomial background is degenerate. The reported chi2 is
    for the resulting model; significance is S / sqrt(S + B) over the window.
    """
    x_lo, x_hi = fit_range
    xs, ys, es = [], [], []
    for i in range(histogram.n_bins):
        xc = histogram.bin_centre(i)
        if x_lo <= xc <= x_hi:
            xs.append(xc)
            ys.append(histogram.counts[i])
            es.append(max(math.sqrt(histogram.sum_w2[i]), 1.0))

    empty = GaussianFitResult(
        mean=peak_mass, sigma=2.0, amplitude=0.0, background_coeffs=[0.0, 0.0, 0.0],
        chi2=0.0, n_dof=0, n_signal=0.0, n_signal_err=0.0, significance=0.0,
    )
    if len(xs) < 5:
        return empty

    half = 0.25 * (x_hi - x_lo)
    sidebands = [(x, y) for x, y in zip(xs, ys) if abs(x - peak_mass) > half]
    a0, a1 = _linear_lsq(sidebands)

    def bkg(x: float) -> float:
        return a0 + a1 * x

    peak_region = [(x, max(y - bkg(x), 0.0)) for x, y in zip(xs, ys)
                   if abs(x - peak_mass) <= half]
    total_s = sum(s for _, s in peak_region)
    if total_s <= 0:
        return empty

    mu = sum(x * s for x, s in peak_region) / total_s
    var = sum((x - mu) ** 2 * s for x, s in peak_region) / total_s
    sigma = math.sqrt(max(var, histogram.bin_width ** 2))

    window = signal_window * sigma
    in_window = [(x, y) for x, y in zip(xs, ys) if abs(x - mu) <= window]
    n_signal = sum(max(y - bkg(x), 0.0) for x, y in in_window)
    n_bkg = max(sum(bkg(x) for x, _ in in_window), 0.0)
    amplitude = n_signal * histogram.bin_width / (sigma * math.sqrt(2 * math.pi))

    def model(x: float) -> float:
        return amplitude * math.exp(-0.5 * ((x - mu) / sigma) ** 2) + bkg(x)

    chi2 = sum(((y - model(x)) / e) ** 2 for x, y, e in zip(xs, ys, es))
    n_dof = max(len(xs) - 5, 1)
    n_signal_err = math.sqrt(max(n_signal, 0.0))
    significance = n_signal / math.sqrt(max(n_signal + n_bkg, 1.0))

    return GaussianFitResult(
        mean=mu,
        sigma=sigma,
        amplitude=amplitude,
        background_coeffs=[a0, a1, 0.0],
        chi2=chi2,
        n_dof=n_dof,
        n_signal=n_signal,
        n_signal_err=n_signal_err,
        significance=significance,
    )


class PhysicsAnalysis:
    """Accumulates invariant-mass spectra and kinematic distributions from events."""

    def __init__(self) -> None:
        # Invariant-mass spectra
        self.h_dimuon_mass = Histogram1D("Di-muon invariant mass", 100, 60.0, 120.0)
        self.h_jpsi_mass = Histogram1D("J/psi -> mumu invariant mass", 60, 2.5, 3.7)
        self.h_dielectron_mass = Histogram1D("Di-electron invariant mass", 100, 60.0, 120.0)
        self.h_diphoton_mass = Histogram1D("Di-photon invariant mass", 50, 115.0, 135.0)

        # pT distributions
        self.h_mu_pt = Histogram1D("Muon pT", 50, 0.0, 150.0)
        self.h_el_pt = Histogram1D("Electron ET", 50, 0.0, 150.0)
        self.h_photon_pt = Histogram1D("Photon ET", 50, 0.0, 150.0)
        self.h_leading_mu_pt = Histogram1D("Leading muon pT", 50, 0.0, 150.0)

        # Pseudorapidity
        self.h_mu_eta = Histogram1D("Muon eta", 50, -3.0, 3.0)
        self.h_el_eta = Histogram1D("Electron eta", 50, -3.0, 3.0)

        # Event-level
        self.h_n_charged = Histogram1D("Charged multiplicity", 60, 0.0, 120.0)
        self.h_ht = Histogram1D("Scalar HT (GeV)", 50, 0.0, 500.0)
        self.h_met = Histogram1D("Missing ET (GeV)", 50, 0.0, 200.0)
        self.h_vertex_z = Histogram1D("Primary vertex z (mm)", 60, -90.0, 90.0)

        self.n_events_total = 0
        self.n_z_candidates = 0
        self.n_jpsi_candidates = 0
        self.n_higgs_candidates = 0

    def process_event(self, event: Event) -> None:
        """Accumulate observables from one reconstructed event."""
        self.n_events_total += 1

        self.h_n_charged.fill(event.n_charged)
        self.h_ht.fill(event.scalar_ht)
        self.h_met.fill(event.missing_et)
        self.h_vertex_z.fill(event.vertex.z)

        muons = event.muons()
        electrons = event.electrons()
        photons = event.photons()

        for mu in muons:
            if mu.status == 1:
                self.h_mu_pt.fill(mu.pt)
                self.h_mu_eta.fill(mu.eta)

        sorted_mu = sorted(muons, key=lambda p: p.pt, reverse=True)
        if sorted_mu:
            self.h_leading_mu_pt.fill(sorted_mu[0].pt)

        for el in electrons:
            if el.status == 1:
                self.h_el_pt.fill(el.pt)
                self.h_el_eta.fill(el.eta)

        for g in photons:
            if g.status == 1:
                self.h_photon_pt.fill(g.pt)

        # Di-muon mass (Z window)
        good_muons = [m for m in muons if m.status == 1 and m.pt > 20.0 and abs(m.eta) < 2.4]
        self.n_z_candidates += self._fill_os_pairs(
            self.h_dimuon_mass, good_muons, 60.0, 120.0
        )

        # J/psi -> mumu
        jpsi_muons = [m for m in muons if m.status == 1 and m.pt > 4.0 and abs(m.eta) < 2.4]
        self.n_jpsi_candidates += self._fill_os_pairs(
            self.h_jpsi_mass, jpsi_muons, 2.5, 3.7
        )

        # Di-electron mass (Z window)
        good_el = [e for e in electrons if e.status == 1 and e.pt > 25.0 and abs(e.eta) < 2.5]
        self._fill_os_pairs(self.h_dielectron_mass, good_el, 60.0, 120.0)

        # Di-photon (H -> gamma gamma): photons are neutral, use the two hardest
        good_photons = [g for g in photons if g.status == 1 and g.pt > 25.0 and abs(g.eta) < 2.5]
        if len(good_photons) >= 2:
            sorted_g = sorted(good_photons, key=lambda p: p.pt, reverse=True)
            m_inv = invariant_mass(sorted_g[0].four_momentum, sorted_g[1].four_momentum)
            if 115.0 < m_inv < 135.0:
                self.h_diphoton_mass.fill(m_inv)
                self.n_higgs_candidates += 1

    @staticmethod
    def _fill_os_pairs(
        histogram: Histogram1D,
        candidates: list,
        mass_lo: float,
        mass_hi: float,
    ) -> int:
        """Fill the histogram with opposite-sign pair masses in (lo, hi); return the count filled."""
        filled = 0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                if a.charge * b.charge < 0:
                    m_inv = invariant_mass(a.four_momentum, b.four_momentum)
                    if mass_lo < m_inv < mass_hi:
                        histogram.fill(m_inv)
                        filled += 1
        return filled

    def fit_z_peak(self) -> GaussianFitResult:
        """Fit the Z -> mumu peak in the di-muon spectrum."""
        return fit_gaussian_peak(self.h_dimuon_mass, peak_mass=91.19, fit_range=(70.0, 110.0))

    def fit_jpsi_peak(self) -> GaussianFitResult:
        """Fit the J/psi -> mumu peak."""
        return fit_gaussian_peak(self.h_jpsi_mass, peak_mass=3.097, fit_range=(2.7, 3.5))

    def fit_higgs_peak(self) -> GaussianFitResult:
        """Fit the H -> gamma gamma peak."""
        return fit_gaussian_peak(self.h_diphoton_mass, peak_mass=125.25, fit_range=(118.0, 132.0))

    def summary(self) -> dict:
        """Return a JSON-serialisable summary of counters and histograms."""
        return {
            "n_events_total": self.n_events_total,
            "n_z_candidates": self.n_z_candidates,
            "n_jpsi_candidates": self.n_jpsi_candidates,
            "n_higgs_candidates": self.n_higgs_candidates,
            "histograms": {
                "dimuon_mass": self.h_dimuon_mass.to_dict(),
                "jpsi_mass": self.h_jpsi_mass.to_dict(),
                "dielectron_mass": self.h_dielectron_mass.to_dict(),
                "diphoton_mass": self.h_diphoton_mass.to_dict(),
                "muon_pt": self.h_mu_pt.to_dict(),
                "electron_pt": self.h_el_pt.to_dict(),
                "muon_eta": self.h_mu_eta.to_dict(),
                "n_charged": self.h_n_charged.to_dict(),
                "scalar_ht": self.h_ht.to_dict(),
                "missing_et": self.h_met.to_dict(),
                "vertex_z": self.h_vertex_z.to_dict(),
            },
        }
