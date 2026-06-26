# ParticlePipe

**Project**: ParticlePipe - a high-energy-physics (HEP) data pipeline and analysis backend (Python).

**About**: Experimental particle physics produces collision data that must be generated, filtered, reconstructed and analysed through a chain of well-separated processing stages. ParticlePipe builds that chain as a layered Python backend, implementing the physics from first principles with no `numpy`, `ROOT` or `Pythia` in the domain core. It provides relativistic four-vector kinematics, a PDG-accurate particle database, and a collision-event model, a seedable Monte Carlo generator that produces Z, J/psi, Higgs-diphoton and minimum-bias events, an asynchronous three-level (L1/L2/L3) trigger that filters events, fits tracks and reconstructs resonances, and an analysis engine that builds invariant-mass spectra and extracts peak positions, widths and yields.

**Technical Expectations**

1. Layered architecture with a single, downward dependency direction from the domain core outward.
1. Physics algorithms implemented from first principles, with no scientific-computing dependency in the domain core.
1. Deterministic, reproducible event generation and reconstruction from an explicit seed.
1. Asynchronous event processing with bounded concurrency built on `asyncio`.
1. Static typing and consistent formatting enforced through `mypy`, `ruff` and `black`.
1. ASCII-only, dependency-light source that stays portable across platforms and terminals.

---

## Table of Contents

1. [General Description](#general-description)
2. [Usage](#usage)
3. [Installation Instructions](#installation-instructions)
4. [References](#references)

---

## General Description

### Overview

The repository is organised as a layered Python package under `src/`, with an accompanying test tree under `tests/`. Each layer is an importable package, and the dependency direction runs strictly downward: an interface layer may import an engine, an engine may import the domain core, but never the reverse.

```
+--------------------------------------------------------------+
| Interface   src/api        FastAPI application, SSE streaming |
+--------------------------------------------------------------+
| Engines     src/pipeline   event generation + trigger chain  |
|             src/analysis   histograms, spectra, peak finding  |
+--------------------------------------------------------------+
| Domain      src/core       4-vectors, particles, events      |
+--------------------------------------------------------------+
| Support     src/models     Pydantic data contracts           |
|             src/utils      logging, metrics, helpers          |
+--------------------------------------------------------------+
        dependencies point downward; lower layers never import upper
```

### Implemented modules

| Module | Key types | Responsibility |
|--------|-----------|----------------|
| `src/core/four_vector.py` | `FourVector`, `invariant_mass` | Immutable relativistic 4-momentum in the (+,-,-,-) metric: invariant mass, general and rest-frame Lorentz boosts, `pt`/`eta`/`phi`/`theta`/`rapidity`, and the `dR` angular separation. |
| `src/core/particles.py` | `PDG`, `ParticleType`, `Particle` | PDG-2022 masses and charges for 27 species, with a particle object carrying kinematics and reconstruction state. |
| `src/core/event.py` | `Event`, `TriggerBit`, `PrimaryVertex`, `EventMetadata` | The collision-event record: particle collection, trigger bitmask, primary vertex, run metadata, and event-level observables (scalar HT, missing ET, latency). |
| `src/pipeline/generator.py` | `CollisionGenerator`, `GeneratorConfig` | Seedable Monte Carlo generator: Z->ll, J/psi->mumu, H->gamma gamma and minimum-bias events, with isotropic two-body decay boosted to the lab frame and beam-profile vertex smearing. |
| `src/pipeline/trigger.py` | `TriggerPipeline`, `TriggerThresholds`, `apply_l1_trigger`, `apply_l2_trigger`, `apply_l3_reconstruction` | Three-level trigger as pure functions over an `Event` (pT/ET/MET cuts, dR-cone isolation, a simplified Kalman track fit, and OS-pair mass-window reconstruction), wrapped in an async bounded-concurrency pipeline with efficiency and latency stats. |
| `src/analysis/analysis.py` | `PhysicsAnalysis`, `Histogram1D`, `fit_gaussian_peak`, `GaussianFitResult` | A from-scratch fixed-bin histogram with Poisson errors, a robust peak estimator (sideband background subtraction plus moments) returning mean, width, yield and significance, and an accumulator that builds the mass spectra and kinematic distributions and serialises them to JSON. |

The package version is declared in `src/__init__.py` (`__version__ = "1.0.0"`).

**Dependencies**: the domain core and generator depend only on the Python standard library (`math`, `random`, `dataclasses`, `enum`). The project manifest (`requirements.txt`) additionally declares `fastapi`, `uvicorn` and `pydantic` for the service layer, `aiosqlite` for persistence, `pytest`, `pytest-asyncio`, `pytest-cov` and `hypothesis` for testing, and `ruff`, `mypy` and `black` for tooling.

---

## Usage

Reconstruct the Z mass from two back-to-back muons:

```python
import math
from src.core.four_vector import FourVector

m_z, m_mu = 91.1876, 0.105658
p = math.sqrt((m_z / 2) ** 2 - m_mu ** 2)
mu_plus = FourVector(m_z / 2, p, 0.0, 0.0)
mu_minus = FourVector(m_z / 2, -p, 0.0, 0.0)
print((mu_plus + mu_minus).invariant_mass)   # 91.1876
```

Generate collision events and read event-level observables:

```python
from src.pipeline.generator import CollisionGenerator, GeneratorConfig

gen = CollisionGenerator(GeneratorConfig(seed=42, z_fraction=0.1))
for event in gen.generate(1000):
    print(event.n_particles, round(event.missing_et, 2))
```

Run generated events through the trigger pipeline and read the per-level efficiencies:

```python
import asyncio
from src.pipeline.generator import CollisionGenerator, GeneratorConfig
from src.pipeline.trigger import TriggerPipeline

async def run():
    pipeline = TriggerPipeline()
    gen = CollisionGenerator(GeneratorConfig(seed=42, z_fraction=0.5))
    for event in gen.generate(200):
        await pipeline.process_one(event)
    print(pipeline.stats.l1_rate, pipeline.stats.l2_rate, pipeline.stats.l3_rate)

asyncio.run(run())
```

Accumulate a mass spectrum and fit the Z peak:

```python
from src.pipeline.generator import CollisionGenerator, GeneratorConfig
from src.analysis.analysis import PhysicsAnalysis

analysis = PhysicsAnalysis()
gen = CollisionGenerator(GeneratorConfig(seed=42, z_fraction=0.3))
for event in gen.generate(2000):
    analysis.process_event(event)

z = analysis.fit_z_peak()
print(round(z.mean, 2), round(z.sigma, 2))   # peak near 91 GeV
```

Generation and reconstruction are reproducible: the same seed produces identical events and identical trigger decisions. Run examples with the package on the path, for example `PYTHONPATH=. python your_script.py`.

---

## Installation Instructions

**Prerequisites**

- Python 3.11 or newer
- `pip` and the `venv` module (bundled with CPython)

**Step-by-step setup**

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On unix systems:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Installation has no side effects beyond populating the virtual environment; there is no schema migration or data download at this stage.

---

## References

- Particle Data Group, Workman, R.L. et al. (2022) Review of Particle Physics. *Progress of Theoretical and Experimental Physics*, 2022(8), 083C01. DOI: 10.1093/ptep/ptac097. Available at: https://pdg.lbl.gov/ (Accessed: 15 June 2026).
