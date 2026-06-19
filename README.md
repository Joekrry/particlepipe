# ParticlePipe

**Project**: ParticlePipe - a high-energy-physics (HEP) data pipeline and analysis backend (Python, FastAPI).

**About**: Experimental particle physics produces collision data that must be generated, filtered, reconstructed and analysed through a chain of well-separated processing stages. ParticlePipe models that chain as a Python backend: an LHC-style collision generator, a multi-level trigger, reconstruction and analysis, and an HTTP service for results. The physics mathematics - Lorentz 4-vectors, invariant-mass spectra, histogramming and track fitting - is implemented from first principles, with no `numpy`, `ROOT` or `Pythia` in the domain core, so the algorithms are explicit and the package stays deployable in restricted environments.

**Technical Expectations**

1. Layered architecture with a single, downward dependency direction from the domain core to the service interface.
1. Physics algorithms implemented from first principles, without a scientific-computing dependency in `src/core`, `src/pipeline` or `src/analysis`.
1. Static typing and consistent formatting enforced through `mypy`, `ruff` and `black`.
1. An asynchronous HTTP interface built on `fastapi` and `pydantic` v2.
1. An automated test harness using `pytest`, `pytest-asyncio` and property-based testing with `hypothesis`.
1. Version control with a Python-oriented `.gitignore`.

---

## Table of Contents

1. [General Description](#general-description)
2. [Installation Instructions](#installation-instructions)
3. [References](#references)

---

## General Description

### Overview

The repository is organised as a layered Python package under `src/`, with an accompanying test tree under `tests/`. Each layer is an importable package, and the dependency direction runs strictly downward: an interface layer may import an engine, an engine may import the domain core, but never the reverse. Cross-cutting concerns (data contracts and helpers) sit alongside the layers and are used by all of them.

```
┌──────────────────────────────────────────────────────────────┐
│ Interface   src/api       FastAPI application, SSE streaming │
├──────────────────────────────────────────────────────────────┤
│ Engines     src/pipeline   event generation + trigger chain  │
│             src/analysis   histograms, spectra, peak finding │
├──────────────────────────────────────────────────────────────┤
│ Domain      src/core       4-vectors, particles, events      │
├──────────────────────────────────────────────────────────────┤
│ Support     src/models     Pydantic data contracts           │
│             src/utils      logging, metrics, helpers         │
└──────────────────────────────────────────────────────────────┘
        dependencies point downward; lower layers never import upper
```

| Layer | Folder | Responsibility |
|-------|--------|----------------|
| Domain core | `src/core` | Particles, events and Lorentz 4-vectors |
| Generation and triggering | `src/pipeline` | Monte Carlo event generator and multi-level trigger chain |
| Analysis | `src/analysis` | Histograms, invariant-mass spectra and peak finding |
| Interface | `src/api` | FastAPI REST application with server-sent-event streaming |
| Data contracts | `src/models` | Pydantic request and response schemas |
| Support | `src/utils` | Logging, metrics and shared helpers |
| Tests | `tests/unit`, `tests/integration` | Unit and end-to-end test suites |

The package version is declared in `src/__init__.py` (`__version__ = "1.0.0"`).

**Extra (3rd-party) tools and packages**

Dependencies are pinned by lower bound in `requirements.txt`.

- Web and API
  - `fastapi` (>=0.110.0) - asynchronous web framework for the service layer
  - `uvicorn[standard]` (>=0.29.0) - ASGI server
  - `pydantic` (>=2.6.0) - typed data contracts
  - `httpx` (>=0.27.0) - async HTTP client used in tests
- Persistence
  - `aiosqlite` (>=0.20.0) - async SQLite access (PostgreSQL via `asyncpg` in production)
- Testing
  - `pytest` (>=8.0.0), `pytest-asyncio` (>=0.23.0), `pytest-cov` (>=5.0.0)
  - `hypothesis` (>=6.100.0) - property-based testing
- Tooling
  - `ruff` (>=0.3.0) - linter
  - `mypy` (>=1.9.0) - static type checker
  - `black` (>=24.0.0) - formatter

### Installation Instructions

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

- Ramírez, S. (2024) *FastAPI documentation*. Available at: https://fastapi.tiangolo.com (Accessed: 15 June 2026).
- Pydantic (2024) *Pydantic v2 documentation*. Available at: https://docs.pydantic.dev/latest/ (Accessed: 15 June 2026).
- pytest Development Team (2024) *pytest documentation*. Available at: https://docs.pytest.org/ (Accessed: 15 June 2026).
- MacIver, D.R. (2024) *Hypothesis documentation*. Available at: https://hypothesis.readthedocs.io/ (Accessed: 15 June 2026).
