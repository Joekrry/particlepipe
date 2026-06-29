"""FastAPI REST API for the ParticlePipe platform.

Exposes endpoints for running the generation and trigger pipeline, streaming
events over Server-Sent Events, inspecting single events, and retrieving
analysis results and fitted mass spectra. This is the only module permitted to
import FastAPI and Pydantic; all request and response bodies are typed with
Pydantic v2 models.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.analysis import PhysicsAnalysis
from src.core.event import Event, TriggerBit
from src.pipeline.generator import CollisionGenerator, GeneratorConfig
from src.pipeline.trigger import PipelineStats, TriggerPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AppState:
    """Singleton application state shared across requests."""

    generator: CollisionGenerator
    pipeline: TriggerPipeline
    analysis: PhysicsAnalysis
    is_running: bool = False


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Construct the generator, pipeline and analysis singletons on startup."""
    logger.info("Initialising ParticlePipe services")
    app_state.generator = CollisionGenerator(GeneratorConfig(seed=42))
    app_state.pipeline = TriggerPipeline(n_workers=4)
    app_state.analysis = PhysicsAnalysis()
    logger.info("ParticlePipe ready: LHC Run 3 simulation at sqrt(s) = 13.6 TeV")
    yield
    logger.info("Shutting down ParticlePipe")


app = FastAPI(
    title="ParticlePipe - HEP Data Pipeline API",
    description=(
        "High-energy physics event simulation, trigger pipeline, and analysis "
        "platform. Simulates LHC-like pp collisions at sqrt(s) = 13.6 TeV."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic response models ---
class ParticleOut(BaseModel):
    """Serialised particle for API responses."""

    model_config = ConfigDict(from_attributes=True)

    particle_id: str
    type: str
    pt: float = Field(description="Transverse momentum (GeV/c)")
    eta: float = Field(description="Pseudorapidity")
    phi: float = Field(description="Azimuthal angle (radians)")
    energy: float = Field(description="Energy (GeV)")
    charge: int
    status: int
    is_reconstructed: bool
    detector_hits: int
    track_chi2: float
    isolation: float


class EventOut(BaseModel):
    """Serialised event for API responses."""

    event_id: str
    run_number: int
    event_number: int
    sqrt_s_gev: float
    n_particles: int
    n_charged: int
    scalar_ht_gev: float
    missing_et_gev: float
    passed_l1: bool
    passed_l2: bool
    passed_l3: bool
    trigger_bits: list[str]
    vertex_x_mm: float
    vertex_y_mm: float
    vertex_z_mm: float
    particles: list[ParticleOut]
    processing_latency_ms: Optional[float]

    @classmethod
    def from_event(cls, event: Event) -> EventOut:
        aggregates = {TriggerBit.NONE, TriggerBit.ANY_L1, TriggerBit.ANY_L2, TriggerBit.ANY_L3}
        trigger_bits = [
            bit.name for bit in TriggerBit
            if bit not in aggregates and (event.trigger & bit)
        ]
        return cls(
            event_id=event.event_id,
            run_number=event.metadata.run_number,
            event_number=event.metadata.event_number,
            sqrt_s_gev=event.metadata.sqrt_s,
            n_particles=event.n_particles,
            n_charged=event.n_charged,
            scalar_ht_gev=event.scalar_ht,
            missing_et_gev=event.missing_et,
            passed_l1=event.passed_l1(),
            passed_l2=event.passed_l2(),
            passed_l3=event.passed_l3(),
            trigger_bits=trigger_bits,
            vertex_x_mm=event.vertex.x,
            vertex_y_mm=event.vertex.y,
            vertex_z_mm=event.vertex.z,
            particles=[
                ParticleOut(
                    particle_id=p.particle_id,
                    type=p.ptype.value,
                    pt=p.pt,
                    eta=p.eta,
                    phi=p.phi,
                    energy=p.energy,
                    charge=p.charge,
                    status=p.status,
                    is_reconstructed=p.is_reconstructed,
                    detector_hits=p.detector_hits,
                    track_chi2=p.track_chi2,
                    isolation=p.isolation,
                )
                for p in event.particles
            ],
            processing_latency_ms=event.processing_latency_ms,
        )


class PipelineStatsOut(BaseModel):
    """Pipeline performance metrics."""

    n_input: int
    n_l1_pass: int
    n_l2_pass: int
    n_l3_pass: int
    l1_efficiency: float
    l2_efficiency: float
    l3_efficiency: float
    throughput_hz: float
    mean_latency_ms: float

    @classmethod
    def from_stats(cls, stats: PipelineStats) -> PipelineStatsOut:
        return cls(
            n_input=stats.n_input,
            n_l1_pass=stats.n_l1_pass,
            n_l2_pass=stats.n_l2_pass,
            n_l3_pass=stats.n_l3_pass,
            l1_efficiency=stats.l1_rate,
            l2_efficiency=stats.l2_rate,
            l3_efficiency=stats.l3_rate,
            throughput_hz=stats.throughput_hz,
            mean_latency_ms=stats.mean_latency_ms,
        )


class RunRequest(BaseModel):
    """Request body for the /run endpoint."""

    n_events: int = Field(default=1000, ge=1, le=100_000,
                          description="Number of events to generate and process")
    z_fraction: float = Field(default=0.05, ge=0.0, le=1.0,
                              description="Fraction of events with a Z boson")
    jpsi_fraction: float = Field(default=0.03, ge=0.0, le=1.0)
    higgs_fraction: float = Field(default=0.001, ge=0.0, le=0.1)
    seed: int = Field(default=42, description="Random seed for reproducibility")


class FitResultOut(BaseModel):
    """Gaussian peak fit result."""

    peak_name: str
    mean_gev: float
    sigma_gev: float
    amplitude: float
    chi2_per_ndf: float
    n_signal: float
    significance: float
    resolution_percent: float


class AnalysisSummaryOut(BaseModel):
    """Full analysis summary with histograms and fits."""

    n_events_total: int
    n_z_candidates: int
    n_jpsi_candidates: int
    n_higgs_candidates: int
    z_peak_fit: Optional[FitResultOut]
    jpsi_peak_fit: Optional[FitResultOut]
    higgs_peak_fit: Optional[FitResultOut]
    histograms: dict


# --- Endpoints ---
@app.get("/", summary="Service info")
async def root() -> dict:
    return {
        "service": "ParticlePipe",
        "status": "operational",
        "sqrt_s_TeV": 13.6,
        "experiment": "LHC Run 3 Simulation",
    }


@app.post("/run", response_model=dict, summary="Run the generation and trigger pipeline")
async def run_pipeline(req: RunRequest) -> dict:
    """Generate N events, process them through the trigger, and accumulate analysis.

    Runs synchronously and returns the pipeline statistics. A concurrent run is
    rejected with HTTP 409.
    """
    if app_state.is_running:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    config = GeneratorConfig(
        seed=req.seed,
        z_fraction=req.z_fraction,
        jpsi_fraction=req.jpsi_fraction,
        higgs_fraction=req.higgs_fraction,
    )
    app_state.generator = CollisionGenerator(config)
    app_state.pipeline = TriggerPipeline(n_workers=4)
    app_state.analysis = PhysicsAnalysis()
    app_state.is_running = True

    try:
        t0 = time.perf_counter()
        for event in app_state.generator.generate(req.n_events):
            event = await app_state.pipeline.process_one(event)
            app_state.analysis.process_event(event)
        elapsed = time.perf_counter() - t0
    finally:
        app_state.is_running = False

    return {
        "status": "completed",
        "n_events": req.n_events,
        "elapsed_s": round(elapsed, 3),
        "pipeline": PipelineStatsOut.from_stats(app_state.pipeline.stats).model_dump(),
    }


@app.get("/events/stream", summary="Stream events in real time via SSE")
async def stream_events(
    n: int = Query(default=50, ge=1, le=500, description="Number of events to stream"),
    l3_only: bool = Query(default=True, description="Only stream events passing L3"),
) -> StreamingResponse:
    """Stream processed collision events as Server-Sent Events."""

    async def event_generator() -> AsyncIterator[str]:
        gen = CollisionGenerator(GeneratorConfig(seed=int(time.time())))
        pipe = TriggerPipeline()
        sent = 0
        # Generate a surplus so the L3 filter can still reach n events.
        for raw_event in gen.generate(n * 10):
            if sent >= n:
                break
            processed = await pipe.process_one(raw_event)
            if l3_only and not processed.passed_l3():
                continue
            yield f"data: {EventOut.from_event(processed).model_dump_json()}\n\n"
            sent += 1
            await asyncio.sleep(0.005)   # pace output for dashboards
        yield 'data: {"type": "END_OF_STREAM"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/events/generate", response_model=EventOut, summary="Generate and process one event")
async def generate_single_event() -> EventOut:
    """Generate, trigger-process, and return a single collision event."""
    event = app_state.generator.generate_one()
    event = await app_state.pipeline.process_one(event)
    app_state.analysis.process_event(event)
    return EventOut.from_event(event)


@app.get("/analysis", response_model=AnalysisSummaryOut, summary="Analysis results and peak fits")
async def get_analysis() -> AnalysisSummaryOut:
    """Return candidate yields, histograms, and Gaussian fits once enough candidates exist."""
    if app_state.analysis.n_events_total == 0:
        raise HTTPException(status_code=404, detail="No events processed yet. Call POST /run first.")

    summary = app_state.analysis.summary()

    z_fit = jpsi_fit = higgs_fit = None
    if app_state.analysis.n_z_candidates >= 10:
        z_fit = _fit_out("Z -> mu+ mu-", app_state.analysis.fit_z_peak())
    if app_state.analysis.n_jpsi_candidates >= 5:
        jpsi_fit = _fit_out("J/psi -> mu+ mu-", app_state.analysis.fit_jpsi_peak())
    if app_state.analysis.n_higgs_candidates >= 3:
        higgs_fit = _fit_out("H -> gamma gamma", app_state.analysis.fit_higgs_peak())

    return AnalysisSummaryOut(
        n_events_total=summary["n_events_total"],
        n_z_candidates=summary["n_z_candidates"],
        n_jpsi_candidates=summary["n_jpsi_candidates"],
        n_higgs_candidates=summary["n_higgs_candidates"],
        z_peak_fit=z_fit,
        jpsi_peak_fit=jpsi_fit,
        higgs_peak_fit=higgs_fit,
        histograms=summary["histograms"],
    )


def _fit_out(peak_name: str, r) -> FitResultOut:
    return FitResultOut(
        peak_name=peak_name,
        mean_gev=r.mean,
        sigma_gev=r.sigma,
        amplitude=r.amplitude,
        chi2_per_ndf=r.chi2_per_ndf,
        n_signal=r.n_signal,
        significance=r.significance,
        resolution_percent=r.resolution_percent,
    )


@app.get("/pipeline/stats", response_model=PipelineStatsOut, summary="Pipeline performance metrics")
async def get_pipeline_stats() -> PipelineStatsOut:
    """Return the current trigger pipeline performance metrics."""
    return PipelineStatsOut.from_stats(app_state.pipeline.stats)


@app.delete("/analysis/reset", summary="Reset analysis and pipeline state")
async def reset_analysis() -> dict:
    """Clear accumulated analysis histograms and pipeline counters."""
    app_state.analysis = PhysicsAnalysis()
    app_state.pipeline = TriggerPipeline()
    return {"status": "reset", "message": "Analysis histograms cleared"}


@app.get("/health", summary="Detailed health check")
async def health() -> dict:
    return {
        "status": "ok",
        "events_processed": app_state.analysis.n_events_total,
        "pipeline_running": app_state.is_running,
        "l3_pass_rate": app_state.pipeline.stats.l3_rate,
    }
