"""FastAPI inference service.

    GET  /health         liveness: the process is up (never touches the model)
    GET  /ready          readiness: the model is loaded and can score (503 until then)
    GET  /metadata       which model/policy is serving
    POST /predict        one LoanApplication -> PD, decision, reason codes, policy flags
    POST /predict/batch  up to MAX_BATCH applications

Config (env vars): RISKFLUX_MODEL_DIR (default: models), PREDICTION_LOG_PATH (optional
JSONL mirror of prediction logs), LOG_LEVEL (default: INFO).

Run locally: uv run uvicorn riskflux.serving.app:app --port 8080
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from riskflux.features.engineer import MODEL_FEATURES
from riskflux.schemas import LoanApplication
from riskflux.serving.logs import PredictionSink, configure_logging
from riskflux.serving.predictor import ModelBundle, Predictor

MAX_BATCH = 1_000
log = structlog.get_logger()


class RiskFactor(BaseModel):
    feature: str
    description: str
    contribution: float = Field(description="Push toward default, in log-odds")


class Prediction(BaseModel):
    request_id: str
    pd: float = Field(description="Calibrated probability of default")
    decision: Literal["approve", "reject", "refer"]
    threshold: float
    expected_loss: float = Field(description="PD x LGD x loan amount, in dollars")
    risk_factors: list[RiskFactor]
    policy_flags: list[str] = Field(
        description="Reasons the applicant is outside the model's range"
    )
    model_version: str


class BatchRequest(BaseModel):
    applications: list[LoanApplication] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


def create_app(model_dir: Path | None = None, prediction_log: Path | None = None) -> FastAPI:
    model_dir = model_dir or Path(os.environ.get("RISKFLUX_MODEL_DIR", "models"))
    if prediction_log is None and os.environ.get("PREDICTION_LOG_PATH"):
        prediction_log = Path(os.environ["PREDICTION_LOG_PATH"])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
        app.state.predictor = None
        app.state.sink = PredictionSink(prediction_log)
        try:
            bundle = ModelBundle.load(model_dir)
            app.state.predictor = Predictor(bundle)
            log.info(
                "model_loaded",
                model_dir=str(model_dir),
                model_version=bundle.version,
                threshold=bundle.policy["threshold"],
            )
        except Exception:
            # Stay up so /health answers and /ready reports 503 with the cause in the logs.
            log.exception("model_load_failed", model_dir=str(model_dir))
        yield

    app = FastAPI(title="RiskFlux", version="1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        return response

    def predictor(request: Request) -> Predictor:
        if request.app.state.predictor is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return request.app.state.predictor

    def score(request: Request, applications: list[LoanApplication]) -> list[Prediction]:
        request_id = structlog.contextvars.get_contextvars()["request_id"]
        results = predictor(request).predict(applications)
        for application, result in zip(applications, results, strict=True):
            fields = application.model_dump(mode="json")
            log.info(
                "prediction",
                model_version=result["model_version"],
                pd=result["pd"],
                decision=result["decision"],
                policy_flags=result["policy_flags"],
                features=fields,
            )
            request.app.state.sink.write(request_id, fields, result)
        return [Prediction(request_id=request_id, **r) for r in results]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        if request.app.state.predictor is None:
            return JSONResponse(status_code=503, content={"status": "model not loaded"})
        return {"status": "ready", "model_version": request.app.state.predictor.bundle.version}

    @app.get("/metadata")
    def metadata(request: Request) -> dict:
        bundle = predictor(request).bundle
        return {
            "model_version": bundle.version,
            "registry": bundle.metadata,
            "threshold": bundle.policy["threshold"],
            "cost_model": bundle.policy["cost_model"],
            "n_features": len(MODEL_FEATURES),
        }

    @app.post("/predict", response_model=Prediction)
    def predict(request: Request, application: LoanApplication) -> Prediction:
        return score(request, [application])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: Request, body: BatchRequest) -> BatchResponse:
        return BatchResponse(predictions=score(request, body.applications))

    return app


app = create_app()
