"""
main.py
-------
FastAPI backend for the AI Patient Readmission Prediction System.
Exposes a single POST /predict endpoint that accepts patient health
data and returns a readmission risk assessment.

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from routes.prediction import router as prediction_router
from model_loader import load_all_artifacts

# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts once at startup."""
    print("[startup] Loading ML model and preprocessing artifacts …")
    app.state.artifacts = load_all_artifacts()
    print("[startup] Artifacts ready ✓")
    yield
    print("[shutdown] Cleaning up …")


# ── FastAPI instance ───────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Patient Readmission Prediction API",
    description=(
        "Predicts the probability that a hospital patient will be "
        "readmitted within 30 days of discharge."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow React dev server ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(prediction_router, prefix="/api", tags=["Prediction"])


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": True}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
