# """
# main.py
# -------
# FastAPI backend for the AI Patient Readmission Prediction System.
# Exposes a single POST /predict endpoint that accepts patient health
# data and returns a readmission risk assessment.

# Run with:
#     uvicorn main:app --reload --port 8000
# """

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager
# import uvicorn

# from routes.prediction import router as prediction_router
# from model_loader import load_all_artifacts

# # ── App lifecycle ──────────────────────────────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Load ML artifacts once at startup."""
#     print("[startup] Loading ML model and preprocessing artifacts …")
#     app.state.artifacts = load_all_artifacts()
#     print("[startup] Artifacts ready ✓")
#     yield
#     print("[shutdown] Cleaning up …")


# # ── FastAPI instance ───────────────────────────────────────────────────────────
# app = FastAPI(
#     title="AI Patient Readmission Prediction API",
#     description=(
#         "Predicts the probability that a hospital patient will be "
#         "readmitted within 30 days of discharge."
#     ),
#     version="1.0.0",
#     lifespan=lifespan,
# )

# # ── CORS — allow React dev server ──────────────────────────────────────────────
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ── Routers ────────────────────────────────────────────────────────────────────
# app.include_router(prediction_router, prefix="/api", tags=["Prediction"])


# # ── Health check ───────────────────────────────────────────────────────────────
# @app.get("/health", tags=["Health"])
# def health():
#     return {"status": "ok", "model_loaded": True}


# # ── Entry point ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




"""
main.py
-------
FastAPI backend for the AI Patient Readmission Prediction System.

Exposes:
    POST /api/predict  -> Predict patient readmission risk
    GET  /health       -> Health check endpoint

Run locally with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from routes.prediction import router as prediction_router
from model_loader import load_all_artifacts


# ─────────────────────────────────────────────────────────────
# Application lifecycle
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model and preprocessing artifacts at startup."""
    print("[startup] Loading ML model and preprocessing artifacts …")

    app.state.artifacts = load_all_artifacts()

    print("[startup] Artifacts ready ✓")
    yield
    print("[shutdown] Cleaning up …")


# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Patient Readmission Prediction API",
    description="Predicts whether a patient will be readmitted within 30 days.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────
# CORS Configuration
# ─────────────────────────────────────────────────────────────
origins = [
    "http://localhost:3000",  # local React development
    "https://ai-readmission-system.vercel.app",  # deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────
app.include_router(
    prediction_router,
    prefix="/api",
    tags=["Prediction"]
)


# ─────────────────────────────────────────────────────────────
# Health Check Endpoint
# ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    """Simple health check for deployment monitoring."""
    return {
        "status": "ok",
        "model_loaded": True
    }


# ─────────────────────────────────────────────────────────────
# Run locally
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
