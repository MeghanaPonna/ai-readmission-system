// src/services/api.js
// ------------------------------------------------------------------
// Thin wrapper around the FastAPI /api/predict endpoint.
// ------------------------------------------------------------------

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

/**
 * Send patient data to the ML backend and return a prediction.
 *
 * @param {Object} patientData  - Object matching the PatientData Pydantic schema
 * @returns {Promise<{readmission_risk, probability, risk_score, recommendation}>}
 */
export async function predictReadmission(patientData) {
  const response = await fetch(`${BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patientData),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error: ${response.status}`);
  }

  return response.json();
}

/**
 * Health-check — verifies the backend is reachable.
 */
export async function checkHealth() {
  const response = await fetch(`${BASE_URL}/health`);
  return response.json();
}
