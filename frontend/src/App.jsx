// src/App.jsx
import React, { useState } from "react";
import PatientForm    from "./components/PatientForm";
import ResultDisplay  from "./components/ResultDisplay";
import { predictReadmission } from "./services/api";
import "./App.css";

export default function App() {
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const handleSubmit = async (data) => {
    setLoading(true);
    setError(null);
    try {
      const prediction = await predictReadmission(data);
      setResult(prediction);
    } catch (err) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <span className="header-logo">🏥</span>
            <div>
              <h1 className="header-title">ReadmissionAI</h1>
              <p className="header-sub">
                AI-Powered 30-Day Patient Readmission Predictor
              </p>
            </div>
          </div>
          <div className="header-badge">
            <span className="badge-dot" />
            ML Model Active
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="app-main">

        {/* Info banner */}
        <div className="info-banner">
          <div className="info-grid">
            {[
              { icon: "🤖", label: "Model",   val: "XGBoost Classifier" },
              { icon: "📈", label: "Dataset", val: "Diabetes 130-US Hospitals" },
              { icon: "🎯", label: "ROC-AUC", val: "~0.83" },
              { icon: "📋", label: "Features",val: "47 clinical variables" },
            ].map(({ icon, label, val }) => (
              <div key={label} className="info-cell">
                <span className="info-icon">{icon}</span>
                <div>
                  <p className="info-label">{label}</p>
                  <p className="info-val">{val}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Error state */}
        {error && (
          <div className="error-banner">
            <span>❌</span>
            <div>
              <strong>Prediction Failed</strong>
              <p>{error}</p>
            </div>
            <button className="error-close" onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {/* Result or Form */}
        {result ? (
          <ResultDisplay result={result} onReset={handleReset} />
        ) : (
          <>
            <div className="form-header">
              <h2 className="form-title">Patient Health Data</h2>
              <p className="form-desc">
                Complete the form below to assess a patient's 30-day
                readmission risk using our trained ML model.
              </p>
            </div>
            <PatientForm onSubmit={handleSubmit} loading={loading} />
          </>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <p>
          ⚠️ For research &amp; educational purposes only. Not a substitute
          for clinical judgment.
        </p>
      </footer>
    </div>
  );
}
