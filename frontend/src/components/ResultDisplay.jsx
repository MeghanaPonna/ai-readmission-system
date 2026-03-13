// src/components/ResultDisplay.jsx
// ------------------------------------------------------------------
// Displays the prediction result with risk level, probability gauge,
// and clinical recommendation.
// ------------------------------------------------------------------
import React, { useEffect, useState } from "react";

const RISK_CONFIG = {
  Low: {
    color: "#22c55e",
    bg:    "#f0fdf4",
    border:"#86efac",
    icon:  "✅",
    label: "LOW RISK",
  },
  Medium: {
    color: "#f59e0b",
    bg:    "#fffbeb",
    border:"#fcd34d",
    icon:  "⚠️",
    label: "MEDIUM RISK",
  },
  High: {
    color: "#ef4444",
    bg:    "#fef2f2",
    border:"#fca5a5",
    icon:  "🚨",
    label: "HIGH RISK",
  },
};

function GaugeBar({ score, color }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(score), 100);
    return () => clearTimeout(t);
  }, [score]);

  return (
    <div className="gauge-wrapper">
      <div className="gauge-track">
        <div
          className="gauge-fill"
          style={{
            width: `${width}%`,
            background: color,
            transition: "width 1s cubic-bezier(0.34, 1.56, 0.64, 1)",
          }}
        />
      </div>
      <div className="gauge-labels">
        <span>0%</span>
        <span style={{ color, fontWeight: 700 }}>{score}%</span>
        <span>100%</span>
      </div>
    </div>
  );
}

export default function ResultDisplay({ result, onReset }) {
  if (!result) return null;

  const { readmission_risk, probability, risk_score, recommendation } = result;
  const cfg = RISK_CONFIG[readmission_risk] || RISK_CONFIG.Medium;

  return (
    <div
      className="result-card"
      style={{ borderColor: cfg.border, background: cfg.bg }}
    >
      {/* Header */}
      <div className="result-header">
        <span className="result-icon">{cfg.icon}</span>
        <div>
          <p className="result-subtitle">Readmission Risk Assessment</p>
          <h2
            className="result-risk-label"
            style={{ color: cfg.color }}
          >
            {cfg.label}
          </h2>
        </div>
      </div>

      {/* Probability gauge */}
      <div className="result-gauge-section">
        <p className="gauge-title">
          30-Day Readmission Probability
          <span className="prob-value" style={{ color: cfg.color }}>
            {(probability * 100).toFixed(1)}%
          </span>
        </p>
        <GaugeBar score={risk_score} color={cfg.color} />
      </div>

      {/* Stats row */}
      <div className="result-stats">
        <div className="stat-box" style={{ borderColor: cfg.border }}>
          <p className="stat-label">Probability Score</p>
          <p className="stat-value" style={{ color: cfg.color }}>
            {probability.toFixed(4)}
          </p>
        </div>
        <div className="stat-box" style={{ borderColor: cfg.border }}>
          <p className="stat-label">Risk Category</p>
          <p className="stat-value" style={{ color: cfg.color }}>
            {readmission_risk}
          </p>
        </div>
        <div className="stat-box" style={{ borderColor: cfg.border }}>
          <p className="stat-label">Risk Score</p>
          <p className="stat-value" style={{ color: cfg.color }}>
            {risk_score} / 100
          </p>
        </div>
      </div>

      {/* Recommendation */}
      <div
        className="recommendation-box"
        style={{ borderLeft: `4px solid ${cfg.color}` }}
      >
        <h4 className="rec-title">
          <span>📋</span> Clinical Recommendation
        </h4>
        <p className="rec-text">{recommendation}</p>
      </div>

      <button className="reset-btn" onClick={onReset}>
        ← Assess Another Patient
      </button>
    </div>
  );
}
