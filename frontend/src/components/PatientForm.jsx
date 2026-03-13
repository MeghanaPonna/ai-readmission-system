// src/components/PatientForm.jsx
// ------------------------------------------------------------------
// Multi-section patient data input form.
// ------------------------------------------------------------------
import React, { useState } from "react";

const AGE_BRACKETS = [
  "[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
  "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)",
];
const RACES    = ["Caucasian","AfricanAmerican","Hispanic","Asian","Other","Unknown"];
const GENDERS  = ["Male","Female","Unknown/Invalid"];
const MED_OPTS = ["No","Steady","Up","Down"];
const GLU_OPTS = ["None",">200",">300","Norm"];
const A1C_OPTS = ["None",">8",">7","Norm"];
const DIAG_CATS = ["Circulatory","Respiratory","Digestive","Diabetes",
                   "Injury","Musculoskeletal","Genitourinary","Neoplasms","Other","External","Unknown"];

const MED_FIELDS = [
  "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
  "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
  "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","insulin",
];

const COMBO_FIELDS = [
  "glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
  "metformin-rosiglitazone","metformin-pioglitazone",
];

const DEFAULT_FORM = {
  race: "Caucasian", gender: "Male", age: "[50-60)",
  admission_type_id: 1, discharge_disposition_id: 1, admission_source_id: 7,
  time_in_hospital: 3, num_lab_procedures: 43, num_procedures: 1,
  num_medications: 15, number_outpatient: 0, number_emergency: 0,
  number_inpatient: 0, number_diagnoses: 9,
  diag_1: "Circulatory", diag_2: "Circulatory", diag_3: "Other",
  max_glu_serum: "None", A1Cresult: "None",
  change: "No", diabetesMed: "Yes",
  metformin: "No", repaglinide: "No", nateglinide: "No",
  chlorpropamide: "No", glimepiride: "No", acetohexamide: "No",
  glipizide: "No", glyburide: "No", tolbutamide: "No",
  pioglitazone: "No", rosiglitazone: "No", acarbose: "No",
  miglitol: "No", troglitazone: "No", tolazamide: "No", insulin: "No",
  "glyburide-metformin": "No", "glipizide-metformin": "No",
  "glimepiride-pioglitazone": "No", "metformin-rosiglitazone": "No",
  "metformin-pioglitazone": "No",
};

function Field({ label, children }) {
  return (
    <div className="field-group">
      <label className="field-label">{label}</label>
      {children}
    </div>
  );
}

function Select({ name, value, options, onChange, numeric }) {
  return (
    <select
      className="input-control"
      name={name}
      value={value}
      onChange={e => onChange(name, numeric ? Number(e.target.value) : e.target.value)}
    >
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

function NumberInput({ name, value, min, max, onChange }) {
  return (
    <input
      type="number"
      className="input-control"
      name={name}
      value={value}
      min={min}
      max={max}
      onChange={e => onChange(name, Number(e.target.value))}
    />
  );
}

export default function PatientForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULT_FORM);

  const set = (name, value) => setForm(prev => ({ ...prev, [name]: value }));

  const handleSubmit = e => {
    e.preventDefault();
    // Rename hyphenated fields to match Pydantic aliases
    const payload = { ...form };
    COMBO_FIELDS.forEach(k => {
      const alias = k.replace(/-/g, "_");
      payload[alias] = payload[k];
      delete payload[k];
    });
    onSubmit(payload);
  };

  return (
    <form className="patient-form" onSubmit={handleSubmit}>

      {/* ── Section 1: Demographics ── */}
      <section className="form-section">
        <h3 className="section-title">
          <span className="section-icon">👤</span> Demographics
        </h3>
        <div className="form-grid-3">
          <Field label="Race">
            <Select name="race" value={form.race} options={RACES} onChange={set}/>
          </Field>
          <Field label="Gender">
            <Select name="gender" value={form.gender} options={GENDERS} onChange={set}/>
          </Field>
          <Field label="Age Bracket">
            <Select name="age" value={form.age} options={AGE_BRACKETS} onChange={set}/>
          </Field>
        </div>
      </section>

      {/* ── Section 2: Admission Details ── */}
      <section className="form-section">
        <h3 className="section-title">
          <span className="section-icon">🏥</span> Admission Details
        </h3>
        <div className="form-grid-3">
          <Field label="Admission Type (1-8)">
            <NumberInput name="admission_type_id" value={form.admission_type_id}
                         min={1} max={8} onChange={set}/>
          </Field>
          <Field label="Discharge Disposition (1-30)">
            <NumberInput name="discharge_disposition_id"
                         value={form.discharge_disposition_id}
                         min={1} max={30} onChange={set}/>
          </Field>
          <Field label="Admission Source (1-26)">
            <NumberInput name="admission_source_id" value={form.admission_source_id}
                         min={1} max={26} onChange={set}/>
          </Field>
        </div>
      </section>

      {/* ── Section 3: Hospital Stay ── */}
      <section className="form-section">
        <h3 className="section-title">
          <span className="section-icon">📊</span> Hospital Stay Metrics
        </h3>
        <div className="form-grid-4">
          {[
            ["time_in_hospital",    "Days in Hospital",   1, 14],
            ["num_lab_procedures",  "Lab Procedures",     0, 132],
            ["num_procedures",      "Procedures",         0, 6],
            ["num_medications",     "Medications",        0, 81],
            ["number_outpatient",   "Outpatient Visits",  0, 50],
            ["number_emergency",    "Emergency Visits",   0, 50],
            ["number_inpatient",    "Prior Inpatient",    0, 50],
            ["number_diagnoses",    "Diagnoses",          1, 16],
          ].map(([name, label, min, max]) => (
            <Field key={name} label={label}>
              <NumberInput name={name} value={form[name]} min={min} max={max} onChange={set}/>
            </Field>
          ))}
        </div>
      </section>

      {/* ── Section 4: Diagnoses & Labs ── */}
      <section className="form-section">
        <h3 className="section-title">
          <span className="section-icon">🔬</span> Diagnoses &amp; Lab Results
        </h3>
        <div className="form-grid-3">
          <Field label="Primary Diagnosis">
            <Select name="diag_1" value={form.diag_1} options={DIAG_CATS} onChange={set}/>
          </Field>
          <Field label="Secondary Diagnosis">
            <Select name="diag_2" value={form.diag_2} options={DIAG_CATS} onChange={set}/>
          </Field>
          <Field label="Tertiary Diagnosis">
            <Select name="diag_3" value={form.diag_3} options={DIAG_CATS} onChange={set}/>
          </Field>
          <Field label="Max Glucose Serum">
            <Select name="max_glu_serum" value={form.max_glu_serum}
                    options={GLU_OPTS} onChange={set}/>
          </Field>
          <Field label="HbA1c Result">
            <Select name="A1Cresult" value={form.A1Cresult}
                    options={A1C_OPTS} onChange={set}/>
          </Field>
          <Field label="Diabetes Med?">
            <Select name="diabetesMed" value={form.diabetesMed}
                    options={["Yes","No"]} onChange={set}/>
          </Field>
        </div>
      </section>

      {/* ── Section 5: Medications ── */}
      <section className="form-section">
        <h3 className="section-title">
          <span className="section-icon">💊</span> Medications
        </h3>
        <div className="form-grid-4">
          {MED_FIELDS.map(med => (
            <Field key={med} label={med.charAt(0).toUpperCase() + med.slice(1)}>
              <Select name={med} value={form[med]} options={MED_OPTS} onChange={set}/>
            </Field>
          ))}
        </div>
        <div className="form-grid-3" style={{ marginTop: "1rem" }}>
          <Field label="Medication Change">
            <Select name="change" value={form.change} options={["No","Ch"]} onChange={set}/>
          </Field>
          {COMBO_FIELDS.map(f => (
            <Field key={f} label={f}>
              <Select name={f} value={form[f]} options={MED_OPTS} onChange={set}/>
            </Field>
          ))}
        </div>
      </section>

      <div className="submit-row">
        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? (
            <><span className="spinner"/> Analyzing…</>
          ) : (
            <><span>⚡</span> Predict Readmission Risk</>
          )}
        </button>
      </div>

    </form>
  );
}
