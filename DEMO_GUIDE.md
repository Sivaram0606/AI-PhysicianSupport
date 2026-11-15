# Live Demo Guide & Presenter Talk Track — AI Physician Emergency Support

This document provides a **complete live demonstration script, click-by-click walkthrough, and presenter talk track** for showcasing the **Emergency Physician Assist UI** ([`physician_assist_app.py`](physician_assist_app.py)) and **Google ADK Multi-Agent Clinical Pipelines** ([`ehr_chart_summary_agent`](ehr_chart_summary_agent/agent.py)).

> **Companion Slide Deck**: Open [`DEMO_SLIDES.html`](DEMO_SLIDES.html) in any browser for a 10-slide interactive presentation deck (`←` / `→` keys to navigate, `N` to toggle Speaker Notes, `F` for Fullscreen, `Cmd/Ctrl + P` to export to PDF).

---

## 1. 60-Second Executive Framing (Opening Pitch)

> *"When a complex patient arrives in the Emergency Department, critical clues are buried across years of fragmented EHR notes, outside urgent-care prescriptions, OTC brand-vs-generic medications, unreviewed send-out labs, and active neighborhood respiratory surges. Dumping a 20-page summary overwhelms the physician.*
>
> *We built **Emergency Physician Assist** on **Google ADK**, **Vertex AI (`gemini-3.8-flash`)**, and **Google Cloud Healthcare HL7 FHIR R4** to mirror how an attending ED physician actually thinks—progressively:*
> 1. *Select the patient from the live **Emergency Ward Roster** and review **Longitudinal Medical Charts & Lab Trajectories**.*
> 2. *Correlate events across time on an **Interactive 4-Track Clinical Swimlane**.*
> 3. *Audit **Brand vs. Generic Drugs & OpenFDA Adverse Effects**—and only after the **Physician Validation Gate** confirms unexplained symptoms, unlock **Step 4: Hyperlocal 5-Digit ZIP-Code Outbreak Surveillance**.*
> 4. *Throughout every step, a dedicated **Google ADK Symptom Mapping Agent UI** tracks which symptoms are explained by EHR history, drug adverse effects, or local outbreaks in real time."*

---

## 2. Pre-Demo Checklist (30 Seconds Before Going Live)

Choose either the **Cloud Run URL** or **Local Streamlit**:

- **Cloud Run (Live Cloud Environment)**:
  1. Open **`https://emergency-physician-assist-<PROJECT_NUMBER>.us-central1.run.app`** in Chrome while signed into your authorized Google account (IAP authenticates automatically).
  2. *(Optional warm-up)* Click once on the ward roster 30 seconds before presenting so the container is warm.
- **Local Backup (Zero-Dependency Fallback)**:
  ```bash
  .venv/bin/streamlit run physician_assist_app.py
  ```
  Open **`http://localhost:8501`**.

---

## 3. Flagship 7-Minute Live Demo Walkthrough (`pat-001` — Alex Jordan Conn)

**Why lead with `pat-001`?** Alex showcases **every single capability** end-to-end: longitudinal asthma/liver lab trends, a 4-track causal swimlane, a hidden **Brand vs. Generic Duplicate Active Ingredient hazard**, an **ACE-inhibitor cough vs. infectious fever split**, the **Physician Validation Gate**, **Boston `02118` wastewater/ED outbreak correlation**, and progressive **ADK Symptom Mapping**.

---

### Act 1 — Emergency Ward Patient Roster (1 min)

1. **Where to click**: Start on Tab **`1 · Emergency Ward Patient Roster`**.
2. **What to point out on screen**:
   - Live **7-patient Emergency Ward cohort (`pat-001` – `pat-007`)** pulled from **Google Cloud Healthcare FHIR R4 (`synthe_data`)**.
   - Each card surfaces **Triage Vitals**, **Chief Complaint**, **High-Criticality Allergies**, and the patient's **5-Digit Residential ZIP Code**.
   - Point out **`pat-001` — Alex Jordan Conn (48M, Boston, MA `02118`)**: presenting with *acute shortness of breath, wheezing, low-grade fever (`100.2°F`), `SpO2 91%`, and a 3-day dry cough*, plus a high-criticality **Sulfonamide allergy**.
3. **Action**: Click **Select Patient** on **`pat-001` (Alex Jordan Conn)** (or switch to Tab **`2 · Guided Clinical Investigation (Alex Jordan Conn)`**).

---

### Act 2 — Step 1: Longitudinal Medical Charts + Side-by-Side Symptom Panel (1.5 min)

1. **What to point out on the Left (Main Clinical Workspace — Step 1)**:
   - **Notice the Sequence**: Drug checks and ZIP outbreaks are *intentionally not dumped upfront*—Steps 3 and 4 remain gated while the physician reviews baseline longitudinal context first.
   - **Longitudinal Lab Trajectories (Plotly Curves)**:
     - `Peak Expiratory Flow` dropping `88% -> 54% predicted`
     - `Absolute Eosinophils` rising `280 -> 520 /uL`
     - Mild new transaminitis: `ALT 22 -> 58 U/L` (plant this clue for Step 3!).
   - **Diagnostic Imaging, Hospitalization Notes & Open Clinical Loops**: Chest X-ray hyperinflation without focal lobar consolidation, prior asthma/HTN visits, and open care-gap flags—all citing exact HL7 FHIR Resource IDs (`Observation/...`, `DiagnosticReport/...`, `Flag/...`).
2. **What to point out on the Right (`Symptom Mapping Agent UI` Side Panel)**:
   - Point out that symptoms are mapped **separately in the UI** alongside the workflow.
   - At Stage 1, `Wheezing / Peak Flow 54%` maps to intrinsic **Asthma (`MATCHED_EHR`)**, whereas `Low-Grade Fever (100.2°F)` and `ALT Elevation (58 U/L)` remain flagged for downstream drug/infectious investigation!
3. **Action**: Click **`Proceed to Step 2: Review 4-Track Clinical Swimlane Chart for Alex Jordan Conn →`**.

---

### Act 3 — Step 2: Interactive 4-Track Clinical Swimlane Chart (1 min)

1. **What to point out on screen**:
   - The **CSP-sandboxed Interactive HTML/SVG 4-Track Swimlane Viewer** synchronizes four parallel tracks across shared dates:
     - **Track 1**: Encounters, Admissions & Triage
     - **Track 2**: Medication Changes & Real-World Adherence
     - **Track 3**: Longitudinal Labs, Vitals & Imaging
     - **Track 4**: Open Clinical Loops & Safety Alerts
   - Toggle **`Causal Connectors: ON`** / **`Critical Only`** inside the interactive viewer to highlight the temporal link: **Lisinopril dose doubled (`5 mg -> 10 mg`) on Oct 8** $\rightarrow$ preceded the non-productive cough!
2. **Action**: Click **`Swimlane Reviewed → Initiate Step 3: Check Drugs & Adverse Effects for Alex Jordan Conn →`**.

---

### Act 4 — Step 3: Drug Brand/Generic Audit, Duplicate Ingredient Hazard & Validation Gate (2 min)

1. **What to point out on screen ("Aha!" Moment #1 — Medication Safety)**:
   - **Red Safety Alert Banner**: **`DUPLICATE ACTIVE INGREDIENT SAFETY HAZARD · ACETAMINOPHEN`**!
     - **NIH RxNorm** normalized **Tylenol 500 MG** as **`BRAND NAME`** (`ingredient: Acetaminophen`) and caught that Alex was simultaneously taking leftover **`ORIGINAL GENERIC` Acetaminophen** tablets at home—explaining his `ALT 22 -> 58 U/L` transaminitis!
   - **OpenFDA Label & FAERS Chart**:
     - **Lisinopril** (`ORIGINAL GENERIC`) lists **dry bradykinin cough** in OpenFDA SPL warnings and top FAERS reports.
2. **Demonstrate Bedside Physician Judgment (The Validation Gate)**:
   - Explain: *"Lisinopril explains chronic dry cough and Tylenol+Acetaminophen explains mild ALT elevation—but **neither drug causes a `100.2°F` fever, `SpO2 91%` hypoxia, and acute viral bronchospasm**!"*
   - Select **`Rule Out (No Match)`** on medications as sole cause (or enter a quick note in **`Optional Physician Validation Note`**, e.g., *"Fever 100.2F & acute hypoxia not explained by meds — check Boston 02118 outbreaks"*).
3. **Action**: Click **`✓ Validate: Nothing Matches / Unexplained Symptoms → Check ZIP 02118 Outbreaks →`**.

---

### Act 5 — Step 4: Hyperlocal 5-Digit ZIP-Code Outbreak Surveillance & Live ADK Agent Synthesis (1.5 min)

1. **What to point out on screen ("Aha!" Moment #2 — Hyperlocal Epidemiology)**:
   - Step 4 unlocks and queries **strictly 5-digit ZIP `02118` (Boston, Suffolk County, MA)** in parallel across four live public health feeds:
     - **CDC NSSP Local Hospital ED Trajectories (`rdmq-nq56`)**: Respiratory ED visit % and trajectory (`Increasing` / `No Change` / `Decreasing`) for **COVID-19, Influenza, and RSV**.
     - **CDC NWSS Local Sewershed Wastewater PCR**: Viral surges in the local sewershed serving `02118`.
     - **CDC PLACES ZCTA5 (`qnzd-25i4`)**: Exact `02118` neighborhood asthma/COPD baseline prevalence.
     - **Open-Meteo Hyperlocal Air Quality**: Real-time `US AQI`, `PM2.5`, and `Ozone` at `02118` coordinates.
2. **Run the Live Google ADK Symptom Mapping Pipeline**:
   - Switch to Tab **`3 · Dedicated Symptom Mapping Matrix UI`** (or use the right-hand Symptom Panel).
   - Click **`✦ Run Live ADK Symptom Mapper Agent (Stage 4)`**.
   - Show how Google ADK's `symptom_mapping_pipeline` (`ParallelAgent` intake $\rightarrow$ `clinical_symptom_mapper_agent` on `gemini-3.8-flash`) synthesizes the complete **3-bucket differential**:
     1. **Intrinsic EHR**: Underlying Asthma exacerbation (`PEF 54%`, `Eos 520`)
     2. **Drug Adverse Effect**: Duplicate `Tylenol` + `Acetaminophen` (`ALT 58`) & `Lisinopril` cough contribution
     3. **ZIP `02118` Outbreak Trigger**: Acute viral respiratory trigger (`Fever 100.2°F`, `SpO2 91%`) — guiding targeted **Flu/COVID/RSV multiplex PCR** + holding Lisinopril/Acetaminophen (while avoiding **Sulfonamides**!).

---

## 4. Two Fast Contrast Scenarios (90 Seconds Each)

If the audience asks *"What happens when drugs DO 100% explain the presentation?"* or *"How does it handle complex multi-hospital heart failure?"*, switch patients in the top dropdown:

### Contrast Case A — `pat-002` (Maria Elena Reynolds, `97205`) — Pure Drug-Drug Toxicity Stopped at Step 3
- **Presentation**: 62F with proximal thigh weakness, dark cola-colored urine, `CK 110 -> 4,850 U/L`, and `Creatinine 0.85 -> 1.92 mg/dL`.
- **Swimlane + Drug Audit Payoff**: Outside Urgent Care prescribed **Clarithromycin** (strong CYP3A4 inhibitor) without holding brand **Lipitor (Atorvastatin 80mg)** $\rightarrow$ **Statin-Induced Rhabdomyolysis & Myoglobinuric AKI** (requiring immediate hold of both **Lipitor** and **Metformin ER**).
- **Demo Highlight**: Click **`Log Confirmed Drug Adverse-Effect Match Only`** at Step 3—showing the physician does *not* need to chase neighborhood infections when Step 3 proves a definitive drug interaction!

### Contrast Case B — `pat-005` (Arthur Pendelton, `60611`) — Adherence Barriers, Unlocked Held GDMT & Open Loops
- **Presentation**: 68M with HFrEF (`EF 30%`), `+7.5 lb` edema, `NT-proBNP 920 -> 1,850 pg/mL`.
- **Longitudinal + Swimlane Payoff**:
  1. Self-cut **Lasix** `40mg -> 20mg` due to urinary frequency and never started **Jardiance** (`$180/mo` Prior Auth barrier).
  2. **Lisinopril** and **Metformin** were held during a prior hospital AKI (`Cr 2.30`), **but today's `Cr 1.50` (`eGFR 48`) crosses the exact nephrology restart threshold**—unlocking GDMT re-initiation!
  3. Surfaces a **dropped open loop**: unordered 6-month follow-up chest CT for an incidental `7mm` RUL lung nodule.

---

## 5. Bonus Track — Conversational ADK Web Demo (`adk web`)

If demonstrating conversational multi-agent orchestration for technical/AI audiences (`.venv/bin/adk web` on `http://127.0.0.1:8000`):

1. **`Hi`** $\rightarrow$ Agent greets physician and renders the live 7-patient FHIR ward table.
2. **`Select pat-007 (Marcus Vance)`** $\rightarrow$ Surfaces ER-prescribed potassium-wasting **Chlorthalidone** despite `K+ 3.2 mEq/L` AND catches an **unreviewed August send-out `Aldosterone-to-Renin Ratio = 95`** confirming **Primary Hyperaldosteronism (Conn's Syndrome)**.
3. **`Generate a complete 1-page physician chart brief for pat-007`** $\rightarrow$ Delegates to `chart_review_pipeline` (`SequentialAgent` + `ParallelAgent`) to synthesize a structured 1-page brief.

---

## 6. Audience Q&A Cheat Sheet (Architecture & Security)

- **Q: How do you avoid slow FHIR latency when multiple sub-agents run in parallel?**
  - **A**: Single-flight per-patient locking (`_FHIR_INFLIGHT_LOCKS`) + 90s TTL caching collapses 6 concurrent compartment reads into **1** GCP Cloud Healthcare `Patient/$everything` HTTP call (`<0.1 ms` warm), backed by a pooled singleton OAuth2 `AuthorizedSession`.
- **Q: How many LLM round-trips does medication auditing take?**
  - **A**: Just **1 tool call** (`audit_patient_medications_rxnorm_openfda`), which fans out RxNorm (`properties` + `related`) and OpenFDA (`SPL` + `FAERS`) queries across all active and stopped medications concurrently in Python threads.
- **Q: How is PHI protected in Cloud Run and the UI?**
  - **A**: Cloud Run runs behind native **Identity-Aware Proxy (`--iap`)** under a dedicated least-privilege service account (`physician-assist-sa`), blocks SSRF/dot-segment traversal (`CWE-22`/`CWE-918`), strips FHIR payloads from error logs (`CWE-532`), disables public HAPI fallback (`ALLOW_PUBLIC_HAPI_FALLBACK=false`), sandboxes Swimlane iframes with `Content-Security-Policy: connect-src 'none'`, and enforces `OWASP LLM01` prompt isolation on untrusted clinical/FDA text.
