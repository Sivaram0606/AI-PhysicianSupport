#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Google Cloud Run Deployment Script — Emergency Physician Assist & ADK Agents
# ==============================================================================
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${CLOUD_RUN_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-emergency-physician-assist}"
FHIR_LOCATION="${FHIR_DATASET_LOCATION:-us-central1}"
FHIR_DATASET="${FHIR_DATASET_ID:-test}"
FHIR_STORE="${FHIR_STORE_ID:-synthe_data}"
MODEL_NAME="${MODEL_NAME:-gemini-3.8-flash}"
SA_NAME="physician-assist-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: GOOGLE_CLOUD_PROJECT is not set and no active gcloud project was found." >&2
  exit 1
fi

echo "==> Deploying '${SERVICE_NAME}' to project '${PROJECT_ID}' (${REGION})..."

# 1. Enable required Google Cloud APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  healthcare.googleapis.com \
  aiplatform.googleapis.com \
  iap.googleapis.com \
  --project="${PROJECT_ID}"

# 1b. Ensure Cloud Healthcare FHIR Dataset & R4 Store exist and are seeded
if command -v python3 >/dev/null 2>&1; then
  echo "==> Ensuring FHIR dataset '${FHIR_DATASET}' and store '${FHIR_STORE}' (${FHIR_LOCATION}) are created & seeded..."
  python3 ehr_chart_summary_agent/seed_fhir_store.py \
    --project="${PROJECT_ID}" \
    --location="${FHIR_LOCATION}" \
    --dataset="${FHIR_DATASET}" \
    --fhir-store="${FHIR_STORE}" || echo "WARNING: FHIR store seeding skipped (operator Healthcare IAM restricted or store already seeded); continuing Cloud Run deploy."
else
  echo "WARNING: python3 not found; skipping automatic FHIR seeding."
fi

# 2. Provision dedicated least-privilege runtime Service Account (falls back to default SA if setIamPolicy is restricted)
SA_FLAG=()
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "==> Creating least-privilege runtime service account: ${SA_EMAIL}"
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="Emergency Physician Assist Cloud Run Runtime SA" \
    --project="${PROJECT_ID}" || true
fi

IAM_OK=true
for ROLE in \
  "roles/healthcare.fhirStoreViewer" \
  "roles/healthcare.fhirResourceReader" \
  "roles/aiplatform.user" \
  "roles/logging.logWriter"; do
  if ! gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None \
    --quiet >/dev/null 2>&1; then
    IAM_OK=false
    break
  fi
done

if [[ "${IAM_OK}" == "true" ]]; then
  SA_FLAG=(--service-account="${SA_EMAIL}")
else
  echo "WARNING: Project-level IAM binding (resourcemanager.projects.setIamPolicy) is restricted; deploying Cloud Run with default Compute Engine service account."
fi

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
# Ensure Cloud Build's default Compute Engine SA can read uploaded source tarballs and push images when project-level IAM is restricted
gcloud storage buckets add-iam-policy-binding "gs://run-sources-${PROJECT_ID}-${REGION}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer" \
  --project="${PROJECT_ID}" >/dev/null 2>&1 || true
gcloud artifacts repositories add-iam-policy-binding cloud-run-source-deploy \
  --location="${REGION}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.writer" \
  --project="${PROJECT_ID}" >/dev/null 2>&1 || true

# 3. Deploy container from source to Cloud Run (with sticky sessions for Streamlit WebSockets & IAP)
gcloud run deploy "${SERVICE_NAME}" \
  --source=. \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  "${SA_FLAG[@]+"${SA_FLAG[@]}"}" \
  --memory=2Gi \
  --cpu=2 \
  --concurrency=20 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=5 \
  --session-affinity \
  --iap \
  --set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,MODEL_NAME=${MODEL_NAME},FHIR_DATASET_LOCATION=${FHIR_LOCATION},FHIR_DATASET_ID=${FHIR_DATASET},FHIR_STORE_ID=${FHIR_STORE},ALLOW_PUBLIC_HAPI_FALLBACK=false,USE_EXTERNAL_MCP_SERVER=false,APP_MODE=ui" \
  --quiet

# 4. Grant IAP Service Agent & active operator Cloud Run Invoker + IAP Web App User access
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com" \
  --role="roles/run.invoker" \
  --quiet >/dev/null 2>&1 || true

ACTIVE_USER="$(gcloud config get-value account 2>/dev/null || true)"
if [[ -n "${ACTIVE_USER}" && "${ACTIVE_USER}" == *"@"* ]]; then
  echo "==> Granting Cloud Run Invoker & IAP Web App User access to ${ACTIVE_USER}..."
  gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --member="user:${ACTIVE_USER}" \
    --role="roles/run.invoker" \
    --quiet >/dev/null || true

  gcloud iap web add-iam-policy-binding \
    --resource-type=cloud-run \
    --service="${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --member="user:${ACTIVE_USER}" \
    --role="roles/iap.httpsResourceAccessor" \
    --quiet >/dev/null || true

  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="user:${ACTIVE_USER}" \
    --role="roles/iap.httpsResourceAccessor" \
    --condition=None \
    --quiet >/dev/null || true
fi

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --project="${PROJECT_ID}" --format='value(status.url)')"
echo ""
echo "=============================================================================="
echo "Cloud Run Deployment Complete!"
echo "Service URL (Direct Browser via Google Sign-In / IAP): ${SERVICE_URL}"
echo "Local Authenticated Proxy Option: gcloud run services proxy ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID} --port=8080"
echo "=============================================================================="
