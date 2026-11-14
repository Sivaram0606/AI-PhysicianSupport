"""
Hardened GCP Cloud Healthcare FHIR R4 Client Helpers & FastMCP Tool Server.

Security & Efficiency Architecture:
- Validates all FHIR resource IDs and GCP project/location/dataset/store identifiers against strict
  character allowlists + URL path-segment encoding to prevent SSRF / path traversal (CWE-22 / CWE-918).
- Reuses a thread-safe pooled `AuthorizedSession` (`HTTPAdapter`) so OAuth2 tokens and TLS connections
  to `healthcare.googleapis.com` are shared instead of re-authenticating on every request.
- Suppresses raw FHIR response payloads from error logs to prevent PHI leakage (CWE-532).
- Enforces explicit connect/read timeouts and Lucene term sanitization on OpenFDA queries.
"""

from __future__ import annotations

import logging
import os
import re
import ssl
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

import certifi
import google.auth
import google.auth.exceptions
import requests
from dotenv import load_dotenv
from google.auth.transport import requests as grequests
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, RequestException
import uvicorn

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # type: ignore

if not os.environ.get("SSL_CERT_FILE") and ssl.get_default_verify_paths().cafile is None:
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_GCP_IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_FHIR_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,63}$")
_SAFE_DRUG_RE = re.compile(r"^[A-Za-z0-9\s.,'()\-/\[\]:]{1,160}$")

_HEALTHCARE_BASE_URL = "https://healthcare.googleapis.com/v1"
_SESSION_LOCK = threading.Lock()
_AUTH_SESSION: grequests.AuthorizedSession | None = None

_PUBLIC_HTTP = requests.Session()
_PUBLIC_HTTP.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=16))


def _get_fhir_store_base_url() -> str | None:
    """Validates GCP Healthcare config identifiers and returns the base `/fhir` URL."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.getenv("FHIR_DATASET_LOCATION", "asia-east1").strip()
    dataset_id = os.getenv("FHIR_DATASET_ID", "test").strip()
    fhir_store_id = os.getenv("FHIR_STORE_ID", "synthe_data").strip()

    if not project_id:
        return None
    for label, val in (
        ("GOOGLE_CLOUD_PROJECT", project_id),
        ("FHIR_DATASET_LOCATION", location),
        ("FHIR_DATASET_ID", dataset_id),
        ("FHIR_STORE_ID", fhir_store_id),
    ):
        if not _GCP_IDENT_RE.fullmatch(val) or ".." in val:
            logger.error("Rejected invalid %s identifier format.", label)
            return None

    return (
        f"{_HEALTHCARE_BASE_URL}/projects/{quote(project_id, safe='')}"
        f"/locations/{quote(location, safe='')}"
        f"/datasets/{quote(dataset_id, safe='')}"
        f"/fhirStores/{quote(fhir_store_id, safe='')}/fhir"
    )


def _get_authorized_fhir_session() -> grequests.AuthorizedSession:
    """Returns a thread-safe, connection-pooled AuthorizedSession for Cloud Healthcare FHIR."""
    global _AUTH_SESSION
    with _SESSION_LOCK:
        if _AUTH_SESSION is None:
            scoped_credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            sess = grequests.AuthorizedSession(scoped_credentials)
            adapter = HTTPAdapter(pool_connections=12, pool_maxsize=16)
            sess.mount("https://", adapter)
            _AUTH_SESSION = sess
        return _AUTH_SESSION


def get_patient_everything(resource_id: str) -> dict[str, Any] | None:
    """
    Fetches all resources in a patient's compartment ($everything) from the GCP FHIR store.

    Validates `resource_id` strictly to block path traversal / query injection and avoids
    logging raw FHIR response payloads (PHI protection).
    """
    clean_id = (resource_id or "").strip()
    if not _FHIR_RESOURCE_ID_RE.fullmatch(clean_id) or ".." in clean_id:
        logger.warning("Blocked invalid FHIR Patient resource_id format.")
        return None

    fhir_base = _get_fhir_store_base_url()
    if not fhir_base:
        return None

    resource_path = f"{fhir_base}/Patient/{quote(clean_id, safe='')}/$everything"
    headers = {"Accept": "application/fhir+json;charset=utf-8"}

    try:
        session = _get_authorized_fhir_session()
        response = session.get(resource_path, headers=headers, timeout=(4, 10))
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except (
        google.auth.exceptions.DefaultCredentialsError,
        google.auth.exceptions.RefreshError,
    ):
        logger.warning("GCP ADC credentials unavailable or expired; falling back to local FHIR store.")
        return None
    except HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response is not None else 0
        logger.warning("GCP FHIR $everything returned HTTP %d for requested Patient resource.", status_code)
        return None
    except RequestException as req_err:
        logger.warning("GCP FHIR network error (%s).", type(req_err).__name__)
        return None
    except Exception as exc:
        logger.error("Unexpected error in get_patient_everything (%s).", type(exc).__name__)
        return None


def list_fhir_patients() -> dict[str, Any] | None:
    """Fetches all Patient resources currently registered in the GCP Cloud Healthcare FHIR store."""
    fhir_base = _get_fhir_store_base_url()
    if not fhir_base:
        return None

    try:
        session = _get_authorized_fhir_session()
        response = session.get(
            f"{fhir_base}/Patient",
            headers={"Accept": "application/fhir+json"},
            timeout=(4, 10),
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Unable to list GCP FHIR patients (%s).", type(exc).__name__)
        return None


def list_active_triage_encounters() -> dict[str, Any] | None:
    """Fetches active (`status=in-progress`) Emergency Ward triage Encounter resources from GCP FHIR store."""
    fhir_base = _get_fhir_store_base_url()
    if not fhir_base:
        return None

    try:
        session = _get_authorized_fhir_session()
        response = session.get(
            f"{fhir_base}/Encounter",
            params={"status": "in-progress"},
            headers={"Accept": "application/fhir+json"},
            timeout=(4, 10),
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Unable to list GCP FHIR triage encounters (%s).", type(exc).__name__)
        return None


# ---------------------------------------------------------------------------
# FastMCP Tool Server (Loopback-Bound)
# ---------------------------------------------------------------------------
mcp = FastMCP()


@mcp.tool()
def get_medication_details(patient_id: str) -> dict[str, Any]:
    """
    Retrieves MedicationRequest resources for a given patient from FHIR (with local cohort fallback).
    """
    clean_id = (patient_id or "").strip()
    if not _FHIR_RESOURCE_ID_RE.match(clean_id):
        return {"error": "Invalid patient_id format."}

    try:
        from .ehr_chart_tools import _fetch_fhir_everything_bundle

        _, patient_details, _ = _fetch_fhir_everything_bundle(clean_id)
    except Exception:
        patient_details = get_patient_everything(clean_id)

    if not patient_details or "entry" not in patient_details:
        return {"error": "Patient details not found, please check the patient id."}

    medication_list = [
        entry["resource"]
        for entry in patient_details.get("entry", [])
        if entry.get("resource", {}).get("resourceType") == "MedicationRequest"
    ]
    if medication_list:
        return {"medication list": medication_list}
    return {"status": "Patient details found, but medication details not found"}


@mcp.tool()
def get_adverse_effects(drug_name: str) -> dict[str, Any]:
    """
    Fetches the top 10 reported adverse effects for a given drug from OpenFDA FAERS
    with strict input sanitization and network timeouts.
    """
    if not isinstance(drug_name, str) or not _SAFE_DRUG_RE.match(drug_name.strip()):
        return {"error": "Invalid drug_name format."}

    # Strip Lucene special operators / quotes before embedding in exact phrase search
    sanitized_term = re.sub(r"[^a-zA-Z0-9\s.\-]", " ", drug_name.strip().lower())
    sanitized_term = " ".join(sanitized_term.split())
    if not sanitized_term:
        return {"error": "Invalid drug_name provided."}

    params = {
        "search": f'patient.drug.medicinalproduct:"{sanitized_term}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": 10,
    }

    try:
        response = _PUBLIC_HTTP.get(
            "https://api.fda.gov/drug/event.json",
            params=params,
            timeout=(4, 10),
        )
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                return {
                    effect.get("term", "Unknown Effect"): int(effect.get("count", 0))
                    for effect in results
                }
            return {"error": "Unable to retrieve adverse effects for the drug."}
        if response.status_code == 404:
            return {"error": f"No OpenFDA adverse event records found for '{sanitized_term}'."}
        return {"error": f"OpenFDA API returned HTTP {response.status_code}."}
    except RequestException as exc:
        return {"error": f"OpenFDA network error ({type(exc).__name__})."}


@mcp.tool()
def list_ward_patients() -> dict[str, Any]:
    """Lists all patients currently registered in the Emergency Ward FHIR store."""
    try:
        from .clinical_tools import list_emergency_ward_patients
    except ImportError:
        from clinical_tools import list_emergency_ward_patients  # type: ignore
    return list_emergency_ward_patients()


@mcp.tool()
def fetch_patient_clinical_profile(patient_id: str, patient_name: str = "") -> dict[str, Any]:
    """Retrieves the complete FHIR ($everything) clinical record for an emergency ward patient."""
    try:
        from .clinical_tools import get_patient_clinical_profile
    except ImportError:
        from clinical_tools import get_patient_clinical_profile  # type: ignore
    return get_patient_clinical_profile(patient_id=patient_id, patient_name=patient_name)


@mcp.tool()
def verify_drug_brand_or_generic(drug_name: str, rxcui_code: str = "") -> dict[str, Any]:
    """Checks whether a prescribed drug in FHIR is an Original Generic or Brand name via RxNorm & OpenFDA."""
    try:
        from .clinical_tools import check_drug_brand_or_generic
    except ImportError:
        from clinical_tools import check_drug_brand_or_generic  # type: ignore
    return check_drug_brand_or_generic(drug_name=drug_name, rxcui_code=rxcui_code)


@mcp.tool()
def audit_all_patient_medications(patient_id: str) -> dict[str, Any]:
    """Performs a parallelized FHIR + NIH RxNorm + OpenFDA medication & adverse-effects audit for a patient."""
    try:
        from .clinical_tools import audit_patient_medications_rxnorm_openfda
    except ImportError:
        from clinical_tools import audit_patient_medications_rxnorm_openfda  # type: ignore
    return audit_patient_medications_rxnorm_openfda(patient_id=patient_id)


@mcp.tool()
def fetch_location_outbreak_and_disease_data(
    postal_code: str,
    latitude: float | None = None,
    longitude: float | None = None,
    city: str = "",
) -> dict[str, Any]:
    """Queries hyperlocal 5-digit ZIP-code outbreaks, sewershed pathogens, persistent diseases, and AQI."""
    try:
        from .clinical_tools import get_location_outbreak_and_disease_data
    except ImportError:
        from clinical_tools import get_location_outbreak_and_disease_data  # type: ignore
    return get_location_outbreak_and_disease_data(
        postal_code=postal_code, latitude=latitude, longitude=longitude, city=city
    )


@mcp.tool()
def fetch_hospitalizations_and_notes(patient_id: str) -> dict[str, Any]:
    """Retrieves longitudinal hospitalizations, ED visits, and specialist consult notes from FHIR."""
    try:
        from .ehr_chart_tools import get_hospitalizations_and_clinical_notes
    except ImportError:
        from ehr_chart_tools import get_hospitalizations_and_clinical_notes  # type: ignore
    return get_hospitalizations_and_clinical_notes(patient_id=patient_id)


@mcp.tool()
def fetch_medication_reconciliation(patient_id: str) -> dict[str, Any]:
    """Retrieves active prescriptions, dose changes, held drugs, and pharmacy adherence barriers from FHIR."""
    try:
        from .ehr_chart_tools import get_medication_reconciliation_history
    except ImportError:
        from ehr_chart_tools import get_medication_reconciliation_history  # type: ignore
    return get_medication_reconciliation_history(patient_id=patient_id)


@mcp.tool()
def fetch_longitudinal_labs_and_imaging(patient_id: str) -> dict[str, Any]:
    """Retrieves multi-date lab trajectories and diagnostic imaging reports (CT, TTE, CXR) from FHIR."""
    try:
        from .ehr_chart_tools import get_longitudinal_labs_and_imaging
    except ImportError:
        from ehr_chart_tools import get_longitudinal_labs_and_imaging  # type: ignore
    return get_longitudinal_labs_and_imaging(patient_id=patient_id)


@mcp.tool()
def fetch_open_loops_and_care_gaps(patient_id: str) -> dict[str, Any]:
    """Retrieves unresolved clinical safety alerts, unreviewed labs, and care gaps from FHIR Flag resources."""
    try:
        from .ehr_chart_tools import get_open_loops_and_care_gaps
    except ImportError:
        from ehr_chart_tools import get_open_loops_and_care_gaps  # type: ignore
    return get_open_loops_and_care_gaps(patient_id=patient_id)


@mcp.tool()
def build_clinical_swimlane_graph(patient_id: str) -> dict[str, Any]:
    """Generates the 4-Track Interactive Clinical Swimlane Graph (Mermaid, Markdown matrix, HTML/SVG) from FHIR."""
    try:
        from .ehr_chart_tools import generate_clinical_swimlane_graph
    except ImportError:
        from ehr_chart_tools import generate_clinical_swimlane_graph  # type: ignore
    return generate_clinical_swimlane_graph(patient_id=patient_id)


if __name__ == "__main__":
    mcp_port = int(os.getenv("MCP_PORT", "8080"))
    uvicorn.run(mcp.streamable_http_app, host="127.0.0.1", port=mcp_port)