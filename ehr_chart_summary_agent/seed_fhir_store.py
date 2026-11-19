"""
Seeds a Google Cloud Healthcare FHIR R4 Store with the unified 7-patient
Emergency Ward + Longitudinal EHR Chart (`ehr_chart_summary_agent`) cohort (`pat-001` through `pat-007`).

Reads `GOOGLE_CLOUD_PROJECT`, `FHIR_DATASET_LOCATION`, `FHIR_DATASET_ID`, and `FHIR_STORE_ID`
from `.env` (or CLI flags `--project`, `--location`, `--dataset`, `--fhir-store`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional when running via gcloud fallback
    load_dotenv = None

try:
    import google.auth
    from google.auth.transport import requests as grequests
except ImportError:  # pragma: no cover - optional when running via gcloud fallback
    google = None
    grequests = None

try:
    from .fhir_cohort_data import get_emergency_ward_resources
except ImportError:
    from fhir_cohort_data import get_emergency_ward_resources  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import re
from urllib.parse import quote

BASE_URL = "https://healthcare.googleapis.com/v1"
_GCP_IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")


def _safe_ident(value: str, label: str) -> str:
    cleaned = (value or "").strip()
    if not _GCP_IDENT_RE.fullmatch(cleaned) or ".." in cleaned:
        raise ValueError(f"Invalid {label} identifier: {value!r}")
    return quote(cleaned, safe="")


class _StdlibResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}: {self._body.decode('utf-8', errors='replace')[:300]}")


class _GcloudTokenSession:
    """Minimal AuthorizedSession-compatible HTTP client using gcloud ADC/CLI token + curl."""

    def __init__(self) -> None:
        token = ""
        for cmd in (
            ["gcloud", "auth", "application-default", "print-access-token"],
            ["gcloud", "auth", "print-access-token"],
        ):
            try:
                token = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
                if token:
                    break
            except subprocess.CalledProcessError:
                continue
        if not token:
            raise RuntimeError("Unable to obtain access token from gcloud.")
        self._token = token

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        json_body: object | None = None,
        timeout: int = 15,
    ) -> _StdlibResponse:
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        cmd = [
            "curl",
            "-sS",
            "--max-time",
            str(timeout),
            "-X",
            method,
            "-H",
            f"Authorization: Bearer {self._token}",
            "-w",
            "\n%{http_code}",
        ]
        req_headers = dict(headers or {})
        if json_body is not None:
            req_headers.setdefault("Content-Type", "application/json")
        for k, v in req_headers.items():
            cmd.extend(["-H", f"{k}: {v}"])
        input_data: bytes | None = None
        if json_body is not None:
            input_data = json.dumps(json_body).encode("utf-8")
            cmd.extend(["--data-binary", "@-"])
        cmd.append(url)
        proc = subprocess.run(cmd, input=input_data, capture_output=True, check=True)
        out = proc.stdout
        body, _, code_bytes = out.rpartition(b"\n")
        return _StdlibResponse(int(code_bytes.decode("ascii").strip()), body)

    def get(self, url: str, timeout: int = 15) -> _StdlibResponse:
        return self._request("GET", url, timeout=timeout)

    def post(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json: object | None = None,
        timeout: int = 15,
    ) -> _StdlibResponse:
        return self._request("POST", url, params=params, json_body=json, timeout=timeout)

    def put(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: object | None = None,
        timeout: int = 15,
    ) -> _StdlibResponse:
        return self._request("PUT", url, headers=headers, json_body=json, timeout=timeout)


def get_authorized_session():
    if google is not None and grequests is not None:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return grequests.AuthorizedSession(creds)
    logging.info("google-auth not installed; using gcloud auth print-access-token fallback.")
    return _GcloudTokenSession()


def ensure_dataset_and_fhir_store(
    session: grequests.AuthorizedSession,
    project_id: str,
    location: str,
    dataset_id: str,
    fhir_store_id: str,
) -> str:
    safe_proj = _safe_ident(project_id, "project_id")
    safe_loc = _safe_ident(location, "location")
    safe_ds = _safe_ident(dataset_id, "dataset_id")
    safe_store = _safe_ident(fhir_store_id, "fhir_store_id")

    parent = f"{BASE_URL}/projects/{safe_proj}/locations/{safe_loc}"
    dataset_url = f"{parent}/datasets/{safe_ds}"

    r_ds = session.get(dataset_url, timeout=15)
    if r_ds.status_code == 404:
        logging.info("Creating Healthcare Dataset '%s' in %s/%s...", dataset_id, project_id, location)
        r_create_ds = session.post(
            f"{parent}/datasets",
            params={"datasetId": dataset_id},
            json={},
            timeout=15,
        )
        if r_create_ds.status_code not in (200, 201, 409):
            logging.error("Failed creating dataset (HTTP %d).", r_create_ds.status_code)
            r_create_ds.raise_for_status()
        for _ in range(45):
            if session.get(dataset_url, timeout=10).status_code == 200:
                break
            time.sleep(4)
    else:
        if not r_ds.ok:
            logging.error("Failed checking dataset (HTTP %d).", r_ds.status_code)
        r_ds.raise_for_status()
        logging.info("Healthcare Dataset '%s' verified.", dataset_id)

    fhir_store_url = f"{dataset_url}/fhirStores/{safe_store}"
    r_fs = session.get(fhir_store_url, timeout=15)
    if r_fs.status_code in (404, 403):
        logging.info("Creating FHIR R4 Store '%s'...", fhir_store_id)
        r_create_fs = session.post(
            f"{dataset_url}/fhirStores",
            params={"fhirStoreId": fhir_store_id},
            json={
                "version": "R4",
                "enableUpdateCreate": True,
                "disableReferentialIntegrity": False,
            },
            timeout=15,
        )
        if r_create_fs.status_code not in (200, 201, 409):
            logging.error("Failed creating FHIR store (HTTP %d).", r_create_fs.status_code)
            r_create_fs.raise_for_status()
        time.sleep(2)
    else:
        if not r_fs.ok:
            logging.error("Failed checking FHIR store (HTTP %d).", r_fs.status_code)
        r_fs.raise_for_status()
        logging.info("FHIR Store '%s' verified.", fhir_store_id)

    return f"{fhir_store_url}/fhir"


def seed_all(
    project_id: str,
    location: str,
    dataset_id: str,
    fhir_store_id: str,
) -> None:
    session = get_authorized_session()
    fhir_base = ensure_dataset_and_fhir_store(
        session=session,
        project_id=project_id,
        location=location,
        dataset_id=dataset_id,
        fhir_store_id=fhir_store_id,
    )
    headers = {"Content-Type": "application/fhir+json;charset=utf-8"}

    resources = get_emergency_ward_resources()
    ok_count = 0
    err_count = 0
    for rtype, rid, body in resources:
        safe_rtype = _safe_ident(rtype, "FHIR resourceType")
        safe_rid = _safe_ident(rid, "FHIR resourceId")
        url = f"{fhir_base}/{safe_rtype}/{safe_rid}"
        resp = session.put(url, headers=headers, json=body, timeout=15)
        if resp.status_code in (200, 201):
            ok_count += 1
        else:
            err_count += 1
            logging.error("Failed %s/%s (HTTP %d).", rtype, rid, resp.status_code)

    logging.info(
        "Completed FHIR R4 seeding in projects/%s/locations/%s/datasets/%s/fhirStores/%s: %d succeeded, %d failed.",
        project_id,
        location,
        dataset_id,
        fhir_store_id,
        ok_count,
        err_count,
    )


def main() -> None:
    default_project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    default_location = os.getenv("FHIR_DATASET_LOCATION", "us-central1").strip()
    default_dataset = os.getenv("FHIR_DATASET_ID", "test").strip()
    default_fhir_store = os.getenv("FHIR_STORE_ID", "synthe_data").strip()

    parser = argparse.ArgumentParser(description="Seed GCP Cloud Healthcare FHIR R4 Store with pat-001..pat-007")
    parser.add_argument("--project", default=default_project)
    parser.add_argument("--location", default=default_location)
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--fhir-store", default=default_fhir_store)
    args = parser.parse_args()

    project_id = (args.project or "").strip() or default_project
    location = (args.location or "").strip() or default_location
    dataset_id = (args.dataset or "").strip() or default_dataset
    fhir_store_id = (args.fhir_store or "").strip() or default_fhir_store

    if not project_id:
        parser.error(
            "GCP project ID is empty. Export PROJECT_ID, pass --project <PROJECT_ID>, "
            "or set GOOGLE_CLOUD_PROJECT in ehr_chart_summary_agent/.env."
        )

    seed_all(
        project_id=project_id,
        location=location,
        dataset_id=dataset_id,
        fhir_store_id=fhir_store_id,
    )


if __name__ == "__main__":
    main()
