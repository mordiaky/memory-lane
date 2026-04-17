"""HTTP API smoke tests — ensures wiring is correct end to end."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_file = tmp_path / "api-test.db"
    monkeypatch.setenv("MEMORY_LANE_DB", f"sqlite:///{db_file}")
    monkeypatch.setenv("MEMORY_LANE_MEDIA_DIR", str(tmp_path / "media"))
    # Re-import so the module-level engine picks up the env var.
    import importlib

    from memory_lane import api as api_module
    from memory_lane import web as web_module

    importlib.reload(web_module)
    importlib.reload(api_module)
    return TestClient(api_module.app)


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_full_flow_via_http(client) -> None:
    # Create patient.
    resp = client.post("/api/patients", json={"display_name": "Eleanor"})
    assert resp.status_code == 201, resp.text
    patient_id = resp.json()["id"]

    # Add two memories.
    mem_resp = client.post(
        "/api/memories",
        json={
            "patient_id": patient_id,
            "title": "Garden roses",
            "description": "Late summer rose garden.",
            "tone": "joyful",
            "valence_start": 0.5,
            "valence_peak": 0.95,
            "valence_end": 0.7,
        },
    )
    assert mem_resp.status_code == 201, mem_resp.text
    mem_id = mem_resp.json()["id"]

    client.post(
        "/api/memories",
        json={
            "patient_id": patient_id,
            "title": "Hospital year",
            "description": "A long illness.",
            "tone": "difficult",
        },
    )

    # Start a session, log a positive reaction, end the session.
    sess_resp = client.post(
        "/api/sessions",
        json={"patient_id": patient_id, "caregiver_name": "Daughter"},
    )
    assert sess_resp.status_code == 201
    sess_id = sess_resp.json()["id"]

    react_resp = client.post(
        "/api/reactions",
        json={
            "session_id": sess_id,
            "memory_id": mem_id,
            "kind": "recognized_positive",
            "notes": "She smiled and started naming the rose varieties.",
        },
    )
    assert react_resp.status_code == 201, react_resp.text

    end_resp = client.post(
        f"/api/sessions/{sess_id}/end",
        json={"summary": "Good visit."},
    )
    assert end_resp.status_code == 200

    # Anchor excludes the difficult memory.
    anchor_resp = client.get(f"/api/patients/{patient_id}/anchor")
    assert anchor_resp.status_code == 200
    anchors = anchor_resp.json()
    assert len(anchors) >= 1
    assert anchors[0]["title"] == "Garden roses"
    assert all(a["title"] != "Hospital year" for a in anchors)

    # Visit plan endpoint responds.
    plan_resp = client.get(f"/api/patients/{patient_id}/visit-plan")
    assert plan_resp.status_code == 200
    assert isinstance(plan_resp.json(), list)


def test_flag_memory_hides_from_anchor(client) -> None:
    patient = client.post(
        "/api/patients",
        json={"display_name": "Eleanor"},
    ).json()
    mem = client.post(
        "/api/memories",
        json={
            "patient_id": patient["id"],
            "title": "Sibling argument",
            "description": "A long-running family dispute.",
            "tone": "bittersweet",
        },
    ).json()

    anchors_before = client.get(f"/api/patients/{patient['id']}/anchor").json()
    assert any(a["memory_id"] == mem["id"] for a in anchors_before)

    flag_resp = client.post(
        f"/api/memories/{mem['id']}/flag",
        json={"note": "Causes agitation."},
    )
    assert flag_resp.status_code == 200

    anchors_after = client.get(f"/api/patients/{patient['id']}/anchor").json()
    assert all(a["memory_id"] != mem["id"] for a in anchors_after)


# ---- Web UI smoke tests -----------------------------------------


def test_web_index_renders(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "MemoryLane" in resp.text
    assert "Not a medical device" in resp.text


def test_web_create_patient_via_form_redirects_to_dashboard(client) -> None:
    resp = client.post(
        "/patients",
        data={"display_name": "Eleanor", "birth_year": "1942"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/patients/")
    dashboard = client.get(resp.headers["location"])
    assert dashboard.status_code == 200
    assert "Eleanor" in dashboard.text
    assert "Add a memory" in dashboard.text


def test_web_add_memory_flow(client) -> None:
    # Create patient via the web form.
    created = client.post(
        "/patients",
        data={"display_name": "Eleanor"},
        follow_redirects=False,
    )
    patient_url = created.headers["location"]
    patient_id = patient_url.rsplit("/", 1)[-1]

    # Add a memory via the form.
    resp = client.post(
        f"/patients/{patient_id}/memories",
        data={
            "title": "Wedding",
            "description": "Sunny afternoon in 1965.",
            "tone": "joyful",
            "valence_start": "0.5",
            "valence_peak": "0.9",
            "valence_end": "0.7",
            "approximate_year": "1965",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    mem_page = client.get(resp.headers["location"])
    assert mem_page.status_code == 200
    assert "Wedding" in mem_page.text
    # Patient dashboard now lists the memory.
    dash = client.get(patient_url)
    assert "Wedding" in dash.text


def test_web_session_flow_and_report(client) -> None:
    created = client.post(
        "/patients",
        data={"display_name": "Eleanor"},
        follow_redirects=False,
    )
    patient_id = created.headers["location"].rsplit("/", 1)[-1]

    # Add a memory.
    mem_created = client.post(
        f"/patients/{patient_id}/memories",
        data={
            "title": "Roses",
            "description": "Late summer garden.",
            "tone": "joyful",
        },
        follow_redirects=False,
    )
    memory_id = mem_created.headers["location"].rsplit("/", 1)[-1]

    # Start a session.
    sess_resp = client.post(
        f"/patients/{patient_id}/sessions",
        data={"caregiver_name": "Daughter"},
        follow_redirects=False,
    )
    session_url = sess_resp.headers["location"]
    assert session_url.startswith("/sessions/")

    session_id = session_url.split("/")[-1]

    # Page renders.
    session_page = client.get(session_url)
    assert session_page.status_code == 200
    assert "Visit in progress" in session_page.text
    assert "Roses" in session_page.text

    # Log a reaction via the HTMX endpoint.
    react_resp = client.post(
        f"/sessions/{session_id}/reactions",
        data={"memory_id": memory_id, "kind": "recognized_positive"},
    )
    assert react_resp.status_code == 200
    # Response is a fragment — contains the "Logged" marker.
    assert "Logged" in react_resp.text

    # End the session; should redirect to the report page.
    end_resp = client.post(
        f"/sessions/{session_id}/end",
        data={"summary": "Lovely visit."},
        follow_redirects=False,
    )
    assert end_resp.status_code == 303
    report_page = client.get(end_resp.headers["location"])
    assert report_page.status_code == 200
    assert "Visit report" in report_page.text
    assert "Roses" in report_page.text
