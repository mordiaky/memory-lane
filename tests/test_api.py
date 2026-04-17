"""HTTP API smoke tests — ensures wiring is correct end to end."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_file = tmp_path / "api-test.db"
    monkeypatch.setenv("MEMORY_LANE_DB", f"sqlite:///{db_file}")
    # Re-import so the module-level engine picks up the env var.
    import importlib

    from memory_lane import api as api_module

    importlib.reload(api_module)
    return TestClient(api_module.app)


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_full_flow_via_http(client) -> None:
    # Create patient.
    resp = client.post("/patients", json={"display_name": "Eleanor"})
    assert resp.status_code == 201, resp.text
    patient_id = resp.json()["id"]

    # Add two memories.
    mem_resp = client.post(
        "/memories",
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
        "/memories",
        json={
            "patient_id": patient_id,
            "title": "Hospital year",
            "description": "A long illness.",
            "tone": "difficult",
        },
    )

    # Start a session, log a positive reaction, end the session.
    sess_resp = client.post(
        "/sessions",
        json={"patient_id": patient_id, "caregiver_name": "Daughter"},
    )
    assert sess_resp.status_code == 201
    sess_id = sess_resp.json()["id"]

    react_resp = client.post(
        "/reactions",
        json={
            "session_id": sess_id,
            "memory_id": mem_id,
            "kind": "recognized_positive",
            "notes": "She smiled and started naming the rose varieties.",
        },
    )
    assert react_resp.status_code == 201, react_resp.text

    end_resp = client.post(f"/sessions/{sess_id}/end", json={"summary": "Good visit."})
    assert end_resp.status_code == 200

    # Ask for an anchor. The joyful memory should win; the difficult
    # one should not be recommended.
    anchor_resp = client.get(f"/patients/{patient_id}/anchor")
    assert anchor_resp.status_code == 200
    anchors = anchor_resp.json()
    assert len(anchors) >= 1
    assert anchors[0]["title"] == "Garden roses"
    assert all(a["title"] != "Hospital year" for a in anchors)

    # Visit plan endpoint responds with something reasonable.
    plan_resp = client.get(f"/patients/{patient_id}/visit-plan")
    assert plan_resp.status_code == 200
    assert isinstance(plan_resp.json(), list)


def test_flag_memory_hides_from_anchor(client) -> None:
    patient = client.post("/patients", json={"display_name": "Eleanor"}).json()
    mem = client.post(
        "/memories",
        json={
            "patient_id": patient["id"],
            "title": "Sibling argument",
            "description": "A long-running family dispute.",
            "tone": "bittersweet",
        },
    ).json()

    # Before flag: memory is a candidate.
    anchors_before = client.get(f"/patients/{patient['id']}/anchor").json()
    assert any(a["memory_id"] == mem["id"] for a in anchors_before)

    # Flag it.
    flag_resp = client.post(
        f"/memories/{mem['id']}/flag",
        json={"note": "Causes agitation."},
    )
    assert flag_resp.status_code == 200

    # After flag: excluded.
    anchors_after = client.get(f"/patients/{patient['id']}/anchor").json()
    assert all(a["memory_id"] != mem["id"] for a in anchors_after)
