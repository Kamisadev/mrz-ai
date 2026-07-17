"""The dashboard's HTTP surface. Read-only by construction."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from mrz_ai.serve.dashboard import STALE_AFTER, create_dashboard
from mrz_ai.training.status import Status, StatusWriter


def client(directory: Path) -> TestClient:
    return TestClient(create_dashboard(directory))


def test_a_dashboard_opened_before_the_run_says_waiting(tmp_path: Path) -> None:
    """Not a 404. There is genuinely nothing to report yet, and a page that errors
    at that moment teaches its reader that it errors."""
    response = client(tmp_path).get("/api/status")
    assert response.status_code == 200
    assert response.json()["state"] == "waiting"


def test_the_status_comes_back(tmp_path: Path) -> None:
    StatusWriter(tmp_path, Status(state="training", step=42, updated_at=time.time())).write()
    body = client(tmp_path).get("/api/status").json()
    assert (body["state"], body["step"], body["stale"]) == ("training", 42, False)


def test_a_run_that_stopped_writing_is_reported_stale(tmp_path: Path) -> None:
    """The honest failure signal. A pod that is OOM-killed or simply taken away
    runs no handler, so the file says "training" forever — the only thing left to
    notice is that it stopped moving.
    """
    stale = Status(state="training", step=42, updated_at=time.time() - STALE_AFTER - 10)
    StatusWriter(tmp_path, stale).write()

    body = client(tmp_path).get("/api/status").json()
    assert body["stale"] is True
    assert body["step"] == 42, "the last numbers stay; they are just no longer current"


def test_a_finished_run_is_not_stale(tmp_path: Path) -> None:
    """It is meant to have stopped writing. Only an unfinished run going quiet means
    something."""
    done = Status(state="finished", step=100, updated_at=time.time() - STALE_AFTER * 10)
    StatusWriter(tmp_path, done).write()
    assert client(tmp_path).get("/api/status").json()["stale"] is False


def test_a_failed_run_carries_its_reason(tmp_path: Path) -> None:
    StatusWriter(tmp_path, Status(state="failed", error="RuntimeError: CUDA OOM")).write()
    body = client(tmp_path).get("/api/status").json()
    assert body["state"] == "failed"
    assert "CUDA OOM" in body["error"]


def test_the_dashboard_offers_no_way_to_touch_the_run(tmp_path: Path) -> None:
    """A monitor over a rented pod must not be able to end a 45-minute job by a
    misclick. There is no route that writes, and this is what keeps it that way.
    """
    app = create_dashboard(tmp_path)
    for route in app.routes:
        methods = getattr(route, "methods", set()) or set()
        assert methods <= {"GET", "HEAD"}, f"{route} accepts {methods}"
