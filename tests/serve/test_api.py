"""The HTTP layer, checked against a reader whose answers are known.

No checkpoint is loaded here. What is under test is the route — what it accepts,
what it refuses, and what it hands the page — and a test that needed a trained
network to assert on a status code would be testing two things and diagnosing
neither.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrz_ai.inference.candidates import Reading
from mrz_ai.parser import parse, validate
from mrz_ai.serve.api import create_app

SPECIMEN_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
SPECIMEN_LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
YEAR = 2026


class ScriptedReader:
    """Returns the specimen, and records the crops it was handed."""

    def __init__(self) -> None:
        self.crops: list[tuple[np.ndarray, np.ndarray]] = []
        self.years: list[int | None] = []

    def read(self, line1_crop, line2_crop, *, reference_year=None):
        self.crops.append((line1_crop, line2_crop))
        self.years.append(reference_year)
        document = parse(f"{SPECIMEN_LINE1}\n{SPECIMEN_LINE2}")
        return Reading(document, validate(document, reference_year=reference_year or YEAR), -0.01)


def an_image(width: int = 400, height: int = 100) -> bytes:
    image = np.full((height, width), 255, dtype=np.uint8)
    image[10:40, 20:380] = 0
    image[60:90, 20:380] = 0
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return bytes(buffer.tobytes())


@pytest.fixture
def reader() -> ScriptedReader:
    return ScriptedReader()


@pytest.fixture
def client(reader: ScriptedReader) -> TestClient:
    with TestClient(create_app(reader)) as running:
        yield running


def client_for(reader: ScriptedReader) -> TestClient:
    """A client outside the fixture, for tests that need their own reader."""
    client = TestClient(create_app(reader))
    client.__enter__()
    return client


def post(client: TestClient, image: bytes, **box):
    fields = {"x": 0, "y": 0, "width": 400, "height": 100} | box
    return client.post(
        "/api/read",
        files={"image": ("passport.png", image, "image/png")},
        data={key: str(value) for key, value in fields.items()},
    )


def test_health_answers_without_a_model(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_a_read_returns_the_fields_and_the_mrz(client: TestClient) -> None:
    response = post(client, an_image())

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["mrz"]["line1"] == SPECIMEN_LINE1
    assert len(body["fields"]) == 10


def test_the_reader_is_handed_two_separate_line_crops(
    client: TestClient, reader: ScriptedReader
) -> None:
    """The route's real job: an image and a box become two line crops."""
    post(client, an_image())

    assert len(reader.crops) == 1
    first, second = reader.crops[0]
    assert first.ndim == 2 and second.ndim == 2
    assert first.min() == 0 and second.min() == 0, "a crop held no ink"


def test_the_response_says_where_the_lines_were_found(client: TestClient) -> None:
    """So the page can show the crop it read, not the box the user drew.

    A wrong reading whose crop is visibly wrong explains itself; the same
    reading without it looks like the model cannot read.
    """
    body = post(client, an_image()).json()

    assert len(body["lines"]) == 2
    first, second = body["lines"]
    assert first["y"] < second["y"]
    assert {"x", "y", "width", "height", "clipped"} == set(first)


def test_a_box_cutting_through_the_text_warns_the_page() -> None:
    """A clipped line reads as a broken model unless the page is told otherwise."""
    body = post(client_for(ScriptedReader()), an_image(), x=60, width=340).json()

    assert body["clipped"] is True
    assert all(line["clipped"] for line in body["lines"])


def test_a_clean_selection_raises_no_warning(client: TestClient) -> None:
    body = post(client, an_image()).json()

    assert body["clipped"] is False


def test_a_reference_year_can_be_pinned(client: TestClient, reader: ScriptedReader) -> None:
    """Two-digit years need an anchor, and a test needs a stable one."""
    post(client, an_image(), reference_year=2001)

    assert reader.years == [2001]


def test_the_year_defaults_to_today(client: TestClient, reader: ScriptedReader) -> None:
    from datetime import date

    post(client, an_image())

    assert reader.years == [date.today().year]


def test_a_file_that_is_not_an_image_is_rejected(client: TestClient) -> None:
    response = post(client, b"this is not a picture")

    assert response.status_code == 400
    assert "image" in response.json()["detail"]


def test_a_box_outside_the_image_is_rejected(client: TestClient) -> None:
    """The user's mistake must come back as a message, not a 500."""
    response = post(client, an_image(), x=5000, y=5000, width=10, height=10)

    assert response.status_code == 400
    assert "outside" in response.json()["detail"]


def test_a_box_too_small_is_rejected(client: TestClient) -> None:
    response = post(client, an_image(), width=2, height=2)

    assert response.status_code == 400


def test_a_missing_box_is_rejected(client: TestClient) -> None:
    """The box is required: there is no detection stage to fall back on."""
    response = client.post(
        "/api/read", files={"image": ("p.png", an_image(), "image/png")}, data={"x": "0"}
    )

    assert response.status_code == 422


def test_an_oversized_upload_is_refused(client: TestClient) -> None:
    """An unbounded upload is a denial of service by accident."""
    from mrz_ai.serve.api import MAX_UPLOAD_BYTES

    response = post(client, b"\x89PNG" + b"\x00" * (MAX_UPLOAD_BYTES + 1))

    assert response.status_code == 413


def test_the_page_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_a_rejected_reading_still_returns_its_best_guess() -> None:
    """A scan that fails ICAO is a normal outcome, not an error.

    The page needs the best guess and the reasons to distrust it — that is what
    a human correcting the scan will work from. A 4xx here would throw away the
    only useful thing the model produced.
    """
    torn = "L898902C31" + SPECIMEN_LINE2[10:]

    class Failing(ScriptedReader):
        def read(self, line1_crop, line2_crop, *, reference_year=None):
            document = parse(f"{SPECIMEN_LINE1}\n{torn}")
            return Reading(document, validate(document, reference_year=YEAR), -8.0)

    with TestClient(create_app(Failing())) as client:
        response = post(client, an_image())

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["issues"], "the reasons to distrust the reading were dropped"
    assert body["mrz"]["line2"] == torn
