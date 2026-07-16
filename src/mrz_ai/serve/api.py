"""The HTTP surface: a page, and one route that reads a passport.

Deliberately small. The reading is `inference`, the shaping is `payload`, the
cropping is `crop`; what is left here is the part that has to know about HTTP.

*The model is loaded once.* Building `MRZReader` reads 27MB off disk and
initialises a network. Per-request it would dominate the 25ms the reading itself
takes.

*Nothing is stored.* A passport is about as identifying as a document gets, and
this uploads one. The image lives in memory for the length of the request and is
never written to disk, never logged, and never sent anywhere: the whole point of
a local model is that it can work this way, and a demo that quietly kept the
files would be teaching the opposite lesson.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, AsyncIterator, Protocol

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..inference.candidates import Reading
from .crop import Box, decode_image, locate_lines
from .payload import as_payload

Array = np.ndarray

__all__ = ["CHECKPOINT", "Reader", "create_app"]

CHECKPOINT = Path("recognition_model/recognition.pt")
STATIC = Path(__file__).parent / "static"

#: Uploads are capped well above a phone photo and well below anything that
#: would exhaust memory. An unbounded upload is a denial of service by accident.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class Reader(Protocol):
    """What the route needs of a reader.

    A Protocol rather than `MRZReader` itself, for the same reason `measure` has
    one: the tests get to check the HTTP layer with a reader whose answers are
    known, instead of loading a 27MB checkpoint to assert on a status code.
    """

    def read(self, line1_crop: Array, line2_crop: Array, *, reference_year: int | None = ...
             ) -> Reading: ...


def _load_reader(checkpoint: Path) -> Reader:
    """Import torch and build the reader, at startup rather than at import.

    Keeping the import inside the function means this module can be imported —
    and its routes tested — without the framework present.
    """
    from ..inference.pipeline import MRZReader

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"no checkpoint at {checkpoint}. Train one with notebooks/02_train_recognition.ipynb, "
            "or point MRZ_CHECKPOINT at an existing one."
        )
    return MRZReader.from_checkpoint(checkpoint, use_ema=True, k=8)


def create_app(reader: Reader | None = None, *, checkpoint: Path = CHECKPOINT) -> FastAPI:
    """Build the app. Pass ``reader`` to skip loading a checkpoint."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.reader = reader if reader is not None else _load_reader(checkpoint)
        yield

    app = FastAPI(title="MRZ reader", lifespan=lifespan, docs_url="/api/docs")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "checkpoint": str(checkpoint)}

    @app.post("/api/read")
    async def read(
        image: UploadFile = File(...),
        x: float = Form(...),
        y: float = Form(...),
        width: float = Form(...),
        height: float = Form(...),
        reference_year: int | None = Form(None),
    ) -> JSONResponse:
        """Read the MRZ inside the given box of the uploaded image.

        The box is in the image's own pixels. It is a required parameter rather
        than something guessed: there is no detection stage, and inventing one
        here would produce confident nonsense on the first real photograph.
        """
        data = await image.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="the image is too large")

        try:
            picture = decode_image(data)
            box = Box(x=x, y=y, width=width, height=height)
            first, second = locate_lines(picture, box)
        except ValueError as error:
            # A bad upload or a bad drag is the user's to fix, not a server fault.
            raise HTTPException(status_code=400, detail=str(error)) from error

        year = reference_year or date.today().year
        reading = app.state.reader.read(
            _cut(picture, first.box), _cut(picture, second.box), reference_year=year
        )

        body = as_payload(reading, reference_year=year)
        # Where the lines were found, so the page can show the user what was
        # actually read rather than what they drew. A wrong reading with a
        # visibly wrong crop explains itself.
        body["lines"] = [
            {**_as_dict(line.box), "clipped": line.clipped} for line in (first, second)
        ]
        # A cut-off line is the one failure the reading itself cannot report: the
        # model returns its best guess at characters it was never shown, and the
        # result reads as a bad model rather than a bad crop.
        body["clipped"] = first.clipped or second.clipped
        return JSONResponse(body)

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app


def _cut(image: Array, box: Box) -> Array:
    return image[int(box.y) : int(box.y + box.height), int(box.x) : int(box.x + box.width)]


def _as_dict(box: Box) -> dict[str, float]:
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


app = create_app()
