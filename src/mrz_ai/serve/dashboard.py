"""A window into a training run. Read-only, and unable to disturb it.

Serves `status.json` and a page that polls it. It never imports torch, never
touches the model, and holds nothing: if this process dies the run does not
notice, and if it is restarted it picks up wherever the run has got to. That is
the entire point of going through a file — a rented pod running a 45-minute job
is not somewhere to introduce a web server into the training process.

    python -m mrz_ai.serve.dashboard --dir checkpoints/recognition --port 8080

*Staleness is the honest failure signal.* A pod that is OOM-killed or simply
taken away runs no exception handler, so the last status on disk says "training"
forever. Nothing can be done about that from inside the run; what can be done is
to notice that the file has stopped moving. A status whose `updated_at` is older
than several logging intervals is reported stale, and the page says so rather
than showing a step count that has quietly been a lie for ten minutes.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..training.status import read_status

__all__ = ["DEFAULT_DIR", "DEFAULT_PORT", "create_dashboard"]

DEFAULT_DIR = Path("checkpoints/recognition")
DEFAULT_PORT = 8080
STATIC = Path(__file__).parent / "static"

#: Seconds without a write before a run is called stale. The trainer writes every
#: `log_every` steps — 100 steps, which is a few seconds on a pod and up to a
#: minute on a slow CPU box. Generous, because crying wolf at a busy run would
#: teach whoever is watching to ignore the one time it means something.
STALE_AFTER = 180.0


def create_dashboard(directory: Path = DEFAULT_DIR) -> FastAPI:
    app = FastAPI(title="MRZ training", docs_url=None, redoc_url=None)

    @app.get("/api/status")
    async def status() -> JSONResponse:
        current = read_status(directory)
        if current is None:
            # Not an error. Before the first write there is genuinely nothing to
            # report, and a dashboard opened early should say "waiting", not 404.
            return JSONResponse({"state": "waiting", "watching": str(directory)})

        body: dict[str, Any] = asdict(current)
        body["watching"] = str(directory)
        age = time.time() - current.updated_at if current.updated_at else None
        body["age_seconds"] = age
        body["stale"] = bool(
            current.state == "training" and age is not None and age > STALE_AFTER
        )
        return JSONResponse(body)

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(STATIC / "dashboard.html")

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="the run's output directory")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="0.0.0.0 to reach it from outside a pod. Note what that means: this "
             "page is unauthenticated, so bind it wide only where the port is.",
    )
    args = parser.parse_args()

    import uvicorn

    print(f"watching {args.dir}\nhttp://{args.host}:{args.port}")
    uvicorn.run(create_dashboard(args.dir), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
