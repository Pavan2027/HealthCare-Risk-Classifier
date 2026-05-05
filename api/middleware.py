"""
FastAPI middleware for CORS, request logging, and latency tracking.
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("health_classifier")


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({latency_ms:.1f}ms)")
        return response


def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Response-Time"],
    )


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("health_classifier")
