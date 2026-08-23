# Gunicorn configuration for all PaaS deployments.
# Read PORT from environment (default: 8050 for local dev).
# Most PaaS providers (Render, Fly.io, Railway, etc.) set PORT via environment.
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8050')}"
workers = int(os.getenv('GUNICORN_WORKERS', '1'))  # stateless app — a single sync worker is sufficient
timeout = int(os.getenv('GUNICORN_TIMEOUT', '120'))  # max seconds a worker may spend on one request
graceful_timeout = int(os.getenv('GUNICORN_GRACEFUL_TIMEOUT', '45'))  # seconds to finish in-flight requests after SIGTERM
