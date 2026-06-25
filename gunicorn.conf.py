# Gunicorn configuration for render.com deployments.
# render.com sends SIGTERM and waits 60 s before SIGKILL on each deploy;
# graceful_timeout must stay well below that budget so in-flight requests
# can drain cleanly before the hard kill arrives.

bind = "0.0.0.0:8050"
workers = 1            # stateless app — a single sync worker is sufficient
timeout = 120          # max seconds a worker may spend on one request
graceful_timeout = 45  # seconds to finish in-flight requests after SIGTERM
