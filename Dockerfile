# Production image for the mrktcmp Dash web app.
# Build:  docker build -t mrktcmp .
# Run:   docker run -e BASE_URL=/data -v /srv/mrktcmp-data:/data:ro -p 8050:8050 mrktcmp
FROM python:3.14-slim

# setuptools-scm normally derives the version from git metadata, which is not
# present inside the build context. Pass the version explicitly instead:
#   docker build --build-arg APP_VERSION=$(git describe --tags) -t mrktcmp .
ARG APP_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${APP_VERSION} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8050

WORKDIR /app

COPY pyproject.toml README.md LICENSE gunicorn.conf.py ./
COPY src ./src
RUN pip install .

# Run as a non-root user; UIDs/GIDs are configurable so the container can
# match the host user that owns the mounted parquet data directory.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -u ${UID} -g app -s /usr/sbin/nologin app

USER app
EXPOSE 8050

# gunicorn.conf.py reads PORT and GUNICORN_* variables from the environment.
CMD ["gunicorn", "--config", "gunicorn.conf.py", "src.app:server"]
