FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY web ./web
COPY docs ./docs

RUN pip install --no-cache-dir -e .

ENV DICO_HOST=0.0.0.0 \
    DICO_PORT=7400 \
    DICO_DATA_DIR=/var/lib/dico \
    DICO_AUTH_DISABLED=true

VOLUME ["/var/lib/dico"]
EXPOSE 7400

CMD ["dico", "coordinator", "--host", "0.0.0.0", "--port", "7400", "--data-dir", "/var/lib/dico", "--auth-disabled"]
