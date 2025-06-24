# TokenSlasher runtime image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# System deps for compiling xxhash / spaCy small model
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . /app

# Install package with semantic + dashboard extras and download spaCy model
RUN pip install --no-cache-dir --upgrade pip && \
    pip install .[semantic,dashboard] && \
    python -m spacy download en_core_web_sm --quiet

# Default command shows CLI help; override with docker run … tokenslasher …
ENTRYPOINT ["tokenslasher"] 