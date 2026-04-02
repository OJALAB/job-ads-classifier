# Reference CPU image for release 0.2.0.
# For GPU containers, install the matching torch build for your CUDA runtime before the requirements step.
FROM python:3.12-slim

WORKDIR /job-offers-classifier

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install torch \
    && python -m pip install -r requirements.txt

COPY . .

CMD ["python", "main.py", "--help"]

