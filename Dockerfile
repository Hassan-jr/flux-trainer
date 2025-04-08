#!/usr/bin/env bash
# Base image
FROM runpod/base:0.6.2-cuda12.1.0

ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install Git
RUN apt-get update && apt-get install -y git

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
COPY builder/setup.py /setup.py
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python3.11 /cache_models.py && \
    rm /cache_models.py

# Clone your repository with submodules
RUN git clone --recursive https://github.com/Hassan-jr/flux-trainer.git /app
WORKDIR /app

# Copy source files (Worker Template)
COPY src .

# Check if run.py exists in the ai-toolkit directory, if not create the directory
RUN if [ ! -d "/ai-toolkit" ]; then mkdir -p /ai-toolkit; fi

# Check if the file exists in the repository after cloning
RUN if [ -f "/app/ai-toolkit/run.py" ]; then \
        cp /app/ai-toolkit/run.py /ai-toolkit/ && \
        chmod +x /ai-toolkit/run.py; \
    elif [ -f "ai-toolkit/run.py" ]; then \
        cp ai-toolkit/run.py /ai-toolkit/ && \
        chmod +x /ai-toolkit/run.py; \
    else \
        echo "Warning: run.py not found, creating empty file"; \
        touch /ai-toolkit/run.py && \
        chmod +x /ai-toolkit/run.py; \
    fi

CMD python3.11 -u /rp_handler.py