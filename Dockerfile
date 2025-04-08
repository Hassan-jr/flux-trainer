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

# Clone the main repository and change to its directory
RUN git clone https://github.com/Hassan-jr/flux-trainer.git /app
WORKDIR /app

# Initialize and update all submodules (including nested ones)
RUN git submodule update --init --recursive

# Verify and clone ai-toolkit if needed
RUN if [ ! -d "/app/ai-toolkit" ] || [ ! -f "/app/ai-toolkit/run.py" ]; then \
        echo "Cloning ai-toolkit repository directly..."; \
        rm -rf /app/ai-toolkit; \
        git clone https://github.com/ostris/ai-toolkit.git /app/ai-toolkit; \
        cd /app/ai-toolkit && git submodule update --init --recursive; \
    fi

# Make the run.py file executable
RUN if [ -f "/app/ai-toolkit/run.py" ]; then \
        echo "Found /app/ai-toolkit/run.py - making executable"; \
        chmod +x /app/ai-toolkit/run.py; \
    else \
        echo "ERROR: Could not find ai-toolkit/run.py"; \
        exit 1; \
    fi

# Now create a symbolic link from /app/ai-toolkit to /ai-toolkit
RUN ln -sf /app/ai-toolkit /ai-toolkit

# Copy additional source files (if needed)
# COPY src .

CMD python3.11 -u /rp_handler.py