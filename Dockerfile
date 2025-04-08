# Stage 1: Builder - Fetch code, install dependencies
FROM runpod/base:0.6.2-cuda12.1.0 AS builder

LABEL stage=builder

# Set environment variables for build stage
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Enable faster transfers if dependencies are on Hugging Face Hub
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install Git
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    # Clean up APT cache in the same layer
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the main repository using a shallow clone (--depth 1)
RUN echo "Cloning main repository..." && \
    git clone --depth 1 https://github.com/Hassan-jr/flux-trainer.git . && \
    echo "Cloning complete."

# Initialize and update all submodules recursively
# We will remove .git folders later instead of attempting shallow submodule clones
RUN echo "Initializing submodules..." && \
    git submodule update --init --recursive --jobs 4 && \
    echo "Submodules initialized."

# --- CRITICAL: Remove all .git directories to drastically reduce image size ---
RUN echo "Removing .git directories..." && \
    find . -type d -name ".git" -exec rm -rf {} + && \
    echo ".git directories removed."

# Copy requirements file from the build context
# Assumes requirements.txt is in a 'builder' subdirectory relative to the Dockerfile
COPY builder/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN echo "Installing Python requirements..." && \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r /app/requirements.txt && \
    echo "Python requirements installed."

# --- DO NOT RUN MODEL CACHING SCRIPT HERE ---
# Model downloading will happen at runtime within rp_handler.py


# Stage 2: Final Runtime Image - Copy only necessary artifacts
FROM runpod/base:0.6.2-cuda12.1.0

# Set environment variables for runtime
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # Set path for runtime model downloads (rp_handler.py should use this)
    MODEL_CACHE_DIR="/workspace/models"

WORKDIR /app

# Copy application code (without .git folders) from the builder stage
COPY --from=builder /app /app

# Copy installed Python packages from the builder stage's site-packages
# Adjust the source path if your base image installs packages elsewhere
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy potentially installed command-line tools/scripts from builder
COPY --from=builder /usr/local/bin /usr/local/bin

# Verify ai-toolkit was copied and make its run.py executable
# This check ensures the core code structure is present
RUN if [ ! -f "/app/ai-toolkit/run.py" ]; then \
        echo "ERROR: /app/ai-toolkit/run.py not found after copy from builder stage!"; \
        exit 1; \
    else \
        echo "Found /app/ai-toolkit/run.py - making executable"; \
        chmod +x /app/ai-toolkit/run.py; \
    fi

# Create the symbolic link required by the application
RUN ln -sf /app/ai-toolkit /ai-toolkit

# Copy your RunPod handler script into the working directory
# Assumes rp_handler.py is in the root of your build context
# COPY rp_handler.py /app/rp_handler.py

# --- IMPORTANT ---
# Ensure rp_handler.py contains the logic to:
# 1. Check if models exist in MODEL_CACHE_DIR (/workspace/models)
# 2. Download models using huggingface_hub or diffusers.from_pretrained (with cache_dir=MODEL_CACHE_DIR) if they don't exist
# 3. Load models from MODEL_CACHE_DIR

# Define the command to run your handler
# Ensure the path to rp_handler.py is correct
CMD ["python3.11", "-u", "/app/rp_handler.py"]