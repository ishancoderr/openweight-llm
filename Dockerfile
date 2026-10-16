FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Compiles llama.cpp for the T4 only (compute capability 7.5); takes ~10-20 min on 4 vCPUs.
# There is no GPU during `docker build`, so libcuda is linked against the CUDA stub library.
# The stub is only on the path for this RUN; at runtime the NVIDIA toolkit mounts the real driver.
COPY requirements.txt .
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH \
       CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" \
       pip3 install -r requirements.txt

COPY app ./app

# Run as a normal user, not root; the GPU and the read-only /models mount don't need root.
RUN useradd --create-home --uid 1000 llm
USER llm

EXPOSE 8000

# One worker only: every worker would load its own two copies of the model.
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
