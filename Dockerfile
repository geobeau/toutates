FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    wget \
    btop \
    ca-certificates \
    gnupg \
    gdb \ 
    linux-tools-generic \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA 13 (toolkit 13.0)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-13-0 libcudnn9-cuda-13 libnvinfer-plugin10 libnvinfer-headers-plugin-dev libnvinfer10 libnvinfer-headers-dev libnvinfer-lean10 libnvinfer-dispatch10 libnvonnxparsers10 libnvonnxparsers-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install Rust nightly
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:${PATH}"

# Download and install ONNX Runtime CUDA 13
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-gpu_cuda13-1.26.0.tgz \
    && tar -xzf onnxruntime-linux-x64-gpu_cuda13-1.26.0.tgz -C /opt \
    && rm onnxruntime-linux-x64-gpu_cuda13-1.26.0.tgz

# Set ONNX Runtime environment variables
ENV ORT_LIB_LOCATION="/opt/onnxruntime-linux-x64-gpu-1.26.0/lib"
ENV ORT_DYLIB_PATH="/opt/onnxruntime-linux-x64-gpu-1.26.0/lib/libonnxruntime.so"
ENV LD_LIBRARY_PATH="${ORT_LIB_LOCATION}:${LD_LIBRARY_PATH}"

# Copy project source
WORKDIR /build/inference-server
COPY . .

# Build the project
RUN cargo build --release \
    && cp target/release/inference-server /usr/local/bin/inference-server \
    && rm -rf target /root/.cargo/registry /root/.cargo/git

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser
USER 1001

ENTRYPOINT ["/usr/local/bin/inference-server"]
