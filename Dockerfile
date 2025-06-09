# Intel-Optimized Financial GNN Dockerfile
# Multi-stage build for production-ready container

# Stage 1: Build environment with Intel tools
FROM intel/oneapi-runtime:2024.0.1-devel-ubuntu22.04 AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python3.9 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Intel optimizations
RUN pip install --no-cache-dir \
    openvino>=2023.1.0 \
    intel-extension-for-pytorch>=2.0.0 \
    mkl>=2023.2.0

# Copy source code
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY README.md .
COPY LICENSE .

# Install the package
RUN pip install -e .

# Stage 2: Production runtime
FROM intel/oneapi-runtime:2024.0.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONPATH=/app/src
ENV INTEL_MKL_VERBOSE=0
ENV KMP_BLOCKTIME=1
ENV KMP_SETTINGS=1
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV OMP_NUM_THREADS=4

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.9 -c "import src; print('Health check passed')" || exit 1

# Expose port for web interface
EXPOSE 8080

# Default command
CMD ["python3.9", "src/main.py", "--symbols", "AAPL", "GOOGL", "MSFT", "AMZN"]

# Labels for better container management
LABEL maintainer="vansh.ramani@example.com"
LABEL version="1.0.0"
LABEL description="Intel-Optimized Financial Graph Neural Network"
LABEL org.opencontainers.image.source="https://github.com/VanshRamani/intel-financial-gnn"
LABEL org.opencontainers.image.documentation="https://github.com/VanshRamani/intel-financial-gnn/blob/main/README.md"
LABEL org.opencontainers.image.licenses="MIT" 