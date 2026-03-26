FROM julia:1.11

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libhdf5-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Julia deps first (cached layer)
COPY julia/Project.toml julia/Project.toml
RUN cd julia && julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Python deps (cached layer)
COPY python/pyproject.toml python/pyproject.toml
RUN cd python && python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY . .

# Re-install with source
RUN cd python && .venv/bin/pip install --no-cache-dir -e ".[dev]"
RUN cd julia && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

CMD ["bash"]
