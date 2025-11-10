### uv setup (Python 3.12 + JAX CUDA12)

- Requirements
  - **Python**: 3.12
  - **CUDA**: 12.x toolchain and compatible NVIDIA drivers (for GPU)

- Install uv
  - Linux/macOS:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    uv --version
    ```

 - Install deps (uses JAX CUDA wheels)
   ```bash
   uv sync --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

