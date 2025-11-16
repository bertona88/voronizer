# Repository Guidelines

## Project Structure & Module Organization
- Core scripts live in the repository root (``main.py``, ``voronize.py``, ``voxelize.py``, ``Frep.py``, ``SDF3D.py``, etc.).
- User-configurable inputs are in ``userInput.py``; STL assets should be placed in ``Input/``. Output meshes and logs are written to ``Output/``.
- CUDA-powered kernels sit inside the helper modules (e.g., ``Frep.py``, ``SDF3D.py``, ``voxelize.py``). Keep GPU-specific logic there to maintain separation from orchestration code in ``main.py``.

## Build, Test, and Development Commands
- ``pip install -r requirements.txt`` *(if created)* or install the known dependencies individually: ``numba``, ``numpy``, ``matplotlib``, ``Pillow``, ``scikit-image``.
- ``python main.py`` runs the full Voronization pipeline using the parameters set in ``userInput.py``.
- ``python visualizeSlice.py`` can be invoked with custom hooks for debugging slices; wrap usage in ad-hoc scripts as needed.

## Coding Style & Naming Conventions
- Follow PEP 8 for Python: 4-space indentation, snake_case for functions and variables, PascalCase for classes (if added).
- GPU kernels should reside in modules ending with ``.py`` but keep their ``@cuda.jit`` functions near related host helpers.
- Prefer descriptive module-level constants (e.g., ``TPB``, ``BUFFER``). Document non-obvious math with brief comments.

## Testing Guidelines
- No automated test suite exists; validate changes by running ``python main.py`` against representative STLs (e.g., ``Input/E.stl``).
- When adding utilities, include lightweight sanity checks (assertions or small demo scripts) and reference them in comments.
- Log key metrics (point counts, volume/mass) to ensure regressions are caught quickly.

## Commit & Pull Request Guidelines
- Use clear, imperative commit messages (``Fix CUDA launch grid``, ``Document shell-free scaffold workflow``).
- Keep PRs focused: describe the scenario tested, attach before/after metrics or screenshots, and mention any STL fixtures used.
- Link to GitHub issues when available and note any follow-up work (e.g., TODOs for performance tuning or mesh validation).

## GPU & Configuration Tips
- Ensure the target machine has a compatible NVIDIA GPU with CUDA Toolkit installed; failures often stem from missing drivers.
- Adjust ``RESOLUTION`` in ``userInput.py`` to balance fidelity and runtime. Start low (~120) for debugging, then scale up.
- Open Voronoi foam is the default: keep ``MODEL_SHELL = 0`` and tune ``MODEL_CELL`` / ``SUPPORT_CELL`` to set strut diameters.
