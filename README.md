# Voronizer — Open-Cell Strut Generator

Voronizer turns any STL mesh into an airy, open-cell structure built from Voronoi struts. CUDA kernels voxelize the source model, seed a Voronoi diagram, and emit printable lattices for both the model interior and optional support scaffolding. The latest iteration focuses on **hollow, strut-based Voronization**—dial in shell thickness, strut diameters, and surface nets to produce lightweight foams straight from your GPU (or fall back to a CPU path when CUDA is unavailable).

## What You Get
- **Open Voronoi foam**: fill the model volume with struts or keep only a thin surface net.
- **Per-feature control**: independent parameters for model infill and support lattices (`MODEL_CELL`, `SUPPORT_CELL`, thresholds, shells).
- **GPU-accelerated pipeline**: `numba.cuda` kernels for voxelization, signed-distance evaluation, and Voronoi carving.
- **Debug + export tools**: slice visualizer, raw voxel analysis, and mesh exporters for downstream cleanup.

## Requirements
- NVIDIA GPU + CUDA Toolkit (10.x or newer recommended). When CUDA is missing, Voronizer automatically falls back to a CPU implementation of the strut finder—expect noticeably longer runtimes, but identical outputs.
- Python 3.8+ with `numba`, `numpy`, `matplotlib`, `Pillow`, `scikit-image` installed.

```
pip install numba numpy matplotlib Pillow scikit-image
```

## Repository Layout
| Path | Purpose |
| --- | --- |
| `Input/` | Demo STLs; add your own meshes here. |
| `Output/` | Runtime export bucket (ignored by git). Meshes arrive as `.ply`. |
| `userInput.py` | All toggles: select STL, resolution, strut diameters, shell thickness, perforations, etc. |
| `main.py` | Pipeline driver: voxelize → Voronize → smooth → export. |
| `voxelize.py`, `voronize.py`, `Frep.py`, `SDF3D.py` | CUDA kernels and helpers for SDF + Voronoi math. |
| `visualizeSlice.py`, `analysis.py` | Optional helpers for debugging cross-sections and metrics. |

## Quick Start
1. Drop an STL in `Input/`.
2. In `userInput.py` set `FILE_NAME = "myMesh.stl"` and tweak:
   - `RESOLUTION`: grid density (start ~140 for testing).
   - `MODEL_CELL` / `SUPPORT_CELL`: target Voronoi strut thickness in voxels.
   - `MODEL_THRESH` / `SUPPORT_THRESH`: cell density. Larger = more points/struts.
   - `MODEL_SHELL`: add a solid skin outside the lattice (keep `0` for fully open cells).
   - `NET = True` + `NET_THICKNESS`: surface-only webbing.
   - `PERFORATE = True`: drill holes through support cells for resin flushes.
3. Run `python main.py`. On systems without CUDA, the CPU fallback kicks in automatically; just budget extra time for the strut-finding phase.
4. Collect `.ply` meshes from `Output/` and, if necessary, post-process in MeshLab (clean non-manifold faces, apply HC-Laplacian smoothing).

## Open-Cell Tips
- Balance `MODEL_CELL` with `RESOLUTION`: smaller voxels allow thinner struts without breakup.
- For delicate lattices, keep `SMOOTH = True` and consider `AESTHETIC = True` to remove internal clutter while keeping the outer foam.
- Need two printable parts? Set `SEPARATE_SUPPORTS = True` to emit model + supports independently.
- Keep GPU memory headroom: if you see `CudaAPIError: cuMemcpyDtoH`, drop `RESOLUTION` or simplify the STL (MeshMixer → Simplification → 0.5 reduction works well).

## Troubleshooting
- **UnicodeDecodeError while loading STL** → Re-export your mesh as a binary STL.
- **Runtime exceeds minutes** → Lower `RESOLUTION`, decimate the mesh, and close other GPU-hungry apps.
- **Banding or chunky struts** → Increase `SMOOTH_STEPS` in `main.py` or run HC-Laplacian smoothing in MeshLab after export.

## Contributing
- Keep GPU kernels colocated with their host helpers (e.g., CUDA bits stay in `voxelize.py`, `Frep.py`).
- Add lightweight assertions/logging for new utilities so regressions surface quickly.
- Output directories (`Output/`, caches, `.ply`) are ignored; please keep the repo clean of generated meshes.

Create, tweak, and print latticed art—happy Voronizing!
