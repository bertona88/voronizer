# Voronizer — Open-Cell Strut Generator

Voronizer turns any STL mesh into an airy, open-cell structure built from Voronoi struts. CUDA kernels voxelize the source model, seed a Voronoi diagram, and emit printable lattices for both the model interior and optional support scaffolding. The latest iteration focuses on **hollow, strut-based Voronization**—dial in shell thickness, strut diameters, and surface nets to produce lightweight foams straight from your GPU (or fall back to a CPU path when CUDA is unavailable).

## What You Get
- **Open Voronoi foam**: fill the model volume with struts or keep only a thin surface net.
- **Surface or hybrid nets**: toggle `NET = True` for a classic surface Voronoi tiling; set `NET_CONNECT = True` to fuse that net with a full volumetric lattice inside.
- **Per-feature control**: independent parameters for model infill and support lattices (`MODEL_CELL_MM`, `SUPPORT_CELL_MM`, thresholds, shells).
- **GPU-accelerated pipeline**: `numba.cuda` kernels for voxelization, signed-distance evaluation, and Voronoi carving.
- **Debug + export tools**: slice visualizer, raw voxel analysis, and mesh exporters for downstream cleanup.

## Requirements
- NVIDIA GPU + CUDA Toolkit (10.x or newer recommended). When CUDA is missing, Voronizer automatically falls back to a CPU implementation of the strut finder—expect noticeably longer runtimes on large grids (though tiny lattices can actually finish faster on CPU because they avoid CUDA launch/transfer overhead).
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
   - `RESOLUTION`: grid density (start ~140 for testing). Physical parameters stay consistent across resolutions using `MODEL_SIZE_MM` (10 cm default span).
   - `DECIMATE_KEEP_FRACTION`: 0-1 fraction of vertices/faces to keep during built-in mesh decimation (1.0 disables, e.g., 0.5 roughly halves mesh size).
   - `MODEL_CELL_MM` / `SUPPORT_CELL_MM`: target Voronoi strut thickness in millimeters.
   - `MODEL_THRESH` / `SUPPORT_THRESH`: cell density. Larger = more points/struts.
   - `MODEL_SHELL_MM`: add a solid skin outside the lattice (keep `0` for fully open cells).
   - `NET = True` + `NET_THICKNESS_MM`: surface-only webbing using a surface Voronoi tiling.
   - `NET_CONNECT = True`: fuse the surface net with the full volumetric Voronoi interior.
   - `BUFFER_MM`: empty margin around the model before processing.
   - `SHOW_PLOTS`: turn slicing/contour figures on/off (disable for batch sweeps).
   - `AUTO_EXPORT` + `RUN_LABEL`: skip the interactive export prompt and append a label to auto-named `.ply` outputs.
   - `PERFORATE = True`: drill holes through support cells for resin flushes.
3. Run `python main.py`. On systems without CUDA, the CPU fallback kicks in automatically; just budget extra time for the strut-finding phase (unless you’re experimenting on very small structures, where the CPU route can be quicker).
4. Collect `.ply` meshes from `Output/` and, if necessary, post-process in MeshLab (clean non-manifold faces, apply HC-Laplacian smoothing).

Exports are auto-named with resolution, thresholds, strut diameters, and optional `RUN_LABEL`. Meshes can be decimated in-tool via `DECIMATE_KEEP_FRACTION` to avoid massive `.ply` files.

For quick experiments, try `python run_sweeps.py`—it runs a few preset surface/hybrid net configurations against `Input/sphere.stl` with plotting disabled and auto-export enabled.

## Open-Cell Tips
- Voxel scale now comes from millimeter inputs; adjust `MODEL_SIZE_MM` if your mesh is significantly larger/smaller than 10 cm so buffers and strut diameters stay physically consistent.
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
