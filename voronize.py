from collections import deque
from visualizeSlice import slicePlot, contourPlot
import Frep as f
from SDF3D import SDF3D, jumpFlood
from numba import cuda, njit, prange
import numpy as np
import userInput as u
try: TPB = u.TPB 
except: TPB = 8
try:
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

@njit(parallel=True)
def _strut_finder_cpu(points):
    x, y, z, _ = points.shape
    out = np.ones((x, y, z), dtype=np.float32)
    for i in prange(x):
        for j in range(y):
            for k in range(z):
                m0 = points[i, j, k, 0]
                n0 = points[i, j, k, 1]
                p0 = points[i, j, k, 2]
                sm, sn, sp = m0, n0, p0
                tm, tn, tp = m0, n0, p0
                unique = 1
                for di in range(-1, 2):
                    ii = i + di
                    if ii < 0 or ii >= x:
                        continue
                    for dj in range(-1, 2):
                        jj = j + dj
                        if jj < 0 or jj >= y:
                            continue
                        for dk in range(-1, 2):
                            kk = k + dk
                            if kk < 0 or kk >= z:
                                continue
                            m1 = points[ii, jj, kk, 0]
                            n1 = points[ii, jj, kk, 1]
                            p1 = points[ii, jj, kk, 2]
                            if m1 != m0 or n1 != n0 or p1 != p0:
                                if unique == 1:
                                    sm, sn, sp = m1, n1, p1
                                    unique = 2
                                elif m1 != sm or n1 != sn or p1 != sp:
                                    if unique == 2:
                                        tm, tn, tp = m1, n1, p1
                                        unique = 3
                                    elif m1 != tm or n1 != tn or p1 != tp:
                                        unique = 4
                                        break
                        if unique >= 4:
                            break
                    if unique >= 4:
                        break
                if unique >= 3:
                    out[i, j, k] = -1.0
    return out

def voronize(origObject, seedPoints, cellThickness, shellThickness, scale,
             name = "", sliceLocation = 0, sliceAxis = "X", order = 2):
    #origObject = voxel model of original object, negative = inside
    #seedPoints = same-size matrix with 0s at the location of each seed point, 1s elsewhere
    #cellThickness = approximate strut diameter (in voxels) for the open Voronoi lattice.
    #shellThickness = desired minimum thickness of shell (mm), 0 if no shell
    #name = If given a value, the name of the model, activates progress plots.
    resX, resY, resZ = origObject.shape
    if sliceLocation == 0:
        if sliceAxis == "X" or sliceAxis == "x":
            sliceLocation = resX//2
        elif sliceAxis == "Y" or sliceAxis =="y":
            sliceLocation = resY//2
        else:
            sliceLocation = resZ//2
    seedPoints = jumpFlood(seedPoints,order)
    if name !="":
        contourPlot(seedPoints[:,:,:,3],sliceLocation,titlestring="SDF of the Points for "+name,axis = sliceAxis)
    voronoi = strutFinder(seedPoints)
    voronoi = SDF3D(voronoi)
    if name !="":
        slicePlot(voronoi,sliceLocation,titlestring="Voronoi Structure for "+name,axis = sliceAxis)
    strutRadius = max(cellThickness/2.0, 0.0)
    voronoi = f.intersection(f.thicken(voronoi,strutRadius),origObject)
    if name !="":
        slicePlot(voronoi, sliceLocation, titlestring=(name+' Trimmed and Thinned'),axis = sliceAxis)
    if shellThickness>0:
        u_shell = f.shell(origObject,shellThickness)
        voronoi = f.union(u_shell,voronoi)
        if name !="":
            slicePlot(voronoi, sliceLocation, titlestring=name+' With Shell',axis = sliceAxis)
    if name =="":
        name = "Model"
    print("Voronize for " + name + " Complete!")
    return voronoi

def surface_voronoi_net(orig_sdf, seed_density, net_thickness, cell_thickness, scale, name="Object"):
    """Generate a classic surface Voronoi net: seeds live on the true surface, Voronoi cells are computed over a thin shell, then extruded."""
    shell_thickness = max(1, int(round(net_thickness)))
    shell_sdf = f.shell(orig_sdf, shell_thickness)
    surface_mask = shell_sdf <= 0
    surface_band = np.abs(orig_sdf) <= max(1.0, net_thickness * 0.75)
    prob = seed_density / max(orig_sdf.shape)
    rng = np.random.rand(*orig_sdf.shape)
    seed_mask = surface_band & (rng < prob)
    if not seed_mask.any():
        candidates = np.argwhere(surface_band)
        if candidates.size == 0:
            candidates = np.argwhere(surface_mask)
        if candidates.size:
            sx, sy, sz = candidates[np.random.randint(len(candidates))]
            seed_mask[sx, sy, sz] = True
    seed_coords = np.argwhere(seed_mask)
    labels = -np.ones(orig_sdf.shape, dtype=np.int32)
    dist = np.full(orig_sdf.shape, np.inf, dtype=np.float32)
    q = deque()
    for idx, (x, y, z) in enumerate(seed_coords):
        labels[x, y, z] = idx
        dist[x, y, z] = 0.0
        q.append((x, y, z))
    neighbors = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    while q:
        x, y, z = q.popleft()
        cur_label = labels[x, y, z]
        cur_dist = dist[x, y, z]
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            if nx < 0 or ny < 0 or nz < 0 or nx >= orig_sdf.shape[0] or ny >= orig_sdf.shape[1] or nz >= orig_sdf.shape[2]:
                continue
            if not surface_mask[nx, ny, nz]:
                continue
            nd = cur_dist + 1.0
            if nd < dist[nx, ny, nz]:
                dist[nx, ny, nz] = nd
                labels[nx, ny, nz] = cur_label
                q.append((nx, ny, nz))
    boundary = np.zeros_like(surface_mask, dtype=bool)
    dims = labels.shape
    for dx, dy, dz in neighbors:
        src = (slice(max(0, -dx), dims[0]-max(0, dx)),
               slice(max(0, -dy), dims[1]-max(0, dy)),
               slice(max(0, -dz), dims[2]-max(0, dz)))
        dst = (slice(max(0, dx), dims[0]-max(0, -dx)),
               slice(max(0, dy), dims[1]-max(0, -dy)),
               slice(max(0, dz), dims[2]-max(0, -dz)))
        overlap = surface_mask[src] & surface_mask[dst]
        mismatch = (labels[src] != -1) & (labels[dst] != -1) & (labels[src] != labels[dst])
        boundary[src] |= overlap & mismatch
        boundary[dst] |= overlap & mismatch
    strut_field = np.where(boundary, -1.0, 1.0).astype(np.float32)
    strut_field = SDF3D(strut_field)
    radius = max(cell_thickness/2.0, 0.0)
    surface_net = f.intersection(f.thicken(strut_field, radius), shell_sdf)
    if name:
        slice_axis = "X"
        slice_loc = surface_net.shape[0] // 2
        slicePlot(surface_net, slice_loc, titlestring=f"Surface Voronoi Net for {name}", axis=slice_axis)
    print("Surface Voronoi net complete!")
    return surface_net

@cuda.jit
def strutFinderKernel(d_points,d_struts):
    i,j,k = cuda.grid(3)
    dims = d_struts.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    m0,n0,p0,_ = d_points[i,j,k]
    second_m, second_n, second_p = m0, n0, p0
    third_m, third_n, third_p = m0, n0, p0
    unique_count = 1
    for index in range(27):
        check_i = i + ((index//9)%3 - 1)
        check_j = j + ((index//3)%3 - 1)
        check_k = k + (index%3 - 1)
        if 0 <= check_i < dims[0] and 0 <= check_j < dims[1] and 0 <= check_k < dims[2]:
            m1,n1,p1,_ = d_points[check_i,check_j,check_k]
            if m1!=m0 or n1!=n0 or p1!=p0:
                if unique_count == 1:
                    second_m, second_n, second_p = m1, n1, p1
                    unique_count = 2
                elif m1!=second_m or n1!=second_n or p1!=second_p:
                    if unique_count == 2:
                        third_m, third_n, third_p = m1, n1, p1
                        unique_count = 3
                    elif m1!=third_m or n1!=third_n or p1!=third_p:
                        unique_count = 4
                        break
    if unique_count >= 3:
        d_struts[i,j,k] = -1
        
def strutFinder(voxel):
    #voxel = jump-flood output with nearest-seed metadata per voxel
    #Outputs a voxel model with negative voxels along Voronoi edges/vertices to seed struts.
    if not CUDA_AVAILABLE:
        return _strut_finder_cpu(voxel)
    dims = voxel.shape
    d_points = cuda.to_device(voxel)
    d_struts = cuda.to_device(np.ones(dims[:3]))
    gridSize = (
        (dims[0] + TPB - 1) // TPB,
        (dims[1] + TPB - 1) // TPB,
        (dims[2] + TPB - 1) // TPB,
    )
    blockSize = (TPB, TPB, TPB)
    strutFinderKernel[gridSize, blockSize](d_points,d_struts)
    return d_struts.copy_to_host()
