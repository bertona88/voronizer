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
