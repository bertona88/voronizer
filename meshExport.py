import os #Just used to set up file directory
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create 3d contourplot (and surface tesselation) based on 3d array fvals 
# sampled on grid with coords determined by xvals, yvals, and zvals
# Note that tesselator requires inputs corresponding to grid spacings
def generateMesh(fvals, scale, modelName='', show = False):
    i,j,k = fvals.shape
    xvals = np.linspace(0,i-1, i, endpoint=True)
    yvals = np.linspace(0,j-1, j, endpoint=True) 
    zvals = np.linspace(0,k-1, k, endpoint=True)
    verts, faces = tesselate(fvals, xvals, yvals, zvals, scale)    
    print("Done Tesselate")
    if modelName !='':
        exportPLY(modelName, verts, faces)	
        print('Object exported to Output folder as '+modelName+'.ply')
    if show:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)    
        ax.set_xlim(0, i)
        ax.set_ylim(0, j)
        ax.set_zlim(0, k)
        plt.tight_layout()
        plt.show()    
    
def exportPLY(modelName, verts2, faces):
    filepath = os.path.join(os.path.dirname(__file__),'Output',modelName+'.ply')
    num_vertices = verts2.shape[0]
    num_faces = faces.shape[0]
    with open(filepath, 'w') as plyf:
        plyf.write("ply\n")
        plyf.write("format ascii 1.0\n")
        plyf.write("comment ism.py generated\n")
        plyf.write("element vertex " + str(num_vertices) + "\n")
        plyf.write("property float x\n")
        plyf.write("property float y\n")
        plyf.write("property float z\n")
        plyf.write("element face " + str(num_faces) + "\n")
        plyf.write("property list uchar int vertex_indices\n")
        plyf.write("end_header\n")
        for i in range(num_vertices):
            plyf.write(
                f"{verts2[i][0]} {verts2[i][1]} {verts2[i][2]}\n"
            )
        for i in range(num_faces):
            plyf.write(
                f"3 {faces[i][0]} {faces[i][1]} {faces[i][2]}\n"
            )
    
# Compute a tesselation of the zero isosurface
def tesselate(fvals, xvals, yvals, zvals, scale):
    verts, faces, normals, values = measure.marching_cubes(
        fvals, level=0, spacing=(1.0, 1.0, 1.0), allow_degenerate=False
    )
    ndex = [0,0,0]
    frac = [0,0,0]
    verts2 = np.ndarray(shape=(verts.size//3,3), dtype=float)
    for i in range(0,verts.size//3):
        for j in range(0,3):
            ndex[j] = int(verts[i][j])
            frac[j] = verts[i][j]%1
        verts2[i][0] = (xvals[ndex[0]]+frac[0])*scale[0]
        verts2[i][1] = (yvals[ndex[1]]+frac[1])*scale[1]
        verts2[i][2] = (zvals[ndex[2]]+frac[2])*scale[2]
    return tuple([verts2, faces])
    
