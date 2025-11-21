import os  # Just used to set up file directory
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create 3d contourplot (and surface tesselation) based on 3d array fvals
# sampled on grid with coords determined by xvals, yvals, and zvals
# Note that tesselator requires inputs corresponding to grid spacings
def generateMesh(fvals, scale, modelName='', show=False, decimate_keep=1.0):
    i, j, k = fvals.shape
    xvals = np.linspace(0, i - 1, i, endpoint=True)
    yvals = np.linspace(0, j - 1, j, endpoint=True)
    zvals = np.linspace(0, k - 1, k, endpoint=True)
    verts, faces = tesselate(fvals, xvals, yvals, zvals, scale)
    if decimate_keep < 0.999:
        v0, f0 = verts.shape[0], faces.shape[0]
        verts, faces = decimate_mesh(verts, faces, decimate_keep)
        print(
            f"Decimated mesh: {v0}->{verts.shape[0]} vertices, "
            f"{f0}->{faces.shape[0]} faces (keep={decimate_keep})"
        )
    print("Done Tesselate")
    if modelName != '':
        exportPLY(modelName, verts, faces)
        print('Object exported to Output folder as ' + modelName + '.ply')
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
    ndex = [0, 0, 0]
    frac = [0, 0, 0]
    verts2 = np.ndarray(shape=(verts.size // 3, 3), dtype=float)
    for i in range(0, verts.size // 3):
        for j in range(0, 3):
            ndex[j] = int(verts[i][j])
            frac[j] = verts[i][j] % 1
        verts2[i][0] = (xvals[ndex[0]] + frac[0]) * scale[0]
        verts2[i][1] = (yvals[ndex[1]] + frac[1]) * scale[1]
        verts2[i][2] = (zvals[ndex[2]] + frac[2]) * scale[2]
    return tuple([verts2, faces])


def decimate_mesh(vertices, faces, keep_fraction):
    """Coarse voxel-grid decimation to reduce vertex/face counts."""
    keep_fraction = float(keep_fraction)
    if keep_fraction <= 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)
    if keep_fraction >= 0.999:
        return vertices, faces

    target_vertices = max(int(vertices.shape[0] * keep_fraction), 4)
    bbox_min = vertices.min(axis=0)
    bbox_span = np.maximum(vertices.max(axis=0) - bbox_min, 1e-9)
    bbox_vol = bbox_span[0] * bbox_span[1] * bbox_span[2]
    if bbox_vol <= 1e-12:
        cell_size = bbox_span.max() / max(int(np.sqrt(target_vertices)), 1)
    else:
        cell_size = (bbox_vol / float(target_vertices)) ** (1 / 3)

    best_vertices, best_faces = vertices, faces
    for _ in range(8):
        clustered_vertices, clustered_faces = _cluster_mesh(vertices, faces, cell_size, bbox_min)
        best_vertices, best_faces = clustered_vertices, clustered_faces
        if clustered_vertices.shape[0] <= target_vertices or cell_size >= bbox_span.max():
            break
        cell_size *= 1.35
    return best_vertices, best_faces


def _cluster_mesh(vertices, faces, cell_size, bbox_min):
    """Group vertices into a 3D grid and average positions."""
    keys = np.floor((vertices - bbox_min) / cell_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    unique_count = unique_keys.shape[0]
    reduced_vertices = np.zeros((unique_count, 3), dtype=np.float64)
    np.add.at(reduced_vertices, inverse, vertices)
    counts = np.bincount(inverse).astype(np.float64)
    reduced_vertices /= counts[:, None]

    reduced_faces = inverse[faces]
    valid = (
        (reduced_faces[:, 0] != reduced_faces[:, 1])
        & (reduced_faces[:, 0] != reduced_faces[:, 2])
        & (reduced_faces[:, 1] != reduced_faces[:, 2])
    )
    return reduced_vertices, reduced_faces[valid]
