import os #Just used to set up file directory
import sys
import time
import numpy as np
import Frep as f
import userInput as u
from voronize import voronize, surface_voronoi_net
from SDF3D import SDF3D, xHeight
from pointGen import genRandPoints, explode
from meshExport import generateMesh
from analysis import findVol
from visualizeSlice import slicePlot, contourPlot, generateImageStack
from voxelize import voxelize


def format_param(value):
    """Stable string for filenames; trims trailing zeros."""
    if isinstance(value, float):
        return format(value, ".2f").rstrip("0").rstrip(".")
    return str(value)


def mm_to_voxels(mm_value, voxel_size_mm):
    """Convert a physical distance (mm) to voxels for the current grid spacing."""
    if voxel_size_mm <= 0:
        return mm_value
    return mm_value / voxel_size_mm

def main():
    start = time.time()
    try:    os.mkdir(os.path.join(os.path.dirname(__file__),'Output')) #Creates an output folder if there isn't one yet
    except: pass
    try:    FILE_NAME = u.FILE_NAME
    except: FILE_NAME = ""
    try:    PRIMITIVE_TYPE = u.PRIMITIVE_TYPE #Checks to see if a primitive type has been set
    except: PRIMITIVE_TYPE = ""
    modelImport = False
    scale = [1,1,1]
    assumed_voxel_size_mm = u.MODEL_SIZE_MM / float(u.RESOLUTION)
    buffer_vox = max(int(round(u.BUFFER_MM / assumed_voxel_size_mm)), 0)
    voxel_size_mm = assumed_voxel_size_mm
    if not u.MODEL and not u.SUPPORT:
        print("You need at least the model or the support structure.")
        return

    repo_dir = os.path.dirname(__file__)
    cli_path = sys.argv[1] if len(sys.argv) > 1 else ""
    filepath = ""
    show_plots = getattr(u, "SHOW_PLOTS", True)
    auto_export = getattr(u, "AUTO_EXPORT", False)
    run_label = getattr(u, "RUN_LABEL", "")

    if cli_path:
        candidate_paths = [
            os.path.abspath(cli_path),
            os.path.abspath(os.path.join(repo_dir, cli_path)),
        ]
        filepath = next((p for p in candidate_paths if os.path.exists(p)), "")
        if not filepath:
            print("Input file not found.")
            return
        FILE_NAME = os.path.basename(filepath)
        shortName = os.path.splitext(FILE_NAME)[0]
        modelImport = True
    elif FILE_NAME != "":
        filepath = os.path.join(repo_dir, 'Input', FILE_NAME)
        if not os.path.exists(filepath):
            print("Input file not found.")
            return
        shortName = os.path.splitext(FILE_NAME)[0]
        modelImport = True
    elif PRIMITIVE_TYPE != "":
        shortName = PRIMITIVE_TYPE
        if PRIMITIVE_TYPE == "Heart":
            x0 = np.linspace(-1.5,1.5,u.RESOLUTION)
            y0, z0 = x0, x0
            origShape = f.heart(x0,y0,z0,0,0,0)
        elif PRIMITIVE_TYPE == "Egg":
            x0 = np.linspace(-5,5,u.RESOLUTION)
            y0, z0 = x0, x0
            origShape = f.egg(x0,y0,z0,0,0,0)
            #eggknowledgement to Molly Carton for this feature.
        else:
            x0 = np.linspace(-50,50,u.RESOLUTION)
            y0, z0 = x0, x0
            if PRIMITIVE_TYPE == "Cube":
                origShape = f.rect(x0,y0,z0,80,80,80)
            elif PRIMITIVE_TYPE == "Silo":
                origShape = f.union(f.sphere(x0,y0,z0,40),f.cylinderY(x0,y0,z0,-40,0,40))
            elif PRIMITIVE_TYPE == "Cylinder":
                origShape = f.cylinderX(x0,y0,z0,-40,40,40)
            elif PRIMITIVE_TYPE == "Sphere":
                origShape = f.sphere(x0,y0,z0,40)
            else:
                print("Selected primitive type has not yet been implemented.")
    else:
        print("Provide either a file name or a desired primitive.")
        return

    if modelImport:
        res = max(int(u.RESOLUTION - buffer_vox*2), 1)
        origShape, objectBox = voxelize(filepath, res, buffer_vox)
        gridResX, gridResY, gridResZ = origShape.shape
        scale[0] = objectBox[0]/(gridResX-buffer_vox*2)
        scale[1] = max(objectBox[1:])/(gridResY-buffer_vox*2)
        scale[2] = scale[1]
        voxel_size_mm = float(np.mean(scale))

    print("Initial Bounding Box Dimensions: "+str(origShape.shape))
    origShape = SDF3D(f.condense(origShape,buffer_vox))
    print("Condensed Bounding Box Dimensions: "+str(origShape.shape))

    model_cell_vox = mm_to_voxels(u.MODEL_CELL_MM, voxel_size_mm)
    support_cell_vox = mm_to_voxels(u.SUPPORT_CELL_MM, voxel_size_mm)
    model_shell_vox = mm_to_voxels(u.MODEL_SHELL_MM, voxel_size_mm)
    net_thickness_vox = mm_to_voxels(u.NET_THICKNESS_MM, voxel_size_mm)
    
    if u.SUPPORT:
        projected = f.projection(origShape)
        support = f.subtract(f.thicken(origShape,1),projected)
        support = f.intersection(support, f.translate(support,-1,0,0))
        if show_plots:
            contourPlot(support,30,titlestring='Support',axis ="Z")
        supportPts = genRandPoints(xHeight(support), u.SUPPORT_THRESH)
        supportVoronoi = voronize(support, supportPts, support_cell_vox, 0, scale, name = "Support", sliceAxis = "Z")
        if u.PERFORATE: 
            explosion = f.union(explode(supportPts), f.translate(explode(supportPts),-1,0,0))
            explosion = f.union(explosion,f.translate(explosion,0,1,0))
            explosion = f.union(explosion,f.translate(explosion,0,0,1))
            supportVoronoi = f.subtract(explosion,supportVoronoi)
        table = f.subtract(f.thicken(origShape,1),f.intersection(f.translate(f.subtract(origShape,f.translate(origShape,-3,0,0)),-1,0,0),projected))
        supportVoronoi = f.union(table,supportVoronoi)
        findVol(supportVoronoi,scale,u.MAT_DENSITY,"Support")
    
    if u.MODEL:
        if u.NET:
            surfaceNet = surface_voronoi_net(origShape, u.MODEL_THRESH, net_thickness_vox, model_cell_vox, scale, name="Object")
            if u.NET_CONNECT:
                volumePts = genRandPoints(origShape,u.MODEL_THRESH)
                print("Points Generated (volume)!")
                volumeVoronoi = voronize(origShape, volumePts, model_cell_vox, model_shell_vox, scale, name = "Object Interior")
                objectVoronoi = f.union(surfaceNet, volumeVoronoi)
            else:
                objectVoronoi = surfaceNet
            findVol(objectVoronoi, scale, u.MAT_DENSITY, "Object")
        else:
            if u.AESTHETIC:
                objectPts = genRandPoints(f.shell(origShape,5),u.MODEL_THRESH)
            else:
                objectPts = genRandPoints(origShape,u.MODEL_THRESH)
            print("Points Generated!")
            objectVoronoi = voronize(origShape, objectPts, model_cell_vox, model_shell_vox, scale, name = "Object")
            findVol(objectVoronoi,scale,u.MAT_DENSITY,"Object") #in mm^3
            if u.AESTHETIC:
                objectVoronoi = f.union(objectVoronoi,f.thicken(origShape,-5))
    shortName = shortName+"_Voronoi"
    if u.SUPPORT and u.MODEL:
        complete = f.union(objectVoronoi,supportVoronoi)
        if u.IMG_STACK:
            generateImageStack(objectVoronoi,[255,0,0],supportVoronoi,[0,0,255],name = shortName)
    elif u.SUPPORT:
        complete = supportVoronoi
        if u.IMG_STACK:
            generateImageStack(supportVoronoi,[0,0,0],supportVoronoi,[0,0,255],name = shortName)
    elif u.MODEL:
        complete = objectVoronoi
        if u.IMG_STACK:
            generateImageStack(objectVoronoi,[255,0,0],objectVoronoi,[0,0,0],name = FILE_NAME[:-4])
    if show_plots:
        slicePlot(complete, origShape.shape[0]//2, titlestring='Full Model', axis = "X")
        slicePlot(complete, origShape.shape[1]//2, titlestring='Full Model', axis = "Y")
        slicePlot(complete, origShape.shape[2]//2, titlestring='Full Model', axis = "Z")
    
    print("That took "+str(round(time.time()-start,2))+" seconds.")
    if not auto_export:
        print("AUTO_EXPORT disabled; skipping mesh export.")
        return

    export_parts = [
        f"res{u.RESOLUTION}",
        f"th{format_param(u.MODEL_THRESH)}",
        f"cell{format_param(u.MODEL_CELL_MM)}",
        f"net{format_param(u.NET_THICKNESS_MM)}" if u.NET else "nonet",
        f"shell{format_param(u.MODEL_SHELL_MM)}",
    ]
    if u.NET and u.NET_CONNECT:
        export_parts.append("netconnect")
    if u.AESTHETIC:
        export_parts.append("aesthetic")
    if run_label:
        export_parts.append(run_label)
    export_suffix = "_".join(export_parts)

    base_name = f"{shortName}_{export_suffix}"
    print("Generating Model...")
    if u.SEPARATE_SUPPORTS and u.SUPPORT and u.MODEL:
        if u.SMOOTH:
            objectVoronoi = f.smooth(objectVoronoi)
        generateMesh(objectVoronoi, scale, modelName=base_name, decimate_keep=u.DECIMATE_KEEP_FRACTION)
        print("Generating Supports...")
        if u.SMOOTH:
            supportVoronoi = f.smooth(supportVoronoi)
        generateMesh(supportVoronoi, scale, modelName=base_name+"Support", decimate_keep=u.DECIMATE_KEEP_FRACTION)
    else:
        if u.SMOOTH:
            complete = f.smooth(complete)
        generateMesh(complete, scale, modelName=base_name, decimate_keep=u.DECIMATE_KEEP_FRACTION)
    if u.INVERSE and u.MODEL:
        print("Generating Inverse...")
        inv=f.subtract(objectVoronoi,origShape)
        if u.SMOOTH:
            inv = f.smooth(inv)
        print("Generating Mesh...")
        generateMesh(inv, scale, modelName=base_name+"Inv", decimate_keep=u.DECIMATE_KEEP_FRACTION)

if __name__ == '__main__':
    main()
