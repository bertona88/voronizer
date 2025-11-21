MAT_DENSITY = 1.25  #g/cm^3 (material density), for information only
MODEL = True        #Generates model with infill
SUPPORT = False      #Generates support structure
SEPARATE_SUPPORTS = True #Spits out two files, one for the support and one for the object
PERFORATE = False    #Perforates the support structure to allow fluids into the support cells
IMG_STACK = False   #Outputs an image stack of the model
AESTHETIC = False   #Removes all internal detail, works best with INVERSE
INVERSE = False     #Also includes the inverse of the model as a separate file
NET = True          #Only draws voronoi patterns at the surface of the objec
NET_CONNECT = True  #When NET is True, also fuse in the volumetric Voronoi interior
SMOOTH = True       #Smooths the output meshes, removes the voxelized texture
NET_THICKNESS_MM = 1.0   #Sets the thickness of the net in millimeters (normalized to resolution)
BUFFER_MM = 1.0          #Sets the empty margin around the object in millimeters
TPB = 8             #Threads per block, leave at 8 unless futzing.
SHOW_PLOTS = False   #Disable matplotlib slice/contour plots (useful for batch runs)
AUTO_EXPORT = True   #Skip the interactive export prompt and always write meshes
RUN_LABEL = ""       #Optional label appended to output filenames for sweeps

DECIMATE_KEEP_FRACTION = 0.95  #Keep fraction for in-tool mesh decimation (1.0 disables)

RESOLUTION = 300    #Sets the resolution of the Y and Z axes
MODEL_THRESH = 0.3  #Influences the number of cells in the model, larger values lead to more cells
MODEL_SHELL_MM = 0.0     #Shell thickness in millimeters; keep at 0 for the default open foam lattice
MODEL_CELL_MM = 0.3      #Approximate Voronoi strut diameter within the model (in millimeters)

SUPPORT_THRESH = 1.2 #Influences the number of cells in the supports, larger values lead to more cells
SUPPORT_CELL_MM = 0.35  #Approximate Voronoi strut diameter within the supports (in millimeters)

MODEL_SIZE_MM = 100.0  #Assumed physical model span used to convert mm to voxels (10 cm default)

#Put the name of the desired file below, or uncomment one of the example files
#This file must be in the Input folder, set to be in the same directory as the
#python files.

#FILE_NAME = "sphere.stl"
#FILE_NAME = "Simple heart.stl"
FILE_NAME = "cone.stl"
#FILE_NAME = "3DBenchy_up.stl"
#FILE_NAME = "3DBenchy.stl"
#FILE_NAME = "bust_low.stl"
#FILE_NAME = "wavySurface.stl"


#If you would prefer a simple geometric object, uncomment the one you want and
#make sure that all FILE_NAME options are commented out.

#PRIMITIVE_TYPE = "Heart"
#PRIMITIVE_TYPE = ""
#PRIMITIVE_TYPE = "Cube"
#PRIMITIVE_TYPE = "Sphere"
#PRIMITIVE_TYPE = "Cylinder"
#PRIMITIVE_TYPE = "Silo"
