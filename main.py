from ArterialTree import ArterialTree
from Simulation import Simulation
from Editor import Editor
from Nfurcation import Nfurcation

import numpy as np


# Import centerline data to create an ArterialTree object
# Import from swc file
centerline_filename = "Data/example_centerline_ICA.swc"
tree = ArterialTree("patient1", "Aneurisk", centerline_filename)

# Uncomment to import from txt model file (ideal models)
#centerline_filename = "Data/ideal_model.swc"
#tree = ArterialTree("patient1", "Ideal", centerline_filename)

# Import from edge and node files
#centerline_filename_edg = "Data/example_centerline_ICA_edg.txt"
#centerline_filename_nds = "Data/example_centerline_ICA_nds.txt"
#tree = ArterialTree("patient1", "Aneurisk", [centerline_filename_edg, centerline_filename_nds])
"""
###### MESHING WITHOUT GUI #########

tree.show(True, False, False)
	
tree.model_network()  # Parametric modeling of the network
tree.show(False, True, False)

#tree.add_extensions(size=10) # Uncomment to add flow extensions
#tree.show(False, True, False)

# Define meshing parameters
layer_ratio = [0.2, 0.4, 0.4] # Layer ratio [la, lb, lc] la+lb+lc = 1
num_a , num_b = 4, 4 # Number of layers for each volume area
N = 48 # Number of nodes in the cross section circumference
d = 0.25 # Longitudinal density of cross sections

tree.compute_cross_sections(N, d, parallel=False) # Mesh cross sections
	
surface_mesh = tree.mesh_surface() # Mesh surface
#surface_mesh.plot(show_edges=True)
surface_mesh.save("Output/surface_mesh.vtk")

# Uncomment to add a stenosis to the model
#template_dir = "pathology_templates/default/"
#template, temp_rad, temp_center = tree.load_pathology_template(template_dir)
#tree.deform_surface_to_template((1,2),  0.6, 0.7, template, temp_rad, temp_center, method = "bicubic", rotate = 0)
#surface_mesh_stenosis = tree.get_surface_mesh()
#surface_mesh_stenosis.save("Output/surface_mesh_stenosis.vtk")

# Uncomment to close the inlet and oulets of the surface mesh
#tree.close_surface()
#surface_mesh_closed = tree.get_surface_mesh()
#surface_mesh_closed.save("Output/surface_mesh_closed.vtk")


volume_mesh = tree.mesh_volume([0.2, 0.4, 0.4], 4, 4) # Mesh volume
#volume_mesh.plot(show_edges=True)
volume_mesh.save("Output/volume_mesh.vtk")


# Write the mesh case to OpenFoam for CFD (Can be converted to Ansys using OpenFoam's foamMeshToFluent function)

path_to_OF_casefolder = "Output/OpenFoam/"

# Write OpenFoam file
simu = Simulation(tree, path_to_OF_casefolder)
simu.write_mesh_files()
simu.write_pressure_boundary_condition_file()
simu.write_velocity_boundary_condition_file([0.2])
"""

###### MESHING WITH GUI #########

# Open the user interface for editing, modeling, meshing
e = Editor(tree, 1500, 600)


##### SINGLE BIFURCATION EXAMPLE #####

# Example of modeling and meshing of a single bifurcation

end_sections = [np.array([[23.9598871 , 23.68382314, 30.86132581,  0.67195508], [ 0.78128613, -0.20870371, -0.58805789, -0.01492182]]), np.array([[25.66299566, 22.7253343 , 29.18165789,  0.59579125], [ 0.5452468 , -0.56835998, -0.60898994, -0.09383026]]), np.array([[26.28516204, 24.26930074, 29.93102134,  0.61640294], [ 0.89689347,  0.425852  , -0.11657413, -0.02535074]])]
apex_sections = [[np.array([[25.32911334, 23.09948612, 29.60329911,  0.65734002], [ 0.53181656, -0.52454216, -0.66040725, -0.07673932]])], [np.array([[25.71390248, 24.02351252, 30.03935879,  0.63252578], [ 0.90842008,  0.35094661, -0.22572994, -0.02560124]])]]
apex = [np.array([25.80793453, 23.54893934, 29.63189945])]
R =  3

# Create bifurcation
bifurcation = Nfurcation("crsec", [end_sections, apex_sections, apex, R])

bifurcation_mesh = bifurcation.mesh_surface()
bifurcation_mesh.save("Output/bifurcation_mesh.vtk")
