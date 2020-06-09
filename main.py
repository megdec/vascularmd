import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # 3D display

import pickle 

import pyvista as pv

from Bifurcation import Bifurcation
from ArterialTree import ArterialTree
from Spline import Spline



def test_tree_class():

	tree = ArterialTree("TestPatient", "BraVa", "Data/reference_mesh_aneurisk_centerline.vtp")

	tree.deteriorate_centerline(0.05, [0.0, 0.0, 0.0, 0.0])

	tree.write_swc("Results/refence_mesh_simplified_centerline.swc")

	tree.show()

	tree.spline_approximation()
	tree.show(False)

	tree.mesh_surface(24, 0.2)
	mesh = tree.get_surface_mesh()
	mesh.plot()
	mesh.save("Results/aneurisk_hex_mesh.ply")
	#tree.distance_mesh("model.vtp", display=True)


test_tree_class()