import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # 3D display

import pickle 

import pyvista as pv

from Bifurcation import Bifurcation
from ArterialTree import ArterialTree
from Spline import Spline



def test_tree_class():

	tree = ArterialTree("TestPatient", "BraVa", "Results/refence_mesh_simplified_centerline.swc")

	#tree.deteriorate_centerline(0.05, [0.0, 0.0, 0.0, 0.0])

	#tree.write_swc("Results/refence_mesh_simplified_centerline.swc")

	tree.show()

	tree.spline_approximation()
	tree.show(False)

	tree.mesh_surface(16, 0.2, bifurcation_model=False)
	mesh = tree.get_surface_mesh()
	mesh.plot()
	mesh.save("Results/aneurisk_hex_mesh.ply")
	#tree.distance_mesh("model.vtp", display=True)


def test_ogrid_pattern():
 
	crsec = np.array([[35.50909475775443, 181.0999679588229, 132.85608603166423], [35.537665711699546, 181.02966736868194, 132.6919766752659], [35.53742697145919, 181.02160863277103, 132.5113517793639], [35.508414883070785, 181.07701862057996, 132.34170984699063], [35.45504627399622, 181.18746166375763, 132.20887732454193], [35.38544603117945, 181.33612381014893, 132.13307675943082], [35.31021016066156, 181.5003725955915, 132.1258481005616], [35.24079264171404, 181.65520263142062, 132.1882918457192], [35.187761662325265, 181.7770424482191, 132.31090150075732], [35.15919070838015, 181.84734303836007, 132.47501085715567], [35.15942944862051, 181.85540177427097, 132.65563575305765], [35.18844153700891, 181.79999178646204, 132.82527768543093], [35.24181014608348, 181.68954874328438, 132.95811020787963], [35.31141038890025, 181.54088659689307, 133.03391077299074], [35.386646259418136, 181.3766378114505, 133.04113943185996], [35.45606377836566, 181.2218077756214, 132.97869568670237]])
	center = np.array([ 35.34842821, 181.4385052, 132.58349377])
	tree = ArterialTree("TestPatient", "BraVa")
	tree.ogrid_pattern(center, crsec, [0.05, 0.55, 0.4], 20, 5)




#test_tree_class()
test_ogrid_pattern()