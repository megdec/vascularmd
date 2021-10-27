import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # 3D display
import pickle 
import pyvista as pv
from geomdl import BSpline, operations

from Nfurcation import Nfurcation
from ArterialTree import ArterialTree
from Spline import Spline
from Model import Model
from Simulation import Simulation
from Editor import Editor
from utils import *

from numpy.linalg import norm
from numpy import dot, cross
import time
import vtk
import copy

import os



def test_editor(patient):


	file = "/home/decroocq/Documents/Thesis/Data/Aneurisk/Vessels/Aneurism/" + patient +".vtp"
	#file = "/home/decroocq/Documents/Thesis/Data/BraVa/Centerlines/Registered/P9.swc"
	#file = "/home/decroocq/Documents/Thesis/Teaching/Projects/Shiraishi/data.swc"

	#file = "/home/decroocq/Documents/Thesis/Data/Test/P1-part.swc"
	#file = "/home/decroocq/Documents/Thesis/Data/Aneurisk//Vessels/Healthy/centerline_BA.vtp"
	#file =  "/home/decroocq/Documents/Thesis/Data/Aneurisk/Bifurcations" + patient +".vtp"

	tree = ArterialTree("TestPatient", "BraVa", file)
	#file = open("Results/BraVa/network/P9.obj", 'rb')
	#tree = pickle.load(file)
	
	
	#file = open("tmp/tree_model.obj", 'rb') 
	#tree = pickle.load(file)
	
	e = Editor(tree, 1500, 600)



def test_aneurisk(patient):

	file = "/home/decroocq/Documents/Thesis/Data/Aneurisk/Vessels/Healthy/" + patient +".vtp"
	#file="/home/decroocq/Documents/Thesis/Data/Aneurisk/C0078/morphology/aneurysm/centerline_branches.vtp"

	tree = ArterialTree("TestPatient", "BraVa", file)
	
	tree.low_sample(0.1)
	#tree.add_noise_radius(0.1)
	tree.show(True, False, False)
	#tree.write_swc("centerline.swc")
	tree.resample(1.5)
	#tree.show(True, False, False)
	tree.model_network()
	#tree.spline_approximation()
	#tree.show(False, True, False)
	#tree.correct_topology()
	tree.show(False, True, False)

	#file = open(patient + "_ArterialTree.obj", 'wb') 
	#pickle.dump(tree, file)


	t1 = time.time()
	tree.compute_cross_sections(24, 0.2, False)
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	t1 = time.time()
	mesh = tree.mesh_surface()
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	print("plot mesh")
	pv.set_plot_theme("document")
	mesh.plot(show_edges=True)
	mesh.save("Results/Aneurisk/" + patient + "_mesh_surface.vtk")

	"""
	bifurcations = tree.get_bifurcations()

	for i in range(len(bifurcations)):

		# Mesh bifurcations
		mesh = bifurcations[i].mesh_surface()
		#bifurcations[i].show(True)
		mesh.plot(show_edges = True)
		#file = open(patient + "_bif_" + str(i) + ".obj", 'wb') 
		#pickle.dump(bifurcations[i], file)
	"""


	mesh = tree.mesh_volume([0.2, 0.3, 0.5], 10, 10)
	mesh = mesh.compute_cell_quality()
	mesh['CellQuality'] = np.absolute(mesh['CellQuality'])
	mesh.plot(show_edges=True, scalars= 'CellQuality')
	mesh.save("Results/Aneurisk/" + patient + "_volume_mesh.vtk")
	

def test_remove_branch():

	file = "/home/decroocq/Documents/Thesis/Data/Aneurisk/Bifurcations/C0099.vtp"
	#file="/home/decroocq/Documents/Thesis/Data/Aneurisk/C0078/morphology/aneurysm/centerline_branches.vtp"

	tree = ArterialTree("TestPatient", "BraVa", file)
	
	tree.low_sample(0.1)
	tree.show(True, False, False)
	#tree.resample(1.5)
	tree.model_network()

	tree.show(False, True, False)

	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)

	tree.show(False, True, False)
	tree.remove_branch((4,8), True)

	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)

	tree.show(False, True, False)
	tree.remove_branch((8,9))

	tree.show(False, True, False)
	
	tree.compute_cross_sections(24, 0.2, True)
	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)


def test_remove_trifurcation():

	file = "/home/decroocq/Documents/Thesis/Data/Aneurisk/Bifurcations/C0032.vtp"
	#file="/home/decroocq/Documents/Thesis/Data/Aneurisk/C0078/morphology/aneurysm/centerline_branches.vtp"

	tree = ArterialTree("TestPatient", "BraVa", file)
	
	tree.low_sample(0.1)
	tree.show(True, False, False)
	#tree.resample(1.5)
	tree.model_network()

	tree.show(False, True, False)


	t1 = time.time()
	tree.compute_cross_sections(24, 0.2, False)
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )
	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)

	tree.remove_branch((2,6), False)
	tree.show(False, True, False)

	print("The process took ", t2 - t1, "seconds." )
	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)


def test_bifurcation():

	end_sections = [np.array([[23.9598871 , 23.68382314, 30.86132581, 0.67195508], [ 0.78128613, -0.20870371, -0.58805789, -0.01492182]]), np.array([[25.66299566, 22.7253343 , 29.18165789, 0.59579125], [ 0.5452468 , -0.56835998, -0.60898994, -0.09383026]]), np.array([[26.28516204, 24.26930074, 29.93102134, 0.61640294], [ 0.89689347, 0.425852 , -0.11657413, -0.02535074]])]
	apex_sections = [[np.array([[25.32911334, 23.09948612, 29.60329911, 0.65734002], [ 0.53181656, -0.52454216, -0.66040725, -0.07673932]])], [np.array([[25.71390248, 24.02351252, 30.03935879, 0.63252578], [ 0.90842008, 0.35094661, -0.22572994, -0.02560124]])]]
	apex = [np.array([25.80793453, 23.54893934, 29.63189945])]

	R = 0.2

	# Create bifrucation
	bif = Nfurcation("crsec", [end_sections, apex_sections, apex, R])
	bif.show(True)
	m = bif.mesh_surface()
	m.plot(show_edges=True)

	bif.rotate_apex_section([0,0], -0.2)
	bif.show(True)
	m = bif.mesh_surface()
	m.plot(show_edges=True)


def mesh_aneurism():

	file = "/home/decroocq/Documents/Thesis/Data/Aneurisk/Bifurcations/C0097.vtp"


	tree = ArterialTree("TestPatient", "BraVa", file)
	tree.low_sample(0.1)
	tree.show(True, False, False)
	tree.resample(1.5)

	tree.model_network()
	tree.show(False, True, False)

	tree.compute_cross_sections(24, 0.2, True)

	mesh = pv.read("/home/decroocq/Documents/Thesis/Data/Aneurisk/C0097/surface/model.vtp")
	tree.deform_surface_to_mesh(mesh, [(14, 6), (14, 12)])

	
	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)


def test_merge():

	file = "/home/decroocq/Documents/Thesis/Data/Aneurisk/Bifurcations/C0099.vtp"
	#file="/home/decroocq/Documents/Thesis/Data/Aneurisk/C0078/morphology/aneurysm/centerline_branches.vtp"

	tree = ArterialTree("TestPatient", "BraVa", file)
	
	tree.low_sample(0.1)
	tree.show(False, False, False)
	print([e for e in tree.get_topo_graph().edges()])
	tree.merge_branch((4,5))
	print([e for e in tree.get_topo_graph().edges()])
	tree.show(False, False, False)



def test_evaluation_mesh():

	

	file = open("Results/BraVa/network/P3.obj", 'rb')
	tree = pickle.load(file)
	mesh = tree.get_volume_mesh()
	mesh.plot()
	"""
	bifs = tree.get_bifurcations()
	for b in bifs:
		b.R = b.optimal_smooth_radius(0.5)
		b.compute_cross_sections(24, 0.2)
		m = b.mesh_surface()
		m.plot(show_edges=True)
		"""

	ratio_furcation, ratio_vessels, stats_furcations, stats_vessels = tree.evaluate_mesh_quality(volume = True, display = False)



	print("Ratio furcations: ", ratio_furcation)
	print("Ratio_vessels : ", ratio_vessels)
	print("Stats furcations : ", stats_furcations)
	print("Stats vessels : ", stats_vessels)

	
def test_distance_constraint():
	# Get centerline data
	file = "/home/decroocq/Documents/Thesis/Data/Test/tube.swc"
	tree = ArterialTree("TestPatient", "BraVa", file)
	tree.model_network()
	tree.add_noise_centerline(1, normal = True)
	tree.show(True)

	# Model with different constraints
	tree.model_network(max_distance = 3)
	tree.show(True, True, True)


def test_part_meshing():

	file = open("Results/BraVa/network/P44.obj", 'rb')
	tree = pickle.load(file)

	G = tree.get_crsec_graph()
	for e in G.edges():
		if G.nodes[e[1]]["type"] != "bif":
			m = tree.mesh_volume([0.2, 0.3, 0.5], 10, 10, edg = [e])
			m.plot(show_edges = True)

def show_edges_data():

	file = open("Results/BraVa/network/P1.obj", 'rb')
	tree = pickle.load(file)

	centers = pv.MultiBlock()
	c = 0
		

	G = tree.get_model_graph()
	for e in G.edges():
		if G.nodes[e[0]]["type"]!= "bif" and G.nodes[e[1]]["type"]!= "bif":
			spl = G.edges[e]["spline"]

				
			#fig = plt.figure(figsize=(10,7))
			#ax = Axes3D(fig)
			#ax.set_facecolor('white')

			pts = spl.get_points()
			#ax.plot(pts[:,0], pts[:,1], pts[:,2],  c='black')

			data = spl.get_data()
			if data is not None:
				#ax.scatter(data[:, 0],data[:, 1], data[:, 2],  c='red', s = 20, depthshade=False)
				for coord in data:
					centers[str(c)]= pv.Sphere(radius=0.2, center=(coord[0], coord[1], coord[2]))
					c += 1

				t, d = spl.distance(data)
				print("Maximum distance :", max(d))

			#ax.set_axis_off()
			#plt.show()
	centers.save("data_P1.vtm")
			



def test_brava(filename, patient):


	tree = ArterialTree(patient, "BraVa", filename)
	
	#tree.show(True, False, False)
	#tree.resample(1)
	#tree.show(True, False, False)

	t1 = time.time()
	tree.model_network(max_distance = 4)
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )
	#tree.show(False, True, False)


	t1 = time.time()
	tree.compute_cross_sections(24, 0.2, False)
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	t1 = time.time()
	mesh = tree.mesh_surface()
	t2 = time.time()
	mesh.save("Results/BraVa/surface/" + patient + ".vtk")
	#mesh.plot(show_edges = True)

	"""
	t1 = time.time()
	mesh = tree.mesh_volume([0.2, 0.3, 0.5], 10, 10)
	t2 = time.time()
	mesh.save("Results/BraVa/volume/" + patient + ".vtu")
	"""

	print("The process took ", t2 - t1, "seconds." )
	file = open("Results/BraVa/network/" + patient + ".obj", 'wb') 
	pickle.dump(tree, file)



def launch_brava_meshing():

	root = "/home/decroocq/Documents/Thesis/Data/BraVa/Centerlines/Registered/"
	for i in range(54,59):
		patient = "P" + str(i)

		print(root+patient+".swc", patient)
		try:
			test_brava(root+patient+".swc", patient)
		except:
			print("Not found.")


			
launch_brava_meshing()
#test_part_meshing()
#show_edges_data()

#test_brava("P" + str(i))

#test_topo_correction()
#test_editor("C0006")
#test_brava("P14")
#test_aneurisk("C0082")
#test_evaluation_mesh()
#test_distance_constraint()