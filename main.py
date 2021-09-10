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


	file = "/home/decroocq/Documents/Thesis/Data/Aneurisk//Vessels/Aneurism/" + patient +".vtp"
	#file = "/home/decroocq/Documents/Thesis/Data/Test/P1-part.swc"
	tree = ArterialTree("TestPatient", "BraVa", file)

	#file = open("tree.obj", 'rb') 
	#tree = pickle.load(file)
	
	e = Editor(tree, 500, 200)



def test_aneurisk(patient):

	file = "/home/decroocq/Documents/Thesis/Data/Aneurisk/Bifurcations/" + patient + ".vtp"
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

	bifurcations = tree.get_bifurcations()

	for i in range(len(bifurcations)):

		# Mesh bifurcations
		mesh = bifurcations[i].mesh_surface()
		#bifurcations[i].show(True)
		mesh.plot(show_edges = True)
		#file = open(patient + "_bif_" + str(i) + ".obj", 'wb') 
		#pickle.dump(bifurcations[i], file)


	mesh = tree.mesh_volume([0.2, 0.3, 0.5], 10, 10)
	mesh = mesh.compute_cell_quality()
	mesh['CellQuality'] = np.absolute(mesh['CellQuality'])
	mesh.plot(show_edges=True, scalars= 'CellQuality')
	mesh.save("Results/Aneurisk/" + patient + "_volume_mesh.vtk")


def test_brava(patient):

	filename = "/home/decroocq/Documents/Thesis/Data/BraVa/Centerlines/Registered/" + patient + ".swc"

	tree = ArterialTree(patient, "BraVa", filename)
	tree.show(True, False, False)
	tree.resample(1)
	tree.show(True, False, False)

	t1 = time.time()
	tree.model_network()
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )
	tree.show(False, True, False)

	file = open("Results/BraVa/model/" + patient + ".obj", 'wb') 
	pickle.dump(tree, file)


	t1 = time.time()
	tree.compute_cross_sections(24, 0.2, True)
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	t1 = time.time()
	mesh = tree.mesh_surface()
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	print("plot mesh")
	mesh = mesh.compute_cell_quality()
	mesh.plot(show_edges=True)
	mesh.save("Results/BraVa/surface/" + patient + ".vtk")



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

	tree.compute_cross_sections(24, 0.2, False)

	mesh = pv.read("/home/decroocq/Documents/Thesis/Data/Aneurisk/C0097/surface/model.vtp")
	tree.deform_surface_to_mesh(mesh, [(14, 6), (14, 12)])

	
	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)


test_editor("C0006")