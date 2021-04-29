import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # 3D display
import pickle 
import pyvista as pv
from geomdl import BSpline, operations

from Nfurcation import Nfurcation
from ArterialTreev2 import ArterialTree
from Spline import Spline
from Model import Model
from Simulation import Simulation
from utils import *

from numpy.linalg import norm
from numpy import dot, cross
import time
import vtk
import copy

from combine_nfurcation import *


def dom_points():
	
	patient = "C0078"
	split_tubes("/home/decroocq/Documents/Thesis/Data/Aneurisk/Vessels/", patient + ".vtp", "Results/Centerlines/")

	for n_tube in [2]:

		tree = ArterialTree("C0083", "Aneurisk", "Results/Centerlines/" + patient+ "_tube_" + str(n_tube)+ ".swc")
		tree.low_sample(0.5)
		#tree.add_noise_centerline(1)
		#tree.resample(3)

		# Reference spline and data
		topo_graph = tree.get_topo_graph()

		for e in topo_graph.edges():
			data = np.vstack((topo_graph.nodes[e[0]]['coords'], topo_graph.edges[e]['coords'], topo_graph.nodes[e[1]]['coords']))

		optimal_knot(10, data)


def number_of_control_points():

	patient = "C0078"
	split_tubes("/home/decroocq/Documents/Thesis/Data/Aneurisk/Vessels/", patient + ".vtp", "Results/Centerlines/")

	for n_tube in [2]:

		tree = ArterialTree("C0083", "Aneurisk", "Results/Centerlines/" + patient+ "_tube_" + str(n_tube)+ ".swc")
		tree.low_sample(0.1)
		#tree.resample(3)

		# Reference spline and data
		topo_graph = tree.get_topo_graph()

		for e in topo_graph.edges():
			data = np.vstack((topo_graph.nodes[e[0]]['coords'], topo_graph.edges[e]['coords'], topo_graph.nodes[e[1]]['coords']))

		bmax = 40
		if len(data) < 40:
			bmax = len(data)

		list_nb = np.linspace(4,bmax,20)
		results = np.zeros((len(list_nb), 6))

		k = 0
		p = 0
		nb_plots = 4
		plot_val = list_nb[np.linspace(0, len(list_nb) - 1, nb_plots, dtype = int)].astype(int).tolist()
		

		figure, ax_curv = plt.subplots(nb_plots, 1)
		figure, ax_x = plt.subplots(nb_plots, 1)
		figure, ax_dx = plt.subplots(nb_plots, 1)


		for i in list_nb:
			i = int(i)
			
			tree.model_network(i, criterion="None")
			model_graph = tree.get_model_graph()
			
			for e in model_graph.edges():
				spl_ref = model_graph.edges[e]['spline']

			t = spl_ref.get_times()
			length = spl_ref.get_length()


			mesh = tree.mesh_surface()
			mesh = mesh.compute_cell_quality()
			#mesh.plot(show_edges = True, scalars= 'CellQuality')
			#mesh.save("Results/Centerlines/Control_Points/"+ patient+"_tube_" + str(n_tube) + "_pts_" + str(i) + ".vtk")
			results[k, 0] = mesh['CellQuality'].min()
			results[k, 1] = mesh['CellQuality'].mean()

			ASE = spl_ref.ASE(data)
			ASEder = spl_ref.ASEder(data)
			print("ASE, ASE_der : ", ASE, ASEder)
			print("length : ", spl_ref.length())

			data_estim = spl_ref.estimated_point(data)
			#data_estim = spl_ref.point(t, True)

			l = length_polyline(data)
			#times, dist = spl_ref.distance(data)
			#l =  spl_ref.time_to_length(times)

			der_ref = np.zeros((data.shape[0]-2, data.shape[1]))

			for j in range(1, len(data)-1):
				der_ref[j-1] = (data[j+1] - data[j-1]) / (l[j+1] - l[j-1])
		
			der_estim = spl_ref.estimated_tangent(data[1:-1])
			#der_estim = spl_ref.tangent(t, True)

			curv = spl_ref.curvature(t)

			if i in plot_val:
				ax_curv[p].plot(length, curv, color='black')
				ax_curv[p].set(xlabel="Length (approximation using " + str(i) + " control points)", ylabel="Curvature")

				ax_x[p].plot(l, data, color='black')
				ax_x[p].plot(l, data_estim, '--', color='red')
				ax_x[p].set(xlabel="Length (approximation using " + str(i) + " control points)", ylabel="X")

				ax_dx[p].plot(l[1:-1], der_ref, color='black')
				ax_dx[p].plot(l[1:-1], der_estim, '--',  color='red')
				ax_dx[p].set(xlabel="Length (approximation using " + str(i) + " control points)", ylabel="DX")

				p += 1
				plot_val.remove(i)
			
			results[k, 2] =  ASE[0]
			results[k, 3] =  ASE[1]
			results[k, 4] =  ASEder[0]
			results[k, 5] =  ASEder[1]
			k+=1

		plt.show()

		figure, axes = plt.subplots(4, 1)
		figure.suptitle(" Patient " + patient +  ", vessel number " + str(n_tube) + " of length " + str(int(length[-1])), fontsize="x-large")

		axes[0].plot(list_nb, results[:,0], color='black')
		axes[0].set(xlabel="Nb control points",ylabel="Minimum cell quality")
		axes[1].plot(list_nb, results[:,1], color='black')
		axes[1].set(xlabel="Nb control points",ylabel="Mean cell quality")
		axes[2].plot(list_nb, results[:,2], color='black')
		axes[2].plot(list_nb, results[:,3], color='blue')
		axes[2].set(xlabel="Nb control points",ylabel="Spatial and radius ASE")
		axes[3].plot(list_nb, results[:,4], color='black')
		axes[3].plot(list_nb, results[:,5], color='blue')
		axes[3].set(xlabel="Nb control points",ylabel="Spatial and radius ASEder")

		
		plt.show()
			



def ground_truth(file):
	""" Build ground truth spline for the reference tubes and return them """

	tree = ArterialTree("C0083", "Aneurisk", file)
	#tree.low_sample(0.05)
	#tree.add_noise_centerline(1)
	#tree.resample(3)

	# Reference spline and data
	topo_graph = tree.get_topo_graph()

	for e in topo_graph.edges():
		data = np.vstack((topo_graph.nodes[e[0]]['coords'], topo_graph.edges[e]['coords'], topo_graph.nodes[e[1]]['coords']))

	tree.model_network(criterion="None")

	#mesh = tree.mesh_surface()
	#mesh = mesh.compute_cell_quality()
	#mesh['CellQuality'] = np.absolute(mesh['CellQuality'])
	#mesh.plot(show_edges = True)


	spl = Spline()
	spl.approximation(data, [0,0,0,0], np.zeros((4,4)), False)
	#spl.show(data=data)

	return spl


	
def validation_vessel_model():

	sampling = [1, 0.5, 0.2, 0.02]
	noise_spatial_list = [0, 0.01, 0.05, 0.1]
	noise_radius_list = [0, 0.01, 0.05, 0.1]


	patient = "C0078"
	split_tubes("/home/decroocq/Documents/Thesis/Data/Aneurisk/Vessels/", patient + ".vtp", "Results/Centerlines/")
	f = open("resultat.txt", "w")
	f.write("Patient" + "\t" + "Tube" + "\t" + "Knot" + "\t" + "Criterion" + "\t" + "Sampling" + "\t" + "Noise_spatial" + "\t" + "Noise_radius" + "\t" + "ASE_spatial" + "\t" + "ASE_radius" + "\t" + "ASEder_spatial" + "\t" + "ASEder_radius" + "\t" + "ASEcurv" +  "\t" + "Ldiff" + "\n")

	for n_tube in [0,1,2]:

		for sample in sampling:
			for noise_spatial in noise_spatial_list:
				for noise_radius in noise_radius_list:

					tree = ArterialTree("Test", "Aneurisk", "Results/Centerlines/" + patient+ "_tube_" + str(n_tube)+ ".swc")
					spl_ref = ground_truth("Results/Centerlines/" + patient + "_tube_" + str(n_tube)+ ".swc")

					# Resample / Deteriorate
					#tree.show(centerline=False)
					tree.low_sample(sample)
					tree.add_noise_centerline(noise_spatial)
					tree.add_noise_radius(noise_radius)

					tree.resample(4)
					
					#tree.show(centerline=False)


					# Reference spline and data
					topo_graph = tree.get_topo_graph()

					for e in topo_graph.edges():
						data_ref = np.vstack((topo_graph.nodes[e[0]]['coords'], topo_graph.edges[e]['coords'], topo_graph.nodes[e[1]]['coords']))
					
					tree.model_network(None, criterion="None")
					model_graph = tree.get_model_graph()

					# Model spline
					for e in model_graph.edges():
						spl_model = model_graph.edges[e]['spline']
					#spl_model.show(data=data_ref)

					# Cell quality
					#mesh = tree.mesh_surface()
					#mesh = mesh.compute_cell_quality()
					#mesh.plot(show_edges = True)

					# ASE, ASEder, curv, length
					l_diff = abs(spl_ref.length() - spl_model.length())
					data = spl_ref.get_points()
					ASE = spl_model.ASE(data)
					ASEder = spl_model.ASEder(data)
					

					data_curv = spl_ref.curvature(spl_ref.get_times())
					print(data.shape, spl_ref.get_times().shape, data_curv.shape)
					ASEcurv = spl_model.ASEcurv(data, data_curv)

					print(str(patient) + "\t" + str(n_tube) + "\t" + "Uniform" "\t" + "None" + "\t" + str(sample) + "\t" + str(noise_spatial) + "\t" + str(noise_radius)+ "\t" + str(ASE[0]) + "\t" + str(ASE[1]) + "\t" + str(ASEder[0]) + "\t" + str(ASEder[1]) + "\t" + str(ASEcurv) +  "\t" + str(l_diff) + "\n")
					f.write(str(patient) + "\t" + str(n_tube) + "\t" + "Uniform" "\t" + "None" + "\t" + str(sample) + "\t" + str(noise_spatial) + "\t" + str(noise_radius)+ "\t" + str(ASE[0]) + "\t" + str(ASE[1]) + "\t" + str(ASEder[0]) + "\t" + str(ASEder[1]) + "\t" + str(ASEcurv) +  "\t" + str(l_diff) + "\n")
					

	f.close()


		
		


def test_aneurisk(patient):

	file = "/home/decroocq/Documents/Thesis/Data/Aneurisk/Bifurcations/" + patient + ".vtp"
	#file="/home/decroocq/Documents/Thesis/Data/Aneurisk/C0078/morphology/aneurysm/centerline_branches.vtp"

	tree = ArterialTree("TestPatient", "BraVa", file)
	
	tree.low_sample(0.04)
	#tree.add_noise_radius(0.1)
	tree.show(True, False, False)
	tree.resample(1.5)
	#tree.show(True, False, False)
	tree.model_network()
	#tree.spline_approximation()
	#tree.show(False, True, False)
	#tree.correct_topology()
	tree.show(False, True, True)


	t1 = time.time()
	tree.compute_cross_sections(24, 0.2, False)
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	t1 = time.time()
	mesh = tree.mesh_surface()
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	print("plot mesh")
	mesh.plot(show_edges=True)
	mesh.save("Results/Aneurisk/" + patient + "_mesh_surface.vtk")

	bifurcations = tree.get_bifurcations()

	for i in range(len(bifurcations)):

		# Mesh bifurcations
		mesh = bifurcations[i].mesh_surface()
		#bifurcations[i].show(True)
		mesh.plot(show_edges = True)
		file = open(patient + "_bif_" + str(i) + ".obj", 'wb') 
		pickle.dump(bifurcations[i], file)


	mesh = tree.mesh_volume([0.2, 0.3, 0.5], 10, 10)
	mesh = mesh.compute_cell_quality()
	mesh['CellQuality'] = np.absolute(mesh['CellQuality'])
	mesh.plot(show_edges=True, scalars= 'CellQuality')
	mesh.save("Results/Aneurisk/" + patient + "_volume_mesh.vtk")


def test_brava(patient):

	filename = "/home/decroocq/Documents/Thesis/Data/BraVa/Centerlines/Registered/" + patient + ".swc"

	tree = ArterialTree(patient, "BraVa", filename)
	tree.show(True, False, False)
	tree.spline_approximation()
	#tree.model_network()
	tree.show(False, True, True)

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
	mesh.save("Results/BraVa/registered/crsec/" + patient + ".vtk")


def test_furcation_erwan():
	file = open("C0097_bif_0.obj", 'rb') 	

	bif = pickle.load(file)
	bif.show(True)


#test_brava("P9")
#test_aneurisk("C0097")
#validation_vessel_model()
#number_of_control_points()
#dom_points()
test_furcation_erwan()
