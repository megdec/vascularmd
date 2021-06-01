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
from utils import *

from numpy.linalg import norm
from numpy import dot, cross
import time
import vtk
import copy

import os

def uniform_average():
	patient = "C0078"
	split_tubes("/home/decroocq/Documents/Thesis/Data/Aneurisk/Vessels/Healthy/", patient + ".vtp", "Results/Centerlines/")

	for n_tube in [0]:

		tree = ArterialTree("C0078", "Aneurisk", "Results/Centerlines/" + patient+ "_tube_" + str(n_tube)+ ".swc")
		tree.low_sample(0.02)
		#tree.add_noise_centerline(1)
		#tree.resample(3)
		tree.model_network()
		tree.show()






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
	tree.resample(1)

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


def visu_projection(spl1, spl2):
	""" Visu of both spline projections """

	pts1 = spl1.get_points()
	pts2 = spl2.get_points()

	tproj21, d = spl1.distance(pts2)
	tproj12, d = spl2.distance(pts1)

	ptsproj2 = spl1.point(tproj21)
	ptsproj1 = spl2.point(tproj12)
	
	h = 5

	# 3D plot
	with plt.style.context(('ggplot')):
			
		fig = plt.figure(figsize=(10,7))
		ax = Axes3D(fig)
		ax.set_facecolor('white')

		ax.plot(pts1[:,0], pts1[:,1], pts1[:,2],  c='black')
		ax.plot(pts2[:,0], pts2[:,1], pts2[:,2],  c='red')

		
		for i in np.arange(0, len(pts1), h):
			ax.plot([pts1[i,0], ptsproj1[i, 0]], [pts1[i,1], ptsproj1[i, 1]] , [pts1[i,2], ptsproj1[i, 2]], c='black', linewidth=1)

		for i in np.arange(0, len(pts2), h):
			ax.plot([pts2[i,0], ptsproj2[i, 0]], [pts2[i,1], ptsproj2[i, 1]] , [pts2[i,2], ptsproj2[i, 2]], c='red', linewidth=1)
		

		# Hide the axes
		ax.set_axis_off()
		plt.show()



	
def validation_vessel_model():


	sampling = [0.02, 0.05, 0.2, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	noise_spatial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.1, 0.3, 0.5, 0, 0, 0, 0]
	noise_radius = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.1, 0.3, 0.5]

	root="/home/decroocq/Documents/Thesis/Data/Aneurisk/Vessels/Healthy/Split/"
	validation_files = [f for f in sorted(os.listdir(root))]

	
	f = open("resultats.txt", "w")
	f.write("Patient" + "\t" + "Vessel_number" + "\t" + "Model" + "\t" +  "Criterion" + "\t" + "Sampling" + "\t" + "Noise_spatial" + "\t" + "Noise_radius" + "\t" + "ASE_spatial" + "\t" + "ASE_radius" + "\t" + "ASEder_spatial" + "\t" + "ASEder_radius" + "\t" + "ASEcurv" +  "\t" + "Ldiff" + "\n")

	#validation_files = [validation_files[0]]
	for file in validation_files:

		spl_ref = ground_truth(root + file)

		patient = file[:5] 
		n_tube = int(file[-5])

		
		for i in range(len(sampling)):
			for j in range(3):

				print(sampling[i], noise_radius[i], noise_spatial[i])

				tree = ArterialTree("Test", "Aneurisk", root + file)

				# Resample / Deteriorate
		
				tree.resample(1)
				tree.low_sample(sampling[i])
				tree.add_noise_centerline(noise_spatial[i])
				tree.add_noise_radius(noise_radius[i])

				#tree.resample(1/sampling[i])
				#tree.resample(2)
						
				#tree.show(centerline=False)

				for model in [[False, "None", False], [False, "None", True], [False, "AIC", False], [True, "AIC", False], [True, "AICC", False], [True, "SBC", False], [True, "CV", False], [True, "GCV", False]]:

					try:
					
						tree.model_network(radius_model = model[0], criterion=model[1], akaike=model[2])
						model_graph = tree.get_model_graph()

						# Model spline
						for e in model_graph.edges():
							spl_model = model_graph.edges[e]['spline']
						#spl_model.show(data=spl_ref.get_points())
						

						# Cell quality
						#mesh = tree.mesh_surface()
						#mesh = mesh.compute_cell_quality()
						#mesh.plot(show_edges = True)

						# Project reference on estimation

						# ASE, ASEder, curv, length
						l_diff = abs(spl_ref.length() - spl_model.length())						

						data = spl_ref.get_points()
						ASE = spl_model.ASE(data)

						data_der = spl_ref.tangent(spl_ref.get_times(), radius=True)
						ASEder = spl_model.ASEder(data, data_der = data_der)
							

						data_curv = spl_ref.curvature(spl_ref.get_times())
						ASEcurv = spl_model.ASEcurv(data, data_curv)

						# Project estimation on reference
						# ASE, ASEder, curv, length			

						data = spl_model.get_points()
						ASE2 = spl_ref.ASE(data)

						data_der = spl_model.tangent(spl_model.get_times(), radius=True)
						ASEder2 = spl_ref.ASEder(data, data_der = data_der)
							

						data_curv = spl_model.curvature(spl_model.get_times())
						ASEcurv2 = spl_ref.ASEcurv(data, data_curv)

						ASE[0] = (ASE[0] + ASE2[0]) / 2.
						ASE[1] = (ASE[1] + ASE2[1]) / 2.

						ASEder[0] = (ASEder[0] + ASEder2[0]) / 2.
						ASEder[1] = (ASEder[1] + ASEder2[1]) / 2.

						ASEcurv = (ASEcurv + ASEcurv2) / 2.


						if model[0]:
							model_name = "SpatialRadius"
						else:
							model_name = "Global"

						if model[1] == "None":
							model_name += "NonPenalized"
						else:
							model_name += "Penalized" + model[1]

						if model[2]:
							model_name += "Akaike"

							
						f.write(str(patient) + "\t" + str(n_tube) +  "\t" + model_name + "\t" + str(model[1]) + "\t" + str(sampling[i]) + "\t" + str(noise_spatial[i]) + "\t" + str(noise_radius[i])+ "\t" + str(ASE[0]) + "\t" + str(ASE[1]) + "\t" + str(ASEder[0]) + "\t" + str(ASEder[1]) + "\t" + str(ASEcurv) +  "\t" + str(l_diff) + "\n")
					
					except:		
						print("Convergence Failed", sampling[i], noise_radius[i], noise_spatial[i])
							
	f.close()

	


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
	tree.compute_cross_sections(24, 0.2, True)
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
	#tree.spline_approximation()
	tree.model_network()
	tree.show(False, True, False)


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



def test_remove_branch():

	file = open("TAMU/simple_network.obj", 'rb') 
	tree = pickle.load(file)

	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)

	tree.show(False, True, False)
	tree.remove_branch((4,8), False)

	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)

	tree.show(False, False, False)
	tree.show(False, True, False)
	#tree.remove_branch((8,9))
	#tree.show(False, False, False)
	#tree.show(False, True, False)
	
	#tree.compute_cross_sections(24, 0.2, False)
	#mesh = tree.mesh_surface()
	#mesh.plot(show_edges=True)




#test_brava("P9")
#test_remove_branch()
test_aneurisk("C0099-2")
#validation_vessel_model()
#number_of_control_points()
#dom_points()
#test_furcation_erwan()

#patient = "C0082"
#split_tubes("/home/decroocq/Documents/Thesis/Data/Aneurisk/Vessels/Healthy/", patient + ".vtp", "/home/decroocq/Documents/Thesis/Data/Aneurisk/Vessels/Healthy/Split/")
#uniform_average()