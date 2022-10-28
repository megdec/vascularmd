####################################################################################################
# Author: Meghane Decroocq
#
# This file is part of vascularmd project (https://github.com/megdec/vascularmd)
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3 of the License.
#
####################################################################################################



from multiprocessing import Pool, Process, cpu_count, set_start_method

import pyvista as pv # Meshing
import vtk
from scipy.spatial import KDTree
import warnings
import pickle

# Trigonometry functions
from math import pi
from numpy.linalg import norm 
from numpy import dot, cross

# Plot
import matplotlib.pyplot as plt # Tools for plots
from mpl_toolkits.mplot3d import Axes3D # 3D display
import gc
import os

import networkx as nx

from utils import *
from Nfurcation import Nfurcation
from Spline import Spline




class ArterialTree:


	#####################################
	##########  CONSTRUCTOR  ############
	#####################################

	def __init__(self, patient_name, database_name, filename = None):


		# Initiate attributes
		self.patient_name = patient_name
		self.database_name = database_name
		self._surface_mesh = None
		self._volume_mesh = None


		if filename == None:
			self._full_graph = None
			self._topo_graph = None
			self._model_graph = None
			self._crsec_graph = None
		else:
			self.__load_file(filename)
			self.__set_topo_graph()
			self._model_graph = None
			self._crsec_graph = None



	#####################################
	#############  GETTERS  #############
	#####################################


	def get_full_graph(self):
		""" Return the full graph """

		if self._full_graph is None:
			warnings.warn("No full graph found.")

		return self._full_graph


	def get_topo_graph(self):
		""" Return the topo graph """

		if self._topo_graph is None:
			warnings.warn("No topo graph found.")
			
		return self._topo_graph

	def get_model_graph(self):
		""" Return the model graph """

		if self._model_graph is None:
			warnings.warn("No model graph found.")
			
		return self._model_graph


	def get_crsec_graph(self):
		""" Return the crsec graph """

		if self._crsec_graph is None:
			warnings.warn("No crsec graph found.")
			
		return self._crsec_graph

	def get_surface_mesh(self):
		""" Return the surface mesh """

		if self._surface_mesh is None:
			warnings.warn("No surface mesh found.")

		else:	
			return pv.PolyData(self._surface_mesh[0], self._surface_mesh[1])


	def get_volume_mesh(self, formt = "pyvista"):
		""" Return the volume mesh """

		if self._volume_mesh is None:
			warnings.warn("No volume mesh found.")
		else:	

			if formt == "pyvista":
				#return self.write_pyvista_mesh_from_vtk(self._volume_mesh[2], self._volume_mesh[0])
				cells = self._volume_mesh[0]
				cell_types = self._volume_mesh[1]
				points = self._volume_mesh[2]
				return pv.UnstructuredGrid(cells, cell_types, points)

			elif formt == "meshio":
				import meshio
		
				points = self._volume_mesh[2]
				cells = [("hexahedron", self._volume_mesh[0][:,1:])]
				return meshio.Mesh(points, cells)
			else:
				warnings.warn("Wrong format argument given. The accepted formats are 'pyvista' and 'meshio'.")


	def get_surface_link(self):

		if self._surface_mesh is None:
			warnings.warn("No surface mesh found.")
		else:	
			if len(self._surface_mesh) < 3:
				warnings.warn("The surface mesh links were not computed.")
			else:
				return self._surface_mesh[2]

	def get_volume_link(self):

		if self._volume_mesh is None:
			warnings.warn("No volume mesh found.")
		else:	
			if len(self._volume_mesh) < 4:
				warnings.warn("The volume mesh links were not computed.")
			else:
				return self._volume_mesh[3]


	def get_number_of_faces(self):

		if self._surface_mesh is None:
			warnings.warn("No surface mesh found.")
		else:	
			return self._surface_mesh[0].shape[0]


	def get_number_of_cells(self):

		if self._volume_mesh is None:
			warnings.warn("No volume mesh found.")
		else:	
			return self._volume_mesh[0].shape[0]


	def get_bifurcations(self):
		""" Returns a list of all bifurcation objects in the network """

		if self._crsec_graph is None:
			raise ValueError("No crsec graph found.")
		else:

			bifurcations = []
			for n in self._model_graph.nodes(): 
				if self._model_graph.nodes[n]['type'] == "bif":
					bifurcations.append(self._model_graph.nodes[n]['bifurcation'])
			return bifurcations


	#####################################
	#############  SETTERS  #############
	#####################################

	def check_full_graph(self):
		""" Validity check for the full graph """
		valid = True 

		valid_degrees = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
		for n in self._full_graph.nodes():
			degrees = (self._full_graph.in_degree(n), self._full_graph.out_degree(n))
			if degrees not in valid_degrees:
				valid = False

		return valid

	def reset_model_mesh(self):
		""" Reset the computation of model and mesh (to preserve coherency with topo and full graphs) """
		self._model_graph = None
		self._crsec_graph = None
		self._surface_mesh = None 
		self._volume_mesh = None

	def set_full_graph(self, G):

		""" Set the full graph of the arterial tree."""

		self._full_graph = G
		self.__set_topo_graph()
		self.reset_model_mesh()
		if not self.check_full_graph():
			print("Warning: the input centerline is invalid.")


	def set_topo_graph(self, G, replace = True):

		""" Set the topo graph of the arterial tree."""

		self._topo_graph = G
		if replace:
			self.topo_to_full(replace)
			self.reset_model_mesh()


	def set_model_graph(self, G, replace = True, down_replace = True):

		""" Set the model graph of the arterial tree."""

		self._model_graph = G
		if replace:
			self._crsec_graph = None
			if down_replace:
				self.model_to_full()


	def set_crsec_graph(self, G):

		""" Set the crsec graph of the arterial tree."""

		self._crsec_graph = G


	def __load_file(self, filename):

		"""Converts a centerline file to a graph and set full_graph attribute.

		Keyword arguments:
		filename -- path to centerline file
		"""
		if type(filename) is list:
			print('Loading edg and nds files...')
			self._full_graph = self.__edg_nds_to_graph(filename[0], filename[1])

		else:
			if filename[-4:] == ".swc":
				print('Loading ' + filename[-3:] + ' file...')
				self._full_graph = self.__swc_to_graph(filename)

			elif filename[-4:] == ".vtk" or filename[-4:] == ".vtp":
				print('Loading ' + filename[-3:] + ' file...')
				self._full_graph = self.__vtk_to_graph(filename)

			elif filename[-4:] == ".txt":
				print('Loading ' + filename[-3:] + ' file...')
				self._full_graph = self.__txt_to_graph(filename)

			elif filename[-4:] == ".tre":
				print('Loading ' + filename[-3:] + ' file...')
				self._full_graph = self.__tre_to_graph(filename)

			else:
				raise ValueError("The provided files must be in swc, vtp, tre or vtk format.")

		if not self.check_full_graph():
			print("Warning: the input centerline is invalid.")



	def __set_topo_graph(self):

		""" Set the topo graph of the arterial tree. 
		All edges of but the terminal and bifurcation edges are collapsed.
		The coordinates of the collapsed regular points are stored as an edge attribute. The nodes are labelled (end, bif, reg) """


		self._topo_graph = self._full_graph.copy()
		nx.set_node_attributes(self._topo_graph, "reg", name="type")
		nx.set_node_attributes(self._topo_graph, None, name="full_id")
		nx.set_edge_attributes(self._topo_graph, None, name="full_id")

		#nx.set_node_attributes(self._full_graph, None, name="topo_id")

		for n in self._full_graph.nodes():

			# If regular nodes
			if self._full_graph.in_degree(n) == 1 and self._full_graph.out_degree(n) == 1:

				# Create tables of regular nodes data
				coords = np.vstack((list(self._topo_graph.in_edges(n, data=True))[0][2]['coords'], self._topo_graph.nodes[n]['coords'], list(self._topo_graph.out_edges(n, data=True))[0][2]['coords']))
				# Create new edge by merging the 2 edges of regular point
				self._topo_graph.add_edge(list(self._topo_graph.predecessors(n))[0], list(self._topo_graph.successors(n))[0], coords = coords)

				# Remove regular point
				self._topo_graph.remove_node(n)

			else : 

				# Add node type attribute
				if self._topo_graph.out_degree(n) == 0 or self._topo_graph.in_degree(n) == 0:
					if self._topo_graph.out_degree(n) == 2 or self._topo_graph.in_degree(n) == 2:
						self._topo_graph.nodes[n]['type'] = "sink" 
					else:
						self._topo_graph.nodes[n]['type'] = "end" 	
				else: 
					self._topo_graph.nodes[n]['type'] = "bif" 
				

		# Store the matching of full graph node numbers
		for n in self._topo_graph.nodes():
			self._topo_graph.nodes[n]["full_id"] = n
			#self._full_graph.nodes[n]["topo_id"] = [n]
		for e in self._topo_graph.edges():
			path = list(nx.all_simple_paths(self._full_graph, source=e[0], target=e[1]))[0]
			self._topo_graph.edges[e]["full_id"] = path[1:-1]
			#for i in range(1, len(path)-1):
				#self._full_graph.nodes[path[i]]["topo_id"] = [e, i-1]
		
		# Relabel nodes
		self._topo_graph = nx.convert_node_labels_to_integers(self._topo_graph, first_label=1, ordering='default', label_attribute=None)
		# Remove the previous model and mesh computations as the full graph was modified
		self.reset_model_mesh()


	#####################################
	##########  APPROXIMATION  ##########
	#####################################

	

	def model_network(self, radius_model = True, criterion="AIC", akaike=False, max_distance = 6):

		""" Create Nfurcation objects and approximate centerlines using splines. The network model is stored in the model_graph attribute."""

		print('Modeling the network...')

		# Check if the input data is ok for modelling
		if not self.check_full_graph():
			print("The network cannot be modeled as the input centerline is invalid.")

		else:

			# If there is a crsec_graph, remove it
			if self._crsec_graph != None:
				self._crsec_graph = None


			self._model_graph = self._topo_graph.copy()
			nx.set_node_attributes(self._model_graph, None, name='bifurcation')
			nx.set_node_attributes(self._model_graph, False, name='combine')
			nx.set_node_attributes(self._model_graph, None, name='tangent')
			nx.set_node_attributes(self._model_graph, None, name='ref')
			nx.set_edge_attributes(self._model_graph, None, name='spline')

			# Get inlet id 
			sources = []
			for n in self._topo_graph.nodes():
				if self._topo_graph.in_degree(n) == 0:
					sources.append(n)

			# Bifurcation models
			all_id = [n for n in self._topo_graph.nodes()]
			original_label = dict(zip(all_id, all_id))

			# dfs
			for source in sources:
				dfs = list(nx.dfs_successors(self._topo_graph, source=source).values())

				for l in dfs:
					for n in l:
						if n in [nds for nds in self._topo_graph.nodes()]:
							n = original_label[n]
							if self._topo_graph.nodes[n]['type'] == "bif":
								original_label = self.__model_furcation(n, original_label, criterion, akaike, max_distance)

			# Spline models
			# Straight tube case
			for e in self._model_graph.edges():

				if self._model_graph.nodes[e[0]]['type'] == "end" and  self._model_graph.nodes[e[1]]['type'] == "end" :
					self.__model_vessel(e, criterion=criterion, akaike=akaike, radius_model=radius_model, max_distance = max_distance)

			# Sink case
			nds_list = list(self._model_graph.nodes())
			for n in nds_list:
				if self._model_graph.nodes[n]['type'] == "sink":

					sink_edg = [e for e in self._model_graph.in_edges(n)] + [e for e in self._model_graph.out_edges(n)] 
					self.__model_vessel(sink_edg[0], criterion=criterion, akaike=akaike, radius_model=radius_model, max_distance = max_distance)


			# Add rotation attributes
			nx.set_edge_attributes(self._model_graph, None, name='alpha')
			nx.set_edge_attributes(self._model_graph, 0, name='connect')

			# Add rotation information on bifs and edges
			for n in self._model_graph.nodes():
				if self._model_graph.nodes[n]["type"] == "end" and self._model_graph.in_degree(n) == 0:
					self.__compute_rotations(n)
				if self._model_graph.nodes[n]["type"] == "sep":
					self.__compute_rotations(n)
			


	def __model_furcation(self, n, original_label, criterion, akaike, max_distance):

		""" Extract bifurcation parameters from the data and modify the model graph to add the bifurcation object and edges.

		Keyword arguments:
		n -- bifurcation node (model graph)
		"""

		LEN_OUT = 1.5 # length of daughter branches = LEN_OUT x radius
		LEN_IN = 4 # length of mother branch = LEN_IN x radius
		R_BIF = 3 # smoothing degree for the bifurcations

		MIN_LEN_IN = 3 # Minimum length to consider before merging

		from Model import Model
		original_label_modif = original_label

		# Get original label dictionary
		all_id = [nds for nds in self._topo_graph.nodes()]
		label_dict_topo = dict(zip(all_id, all_id))

		all_id = [nds for nds in self._model_graph.nodes()]
		label_dict_model = dict(zip(all_id, all_id))

		nmax = max(list(self._model_graph.nodes())) + 1

		e_in = [e for e in self._model_graph.in_edges(n)]
		e_out = [e for e in self._model_graph.out_edges(n)]
		e_out.sort()

		apex_found = False
		nb_min = 10
		while not apex_found:

			apex_found = True
			splines = []
		
			# Approximate vessels
			for i in range(len(e_out)):
				data = np.vstack((self._model_graph.nodes[n]['coords'], self._model_graph.edges[e_out[i]]['coords'], self._model_graph.nodes[e_out[i][1]]['coords']))
				nb_data_out = self._model_graph.edges[e_out[i]]['coords'].shape[0] + 1
				

				e_act = e_out[i]
				n_act = e_act[1]

				while data.shape[0] < nb_min:
				
					if self._model_graph.out_degree(n_act)!= 0:
						e_next = (n_act, [e for e in self._model_graph.successors(n_act)][0]) # Get a successor edge		
						data =  np.vstack((data, self._model_graph.edges[e_next]['coords'], self._model_graph.nodes[e_next[1]]['coords'])) # Add data
						e_act = e_next
						n_act = e_next[1]

					#elif self._model_graph.nodes[n_act]["type"] == "sink" and self._model_graph.out_degree(n_act) == 0: # If in sink, cross the next bifurcation
					#	e_next = list(self._model_graph.in_edges(n_act))
					#	if e_act in e_next:
					#		e_next.remove(e_act)
					#	e_next = e_next[0] # Get a successor edge (in in the sink case)		
					#	data =  np.vstack((data, self._model_graph.edges[e_next]['coords'][::-1], self._model_graph.nodes[e_next[0]]['coords'])) # Add data (reversed because in sink)
					#	e_act = e_next
					#	n_act = e_next[0]

					else: break

				values = np.zeros((4,4))
				constraint = [False] * 4

				if self._model_graph.nodes[e_in[0][0]]['type'] != "bif":
					data = np.vstack((self._model_graph.nodes[e_in[0][0]]['coords'], self._model_graph.edges[e_in[0]]['coords'], data))

					if self._model_graph.nodes[e_in[0][0]]['type'] == "sep":
						values[0,:] = self._model_graph.nodes[e_in[0][0]]['coords']
						constraint[0] = True
						values[1,:] = self._model_graph.nodes[e_in[0][0]]['tangent']
						constraint[1] = True
				else: 
					values[0,:] = self._model_graph.nodes[n]['coords']
					constraint[0] = True
					values[1,:] = self._model_graph.nodes[n]['tangent']
					constraint[1] = True


				spl = Spline()
				spl.approximation(data, constraint, values, False, criterion = criterion, max_distance = max_distance) 

				#spl.show(data=data)
				splines.append(spl)
		

			# Reorder nodes
			if len(e_out) > 2: 
				order, label_dict_model, label_dict_topo, original_label_modif = self.__reorder_branches(n, splines, original_label)

				# Reorder splines 
				splines_reordered = []
				for j in range(len(splines)):
					splines_reordered.append(splines[order[j]])
				splines = splines_reordered

			
			AP = []
			tAP = []
			for i in range(len(splines)):
				tAP.append([])

			# Find apex
			for i in range(len(splines)-1):
				# Compute apex value
				ap, t = splines[i].first_intersectionv2(splines[i+1])
				if t[0] == 1.0 or t[1] == 1.0:
					# Splines were included in each other, add more data
					apex_found = False
					nb_min += 10 

				AP.append(ap)
				tAP[i].append(t[0])
				tAP[i+1].append(t[1])

		
		self._topo_graph = nx.relabel_nodes(self._topo_graph, label_dict_topo)
		self._model_graph = nx.relabel_nodes(self._model_graph, label_dict_model)

		combine = False
		merge_next = False

		# Get index of the in spline with max diameter for C0
	
		if len(splines) == 2: # The vessel with the bigger radius is used to set C0
			r = 0
			ind = 0
			for i in range(len(splines)):
				rad = splines[i].radius(max(tAP[i]))
				if rad > r:
					r = rad
					ind = i
		else:

			min_t = 1
			for i in range(len(AP)): # The vessel with the smallest tAP is used to set C0
				t = splines[1].project_point_to_centerline(AP[i])
				if t < min_t:
					min_t = t
					ind = i

		t_ap = min(tAP[ind])
		l_ap = splines[ind].time_to_length(t_ap)


		if l_ap - splines[ind].radius(t_ap)* MIN_LEN_IN > 0.2: # We can cut!

			if l_ap - splines[ind].radius(t_ap)* LEN_IN > 0.2:
				t_cut = splines[ind].length_to_time(l_ap - splines[ind].radius(t_ap)*LEN_IN)
			else:
				t_cut = splines[ind].length_to_time(l_ap - splines[ind].radius(t_ap)*MIN_LEN_IN)

			if t_cut < 10**(-2):

				pt_cut = splines[ind].point(t_cut)
				spline_cut, tmp_spl = splines[ind].split_time(0.1)
				new_t_cut = spline_cut.project_point_to_centerline(pt_cut)
				spline_in, tmp_spl = spline_cut.split_time(new_t_cut)
			else:
				spline_in, tmp_spl = splines[ind].split_time(t_cut)

			model = splines[ind].get_model()
			T = model[0].get_t()
			thres = np.argmax(T>t_cut)
			thres = thres - 1
			data = self._model_graph.edges[e_in[0]]['coords']
			D_in = data[:thres,:]

			C0 = [splines[ind].point(t_cut, True), splines[ind].tangent(t_cut, True)]

		else: # Merge with LAST one to create trifurcation or combine (length criterion)
			print("Combine!")
			
			combine = True
			nbifprec = list(self._model_graph.predecessors(e_in[0][0]))[0]
			branch = list(self._model_graph.successors(nbifprec))
			branch = branch.index(e_in[0][0])
			bifprec = self._model_graph.nodes[nbifprec]['bifurcation']

			C0 = bifprec.get_apexsec()[branch][0] # If merged bifurcations, C0 is the apex section of the previous bifurcation
			

		if not merge_next: # Keep on building the model

			# Compute apical and end cross sections + new data
			C = [C0]
			AC = []

			spl_out = []
			D_out = []

			for i in range(len(splines)):

				t_ap = max(tAP[i])
				t_cut = splines[i].length_to_time(splines[i].radius(t_ap)*LEN_OUT + splines[i].time_to_length(t_ap))

				n_next = e_out[i][1]

				if t_cut > splines[i].project_point_to_centerline(self._topo_graph.nodes[n_next]['coords'][:-1]) and self._topo_graph.nodes[n_next]['type'] == "bif":
					merge_next = True
					print("Merging to next!")
					break

				else:
					if t_cut > 1.0 - 10**(-2):
						pt_cut = splines[i].point(t_cut)
						tmp_spl, spline_cut = splines[i].split_time(0.9)
						new_t_cut = spline_cut.project_point_to_centerline(pt_cut)
						tmp_spl, spl = spline_cut.split_time(new_t_cut)
	
					else:
						tmp_spl, spl = splines[i].split_time(t_cut)

					# Cut spline_out
					spl_out.append(spl)

					# Separate data points 
					model =  splines[i].get_model()
					T = model[0].get_t()
					thres = np.argmax(T>t_cut)
					nb_in = len(np.vstack((self._model_graph.nodes[e_in[0][0]]['coords'], self._model_graph.edges[e_in[0]]['coords'])))
					thres = thres - nb_in - 1

					data = self._model_graph.edges[e_out[i]]['coords']

					if thres >= len(data):
						D_out.append(np.array([]).reshape(0,4))
					elif thres < 0:
						D_out.append(data)
					else:
						D_out.append(data[thres:,:])


					C.append([splines[i].point(t_cut, True), splines[i].tangent(t_cut, True)])
					AC.append([])


					tap_ordered = tAP[i][:]
					tap_ordered.sort()

					for t_ap in tap_ordered:
						AC[i].append([splines[i].point(t_ap, True), splines[i].tangent(t_ap, True)])

		if merge_next:
			self.merge_branch(n_next) # Merge in topo graph and recompute
			self.merge_branch(n_next, mode = "model")
			
		
			# Re-model furcation 
			original_label = self.__model_furcation(n, original_label, criterion, akaike, max_distance)


		else: # Build bifurction and include it in graph
			original_label = original_label_modif
			bif = Nfurcation("crsec", [C, AC, AP, R_BIF])
			#bif.show(True)

			ref = bif.get_reference_vectors()
			endsec = bif.get_endsec()


			if combine:

				# Add in edge and bif
				self._model_graph.remove_node(n)
				self._model_graph.add_node(nmax, coords = bif.get_X(), bifurcation = bif, combine = combine, type = "bif", ref = None, tangent = None) # Add bif node
				self._model_graph.add_edge(e_in[0][0], nmax, coords = np.array([]).reshape(0,4), spline = bif.get_tspl()[0]) # Add in edge

				# Relabel sep node
				all_id = [n for n in self._model_graph.nodes()]
				label_dict = dict(zip(all_id, all_id))
				label_dict[e_in[0][0]] = n
				self._model_graph = nx.relabel_nodes(self._model_graph, label_dict)
				nbif = nmax
				nmax += 1

			else:

				# Add in edge and bif
				self._model_graph.add_node(n, coords = endsec[0][0], bifurcation = None, combine = combine, type = "sep", ref = ref[0], tangent = endsec[0][1]) # Change bif node to sep
				self._model_graph.edges[e_in[0]]['coords'] = D_in
				self._model_graph.edges[e_in[0]]['spline'] = spline_in

				self._model_graph.nodes[e_in[0][0]]['coords'] = spline_in.point(0.0, True)
				self._model_graph.add_node(nmax, coords = bif.get_X(), bifurcation = bif, combine = combine, type = "bif", ref = None, tangent = None) # Add bif node
				self._model_graph.add_edge(n, nmax, coords = np.array([]).reshape(0,4), spline = bif.get_tspl()[0]) # Add in edge
				nbif = nmax
				nmax += 1

			for i in range(len(splines)): # Add out edges

				self._model_graph.add_node(nmax, coords = C[i+1][0], bifurcation = None, combine = False, type = "sep", ref =  ref[i+1], tangent = C[i+1][1]) 
				self._model_graph.add_edge(nbif, nmax, coords = np.array([]).reshape(0,4), spline = bif.get_tspl()[i+1])

				if self._topo_graph.nodes[e_out[i][1]]["type"] == "end": # Add the cut out splines if end segments
					self._model_graph.add_edge(nmax, e_out[i][1], coords = D_out[i], spline = spl_out[i])
					self._model_graph.nodes[e_out[i][1]]['coords'] = spl_out[i].point(1.0, True)
				else:
					self._model_graph.add_edge(nmax, e_out[i][1], coords = D_out[i], spline = None) # Do not add the cut out splines

				if not combine:
					self._model_graph.remove_edge(e_out[i][0], e_out[i][1])
				nmax += 1

		return original_label




	def __reorder_branches(self, n, splines, original_label, dist_threshold = 1.5):

		""" Reorder branches to handle planar furcation case.

		Keyword arguments:
		n -- furcation node
		splines -- list of splines
		dist_threshold -- minimum distance between evaluated points
		"""

		# Get label dictionnaries
		all_id = [nds for nds in self._topo_graph.nodes()]
		label_dict_topo = dict(zip(all_id, all_id))

		all_id = [nds for nds in self._model_graph.nodes()]
		label_dict_model = dict(zip(all_id, all_id))

		# Get edges and nodes ids
		edg_id = [edg for edg in self._model_graph.out_edges(n)]
		edg_id.sort()
		nds_id = [e[1] for e in edg_id]

		# Order by angles
		dist_min = False
		tmax = False
		l = 1

		while not dist_min and not tmax:

			dist_min = True
			# Spline evaluation at length l
			pt = []
			for s in range(len(splines)):
				t = splines[s].length_to_time(l)
				if t == 1.0:
					tmax = True
					# We can't search further

				pt.append(splines[s].point(t))

			# Check distance condition
			for j in range(len(pt)-1):
				if norm(pt[j] - pt[j+1]) < dist_threshold: 
					# Distance condition is not satisfied
					dist_min = False 

			if not dist_min: # Search further points
				l += 1

		# Get angles
		angles = np.zeros((len(pt), len(pt)))
		for j in range(len(pt)):
			for k in range(len(pt)):
				if k > j:
					# Compute angle
					v1 = pt[j] - self._model_graph.nodes[n]['coords'][:-1]
					v2 = pt[k] - self._model_graph.nodes[n]['coords'][:-1]
					a = angle(v1, v2)
					angles[j, k] = a
					angles[k, j] = a
		
		# Get index of maximum angles			
		ind = np.argmax(angles)
		ind = np.unravel_index(ind, (len(pt), len(pt)))

		order = [ind[0]]
		angles[:, ind[0]] = [2*pi] * len(pt)

		for j in range(len(pt)-1):

			ind = np.argmin(angles[order[j-1]])
			order.append(ind)
			angles[:, ind] = [2*pi] * len(pt)
					
		# Relabel nodes
		for j in range(len(nds_id)):
			label_dict_model[nds_id[order[j]]] = nds_id[j]
			label_dict_topo[nds_id[order[j]]] = nds_id[j]
			original_label[nds_id[order[j]]] = nds_id[j]


		return order, label_dict_model, label_dict_topo, original_label



	def __model_vessel(self, e, criterion, akaike, radius_model, max_distance):

		""" Compute vessel spline model with end tangent constraint and add it to model graph

		Keyword arguments:
		e -- vessel (model graph)
		"""


		# Merge data of both sides 
		if self._model_graph.nodes[e[0]]['type'] == 'sink': # Out sink
			sink_edg = [e for e in self._model_graph.out_edges(e[0])] 
			pts = np.vstack((self._model_graph.nodes[sink_edg[0][1]]['coords'], self._model_graph.edges[sink_edg[0]]['coords'][::-1, :], self._model_graph.nodes[sink_edg[0][0]]['coords']))
			pts = np.vstack((pts, self._model_graph.edges[sink_edg[1]]['coords'], self._model_graph.nodes[sink_edg[1][1]]['coords']))
			edg_type = 'out_sink'
			sink = e[0]
			ends = (sink_edg[0][1], sink_edg[1][1])

		elif self._model_graph.nodes[e[1]]['type'] == 'sink': #In sink
			sink_edg = [e for e in self._model_graph.in_edges(e[1])] 
			pts1 = np.vstack((self._model_graph.nodes[sink_edg[0][0]]['coords'], self._model_graph.edges[sink_edg[0]]['coords'], self._model_graph.nodes[sink_edg[0][1]]['coords']))
			pts = np.vstack((pts1, self._model_graph.edges[sink_edg[1]]['coords'][::-1, :], self._model_graph.nodes[sink_edg[1][0]]['coords']))
			edg_type = 'in_sink'
			sink = e[1]
			ends = (sink_edg[0][0], sink_edg[1][0])

		else: # Normal edge
			pts = np.vstack((self._model_graph.nodes[e[0]]['coords'], self._model_graph.edges[e]['coords'], self._model_graph.nodes[e[1]]['coords']))
			edg_type = 'edg'
			ends = (e[0], e[1])

		if len(pts) <=6:
			pts = resample(pts, 6)


		# Fit spline
		values = np.zeros((4,4))
		constraint = [False] * 4

		if self._model_graph.nodes[ends[0]]['type'] != "end":
			values[0,:] = self._model_graph.nodes[ends[0]]['coords']
			constraint[0] = True
			if edg_type == "out_sink":
				values[1,:] = -self._model_graph.nodes[ends[0]]['tangent']
			else:
				values[1,:] = self._model_graph.nodes[ends[0]]['tangent']
			constraint[1] = True
		
		if self._model_graph.nodes[ends[1]]['type'] != "end":
			values[-1,:] =  self._model_graph.nodes[ends[1]]['coords']
			constraint[-1] = True
			if edg_type == "in_sink":
				values[-2,:] = -self._model_graph.nodes[ends[1]]['tangent']
			else:
				values[-2,:] = self._model_graph.nodes[ends[1]]['tangent']
			constraint[-2] = True

		spl = Spline()
		spl.approximation(pts, constraint, values, False, criterion=criterion, akaike=akaike, radius_model=radius_model, max_distance = max_distance)
		
		# If not sink
		if edg_type == "edg":
			self._model_graph.edges[e]['spline'] = spl
			self._model_graph.nodes[e[1]]['coords'] = spl.point(1.0, True)
		else:

			# Remove sink point
			self._model_graph.remove_node(sink)

			# Add new edge
			self._model_graph.add_edge(ends[0], ends[1], spline = spl, data = pts[1:-1, :])


	def __compute_rotations(self, n):

		""" Compute the rotation angle alpha and the connecting node for vessel of edge e

		Keyword arguments:
		e -- vessel (model graph)
		"""
		
		if self._model_graph.nodes[n]['type'] == "end" and self._model_graph.in_degree(n)==0: # Inlet case
			
			sep_end = False
			# path to the next sep node
			path = [n]
			nd = n
			reach_end = False
			while not reach_end:
				succ = list(self._model_graph.successors(nd))[0]
				path.append(succ)
				nd = succ

				if self._model_graph.nodes[succ]['type'] == "end":
					reach_end = True
					
				if self._model_graph.nodes[succ]['type'] == "sep":
					sep_end = True
					reach_end = True
				
			if sep_end: # Inlet vessel case
				ref1 = self._model_graph.nodes[path[-1]]['ref']
				ref_org = self._model_graph.nodes[path[-1]]['ref']
							
				for i in reversed(range(1, len(path))):
					spl = self._model_graph.edges[(path[i-1], path[i])]['spline']
								
					# Transport ref vector back to the inlet
					ref1 = spl.transport_vector(ref1, 1.0, 0.0)

					# Transport vector back up to estimate the rounding error
					ref_back = spl.transport_vector(ref1, 0.0, 1.0)
					a = angle(ref_org, ref_back, axis = spl.tangent(1.0), signed = True)

					self._model_graph.nodes[path[i-1]]['ref'] = ref1
					ref_org = ref1

					# Set angle to correct rounding error
					self._model_graph.edges[(path[i-1], path[i])]['alpha'] = -a

			else: # Simple tube case
				
				spl = self._model_graph.edges[(path[0], path[1])]['spline']
				ref1 =  cross(spl.tangent(0), np.array([0,0,1])) 
				self._model_graph.nodes[path[0]]['ref'] = ref1
				
				# Transport reference along the path
				for i in range(len(path)-1):
					
					spl = self._model_graph.edges[(path[i], path[i+1])]['spline']
					# Transport ref vector 
					ref1 = spl.transport_vector(ref1, 0.0, 1.0)
					self._model_graph.nodes[path[i+1]]['ref'] = ref1


		if self._model_graph.nodes[n]['type'] == "sep":
		
			# path to the next sep node
			end_bif = False

			path = [n]
			nd = n
			reach_end = False
			while not reach_end:
				succ = list(self._model_graph.successors(nd))
				if len(succ) == 0:# in_sink, skip
					reach_end = True
				else:
					succ = succ[0]
					if self._model_graph.nodes[succ]['type'] == "bif":
						reach_end = True
					else:
						path.append(succ)
						nd = succ

					if self._model_graph.nodes[succ]['type'] == "end":
						reach_end = True

					if self._model_graph.nodes[succ]['type'] == "sep":
						end_bif = True
						reach_end = True
						break

			if len(path) > 1: # If path  = 0, the sep node in a bifurcation inlet, no need to compute cross sections

				if end_bif:

					ref0 = self._model_graph.nodes[path[0]]['ref']

					# Get in bif 
					for e in self._model_graph.in_edges(path[0]):
						if self._model_graph.nodes[e[0]]["type"] =="bif":
							in_edge = e

					for e in self._model_graph.out_edges(path[0]):
						if self._model_graph.nodes[e[1]]["type"] =="bif":
							in_edge = e

					# Get out bif 
					for e in self._model_graph.in_edges(path[-1]):
						if self._model_graph.nodes[e[0]]["type"] =="bif":
							out_edge = e

					for e in self._model_graph.out_edges(path[-1]):
						if self._model_graph.nodes[e[1]]["type"] =="bif":
							out_edge = e
					

					length = [self._model_graph.edges[in_edge]['spline'].length()]

					# Transport original reference to the target reference
					for i in range(len(path) - 1):
						# Transport ref vector downstream
						spl = self._model_graph.edges[(path[i], path[i+1])]['spline']
						ref0 = spl.transport_vector(ref0, 0.0, 1.0)
						length.append(spl.length())

					length.append(self._model_graph.edges[out_edge]['spline'].length())

					# Compute target symmetric vectors
					sym_angles = [pi / 2, pi, 3 * pi / 2]
					sym = [self._model_graph.nodes[path[-1]]['ref']]

					if self._model_graph.in_degree(out_edge[0]) == 0 or self._model_graph.out_degree(out_edge[0]) == 0:
						tg = -self._model_graph.edges[out_edge]['spline'].tangent(0.0)
					else:
						tg = self._model_graph.edges[out_edge]['spline'].tangent(0.0)

					for a in sym_angles:
						rot_vect = rotate_vector(sym[0], tg, a)
						sym.append(rot_vect)

					# Find the minimum angle
					min_a = 5.0
					for i in range(len(sym)):
						a = angle(ref0, sym[i], axis = tg, signed =True)

						if abs(a) < abs(min_a):
							min_a = a
							min_ind = i

					# Smoothly distribute the rotations along the path
					coef = np.array(length) / sum(length) * min_a

					# Rotate ref0
					tg = -self._model_graph.edges[in_edge]['spline'].tangent(1.0)
					self._model_graph.nodes[path[0]]['ref'] = rotate_vector(self._model_graph.nodes[path[0]]['ref'], tg, coef[0])

					# Rotate path
					for i in range(len(path) - 1):
						if i == len(path) - 2:
							self._model_graph.edges[(path[i], path[i+1])]['connect'] = min_ind
							self._model_graph.edges[(path[i], path[i+1])]['alpha'] = coef[i + 1]
						else:
							self._model_graph.edges[(path[i], path[i+1])]['alpha'] = coef[i + 1]

							spl = self._model_graph.edges[(path[i], path[i+1])]['spline']
							ref0 = self._model_graph.nodes[path[i]]['ref'] 
							ref0 = spl.transport_vector(ref0, 0.0, 1.0)
							self._model_graph.nodes[path[i+1]]['ref'] = rotate_vector(ref0, spl.tangent(1.0), coef[i + 1])

					# Rotate ref1
					if self._model_graph.in_degree(out_edge[0]) == 0 or self._model_graph.out_degree(out_edge[0]) == 0:
						tg = self._model_graph.edges[out_edge]['spline'].tangent(0.0)
					else:
						tg = -self._model_graph.edges[out_edge]['spline'].tangent(0.0)
					self._model_graph.nodes[path[-1]]['ref'] = rotate_vector(self._model_graph.nodes[path[-1]]['ref'], tg, coef[-1])

				else: 
						
					ref0 = self._model_graph.nodes[n]['ref']

					# Transport original reference to the target reference
					for i in range(len(path) - 1):
						# Transport ref vector downstream
						spl = self._model_graph.edges[(path[i], path[i+1])]['spline']
						ref0 = spl.transport_vector(ref0, 0.0, 1.0)
						self._model_graph.nodes[path[i+1]]['ref'] = ref0
						self._model_graph.edges[(path[i], path[i+1])]['connect'] = 0
						self._model_graph.edges[(path[i], path[i+1])]['alpha'] = 0

						#if self._model_graph.nodes[path[i+1]]['ref'] is None: # Reg node case
							
		elif (self._model_graph.nodes[n]['type'] == "end" and self._model_graph.in_degree(n)!=0) or self._model_graph.nodes[n]['type'] == "reg":
		
			prec_sep = n
			
			nd = n
			reach_end = False
			while not reach_end:
				prec = list(self._model_graph.predecessors(nd))[0]
				nd = prec
				if self._model_graph.nodes[prec]['type'] == "sep" or  self._model_graph.nodes[prec]['type'] == "end":
					prec_sep = prec
					reach_end = True
			self.__compute_rotations(prec_sep)

		else:
			pass

			
	def __combine_nfurcation(self, n):

		""" Returns merged nfurcation from two close bifurcations

		Keyword arguments: 
		n -- node of the bifurcation to combine in model graph
		"""

		bif2 =  self._model_graph.nodes[n]['bifurcation']

		e_in = [e for e in self._model_graph.in_edges(n)]
		nbifprec = list(self._model_graph.predecessors(e_in[0][0]))[0]
		branch = list(self._model_graph.successors(nbifprec))
		branch = branch.index(e_in[0][0])

		bif1 = self._model_graph.nodes[nbifprec]['bifurcation']

		def connect_nodes(bif1, bif2, tspl, P0, P1, n, tsep):

			""" Compute the nodes connecting an end point to a separation point.

			Keyword arguments: 
			P0, P1 -- end points as np array
			n -- number of nodes
			"""

			pts = np.zeros((n, 3))
			# Initial trajectory approximation

			# Method 1 using surface splines
			relax = 0.1
			d = norm(P0 - P1)

			tg = P0 - P1
			tg = tg / norm(tg)

			pint0 = P0 -  tg * norm(P0 - P1) * relax
			pint1 = P1 + tg * norm(P0 - P1) * relax

			# Fit spline
			spl = Spline()
			spl.approximation(np.vstack((P0, pint0, pint1, P1)), [1,1,1,1], np.vstack((P0, -tg, tg, P1)), False, n = 4, radius_model=False, criterion= "None")

			P = spl.get_control_points()
			P = np.hstack((P, np.zeros((4,1))))
				
			trajectory = Spline(P)
			times = trajectory.resample_time(n) #np.linspace(0, 1, n+2)[1:-1]
				
			for i in range(n):
				pts[i, :] = trajectory.point(times[i])
				
			nds = np.zeros((n,3))
			for i in range(n):

				t = tspl.project_point_to_centerline(pts[i])
				P = tspl.point(t)
				n =  pts[i] - P
				n = n / norm(n)
				
				if t >= tsep:
					pt = bif2.send_to_surface(P, n, 1)
				else:
					pt = bif1.send_to_surface(P, n, 0)
		

				nds[i] = pt
				
			return nds

		# Get separation nodes
		end1, bifsec1, nds1, connect1 = bif1.get_crsec()
		end2, bifsec2, nds2, connect2 = bif2.get_crsec()

		sep1 = []
		c1 = connect1[branch + 1]
		for j in range(len(c1)):
			sep1.append(bifsec1[c1[j]])

		sep2 = []
		c2 = connect2[0]
		for j in range(len(c2)):
			sep2.append(bifsec2[c2[j]])

		# Get trajectory spline
		'''
		P = np.zeros((5,4))

		tspl2 = bif2.get_tspl()[0]

		P[0, :3] = bif1.get_X()
		P[1:, :3] = tspl2.point([0,0.3,0.6,1.0])

		tspl = Spline()
		tspl.approximation(P, [1,0,0,1], np.vstack((P[0], np.zeros((2,4)), P[-1])), False, n = 4, radius_model=False, criterion= "None")
		
		'''
		relax = 0.15

		tspl1 = bif1.get_tspl()
		tspl1 = tspl1[branch + 1]
		tg1 = -tspl1.tangent(1.0)

		tspl2 = bif2.get_tspl()
		tspl2 = tspl2[0]
		tg2 = -tspl2.tangent(1.0)

		X1 = bif1.get_X()
		X2 = bif2.get_X()

		D = norm(X1 - X2)

		tg1 = tg1 / norm(tg1)
		tg2 = tg2 / norm(tg2)

		P1 = X1 + tg1 * D * relax
		P2 = X2 + tg2 * D * relax

		spl = Spline()
		spl.approximation(np.vstack((X1, P1, P2, X2)), [1,1,1,1], np.vstack((X1, tg1, tg2, X2)), False, n = 4, radius_model=False, criterion= "None")
		P = spl.get_control_points()
		P = np.hstack((P, np.zeros((4,1))))
		tspl = Spline(P)
		

		# Get separation time tsep
		C0_center = bif2.get_endsec()[0][0][:-1]
		margin = 0.3
		l1 = tspl.time_to_length(tspl.project_point_to_centerline(C0_center))
		tsep = tspl.length_to_time(l1 + margin)

		# Find closest symetric node
		sym_nodes = [0, self._N//4, self._N//2, self._N//4 * 3] 
		tvec = tspl.project_point_to_centerline(sep1[0])
		vec = sep1[0] - tspl.point(tvec)
		vec = vec / norm(vec)

		min_a = 5.0
		for j in range(len(sym_nodes)):
			# Project to tspl to find vector
			tsym = tspl.project_point_to_centerline(sep2[sym_nodes[j]])
			sym = sep2[sym_nodes[j]] - tspl.point(tsym)
				
			# Transport  
			vec_trans = tspl.transport_vector(vec, tvec, tsym)
			a = angle(vec_trans, sym, axis = tspl.tangent(tsym), signed =True)
			if abs(a) < abs(min_a):
				min_a = a
				min_ind = sym_nodes[j]

		connect_sep = np.hstack((np.arange(min_ind, self._N), np.arange(0, min_ind)))
			
		# Number of cross sections
		num = int(tspl.length() / (bif1.get_spl()[branch].radius(1.0) * self._d))
			
		if num <= 1:
			num = 2
		if num//2 != 0:
			num = num + 1
		
		# Connect nodes
		nds = np.zeros((num, len(sep1), 3))
		for j in range(len(sep1)):
			nds[:,j,:] = connect_nodes(bif1, bif2, tspl, sep1[j], sep2[connect_sep[j]], num, tsep)


		new_nds1 = nds[:num//2 + 1][::-1].copy()
		new_nds2 = nds[num//2:].copy()

		
		nds1[branch + 1] = new_nds1[1:]
		nds2[0] = new_nds2[1:]

		end1[branch + 1] = new_nds1[0]
		end2[0] = new_nds2[0]

		# Change bifurcation connection 
		new_connect2 = connect2.copy()
		for j in range(len(sep2)):
			new_connect2[0][j] = connect2[0][connect_sep[j]]

		# Modify furcations
		bif1.set_crsec([end1, bifsec1, nds1, connect1])
		bif2.set_crsec([end2, bifsec2, nds2, new_connect2])


		self._model_graph.nodes[n]['bifurcation'] = bif2
		self._model_graph.nodes[nbifprec]['bifurcation'] = bif1
		

		self._crsec_graph.edges[e_in[0]]['connect'] = new_connect2[0]
		self._crsec_graph.edges[e_in[0]]['crsec'] = nds2[0]
		self._crsec_graph.edges[(e_in[0][0], nbifprec)]['crsec'] = nds1[branch + 1]
		self._crsec_graph.nodes[e_in[0][0]]['crsec'] = end2[0]
		self._crsec_graph.nodes[e_in[0][0]]['coords'] = tspl.point(0.5, True)

		# Cut tspl
		tspl1, tspl2 = tspl.split_time(0.5)
		tspl1.reverse()
		
		self._model_graph.edges[(nbifprec, e_in[0][0])]['spline'] = tspl1
		self._model_graph.edges[e_in[0]]['spline'] = tspl2




	#####################################
	#########  MESHING METHODS  #########
	#####################################


	def compute_cross_sections(self, N, d, parallel=True):

		""" Splits the splines into segments and bifurcation parts and computes surface cross sections.

		Keyword arguments:

		N -- number of nodes in a transverse section (multiple of 4)
		d -- longitudinal density of nodes as a proportion of the radius
		"""

		self._N = N
		self._d = d

		# Compute cross section graph
		
		self._crsec_graph = nx.DiGraph()
		for n in self._model_graph.nodes():
			self._crsec_graph.add_node(n, coords = self._model_graph.nodes[n]['coords'], type =  self._model_graph.nodes[n]['type'], crsec = None)

		for e in self._model_graph.edges():
			if self._model_graph.nodes[e[0]]['type'] == "bif": 
				self._crsec_graph.add_edge(e[1], e[0], crsec = None, connect = None, center = None)
			else:
				self._crsec_graph.add_edge(e[0], e[1], crsec = None, connect = None, center = None)


		if parallel:
			# Compute bifurcation sections (parallel)
			print('Meshing bifurcations.')
			args = []
			bif_id = []
			for n in self._model_graph.nodes():
				if self._model_graph.nodes[n]['type'] == "bif":
					# Get ids
					ids = [n] + [e[0] for e in self._model_graph.in_edges(n)] + [e[1] for e in self._model_graph.out_edges(n)]
					# Get refs
					end_ref = []
					for i in range(1, len(ids)):
						end_ref.append(self._model_graph.nodes[ids[i]]['ref'])

					args.append((self._model_graph.nodes[n]['bifurcation'], N, d, end_ref))
					bif_id.append(ids)

			# Return bifurcations with cross sections
			p = Pool(cpu_count())
			bif_list = p.starmap(parallel_bif, args)	
	
			# Add crsec to graph
			for i in range(len(bif_list)):

				end_crsec, bif_crsec, nds, connect_index = bif_list[i].get_crsec()
				self._crsec_graph.nodes[bif_id[i][0]]['crsec'] = bif_crsec
				self._model_graph.nodes[bif_id[i][0]]['bifurcation'] = bif_list[i]

				for j in range(len(bif_id[i]) - 1):
					if self._model_graph.in_degree(bif_id[i][j+1]) == 0 or self._model_graph.out_degree(bif_id[i][j+1]) ==0:
						order_crsec = [0] + np.arange(1,len(end_crsec[j])).tolist()[::-1]
						self._crsec_graph.nodes[bif_id[i][j+1]]['crsec'] = end_crsec[j][order_crsec]
						self._crsec_graph.edges[(bif_id[i][j+1], bif_id[i][0])]['crsec'] = nds[j][:,order_crsec]
						self._crsec_graph.edges[(bif_id[i][j+1], bif_id[i][0])]['connect'] = connect_index[j][order_crsec]
					else:
						self._crsec_graph.nodes[bif_id[i][j+1]]['crsec'] = end_crsec[j]
						self._crsec_graph.edges[(bif_id[i][j+1], bif_id[i][0])]['crsec'] = nds[j]
						self._crsec_graph.edges[(bif_id[i][j+1], bif_id[i][0])]['connect'] = connect_index[j]
				

			# Compute edges sections (parallel)
			print('Meshing edges.')
			args = []
			centers = []

			for e in self._model_graph.edges():
				if self._model_graph.nodes[e[0]]['type'] != "bif" and self._model_graph.nodes[e[1]]['type'] != "bif":
					
					spl = self._model_graph.edges[e]['spline']

					# Number of cross sections
					num = int(spl.length() / (spl.mean_radius()* d))
					if num <= 1:
						num = 2
					centers.append([0.0] + spl.resample_time(num) + [1.0])#np.linspace(0.0, 1.0, num + 2))
					v0 = self._model_graph.nodes[e[0]]['ref']
					alpha = self._model_graph.edges[e]['alpha']

					args.append((spl, num, N, v0, alpha))

			p = Pool(cpu_count())
			crsec_list  = p.starmap(segment_crsec, args)

			i = 0
			for e in self._model_graph.edges():#self._model_graph.edges():
				if self._model_graph.nodes[e[0]]['type'] != "bif" and self._model_graph.nodes[e[1]]['type'] != "bif":

					connect_id = self._model_graph.edges[e]['connect'] * (N // 4)
					crsec = crsec_list[i]

					self._crsec_graph.edges[e]['crsec'] = crsec[1:-1, :, :]
					self._crsec_graph.edges[e]['center'] = centers[i][1:-1]
					if self._model_graph.nodes[e[0]]['type'] != "sep":
						self._crsec_graph.nodes[e[0]]['crsec'] = crsec[0, :, :]

					if self._model_graph.nodes[e[1]]['type'] != "sep":
						self._crsec_graph.nodes[e[1]]['crsec'] = crsec[-1, :, :]

					# Write connection index
					if connect_id == 0:
						connect = np.arange(0, N)
					else: 
						connect = np.hstack((np.arange(connect_id, N), np.arange(0, connect_id)))

					self._crsec_graph.edges[e]['connect'] = connect.tolist()
					i += 1

		else: 

			# Compute bifurcation sections
			print('Meshing bifurcations.')
			for  n in self._model_graph.nodes():
				if self._model_graph.nodes[n]['type'] == "bif":
					self.furcation_cross_sections(n)
					

			# Compute edges sections
			print('Meshing edges.')
			for e in self._model_graph.edges():
				if self._model_graph.nodes[e[0]]['type'] != "bif" and self._model_graph.nodes[e[1]]['type'] != "bif":
					self.vessel_cross_sections(e)

		#self.show(False, False, False)

		# Combine bifurcations
		for n in self._model_graph.nodes():
			if self._model_graph.nodes[n]['combine']:
				self.__combine_nfurcation(n)

		nx.set_node_attributes(self._crsec_graph, None, name='id') # Add id attribute for meshing


	def recompute_cross_sections(self, n):

		""" Recompute the locally the cross section of a branch, given any node of the branch
		Keyword arguments :
		n -- one node of the branch (model graph)"""
		node_list = []
		edg_list = []

		# Find the path to the previous and next end or sep 
		if self._model_graph.nodes[n]['type'] == "end" and self._model_graph.in_degree(n)==0: # Inlet case
			
			sep_end = False
			# path to the next sep node
			path = [n]
			nd = n
			reach_end = False
			while not reach_end:
				succ = list(self._model_graph.successors(nd))[0]
				path.append(succ)
				nd = succ

				if self._model_graph.nodes[succ]['type'] == "end":
					reach_end = True
					
				if self._model_graph.nodes[succ]['type'] == "sep":
					sep_end = True
					reach_end = True

			for i in range(len(path)-1):
				edg_list.append((path[i], path[i+1]))

				

		if self._model_graph.nodes[n]['type'] == "sep":
		
			# path to the next sep node
			end_bif = False

			path = [n]
			nd = n
			reach_end = False
			while not reach_end:
				succ = list(self._model_graph.successors(nd))
				if len(succ) == 0:# in_sink, skip
					reach_end = True
				else:
					succ = succ[0]
					if self._model_graph.nodes[succ]['type'] == "bif":
						reach_end = True
					else:
						path.append(succ)
						nd = succ

					if self._model_graph.nodes[succ]['type'] == "end":
						reach_end = True

					if self._model_graph.nodes[succ]['type'] == "sep":
						end_bif = True
						reach_end = True
						break

			if len(path) > 1: # If path  = 0, the sep node in a bifurcation inlet, no need to compute cross sections

				if end_bif:

					ref0 = self._model_graph.nodes[path[0]]['ref']

					# Get in bif 
					for e in self._model_graph.in_edges(path[0]):
						if self._model_graph.nodes[e[0]]["type"] =="bif":
							in_bif = e[0]

					for e in self._model_graph.out_edges(path[0]):
						if self._model_graph.nodes[e[1]]["type"] =="bif":
							in_bif = e[1]

					# Get out bif 
					for e in self._model_graph.in_edges(path[-1]):
						if self._model_graph.nodes[e[0]]["type"] =="bif":
							out_bif = e[0]

					for e in self._model_graph.out_edges(path[-1]):
						if self._model_graph.nodes[e[1]]["type"] =="bif":
							out_bif = e[1]

					for i in range(len(path)-1):
						edg_list.append((path[i], path[i+1]))

					node_list.append(in_bif)
					node_list.append(out_bif)
					

				else: 

					for i in range(len(path)-1):
						edg_list.append((path[i], path[i+1]))
						
				
		elif (self._model_graph.nodes[n]['type'] == "end" and self._model_graph.in_degree(n)!=0) or self._model_graph.nodes[n]['type'] == "reg":
		
			prec_sep = n
			
			nd = n
			reach_end = False
			while not reach_end:
				prec = list(self._model_graph.predecessors(nd))[0]
				nd = prec
				if self._model_graph.nodes[prec]['type'] == "sep" or  self._model_graph.nodes[prec]['type'] == "end":
					prec_sep = prec
					reach_end = True

			self.recompute_cross_sections(prec_sep)
					

		else: # the node type is bif or inlet sep
			pass


		# Recompute the cross sections
		print("node_list", node_list)
		print("edge_list", edg_list)
					# Compute bifurcation sections
		print('Meshing bifurcations.')
		for  n in node_list:
			self.furcation_cross_sections(n)
					

		# Compute edges sections
		print('Meshing edges.')
		for e in edg_list:
				self.vessel_cross_sections(e)

		if self._surface_mesh is not None:
			self.mesh_surface()




	def furcation_cross_sections(self, n):

		""" Compute furcation cross sections and add it to crsec graph attribute 
		Keyword arguments: 
		n -- crsec graph node id
		"""

		bif = self._model_graph.nodes[n]['bifurcation']
		ids = [e[0] for e in self._model_graph.in_edges(n)] + [e[1] for e in self._model_graph.out_edges(n)]

		end_ref = []
		for e in ids:
			end_ref.append(self._model_graph.nodes[e]['ref'])

		end_crsec, bif_crsec, nds, ind = bif.compute_cross_sections(self._N, self._d, end_ref=end_ref)
		self._crsec_graph.nodes[n]['crsec'] = bif_crsec

		for j in range(len(ids)):

			if self._model_graph.in_degree(ids[j]) == 0 or self._model_graph.out_degree(ids[j]) == 0:
				order_crsec = [0] + np.arange(1,len(end_crsec[j])).tolist()[::-1]
				self._crsec_graph.nodes[ids[j]]['crsec'] = end_crsec[j][order_crsec]
				self._crsec_graph.edges[(ids[j], n)]['crsec'] = nds[j][:,order_crsec]
				self._crsec_graph.edges[(ids[j], n)]['connect'] = connect_index[j][order_crsec]
			else:

				self._crsec_graph.nodes[ids[j]]['crsec'] = end_crsec[j]
				self._crsec_graph.edges[(ids[j], n)]['crsec'] = nds[j]
				self._crsec_graph.edges[(ids[j], n)]['connect'] = ind[j]

	
	def vessel_cross_sections(self, e):

		""" Compute vessel cross sections and add it to crsec graph attribute 
		Keyword arguments: 
		e -- crsec graph edge id
		"""

		spl = self._model_graph.edges[e]['spline']
		connect_id = self._model_graph.edges[e]['connect'] * (self._N // 4)

		# Number of cross sections
		num = int(spl.length() / (spl.mean_radius()* self._d))
		if num <= 1:
			num = 2
		v0 = self._model_graph.nodes[e[0]]['ref']
		alpha = self._model_graph.edges[e]['alpha']

		crsec, center = self.__segment_crsec(spl, num, self._N, v0, alpha)

		self._crsec_graph.edges[e]['crsec'] = crsec[1:-1, :, :]
		self._crsec_graph.edges[e]['center'] = center[1:-1]
		if self._model_graph.nodes[e[0]]['type'] != "sep":
			self._crsec_graph.nodes[e[0]]['crsec'] = crsec[0, :, :]

		if self._model_graph.nodes[e[1]]['type'] != "sep":
			self._crsec_graph.nodes[e[1]]['crsec'] = crsec[-1, :, :]

		# Write connection index
		if connect_id == 0:
			connect = np.arange(0, self._N)
		else: 
			connect = np.hstack((np.arange(connect_id, self._N), np.arange(0, connect_id)))

		self._crsec_graph.edges[e]['connect'] = connect.tolist()



	def __add_node_id_surface(self, nds = [], edg = []):

		""" Add id attribute to the nodes for surface meshing"""

		if len(nds) == 0:
			nds = [n for n in self._crsec_graph.nodes()]
		if len(edg) == 0:
			edg = [e for e in self._crsec_graph.edges()]

		count = 0

		for n in nds:
			self._crsec_graph.nodes[n]['id'] = count
			count += self._crsec_graph.nodes[n]['crsec'].shape[0]


		for e in edg:
			self._crsec_graph.edges[e]['id'] = count
			count += self._crsec_graph.edges[e]['crsec'].shape[0] * self._crsec_graph.edges[e]['crsec'].shape[1] 

		self._nb_nodes = count 



	def __add_node_id_volume(self, num_a, num_b, nds = [], edg = []):

		""" Add id attribute to the nodes for volume meshing"""

		if len(nds) == 0:
			nds = [n for n in self._crsec_graph.nodes()]
		if len(edg) == 0:
			edg = [e for e in self._crsec_graph.edges()]


		count = 0
		nb_nds_ogrid = int(self._N * (num_a + num_b + 3) + ((self._N - 4)/4)**2)
	

		for n in nds:
			self._crsec_graph.nodes[n]['id'] = count
			if self._crsec_graph.nodes[n]['type'] == "bif":

				nbif = self._crsec_graph.in_degree(n)
				nb_nds_bif_ogrid = int(nb_nds_ogrid + (nbif-2) * ((self._N/2 - 1) * (num_a + num_b + 3) + (self._N - 4)/4 * (((self._N - 4)/4 -1)/2)))

				count +=  nb_nds_bif_ogrid
			else: 
				count += nb_nds_ogrid

		for e in edg:
			self._crsec_graph.edges[e]['id'] = count
			count += self._crsec_graph.edges[e]['crsec'].shape[0] * nb_nds_ogrid

		self._nb_nodes = count 

		

	def __segment_crsec(self, spl, num, N, v0 = [], alpha = None):

		""" Compute the cross section nodes along for a vessel segment.

		Keyword arguments:
		spl -- segment spline
		num -- number of cross sections
		N -- number of nodes in a cross section (multiple of 4)
		v0 -- reference vector
		alpha -- rotation angle
		"""

		#t = np.linspace(0.0, 1.0, num + 2) 
		t = [0.0] + spl.resample_time(num) + [1.0]

		if len(v0) == 0:
			v0 = cross(spl.tangent(0), np.array([0,0,1])) # Random initialisation of the reference vector
			

		if alpha!=None:
			theta = np.linspace(0.0, alpha, num + 2) # Get rotation angles

		crsec = np.zeros((num + 2, N, 3))

		for i in range(num + 2):
			
			tg = spl.tangent(t[i])
			v = spl.transport_vector(v0, 0, t[i]) # Transports the reference vector to time t[i]

			if alpha!=None: 
				v = rotate_vector(v, tg, theta[i]) # Rotation of the reference vector

			crsec[i,:,:] = self.__single_crsec(spl, t[i], v, N)

		return crsec, t




	def __single_crsec(self, spl, t, v, N):


		""" Returns the list of N nodes of a single cross section.

		Keyword arguments:
		spl -- spline of the centerline
		t -- time 
		v -- vector pointing to the first node (reference)
		N -- number of nodes of the cross section (multiple of 4)

		"""

		tg = spl.tangent(t)

		# Test the orthogonality of v and the tangent
		if abs(dot(tg, v)) > 0.01:
			raise ValueError('Non-orthogonal cross section')
	
		angle = 2 * pi / N
		angle_list = angle * np.arange(N)

		nds = np.zeros((N, 3))
		for i in range(N):
			n = rotate_vector(v, tg, angle_list[i])
			nds[i, :] = spl.project_time_to_surface(n, t)

		return nds



	def mesh_surface(self, edg = [], link = True):

		""" Meshes the surface of the arterial tree."""

		if self._crsec_graph is None:

			print('Computing cross sections with default parameters...')
			self.compute_cross_sections(24, 0.2) # Get cross section graph
		

		print('Meshing surface...')

		nds = []
		for e in edg:
			if e[0] not in nds:
				nds.append(e[0])
			if e[1] not in nds:
				nds.append(e[1])

		# Add node id
		self.__add_node_id_surface(nds, edg)
		G = self._crsec_graph

		if len(edg) == 0:
			edg = [e for e in G.edges()]

		if link:
			link_graph = np.zeros((self._nb_nodes, 3), dtype =int)
		

		vertices = np.zeros((self._nb_nodes, 3))
		faces = np.zeros((self._nb_nodes, 5), dtype =int)
		nb_faces = 0

		for e in edg:		

		
			flip_norm = False

			if e not in [e for e in self._model_graph.edges()]:
				flip_norm = True

			# Mesh the first cross section
			id_first = G.nodes[e[0]]['id']
			id_edge = G.edges[e]['id']
			id_last = G.nodes[e[1]]['id']

			# Add vertices
			for i in range(G.nodes[e[0]]['crsec'].shape[0]): # First cross section
				vertices[id_first + i, :] = G.nodes[e[0]]['crsec'][i]

			for i in range(G.edges[e]['crsec'].shape[0]): # Edge sections
				for j in range(G.edges[e]['crsec'].shape[1]):
					vertices[id_edge + (i * self._N) + j, :] = G.edges[e]['crsec'][i, j]

			for i in range(len(G.nodes[e[1]]['crsec'])): # Last cross section
				vertices[id_last + i, :] = G.nodes[e[1]]['crsec'][i]

			# Add faces
			for i in range(G.nodes[e[0]]['crsec'].shape[0]): # First cross section

				if i == G.nodes[e[0]]['crsec'].shape[0] - 1:
					j = 0
				else: 
					j = i + 1

				faces[nb_faces,:] = np.array([4, id_first + i, id_edge + i, id_edge + j, id_first + j])
				if flip_norm:
					faces[nb_faces,:] = np.array([4, id_first + i, id_first + j, id_edge + j, id_edge + i])

				if link:
					link_graph[nb_faces, :] = [e[0], e[1], 0]


				nb_faces += 1

			for k in range(G.edges[e]['crsec'].shape[0] -1): # Edge sections
				for i in range(G.edges[e]['crsec'].shape[1]):

					if i == G.edges[e]['crsec'].shape[1] - 1:
						j = 0
					else: 
						j = i + 1

					faces[nb_faces,:] = np.array([4, id_edge + (k * self._N) + i, id_edge + ((k + 1) * self._N) + i, id_edge + ((k + 1) * self._N) + j, id_edge + (k * self._N) + j])
					if flip_norm:
						faces[nb_faces,:] = np.array([4, id_edge + (k * self._N) + i, id_edge + (k * self._N) + j , id_edge + ((k + 1) * self._N) + j, id_edge + ((k + 1) * self._N) + i])

					if link:
						link_graph[nb_faces, :] = [e[0], e[1], k+1]
					
					nb_faces += 1

			id_last_edge = id_edge + ((G.edges[e]['crsec'].shape[0] -1) * self._N)
			connect = G.edges[e]['connect']


			for i in range(G.edges[e]['crsec'].shape[1]): # Last section

				if i == G.edges[e]['crsec'].shape[1] - 1:
					j = 0
				else:
					j = i + 1

				faces[nb_faces,:] = np.array([4, id_last_edge + i, id_last + connect[i], id_last + connect[j], id_last_edge + j])
				if flip_norm:
					faces[nb_faces,:] = np.array([4, id_last_edge + i, id_last_edge + j , id_last + connect[j], id_last + connect[i]])

				if link:
					link_graph[nb_faces, :] = [e[0], e[1], k+2]

				nb_faces += 1

		
		faces = faces[:nb_faces]
		if link:
			link_graph = link_graph[:nb_faces]

		if link:
			self._surface_mesh = [vertices, faces, link_graph]
			
		else:
			self._surface_mesh = [vertices, faces]

		return pv.PolyData(vertices, faces)



	def mesh_volume(self, layer_ratio = [0.2, 0.3, 0.5], num_a = 10, num_b=10, edg = [], link = True):

		""" Meshes the volume of the arterial tree with O-grid pattern.

		Keyword arguments:

		layer_ratio -- radius ratio of the three O-grid parts [a, b, c] such as a+b+c = 1
		num_a, num_b -- number of layers in the parts a and b
		"""

		if self._crsec_graph is None:

			print('Computing cross sections with default parameters...')
			self.compute_cross_sections(24, 0.2) # Get cross section graph

		G = self._crsec_graph
		
		if self._N%8 != 0:
			raise ValueError('The number of cross section nodes must be a multiple of 8 for volume meshing.')

		print('Meshing volume...')

		nds = []
		for e in edg:
			if e[0] not in nds:
				nds.append(e[0])
			if e[1] not in nds:
				nds.append(e[1])


		# Keep parameters as attributes
		self._layer_ratio = layer_ratio
		self._num_a = num_a
		self._num_b = num_b


		nb_nds_ogrid = int(self._N * (num_a + num_b + 3) + ((self._N - 4)/4)**2)

		# Compute node ids
		self.__add_node_id_volume(num_a, num_b, nds, edg)

		if len(edg) == 0:
			edg = [e for e in G.edges()]

		# Compute faces
		f_ogrid = self.ogrid_pattern_faces(self._N, num_a, num_b)
		f_ogrid_flip = np.copy(f_ogrid[:,[0,1,4,3,2]])

		if link:
			link_graph = np.zeros((self._nb_nodes, 3), dtype =int)
			

		# Add vertices and cells to the mesh
		vertices = np.zeros((self._nb_nodes,3))
		cells = np.zeros((self._nb_nodes,9), dtype=int)
		nb_cells = 0

		for e in edg:

			flip_norm = False

			if e not in [e for e in self._model_graph.edges()]:
				flip_norm = True

			# Mesh the first cross section
			id_first = G.nodes[e[0]]['id']
			id_edge = G.edges[e]['id']
			id_last = G.nodes[e[1]]['id']

			# Add vertices
			# First cross section
			crsec = G.nodes[e[0]]['crsec']
			center = G.nodes[e[0]]['coords'][:-1]
			v = self.ogrid_pattern_vertices(center, crsec, layer_ratio, num_a, num_b)

			vertices[id_first:id_first + nb_nds_ogrid, :] = v
			cells[nb_cells: nb_cells + f_ogrid.shape[0], :] = np.hstack((np.zeros((f_ogrid.shape[0], 1)) + 8, id_first + f_ogrid[:,1:], id_edge + f_ogrid[:,1:]))
			if flip_norm:
				cells[nb_cells: nb_cells + f_ogrid.shape[0], :] = np.hstack((np.zeros((f_ogrid_flip.shape[0], 1)) + 8, id_first + f_ogrid_flip[:,1:], id_edge + f_ogrid_flip[:,1:]))

			if link:
				link_graph[nb_cells: nb_cells + f_ogrid.shape[0], :2] = np.repeat(np.array([e[0],e[1]]).reshape((1,2)), f_ogrid.shape[0], axis = 0)
				link_graph[nb_cells: nb_cells + f_ogrid.shape[0], -1] = np.repeat(0, f_ogrid.shape[0])

			nb_cells += f_ogrid.shape[0]

			# Edge cross sections
			for i in range(G.edges[e]['crsec'].shape[0]):

				crsec = G.edges[e]['crsec'][i]
				center = (crsec[0] + crsec[int(self._N/2)])/2.0
				v = self.ogrid_pattern_vertices(center, crsec, layer_ratio, num_a, num_b)	
				id_crsec = id_edge + (i * nb_nds_ogrid)

				vertices[id_crsec:id_crsec + nb_nds_ogrid, :] = v

			for i in range(G.edges[e]['crsec'].shape[0] - 1):

				cells[nb_cells: nb_cells + f_ogrid.shape[0], :] = np.hstack((np.zeros((f_ogrid.shape[0], 1)) + 8, id_edge + (nb_nds_ogrid * i) + f_ogrid[:,1:], id_edge + (nb_nds_ogrid * (i + 1)) + f_ogrid[:,1:]))
				if flip_norm:
					cells[nb_cells: nb_cells + f_ogrid.shape[0], :] = np.hstack((np.zeros((f_ogrid_flip.shape[0], 1)) + 8, id_edge + (nb_nds_ogrid * i) + f_ogrid_flip[:,1:], id_edge + (nb_nds_ogrid * (i + 1)) + f_ogrid_flip[:,1:]))

				if link:
					link_graph[nb_cells: nb_cells + f_ogrid.shape[0], :2] = np.repeat(np.array([e[0],e[1]]).reshape((1,2)), f_ogrid.shape[0], axis = 0)
					link_graph[nb_cells: nb_cells + f_ogrid.shape[0], -1] = np.repeat(i+1, f_ogrid.shape[0])
				nb_cells += f_ogrid.shape[0]

			# Last cross section
			if G.nodes[e[1]]['type'] == "bif": # Bifurcation case 

				nbif = G.in_degree(e[1])
				nb_nds_bif_ogrid = int(nb_nds_ogrid +  (nbif-2) * ((self._N/2 - 1) * (num_a + num_b + 3) + (self._N - 4)/4 * (((self._N - 4)/4 -1)/2)))

				crsec = G.nodes[e[1]]['crsec']
				center = (np.array(crsec[0]) + crsec[1])/2.0


				# Mesh of the 3 half-sections
				v = self.bif_ogrid_pattern_vertices(center, crsec, self._N, nbif, layer_ratio, num_a, num_b)	
				f_bif_ogrid = self.bif_ogrid_pattern_faces(self._N, nbif, num_a, num_b)
				vertices[id_last:id_last + nb_nds_bif_ogrid, :] = v

				# Get bifurcation half section id and orientation from connection information
				connect = G.edges[e]['connect']


				nodes_num = np.arange(2, len(crsec)).tolist()
				l = len(nodes_num) // nbif

				ind = []
				for k in range(nbif):
					ind.append((nodes_num[k*l:(k+1)*l]))


				h = []
				for s in [1, int(self._N/2) + 1]:
					for k in range(len(ind)):
						if connect[s] in ind[k]:
							h1 = [k]

							if s == 1:
								if connect[s] != ind[k][0] and connect[s]!= ind[k][-1]:
									quarter = 1
								else:
									quarter = 0

							#if connect[s] == ind[k][0]:
							if ind[k].index(connect[s]) < ind[k].index(connect[s + 1]):
								h1+=[0,1]
							else:
								h1+=[1,0]
					h.append(h1)

				f_ogrid_reorder = self.__reorder_faces(h, f_bif_ogrid, self._N, num_a, num_b)

				
				# Reorder faces for rotation
				start_face = int(quarter * f_ogrid_reorder.shape[0] / 4)
		
				if start_face != 0:
					f_ogrid_reorder = np.vstack((f_ogrid_reorder[start_face:, :], f_ogrid_reorder[:start_face, :]))
		
				cells[nb_cells: nb_cells + f_ogrid_reorder.shape[0], :] = np.hstack((np.zeros((f_ogrid_reorder.shape[0], 1)) + 8, id_edge + (nb_nds_ogrid * (G.edges[e]['crsec'].shape[0] - 1)) + np.array(f_ogrid)[:,1:], id_last + np.array(f_ogrid_reorder)[:,1:]))

				if flip_norm:
					f_ogrid_reorder_flip = np.copy(f_ogrid_reorder[:,[0,1,4,3,2]])
					cells[nb_cells: nb_cells + f_ogrid_reorder.shape[0], :] = np.hstack((np.zeros((f_ogrid_reorder.shape[0], 1)) + 8, id_edge + (nb_nds_ogrid * (G.edges[e]['crsec'].shape[0] - 1)) + np.array(f_ogrid_flip)[:,1:], id_last + np.array(f_ogrid_reorder_flip)[:,1:]))

				if link:
					link_graph[nb_cells: nb_cells + f_ogrid_reorder.shape[0], :2] = np.repeat(np.array([e[0],e[1]]).reshape((1,2)), f_ogrid_reorder.shape[0], axis = 0)
					link_graph[nb_cells: nb_cells + f_ogrid_reorder.shape[0], -1] = np.repeat(i+2, f_ogrid_reorder.shape[0])

				nb_cells += f_ogrid_reorder.shape[0]


			else: 

				crsec = G.nodes[e[1]]['crsec']
				center = G.nodes[e[1]]['coords'][:-1]
				v = self.ogrid_pattern_vertices(center, crsec, layer_ratio, num_a, num_b)

				# Reorder faces for rotation
				quarter = [0, int(self._N/4), int(self._N/2), int(3* self._N/4)].index(G.edges[e]['connect'][0])
				start_face = int(quarter * f_ogrid.shape[0] / 4)
		
				if start_face != 0:
					f_ogrid_rot = np.vstack((f_ogrid[start_face:, :], f_ogrid[:start_face, :]))
				else: 
					f_ogrid_rot = f_ogrid

				vertices[id_last:id_last + nb_nds_ogrid, :] = v
				cells[nb_cells: nb_cells + f_ogrid.shape[0], :] = np.hstack((np.zeros((f_ogrid.shape[0], 1)) + 8, id_edge + (nb_nds_ogrid * (G.edges[e]['crsec'].shape[0] - 1)) + np.array(f_ogrid)[:,1:], id_last + np.array(f_ogrid_rot)[:,1:]))

				if link:
					link_graph[nb_cells: nb_cells + f_ogrid.shape[0], :2] = np.repeat(np.array([e[0],e[1]]).reshape((1,2)), f_ogrid.shape[0], axis = 0)
					link_graph[nb_cells: nb_cells + f_ogrid.shape[0], -1] = np.repeat(i+2, f_ogrid.shape[0])

				nb_cells += f_ogrid.shape[0]


		cells = cells[:nb_cells]
		if link: 
			link_graph = link_graph[:nb_cells]

		cell_types = np.array([vtk.VTK_HEXAHEDRON] * cells.shape[0])

		# Return volume mesh
		mesh = pv.UnstructuredGrid(cells, cell_types, vertices) 

		if link:
			self._volume_mesh = [cells, cell_types, vertices, link_graph]
		else:
			self._volume_mesh = [cells, cell_types, vertices]
		return mesh


	def write_pyvista_mesh_from_vtk(self, vertices, cells):

		points = vtk.vtkPoints()		
		cell = vtk.vtkCellArray()

		for i in range(len(vertices)):
			points.InsertNextPoint(vertices[i, 0],vertices[i, 1],vertices[i, 2])

		for i in range(len(cells)):
			quad = vtk.vtkHexahedron()
			for j in range(1, 9):
				quad.GetPointIds().SetId(j-1,cells[i, j])
			cell.InsertNextCell(quad)

		grid = vtk.vtkUnstructuredGrid()
		grid.SetPoints(points)
		grid.SetCells(vtk.VTK_HEXAHEDRON, cell)

		writer = vtk.vtkXMLUnstructuredGridWriter()
		writer.SetFileName('tmp.vtu')
		writer.SetInputData(grid)
		writer.Write() 

		mesh = pv.read("tmp.vtu")

		return mesh


	def bif_ogrid_pattern(self, center, crsec, N, nbif, layer_ratio, num_a, num_b):


		""" Computes the nodes of a O-grid pattern from the bifurcation separation nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array 
		N -- number of nodes of the cross section (multiple of 4)
		nbif -- id of bifurcation node
		layer_ratio -- radius ratio of the three O-grid parts [a, b, c] such as a+b+c = 1
		num_a, num_b -- number of layers in the parts a and b
		"""
		vertices = self.bif_ogrid_pattern_vertices(center, crsec, N, nbif, layer_ratio, num_a, num_b) 
		faces = self.bif_ogrid_pattern_faces(N, nbif, num_a, num_b)


		return vertices, faces



	def bif_ogrid_pattern_faces(self, N, nbif, num_a, num_b):


		""" Computes the nodes of a O-grid pattern from the bifurcation separation nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array 
		N -- number of nodes of the cross section (multiple of 4)
		nbif -- id of bifurcation node
		num_a, num_b -- number of layers in the parts a and b
		"""

		# Get the suface nodes of each individual half section
		Nh = int((N - 2) / 2) 
		
		
		nb = int(N/8) * (2 * (num_a + num_b + 2) + int(N/8))
		faces = np.zeros((nbif, 2, nb, 5), dtype = int)
			
		shared_edge2 = []
		count = 0

		for h in range(nbif):

			# Separate in quarters
			for q in range(2):
				nb_faces = 0

				if q == 0:
					# Point of the horizontal shared line
					if h == 0:
						nb_vertices = int(num_a + num_b + 2 + N/8 + 1)
						shared_edge1 = list(range(count, count + nb_vertices))
						center = shared_edge1[-1]
					else: 
						nb_vertices = int(num_a + num_b + 2 + N/8)
						shared_edge1 = list(range(count, count + nb_vertices)) + [center]
					
					count += nb_vertices

				for i in range(int(N/4)):
					
					# First half points	
					if i <= int(N/8): 
						if i == 0:
							if h == 0:
								nb_vertices = int(num_a + num_b + 2 + N/8)
								ray_ind = list(range(count, count + nb_vertices)) + [shared_edge1[-i-1]]
								count += nb_vertices
								shared_edge2.append(ray_ind)
							else: 
								ray_ind = shared_edge2[q] # Common edge of the bifurcation plan

						else:
							nb_vertices = int(num_a + num_b + 2 + N/8)
							ray_ind = list(range(count, count + nb_vertices)) + [shared_edge1[-i-1]]
							count += nb_vertices

						if i == int(N/8):
							shared_edge3 = ray_ind[-int(N/8):]
							
					else: # Second half points
						nb_vertices = int(num_a + num_b + 2)
						ray_ind = list(range(count, count + nb_vertices)) + [shared_edge3[i-int(N/8)-1]]
						count += nb_vertices

					if i > 0:
						for k in range(min([len(ray_prec)-1, len(ray_ind)-1])):
							faces[h, q, nb_faces, :] = np.array([4, ray_prec[k], ray_ind[k], ray_ind[k+1], ray_prec[k+1]])
							nb_faces += 1
				
					ray_prec = ray_ind[:]

				for k in range(min([len(ray_ind)-1, len(shared_edge1)-1])):
					faces[h, q, nb_faces, :] = np.array([4, ray_ind[k], shared_edge1[k], shared_edge1[k+1], ray_ind[k+1]])
					nb_faces += 1


		return faces


	def bif_ogrid_pattern_vertices(self, center, crsec, N, nbif, layer_ratio, num_a, num_b):


		""" Computes the nodes of a O-grid pattern from the bifurcation separation nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array 
		N -- number of nodes of the cross section (multiple of 4)
		nbif -- id of bifurcation node
		layer_ratio -- radius ratio of the three O-grid parts [a, b, c] such as a+b+c = 1
		num_a, num_b -- number of layers in the parts a and b
		"""


		if sum(layer_ratio) != 1.0:
			raise ValueError("The sum of the layer ratios must equal 1.")
			
		# Get the suface nodes of each individual half section
		Nh = int((N - 2)/2)

		nb_nds_ogrid = int(N * (num_a + num_b + 3) + ((N - 4)/4)**2)
		nb_vertices = int(nb_nds_ogrid +  (nbif - 2) * ((N/2 - 1) * (num_a + num_b + 3) + (N - 4)/4 * (((N - 4)/4 -1)/2)))

		half_crsec =  np.zeros((nbif, Nh + 2, 3))
		
		half_crsec[:, 0, :] = [crsec[0]] * nbif
		half_crsec[:, -1, :] = [crsec[1]] * nbif

		for i in range(nbif):
			half_crsec[i, 1 : Nh + 1, :] = crsec[(Nh*i) + 2:(Nh*(i+1)) + 2]


		vertices = np.zeros((nb_vertices, 3))
		count = 0
		for h in range(nbif):
		
			# Separate in quarters
			quarters = np.array([half_crsec[h,:int(N/4)+1], half_crsec[h, int(N/4):][::-1]])
	

			for q in range(2):

				# Computes the coordinates of the corners and the side nodes of the central square
				square_corners = []
				for n in [0, int(N/8), int(N/4)]:

					v = quarters[q, n] - center
					pt = (center + v / norm(v) * (layer_ratio[2] * norm(v))).tolist()
					square_corners.append(pt)
					

				square_sides1 = [lin_interp(square_corners[0], square_corners[1], N/8+1), lin_interp(center, square_corners[2], N/8+1)] # Horizontal square edges
				square_sides2 = [lin_interp(square_corners[0], center, N/8+1), lin_interp(square_corners[1], square_corners[2], N/8+1)] # Vertical square edges

				if q == 0:
					# Point of the horizontal shared line
					v = square_corners[2] - quarters[0, -1] # Direction of the ray
					pb = quarters[0][-1] + v / norm(v) * (layer_ratio[0] * norm(v)) # Starting point of layer b
					if h == 0:
						ray_vertices = lin_interp(quarters[0, -1], pb, num_a + 2)[:-1] + lin_interp(pb, square_corners[2], num_b + 2)[:-1] + lin_interp(square_corners[2], center, N/8 + 1)
					else: 
						ray_vertices = lin_interp(quarters[0, -1], pb, num_a + 2)[:-1] + lin_interp(pb, square_corners[2], num_b + 2)[:-1] + lin_interp(square_corners[2], center, N/8 + 1)[:-1]
					vertices[count: count + len(ray_vertices)]  =  ray_vertices
					count += len(ray_vertices)

				for i in range(quarters.shape[1]-1):
					
					# First half points	
					if i <= (quarters.shape[1]-1) / 2: 
						
						v = square_sides1[0][i] - quarters[q, i] # Direction of the ray
						pb = quarters[q, i] + v / norm(v) * (layer_ratio[0] * norm(v)) # Starting point of layer b

						if i == 0:
							if h == 0:
								ray_vertices = lin_interp(quarters[q, i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][i], num_b + 2)[:-1] + lin_interp(square_sides1[0][i], square_sides1[1][i], N/8 + 1)[:-1]
							else: 
								ray_vertices = []
						else:
							ray_vertices = lin_interp(quarters[q, i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][i], num_b + 2)[:-1] + lin_interp(square_sides1[0][i], square_sides1[1][i], N/8 + 1)[:-1]

					else: # Second half points
						
						v = square_sides2[1][i-int(N/8)] - quarters[q, i] # Direction of the ray
						pb = quarters[q, i] + v / norm(v) * (layer_ratio[0] * norm(v)) # Starting point of layer b

						ray_vertices = lin_interp(quarters[q, i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides2[1][i-int(N/8)], num_b + 2)[:-1]
					
					if len(ray_vertices) != 0:
						vertices[count: count + len(ray_vertices)]  =  ray_vertices
						count += len(ray_vertices)

		return vertices




	def __reorder_faces(self, h, f, N, num_a, num_b):

		""" Reorder the faces of the separation mesh to connect with different branches """
				
		f_ord = np.zeros((2, f.shape[1], f.shape[2], f.shape[3]), dtype = int) 

		nb_short =  num_a + num_b + 2
		nb_long = num_a + num_b + 2 + int(N/8)

	
		for s in range(2): # Both half-sections

			# Write first quarter
			f_ord[s, 0] = f[h[s][0], h[s][1]]
			f_ord[s, 0, :, 0] = [4] * f_ord.shape[2]
			
			# Reorder second quarter
			ind1 = 0
			ind2 = 0
			
			for i in range(int(N/4)): # Inverse the rays of faces
				
				if i < int(N/8):
	
					if ind2 == 0:
						f_ord[s, 1, ind1:ind1 + nb_short , :] = np.transpose(f[h[s][0], h[s][2], -(nb_short + ind2):, [0, 2, 1, 4, 3]]) # Inverse rays
					else: 
						f_ord[s, 1, ind1:ind1 + nb_short , :] = np.transpose(f[h[s][0], h[s][2], -(nb_short + ind2):-ind2, [0, 2, 1, 4, 3]]) # Inverse rays
					ind1 += nb_long 
					ind2 += nb_short
				
				else:
					if i == int(N/8):
						ind2 = ind2 + int(N/8)

					f_ord[s, 1, ind1:ind1 + nb_short , :] = np.transpose(f[h[s][0], h[s][2], -(nb_short + ind2):-ind2, [0, 2, 1, 4, 3]]) # Inverse rays

					for j in range(int(N/8)):
						f_ord[s, 1, j*(nb_long-1) + nb_short + j + i - int(N/8) , :] = np.transpose(f[h[s][0], h[s][2], -(ind2 + j -int(N/8) +1),[0, 3, 2, 1, 4]])
					

					ind1 += nb_short
					ind2 += nb_long
	
		return f_ord.reshape((-1, 5))



	def ogrid_pattern(self, center, crsec):

		""" Computes the nodes of a O-grid pattern from the cross section surface nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array
		"""
		
		vertices = self.ogrid_pattern_vertices(center, crsec, self._layer_ratio, self._num_a, self._num_b)
		faces = self.ogrid_pattern_faces(self._N, self._num_a, self._num_b)

		return vertices, faces




	def ogrid_pattern_vertices(self, center, crsec, layer_ratio, num_a, num_b):

		""" Computes the nodes of a O-grid pattern from the cross section surface nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array
		layer_ratio, num_a, num_b -- parameters of the O-grid
		"""

		if sum(layer_ratio) != 1.0:
			raise ValueError("The sum of the layer ratios must equal 1.")
		
		
		N = crsec.shape[0]
		nb_vertices = int(N * (num_a + num_b + 3) + ((N - 4)/4)**2)

		vertices = np.zeros((nb_vertices, 3))
		
		# Get the symmetric nodes of the pattern
		sym_nodes = np.array([0, int(N/8), int(N/4)])
		count = 0

		for s in range(4): # For each quarter
			j = 0

			# Computes the coordinates of the corners and the side nodes of the central square
			square_corners = []
			for n in sym_nodes:

				if n == N:
					n = 0
				v = crsec[n] - center
				pt = (center + v / norm(v) * (layer_ratio[2] * norm(v))).tolist()
				square_corners.append(pt)
				

			square_sides1 = [lin_interp(square_corners[0], square_corners[1], N/8+1), lin_interp(center, square_corners[2], N/8+1)] # Horizontal square edges
			square_sides2 = [lin_interp(square_corners[0], center, N/8+1), lin_interp(square_corners[1], square_corners[2], N/8+1)] # Vertical square edges

			for i in range(int(s * N/4), int((s+1) * N/4)):

				# First half points	
				if j <= N/8: 
					
					v = square_sides1[0][j] - crsec[i] # Direction of the ray
					pb = crsec[i] + v / norm(v) * (layer_ratio[0] * norm(v)) # Starting point of layer b

					if s != 0 and j == 0:
						ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][j], num_b + 2)[:-1]  # Shared square

					elif s==3:
						ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][j], num_b + 2)[:-1] + lin_interp(square_sides1[0][j], square_sides1[1][j], N/8 + 1)[:-1]
						
					else:
						ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][j], num_b + 2)[:-1] + lin_interp(square_sides1[0][j], square_sides1[1][j], N/8 + 1)

				else: 
					v = square_sides2[1][j-int(N/8)] - crsec[i] # Direction of the ray
					pb = crsec[i] + v / norm(v) * (layer_ratio[0] * norm(v)) # Starting point of layer b

					ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides2[1][j-int(N/8)], num_b + 2)[:-1]
					
				vertices[count: count + len(ray_vertices)]  = ray_vertices
				count += len(ray_vertices)

				j = j + 1

			sym_nodes = sym_nodes + int(N/4)

		
		return np.array(vertices)



	def ogrid_pattern_faces(self, N, num_a, num_b):

		""" Computes the nodes of a O-grid pattern from the cross section surface nodes.

		Keyword arguments: 
		N -- number of nodes of the cross section
		num_a, num_b -- parameters of the O-grid
		"""

		count = 0
		tot_faces = int(N/8) * (2 * (num_a + num_b + 2) + int(N/8)) * 4
		faces = np.zeros((tot_faces, 5), dtype = int)

		shared_edge1 = []
		nb_faces = 0

		for s in range(4): # For each quarter
			j = 0

			for i in range(int(s * N/4), int((s+1) * N/4)):

				# First half points	
				if j <= int(N/8): 
				
					if s != 0 and j == 0:
						nb_vertices = int(num_a + num_b + 2)
						ray_ind = list(range(count, count + nb_vertices)) + shared_edge1[::-1] # Common nodes of the previous quarter
						shared_edge1 = [ray_ind[-1]]

					elif s==3:
						nb_vertices = int(num_a + num_b + 2 + N/8)
						ray_ind = list(range(count, count + nb_vertices)) + [first_ray[-j - 1]] # Common nodes to close the circle
					else:
						nb_vertices = int(num_a + num_b + 3 + N/8)
						ray_ind = list(range(count, count + nb_vertices))
						shared_edge1.append(ray_ind[-1]) # Store the indices of the shared nodes
					
					if j == int(N/8):
						shared_edge2 = ray_ind[-int(N/8):]
						
				else: 
				
					nb_vertices = num_a + num_b + 2
					ray_ind = list(range(count, count + nb_vertices)) + [shared_edge2[j-int(N/8)-1]]

				if s==0 and j == 0:
					first_ray = ray_ind
					ray_prec = ray_ind
				else:
					for k in range(min([len(ray_prec)-1, len(ray_ind)- 1])):
						faces[nb_faces, :] = np.array([4, ray_prec[k], ray_ind[k], ray_ind[k+1], ray_prec[k+1]])
						nb_faces += 1
					
				count += nb_vertices
				ray_prec = ray_ind
				j = j + 1

		for k in range(min([len(ray_ind)-1, len(first_ray) - 1])):
			faces[nb_faces, :] = np.array([4, ray_ind[k], first_ray[k], first_ray[k+1], ray_ind[k+1]])
			nb_faces += 1

		return faces



	#####################################
	###########  CONVERSIONS  ###########
	#####################################

	def topo_to_full(self, replace = True):

		""" Converts topo_graph to a full graph.""" 
			
		G = nx.DiGraph()
		k = 1

		ndict = {}
		for n in self._topo_graph.nodes():
			G.add_node(k, coords = self._topo_graph.nodes[n]['coords']) #, topo_id = [n])
			self._topo_graph.nodes[n]["full_id"] = k
			ndict[n] = k
			k  = k + 1

		for e in self._topo_graph.edges():
			pts = self._topo_graph.edges[e]['coords']

			if len(pts) == 0:

				G.add_edge(ndict[e[0]], ndict[e[1]], coords = np.array([]).reshape(0,4))
				self._topo_graph.edges[e]["full_id"] = []

			else: 

				G.add_node(k, coords = pts[0])
				G.add_edge(ndict[e[0]], k, coords = np.array([]).reshape(0,4))
				self._topo_graph.edges[e]["full_id"][0] = k

				k = k + 1

				for i in range(1, len(pts)):

					G.add_node(k, coords = pts[i]) #,topo_id = [e, i])
					self._topo_graph.edges[e]["full_id"][i] = k

					G.add_edge(k - 1, k, coords = np.array([]).reshape(0,4))
					k = k + 1

				G.add_edge(k - 1, ndict[e[1]], coords = np.array([]).reshape(0,4))

		if replace:
			self._full_graph = G
		else:
			return G
	

	'''

	def topo_to_full(self, replace=True):

		""" Converts topo_graph to a full graph.""" 
			
		G = nx.DiGraph()
		k = 1

		ndict = {}
		for n in self._topo_graph.nodes():
			G.add_node(self._topo_graph.nodes[n]["full_id"], coords = self._topo_graph.nodes[n]['coords'])

		for e in self._topo_graph.edges():
			pts = self._topo_graph.edges[e]['coords']

			if len(pts) == 0:

				G.add_edge(self._topo_graph.nodes[e[0]]["full_id"], self._topo_graph.nodes[e[1]]["full_id"], coords = np.array([]).reshape(0,4))

			else: 
				G.add_node(self._topo_graph.edges[e]["full_id"][0], coords = pts[0])
				G.add_edge(self._topo_graph.nodes[e[0]]["full_id"], self._topo_graph.edges[e]["full_id"][0], coords = np.array([]).reshape(0,4))
				

				for i in range(1, len(pts)):

					G.add_node(self._topo_graph.edges[e]["full_id"][i], coords = pts[i])
					G.add_edge(self._topo_graph.edges[e]["full_id"][i-1], self._topo_graph.edges[e]["full_id"][i], coords = np.array([]).reshape(0,4))
		
				G.add_edge(self._topo_graph.edges[e]["full_id"][-1],self._topo_graph.nodes[e[1]]["full_id"], coords = np.array([]).reshape(0,4))

		if replace:
			self._full_graph = G
		else:
			return G
	'''




	def model_to_full(self, replace = True):

		""" Converts spline_graph to a full graph and set the full and topo graphs accordingly"""

		G = nx.DiGraph()
		k = 1

		ndict = {}
		for n in self._model_graph.nodes():
			G.add_node(k, coords = self._model_graph.nodes[n]['coords'])
			ndict[n] = k
			k = k + 1

		for e in self._model_graph.edges():
			spl = self._model_graph.edges[e]['spline']

			if spl != None:

				pts = spl.get_points()

				G.add_node(k, coords = pts[0])
				G.add_edge(ndict[e[0]], k, coords = np.array([]).reshape(0,4))
				k = k + 1

				for i in range(1, len(pts)):

					G.add_node(k, coords = pts[i])
					G.add_edge(k - 1, k, coords = np.array([]).reshape(0,4))
					k = k + 1

				G.add_edge(k - 1, ndict[e[1]], coords = np.array([]).reshape(0,4))

			else:

				G.add_edge(ndict[e[0]], ndict[e[1]], coords = np.array([]).reshape(0,4))
		if replace:
			self._full_graph = G
			self.__set_topo_graph()
		else:
			return G



	#####################################
	######### POST PROCESSING  ##########
	#####################################

	def close_surface(self, edg = [], layer_ratio = [0.2, 0.3, 0.5], num_a = 10, num_b=10):

		""" Close the open ends of the surface mesh with a ogrid pattern """
		if len(edg) == 0:
			nds = list(self._crsec_graph.nodes())
		else:
			nds = []
			for e in edg:
				if e[0] not in nds:
					nds.append(e[0])
				if e[1] not in nds:
					nds.append(e[1])

		for n in nds:
			if self._crsec_graph.nodes[n]['type'] == "end":
				# Compute O-grid
				nb_nds_ogrid = int(self._N * (num_a + num_b + 3) + ((self._N - 4)/4)**2)
				f_ogrid = self.ogrid_pattern_faces(self._N, num_a, num_b)

				crsec = self._crsec_graph.nodes[n]['crsec']
				center = self._crsec_graph.nodes[n]['coords'][:-1]
				v_ogrid = self.ogrid_pattern_vertices(center, crsec, layer_ratio, num_a, num_b)

				id_first = self._surface_mesh[0].shape[0]
				f_ogrid[:,1:] = f_ogrid[:,1:] + id_first

				self._surface_mesh[0] = np.vstack((self._surface_mesh[0], v_ogrid)) 
				self._surface_mesh[1] = np.vstack((self._surface_mesh[1], f_ogrid))



	def open_surface(self, edg = []):

		""" Reopens the surface """
		if len(edg) == 0:
			edg = list(self._crsec_graph.edges())

		self.mesh_surface(edg=edg)


	def add_extensions(self, edg = [], size=6):

		""" Extend the inlet and outlet nodes 

		Keywords arguments: 
		size -- relative size of the extensions (proportionally to the radius)"""

		if self._model_graph is None:
			warnings.warn("No model found.")
		else:
			
			edg_list = list(self._model_graph.edges())[:]

			for e in edg_list: 
				if self._model_graph.nodes[e[1]]['type'] == "end":
					spl = self._model_graph.edges[e]['spline']
					tg = spl.tangent(1.0)
					r = spl.radius(1.0)

					P = np.zeros((4,4))
					P[:, 3] = [r,r,r,r]
					P[0,:3] = spl.point(1.0)
					P[1,:3] = P[0,:3] + tg*size*r/3
					P[2,:3] = P[0,:3] + tg*size*r*2/3
					P[3,:3] = P[0,:3] + tg*size*r

					ext_spl = Spline(P)

					# Add new edge
					nmax = max(list(self._model_graph.nodes())) + 1
					self._model_graph.nodes[e[1]]['type'] = "reg"
					self._model_graph.add_node(nmax, type = "end", coords = P[-1,:], bifurcation = None, combine = False, tangent = [], ref = [])
					self._model_graph.add_edge(e[1], nmax, spline = ext_spl, connect = 0, alpha = None)
					self.__compute_rotations(e[1])
				

					if self._crsec_graph is not None:
						# Add the cross section of the extensions
						self._crsec_graph.nodes[e[1]]['type'] = "reg"

						self._crsec_graph.add_node(nmax, type = "end", crsec = None, coords = P[-1,:])
						self._crsec_graph.add_edge(e[1], nmax, crsec = None, connect = None, center = None)

						self.recompute_cross_sections(e[1])


				if self._model_graph.nodes[e[0]]['type'] == "end":

					spl = self._model_graph.edges[e]['spline']
					tg = spl.tangent(0.0)
					r = spl.radius(0.0)

					P = np.zeros((4,4))
					P[:, 3] = [r,r,r,r]
					P[3,:3] = spl.point(0.0)
					P[2,:3] = P[3,:3] - tg*size*r/3
					P[1,:3] = P[3,:3] - tg*size*r*2/3
					P[0,:3] = P[3,:3] - tg*size*r

					ext_spl = Spline(P)

					# Add new edge
					nmax = max(list(self._model_graph.nodes())) + 1
					self._model_graph.nodes[e[0]]['type'] = "reg"
					self._model_graph.add_node(nmax, type = "end", coords = P[0,:], bifurcation = None, combine = False, tangent = [], ref = [])
					self._model_graph.add_edge(nmax, e[0], spline = ext_spl, connect = 0, alpha = None)
					self.__compute_rotations(nmax)

					if self._crsec_graph is not None:
						# Add the cross section of the extensions
						self._crsec_graph.nodes[e[0]]['type'] = "reg"
						self._crsec_graph.add_node(nmax, type = "end", crsec = None, coords = P[0,:])
						self._crsec_graph.add_edge(nmax, e[0], crsec = None, connect = None, center = None)
						self.recompute_cross_sections(nmax)
						#self.vessel_cross_sections((nmax, e[0]))

			if self._surface_mesh is not None:
				# Replace the mesh by the mesh with extensions
				if len(edg) == 0:
					edg = list(self._crsec_graph.edges())
				
				self.mesh_surface(edg = edg)

			
					 
	def remove_extensions(self, edg = [], prop=3):

		""" Extend the inlet and outlet nodes 

		Keywords arguments: 
		prop -- relative size of the extensions (proportionally to the radius)"""

		if self._model_graph is None:
			warnings.warn("No model found.")
		else:
			edg_list_inlet = []
			edg_list_outlet = []

			for e in self._model_graph.edges(): 
				if self._model_graph.nodes[e[1]]['type'] == "end":
					edg_list_outlet.append(e)

				if self._model_graph.nodes[e[0]]['type'] == "end":
					edg_list_inlet.append(e)

			for e in edg_list_outlet: 
					
				# Remove extension edges
					
				self._model_graph.remove_edge(e[0], e[1])
				self._model_graph.remove_node(e[1])
				self._model_graph.nodes[e[0]]['type'] = "end"

				if self._crsec_graph is not None:
						
					self._crsec_graph.remove_edge(e[0], e[1])
					if e in edg:
						edg.remove(e)
					self._crsec_graph.remove_node(e[1])
					self._crsec_graph.nodes[e[0]]['type'] = "end"

			for e in edg_list_inlet: 

				self._model_graph.remove_edge(e[0], e[1])
				self._model_graph.remove_node(e[0])
				self._model_graph.nodes[e[1]]['type'] = "end"

				if self._crsec_graph is not None:
						
					self._crsec_graph.remove_edge(e[0], e[1])
					if e in edg:
						edg.remove(e)
					self._crsec_graph.remove_node(e[0])
					self._crsec_graph.nodes[e[1]]['type'] = "end"

			if self._surface_mesh is not None:
				# Replace the mesh by the mesh without extensions
				if len(edg) == 0:
					edg = list(self._crsec_graph.edges())[:]
				
				self.mesh_surface(edg = edg)

	def load_pathology_template(self, path):

		""" Load the pathology file 
		Keyword arguments :
		path -- path to the template folder

		Output :
		template -- list of coordinates of the template curves
		temp_rad -- the radius of the template
		temp_center -- the center of the template
		"""
		try:
			info_file = None 
			crsec_files = []
			for root, dirs, files in os.walk(path):
				for file in files:
					if file == "info.txt":
						info_file = file
					elif file[:5] == "crsec":
						crsec_files.append(file)

			crsec_files.sort()
			if info_file is None:
				raise ValueError("Info file not found.")
				return [False, False, False]
				
			elif len(crsec_files) == 0:
				raise ValueError("Crsec files not found.")
				return [False, False, False]
				
			else:
				data = []
				
				print("Pathology template loaded from " + path + ".")

				for i in range(len(crsec_files)):
					data.append(np.loadtxt(path + crsec_files[i], skiprows=0))

				infos = np.loadtxt(path + info_file, skiprows=1)
				return [data, int(infos[-1]), infos[:-1]]

		except FileNotFoundError:
			raise ValueError("No template found at" + path + ".")
			return [False, False, False]
		

		
	def deform_surface_to_template(self, edg, t0, t1, template, temp_rad, temp_center, method = "bicubic", rotate = 0):

		""" Deforms the original mesh to match a given crsec section template
		Overwrite the cross section graph.

		Keywords arguments: 
		edg -- edge on which to add a stenosis
		pt -- 3D coordinate of the starting point of the pathology
		length -- length of the pathology
		template -- list of coordinates of the template curves
		temp_rad -- the radius of the template
		temp_center -- the center of the template
		method -- interpolation method "linear" or "bicubic"
		"""
		template = template[:]
		nb_slice = len(template)
		nb_pt = len(template[0])

		# If rotate, rotate the template
		if rotate!= 0:
			# Convert to radian
			if rotate < 0:
				rotate = 360 - rotate
			rotate = rotate * (pi / 180)
			ang_list = np.linspace(0, 2*pi, nb_pt - 1)
			idx = np.argwhere(ang_list >= rotate)[0][0]
		
			for i in range(nb_slice):
				template[i] = np.vstack((template[i][idx:-1, :], template[i][:idx, :], template[i][idx, :]))

		# Project point to model graph
		spl_edg = self._model_graph.edges[edg]['spline']

		Lstart = spl_edg.time_to_length(t0)
		length = spl_edg.time_to_length(t1) - Lstart

		# Find the crsec to deform and their distance to the starting point
		t_centers = self._crsec_graph.edges[edg]['center']
		l_centers = spl_edg.time_to_length(t_centers)

		id_crsec0 = np.argwhere(np.array(t_centers) >= t0)[0][0]
		id_crsec1 = np.argwhere(l_centers > Lstart + length)[0][0] - 1

		slice_z = l_centers[id_crsec0:id_crsec1+1] - Lstart

		depth = np.linspace(0, length, nb_slice)

		# Interpolate template 
		depth_interp = []
		depth_spl =  []
		# Interpolate along the vessel (linear interpolation)
		for i in range(nb_pt):
			# Extract curve
			poly = np.zeros((nb_slice, 3))
			for j in range(nb_slice):
				poly[j, :] = template[j][i, :]
				poly[j, -1] = depth[j]

			depth_interp.append(poly)
			if method != "linear ":
				spl = Spline()
				spl.interpolation(poly)
				depth_spl.append(spl)


		crsec_interp = []
		# Interpolate the cross sections (cubic interpolation)
		for i in range(len(slice_z)):
			slice_data = np.zeros((nb_pt, 3))
			# Extract the data from the depth interpolation spline
			id1 = np.argwhere(depth >= slice_z[i])[0][0]
	
			if depth[id1] == slice_z[i]:
				for j in range(nb_pt):
					
					slice_data[j, :] = depth_interp[j][id1, :]
			else:

				if method == "linear":
					for j in range(nb_pt):

						id0 = id1 -1
						# Find the intersection of the line nd the plane z = slice_z[i]
						pt0 = depth_interp[j][id0, :]
						pt1 = depth_interp[j][id1, :]
						vec = pt1 - pt0
						vec = vec / norm(vec)

						a = (slice_z[i]-pt0[2]) / vec[2]

						pt = pt0 + a*vec
						slice_data[j, :] = pt
				else:
					for j in range(nb_pt):
						# Find the intersection of the spline nd the plane z = slice_z[i]
						pts = depth_spl[j].get_points()
						idx = np.argwhere(pts[:, 2] >= slice_z[i])[0][0]
						slice_data[j, :] = pts[idx]
						slice_data[j, 2] = slice_z[i]


			# Interpolate
			spl = Spline()
			spl.interpolation(slice_data)
			crsec_interp.append(spl)

			# Extract the nodes along the new outline (even sampling)
			t_nodes = spl.resample_time(self._N, include_start = True)
			nds = spl.point(t_nodes)

			# Get crsec geometry
			rad = spl_edg.radius(t_centers[id_crsec0 + i])
			tg = spl_edg.tangent(t_centers[id_crsec0 + i])
			center = spl_edg.point(t_centers[id_crsec0 + i])
			loc_center = np.array([temp_center[0], temp_center[1], slice_z[i]])

			crsec = self._crsec_graph.edges[edg]['crsec'][id_crsec0 + i]

			# Rescale the nodes to match the crsec radius
			scaling_factor = temp_rad / rad
			nds = nds / scaling_factor
			loc_center = loc_center / scaling_factor

			# Translate the nodes to match the crsec center
			translation = center - loc_center
			for k in range(len(nds)):
				nds[k, :] = nds[k, :] + translation
			
			# Rotate the nodes to match the crsec tangent 
			temp_tg = np.array([0,0,1])
			rotation_axis = cross(temp_tg, tg)
			ang = angle(temp_tg, tg, axis = rotation_axis, signed = True)

			for k in range(len(nds)):
				ref_nds = nds[k, :] - center
				rot = rotate_vector(ref_nds, rotation_axis, ang)
				nds[k, :] = rot + center

			# Rotate the nodes to match the crsec reference
			ref = crsec[0] - center 
			ref_temp = nds[0, :]  - center
			rotation_axis = tg
			ang = angle(ref_temp, ref, axis = tg, signed = True)

			for k in range(len(nds)):
				ref_nds = nds[k, :] - center
				rot = rotate_vector(ref_nds, tg, ang)
				nds[k, :] = rot + center

			# Show nds and crsec nodes in 3D
			"""
			fig = plt.figure(figsize=(10,7))
			ax = Axes3D(fig)
			ax.set_facecolor('white')
			ax.scatter(nds[:, 0],nds[:, 1], nds[:, 2], c="red", depthshade=False)
			ax.scatter(crsec[:, 0],crsec[:, 1], crsec[:, 2], c="black", depthshade=False)
			plt.show()
			"""
			
			# Replace the cross section
			self._crsec_graph.edges[edg]['crsec'][id_crsec0 + i] = nds
			#spl.show(control_points = False, data = slice_data)

		# For visualization of the result in 3D
		"""
		# Write interpolation mesh
		samp_slices = 200 
		samp_pt = 200

		slice_z_new = np.linspace(0, length, samp_slices)
		vertices = np.array([]).reshape(0,3)
		for i in range(samp_pt):
			slice_data = np.zeros((nb_pt, 3))
			for j in range(nb_pt):
				# Find the intersection of the spline nd the plane z = slice_z[i]
				pts = depth_spl[j].get_points()
				idx = np.argwhere(pts[:, 2] >= slice_z_new[i])[0][0]
				slice_data[j, :] = pts[idx]
				slice_data[j, 2] = slice_z_new[i]

			# Interpolate
			spl = Spline()
			spl.interpolation(slice_data)
			crsec_interp.append(spl)

			# Extract the nodes along the new outline (even sampling)
			t_nodes = spl.resample_time(samp_pt, include_start = True)
			nds = spl.point(t_nodes)
			vertices = np.vstack((vertices, nds))

		edges = []
		for i in range(len(slice_z_new) - 1):
			for j in range(samp_slices-1):
				edges.append([4, i * samp_pt + j,  i * samp_pt + j + 1, (i+1) * samp_pt + j + 1, (i+1) * samp_pt + j])
			edges.append([4, i * samp_pt + j,  i * samp_pt + 0, (i+1) * samp_pt + 0, (i+1) * samp_pt + j])

		mesh = pv.PolyData(vertices, np.array(edges))
		mesh.save("test_stenosis_applied.vtk")

		# Write the cross sections with lines and spheres

		circles = pv.MultiBlock()
		centers = pv.MultiBlock()

		for i in range(nb_slice):
			print(i)
			circle = pv.PolyData()
			v = template[i]
			f = []
			for j in range(nb_pt):
				v[j, 2] = depth[i]
				if j < nb_pt - 1:
					f.append([2, j, j+1])
				centers[str(i) +","+ str(j)]= pv.Sphere(radius=0.2, center=(template[i][j, 0], template[i][j, 1], depth[i]))

			circle.points = v
			circle.lines = np.array(f)
			circles[str(i)] = circle

		circles.save("crsecs.vtm")
		centers.save("points.vtm")
		"""

		if self._surface_mesh is not None:
			self.mesh_surface()


	def deform_surface_to_mesh(self, mesh, edges=[], search_dist = 40):

		""" Deforms the original mesh to match a given surface mesh. 
		Overwrite the cross section graph.

		Keywords arguments: 
		mesh -- a surface mesh in vtk format
		"""

		if len(edges) == 0:
			edges = [e for e in  self._crsec_graph.edges()]

		for e in edges:
 
			# Get the start cross section only if it is a terminating edge
			if self._crsec_graph.in_degree(e[0]) == 0:

				crsec = self._crsec_graph.nodes[e[0]]['crsec']
				center = (crsec[0] + crsec[int(crsec.shape[0]/2)])/2.0 # Compute the center of the section
			
				new_crsec = np.zeros([crsec.shape[0], 3])
				for i in range(crsec.shape[0]):
					new_crsec[i] = self.__intersection(mesh, center, crsec[i], search_dist)
				self._crsec_graph.nodes[e[0]]['crsec'] = new_crsec

			# Get the connection cross sections
			crsec_array = self._crsec_graph.edges[e]['crsec']  # In edges, the sections are stored as an array of cross section arrays
			new_crsec = np.zeros([crsec_array.shape[0], crsec_array.shape[1], 3])

			for i in range(crsec_array.shape[0]):
					crsec = crsec_array[i] # Get indivisual cross section
					center = (crsec[0] + crsec[int(crsec.shape[0]/2)])/2.0
			
					for j in range(crsec.shape[0]):
						new_crsec[i, j, :] = self.__intersection(mesh, center, crsec[j], search_dist)

			self._crsec_graph.edges[e]['crsec'] = new_crsec

			# Get the end cross section
			crsec = self._crsec_graph.nodes[e[1]]['crsec']
			if self._crsec_graph.nodes[e[1]]['type'] == "bif": # If bifurcation
				center = self._crsec_graph.nodes[e[1]]['coords']
			else:
				center = (crsec[0] + crsec[int(crsec.shape[0]/2)])/2.0

			new_crsec = np.zeros([crsec.shape[0], 3])

			for i in range(crsec.shape[0]):
				new_crsec[i, :] = self.__intersection(mesh, center, crsec[i], search_dist)

			self._crsec_graph.nodes[e[1]]['crsec'] = new_crsec
			
			if self._surface_mesh is not None:
				self.mesh_surface()
				



	def __intersection(self, mesh, center, coord, search_dist = 40):

		""" Returns the first intersection point between the mesh and a ray passing through the points center and coord.

		Keyword arguments: 
		mesh -- surface mesh
		center -- coordinate of the starting point of intersection ray
		coord -- coordinate of a point on the intersection ray
		"""
		dist = 1
		inter = coord
		points = []
		while len(points) == 0 and dist < search_dist:

			normal = coord - center
			normal = normal / norm(normal) # Normal=direction of the projection 
			p2 = center + normal * dist
			points, ind = mesh.ray_trace(center, p2)

			if len(points) > 0 :
				inter = points[0]
			else :
				dist += 1

		return inter


	def extract_vessel(self, e, volume = False):

		""" Create a submesh of vessel of edge e """
	
		if volume:
			self.__add_node_id_volume(self._num_a, self._num_b)
			nb_nds_ogrid = int(self._N * (self._num_a + self._num_b + 3) + ((self._N - 4)/4)**2)
		else:
			self.__add_node_id_surface()
		

		start_id = self._crsec_graph.nodes[e[0]]['id']
		if volume:
			end_id = start_id + nb_nds_ogrid
		else:
			end_id = start_id + self._crsec_graph.nodes[e[0]]['crsec'].shape[0]

		node_id_list = np.arange(start_id, end_id)
		

		start_id = self._crsec_graph.edges[e]['id']
		if volume:
			end_id = start_id + self._crsec_graph.edges[e]['crsec'].shape[0] * nb_nds_ogrid
		else:
			end_id = start_id + self._crsec_graph.edges[e]['crsec'].shape[0]*self._crsec_graph.edges[e]['crsec'].shape[1]
		
		node_id_list = np.hstack((node_id_list, np.arange(start_id, end_id)))

		start_id = self._crsec_graph.nodes[e[1]]['id']
		if volume:
			end_id = start_id + nb_nds_ogrid
		else:
			end_id = start_id + self._crsec_graph.nodes[e[1]]['crsec'].shape[0]

		node_id_list = np.hstack((node_id_list, np.arange(start_id, end_id)))
		


		if volume :

			v = self._volume_mesh[2]
			selected_point = np.zeros((len(v), 1))
			for i in node_id_list:
				selected_point[i, :] = 1.0

			#mesh = self.get_volume_mesh()
			#mesh["SelectedPoints"] = selected_point
		
			#inside = mesh.threshold(value = 0.5, scalars="SelectedPoints", preference = "point")

		else:
			v = self._surface_mesh[0]
			selected_point = np.zeros((len(v), 1))
			for i in node_id_list:
				selected_point[i, :] = 1.0
			"""
			mesh = self.get_surface_mesh()
			mesh["SelectedPoints"] = selected_point
		
			inside = mesh.threshold(value = 0.5, scalars="SelectedPoints", preference = "point")
			inside.plot(show_edges = True)
			"""
		
		return selected_point


		


	def extract_furcation(self, n, volume = False):
		""" Create a submesh of furcation of node n"""
		"""
		if volume:
			self.__add_node_id_volume(self._num_a, self._num_b)
			nb_nds_ogrid = int(self._N * (self._num_a + self._num_b + 3) + ((self._N - 4)/4)**2)
		else:
			self.__add_node_id_surface()	
		"""


		start_id = self._crsec_graph.nodes[n]['id']
		if volume:
			nbif = self._crsec_graph.in_degree(n)
			nb_nds_bif_ogrid = int(nb_nds_ogrid + (nbif-2) * ((self._N/2 - 1) * (self._num_a + self._num_b + 3) + (self._N - 4)/4 * (((self._N - 4)/4 -1)/2)))
			end_id = start_id + nb_nds_bif_ogrid
		else:
			end_id = start_id + self._crsec_graph.nodes[n]['crsec'].shape[0]
		node_id_list = np.arange(start_id, end_id)

		for e in self._crsec_graph.in_edges(n):


			start_id = self._crsec_graph.nodes[e[0]]['id']
			if volume:
				end_id = start_id + nb_nds_ogrid
			else:
				end_id = start_id + self._crsec_graph.nodes[e[0]]['crsec'].shape[0]

			node_id_list = np.hstack((node_id_list, np.arange(start_id, end_id)))
			

			start_id = self._crsec_graph.edges[e]['id']
			if volume:
				end_id = start_id + self._crsec_graph.edges[e]['crsec'].shape[0] * nb_nds_ogrid
			else:
				end_id = start_id + self._crsec_graph.edges[e]['crsec'].shape[0]*self._crsec_graph.edges[e]['crsec'].shape[1]
			
			node_id_list = np.hstack((node_id_list, np.arange(start_id, end_id)))


		if volume :

			v = self._volume_mesh[2]
			selected_point = np.zeros((len(v), 1))
			for i in node_id_list:
				selected_point[i, :] = 1.0

			#mesh = self.get_volume_mesh()
			#mesh["SelectedPoints"] = selected_point
		
			#inside = mesh.threshold(value = 0.5, scalars="SelectedPoints", preference = "point")


		else:
			v = self._surface_mesh[0]
			selected_point = np.zeros((len(v), 1))
			for i in node_id_list:
				selected_point[i, :] = 1.0
			
			#mesh = self.get_surface_mesh()
			#mesh["SelectedPoints"] = selected_point
		
			#inside = mesh.threshold(value = 0.5, scalars="SelectedPoints", preference = "point")
			#inside.plot(show_edges = True)
			
		return selected_point


	def subgraph(self, nodes):

		""" Cuts the original graph to a subgraph. 
		Remove any spline approximation of cross section computation performed on the previous graph.

		Keywords arguments: 
		nodes -- list of nodes to keep in the subgraph
		"""
		if self._topo_graph is None:
			raise ValueError('Cannot make subgraph because no network was found.')
		else:
		
			self.set_topo_graph(self._topo_graph.subgraph(nodes).copy())
			self.__set_topo_graph()

	def make_inlet(self, n):

		""" Change an outlet point to inlet by adding a sink if necessary

		Keyword arguments:
		n -- point of the topo graph to change to inlet (must be outlet type point of topo graph)
		"""

		# Check if the point n is an outlet
		if self._topo_graph.out_degree(n) == 0 and self._topo_graph.in_degree(n) == 1:

			# Inverse the orientation of the out edge
			oute = list(self._topo_graph.in_edges(n))[0]

			coords = self._topo_graph.edges[oute]['coords'][::-1, :]
			full_id = self._topo_graph.edges[oute]['full_id'][::-1]

			self._topo_graph.remove_edge(oute[0], oute[1])
			self._topo_graph.add_edge(oute[1], oute[0], coords = coords, full_id = full_id)

			if self._topo_graph.nodes[oute[0]]['type'] != "end" and self._topo_graph.in_degree(oute[0]) == 2 and self._topo_graph.out_degree(oute[0]) == 1:

				# Add sink to the next in edge and inverse half
				outi = list(self._topo_graph.in_edges(oute[0]))
				outi.remove((oute[1], oute[0]))
				outi = outi[0]

				coords = self._topo_graph.edges[outi]['coords'][::-1, :]
				full_id = self._topo_graph.edges[outi]['full_id'][::-1]

				if coords.shape[0] <= 1 :
					if coords.shape[0] == 0:
						sink_coords = (self._topo_graph.nodes[outi[0]]['coords'] + self._topo_graph.nodes[outi[1]]['coords']) / 2
						sink_id = max(list(self._full_graph.nodes())) + 1# Max id of full graph
					else:
						sink_coords = coords
						sink_id = full_id

					coords1 = np.array([]).reshape(0,4)
					full_id1 = []

					coords2 = np.array([]).reshape(0,4)
					full_id2 = []

				else:

					ind = coords.shape[0] // 2
					sink_coords = coords[ind]
					sink_id = full_id[ind]

					coords1 = coords[:ind]
					full_id1 = full_id[:ind]

					if coords.shape[0] == 2:
						coords2 = np.array([]).reshape(0,4)
						full_id2 = []
					else:
						coords2 = coords[ind+1:][::-1]
						full_id2 = full_id[ind+1:][::-1] # Inverse the nodes 
					

				name = max(list(self._topo_graph.nodes())) + 1 # Find name for new node in topo graph
				self._topo_graph.add_node(name, coords = sink_coords, full_id = sink_id, type = "sink") # Add sink node
				# Remove previous edge
				self._topo_graph.remove_edge(outi[0], outi[1])
				# Add both half edges
				self._topo_graph.add_edge(outi[0], name, coords = coords2, full_id = full_id2)
				self._topo_graph.add_edge(outi[1], name, coords = coords1, full_id = full_id1)

			# Propagate changes to full graph
			self.topo_to_full(replace = True)
			# The model and mesh must be recomputed completely after making a new inlet
			self.reset_model_mesh()


		else:
			raise ValueError("The node requested is not an outlet. It cannot be changed to inlet.")



	def crop_branch_degree(self, max_deg):

		""" Removes the extremity branches from the data and topo graphs
		Keyword argments :
		max_deg -- degree threshold (maximum) """
		
		# Find branching degree
		remove_nodes = []

		for n in self._topo_graph.nodes():
			if self._topo_graph.in_degree(n) == 0:
				origin = n
		prec = list(self._topo_graph.out_edges(origin))

		propagate = True
		deg = 0
		while propagate:
			# Label the edges
			suc = []
			for e in prec:
				if deg == max_deg:
					
					remove_nodes.append(e)
					#self._topo_graph.nodes[e[0]]["type"] = "end"
				
				# Find the next edges
				suc += list(self._topo_graph.out_edges(e[1]))
	
			deg+=1
			prec = suc

			if len(prec) == 0:
				propagate = False 
				if deg == max_deg:
					
					remove_nodes.append(e)
					#self._topo_graph.nodes[e[0]]["type"] = "end"
				

		for e in remove_nodes:
			self.remove_branch(e, preserve_shape = True, from_node = True)
			#self._topo_graph.remove_node(n)

		#self.topo_to_full(replace = True)


	def merge_branch(self, n, mode = "topo"):

		""" Merge the branch with the previous one to from a (n+1)-furcation

		Keyword arguments:
		e -- edge of the branch to merge """

		# Get previous node
		if mode == "topo":
			pred = list(self._topo_graph.predecessors(n))[0]
			if self._topo_graph.nodes[pred]["type"] == "end":
				print("Branch cannot be merged.")

			else:

				for e in self._topo_graph.out_edges(n):
					d = np.vstack((self._topo_graph.edges[(pred, e[0])]['coords'], self._topo_graph.nodes[e[0]]['coords'], self._topo_graph.edges[(e)]['coords']))
					full_id = self._topo_graph.edges[(pred, e[0])]['full_id'] + [self._topo_graph.nodes[e[0]]['full_id']] + self._topo_graph.edges[(e)]['full_id']
					self._topo_graph.add_edge(pred, e[1], coords = d, full_id = full_id)


				self._topo_graph.remove_node(n)
				self.topo_to_full()
		else:

			pred = list(self._model_graph.predecessors(n))[0]
			if self._model_graph.nodes[pred]["type"] == "end":
				print("Branch cannot be merged.")

			else:

				for e in self._model_graph.out_edges(n):
					d = np.vstack((self._model_graph.edges[(pred, e[0])]['coords'], self._model_graph.nodes[e[0]]['coords'], self._model_graph.edges[(e)]['coords']))
					self._model_graph.add_edge(pred, e[1], coords = d)


				self._model_graph.remove_node(n)


	def remove_branch(self, e, preserve_shape = True, from_node = False):

		""" Cuts the branch at edge e and all the downstream branches. 
		If a cross section graph is already computed, recomputes the cross sections at the cut part.

		Keyword arguments: 
		e -- edge of the branch in topo graph as a tuple
		"""

		if self._topo_graph is None:
			raise ValueError('No network was found.')

		inlet = False 
		if self._topo_graph.nodes[e[0]]["type"] == "end":
			inlet = True 

		if from_node: 

			# Get all downstream nodes
			downstream_nodes =  list(nx.dfs_preorder_nodes(self._topo_graph, source=e[0]))

			# Remove them from topo graph
			for node in downstream_nodes[1:]:
				#self._full_graph.remove_node(self._topo_graph.nodes[node]["full_id"])
				self._topo_graph.remove_node(node)
				self._topo_graph.nodes[e[0]]['type'] = "end"

						# Inlet case
			if inlet:
				self._topo_graph.remove_node(e[0])

		else:

			if e not in [e for e in self._topo_graph.edges()]:
				raise ValueError('Not a valid edge.')

			is_sink = False
			if self._topo_graph.nodes[e[1]]['type'] == "sink":
				is_sink = True
				bif_n = [e[0] for e in self._topo_graph.in_edges(e[1])]

			self._topo_graph.remove_edge(e[0], e[1])

			# Inlet case
			if inlet:
				self._topo_graph.remove_node(e[0])

			# Get all downstream nodes
			downstream_nodes =  list(nx.dfs_preorder_nodes(self._topo_graph, source=e[1]))	

			# Remove them from topo graph
			for node in downstream_nodes:
				#self._full_graph.remove_node(self._topo_graph.nodes[node]["full_id"])
				if self._topo_graph.nodes[node]['type']!="sink":
					self._topo_graph.remove_node(node)
				else:
					self._topo_graph.nodes[node]['type']="end"

			# Merge data and remove bifurcation point

			if self._topo_graph.out_degree(e[0]) == 1 and self._topo_graph.in_degree(e[0]) == 1:

				eprec = list(self._topo_graph.in_edges(e[0]))[0]
				esucc = list(self._topo_graph.out_edges(e[0]))[0]
	 
				# Create tables of regular nodes data
				coords = np.vstack((self._topo_graph.edges[eprec]['coords'], self._topo_graph.nodes[e[0]]['coords'], self._topo_graph.edges[esucc]['coords']))
				ids = self._topo_graph.edges[eprec]['full_id'] + [self._topo_graph.nodes[e[0]]['full_id']] + self._topo_graph.edges[esucc]['full_id']
				
				# Create new edge by merging the 2 edges of regular point
				self._topo_graph.add_edge(eprec[0], esucc[1], coords = coords, full_id = ids)
				# Remove regular point
				self._topo_graph.remove_node(e[0])

			elif self._topo_graph.out_degree(e[0]) == 2 and self._topo_graph.in_degree(e[0]) == 0: # Out sink
				self._topo_graph.remove_node(e[0])


			# Propagate to full_graph
			self.topo_to_full()
			# Check if valid
			if not self.check_full_graph():
				self.reset_model_mesh()

		
		if self._model_graph is not None:
			

			def remove_bifurcation_branch(nbiftopo, nbif, nsep, apply_crsec, preserve_shape):

				if self._model_graph.nodes[nbif]['bifurcation'].get_n() == 2: # Furcation is a bifurcation

					if preserve_shape:

						# Get correct bifurcation shape spline number
						id_out = list(self._model_graph.successors(nbif))
						if id_out[0] == nsep: # The node to remove
							id_spl = 1
						else:
							id_spl = 0

						spl = self._model_graph.nodes[nbif]['bifurcation'].get_spl()[id_spl]

						# Get in and out bifurcation nodes
						n_in = nbiftopo
						n_out = list(self._model_graph.successors(nbif))[id_spl]


						self._model_graph.remove_node(nbif)

						if apply_crsec:
								self._crsec_graph.remove_node(nbif)

						if preserve_shape: # Keep the bif spline to keep the same trajectory
								
							self._model_graph.nodes[n_in]['type'] = "reg"
							self._model_graph.nodes[n_out]['type'] = "reg"

							self._model_graph.nodes[n_in]['ref'] = None
							self._model_graph.nodes[n_out]['ref'] = None

							# Connect them with bifurcation spline
							self._model_graph.add_edge(n_in, n_out, coords = np.array([]).reshape(0,4), spline = spl, alpha = None, connect = 0)

						
							self.__compute_rotations(n_in)

							if apply_crsec:
								self._crsec_graph.add_edge(n_in, n_out, coords = np.array([]).reshape(0,4), crsec = None, connect = None, center = None)

								self._crsec_graph.nodes[n_in]['type'] = "reg"
								self._crsec_graph.nodes[n_out]['type'] = "reg"

								# Compute crsec
								self.recompute_cross_sections(n_in)

					else: # Remove all bif nodes and merge into a single edge

						# Get the id of edges before and after the bifurcation
						in_edge = [e for e in self._model_graph.in_edges(n_in)][0]
						out_edge = [e for e in self._model_graph.out_edges(n_out)][0]
						# If no in or out edge : very particular case of the sinks (NOT HANDLED YET)

						# Get data coordinates along the path and merge them 
						coords = self._model_graph.edges[in_edge]['coords']
						coords = np.vstack((coords, self._model_graph.edges[out_edge]['coords']))

						# Remove in and out nodes + add new edge
						self._model_graph.remove_node(n_in)
						if apply_crsec:
							self._crsec_graph.remove_node(n_in)

						self._model_graph.remove_node(n_out)
						if apply_crsec:
							self._crsec_graph.remove_node(n_out)

						# Create edge between n_first and n_last
						self._model_graph.add_edge(in_edge[0], out_edge[1], coords = coords, spline = None, alpha = None, connect = 0)
						self.__model_vessel((in_edge[0], out_edge[1])) # Recompute spline
						self.__compute_rotations(in_edge[0])
						

						if apply_crsec:
							self._crsec_graph.add_edge(in_edge[0], out_edge[1], crsec = None, coord = None, center = None)
							self.recompute_cross_sections(in_edge[0])
							

				else: # Furcation is a trifurcation or more # NOT TESTED YET

					# Create (n-1)-furcation
					all_ids = list(self._model_graph.successors(path[1]))
					all_ids.sort()
					cut_id = all_ids.index(path[2])
					
					# Remove downstream nodes
					downstream_nodes = list(nx.dfs_preorder_nodes(self._model_graph, source=path[2]))
					for node in downstream_nodes:
						self._model_graph.remove_node(node)
						if apply_crsec:
							self._crsec_graph.remove_node(node)
					

					old_furcation = self._model_graph.nodes[path[1]]['bifurcation']
					spl_list = old_furcation.get_spl()
					del spl_list[cut_id]
					new_furcation = Nfurcation("spline", [spl_list, old_furcation.get_R()])

					# Update center node
					self._model_graph.nodes[path[1]]['bifurcation'] = new_furcation
					self._model_graph.nodes[path[1]]['coords'] = new_furcation.get_X()

					# Update splines
					self._model_graph.edges[(path[0], path[1])]['spline'] = new_furcation.get_tspl()[0]
					k = 1
					for j in range(len(all_ids)):
						if j != cut_id:
							self._model_graph.edges[(path[1], all_ids[j])]['spline'] = new_furcation.get_tspl()[k]
							k+=1


					if apply_crsec:
						# Update cross sections 
						new_furcation.compute_cross_sections(self._N, self._d)
						self._model_graph.nodes[path[1]]['bifurcation'] = new_furcation
						self.furcation_cross_sections(path[1])



			apply_crsec = False
			if self._crsec_graph is not None:
				apply_crsec = True

			if from_node:

				# Get all downstream nodes
				downstream_nodes = list(nx.dfs_preorder_nodes(self._model_graph, source=e[0]))

				# Remove downstream nodes from model graph
				for node in downstream_nodes[1:]:
					self._model_graph.remove_node(node)
					self._model_graph.nodes[e[0]]['type'] = "end"
								# Inlet case
					if inlet:
						self._model_graph.remove_node(e[0])

					if apply_crsec:
						self._crsec_graph.remove_node(node)
						self._crsec_graph.nodes[e[0]]['type'] = "end"
						if inlet:
							self._crsec_graph.remove_node(e[0])
			else:

				list_edg = [e for e in self._model_graph.edges()]

				if is_sink: # In sink case

					# Get out sep nodes for each bif
					out_sep = []
					mod_bif = []

					for n in bif_n:
						out_e = [e[1] for e in self._model_graph.out_edges(n)][0]
						mod_bif.append(out_e)
						out_sep.append([e[1] for e in self._model_graph.out_edges(out_e)])

					# Find which is connected to which
					for i in range(len(out_sep[0])):
						for j in range(len(out_sep[1])):
							if (out_sep[0][i], out_sep[1][j]) in list_edg:
								sink_edg = (out_sep[0][i], out_sep[1][j])

							if (out_sep[1][j], out_sep[0][i]) in list_edg:
								sink_edg = (out_sep[1][j], out_sep[0][i])
								bif_n = [bif_n[0], bif_n[1]]
								mod_bif = [mod_n[0], mod_n[1]]

					# Remove the connection edge
					self._model_graph.remove_edge(sink_edg[0], sink_edg[1])

					# Remove the bifurcation branches
					remove_bifurcation_branch(bif_n[0], mod_bif[0], sink_edg[0], apply_crsec, preserve_shape)
					remove_bifurcation_branch(bif_n[1], mod_bif[1], sink_edg[1], apply_crsec, preserve_shape)

					# Remove the other sep node
					self._model_graph.remove_node(sink_edg[0])
					self._model_graph.remove_node(sink_edg[1])

				else:
			
					# Get path between e[0] and e[1]
					path = list(nx.all_simple_paths(self._model_graph, source=e[0], target=e[1]))[0]
							
					if self._model_graph.nodes[path[1]]['type']=="bif": # Furcation branch
						if self._model_graph.nodes[path[1]]['bifurcation'].get_n() == 2: # Furcation is a bifurcation

							# Get all downstream nodes
							downstream_nodes = list(nx.dfs_preorder_nodes(self._model_graph, source=path[2]))

							# Remove downstream nodes from model graph
							for node in downstream_nodes[1:]:
								self._model_graph.remove_node(node)

								if apply_crsec:
									self._crsec_graph.remove_node(node)

							# Remove bifurcation branch
							remove_bifurcation_branch(e[0], path[1], path[2], apply_crsec, preserve_shape)

							# Remove downstream nodes from model graph
							self._model_graph.remove_node(downstream_nodes[0])

							if apply_crsec:
								self._crsec_graph.remove_node(downstream_nodes[0])



	def translate(self, val):

		""" Translate the graph coordinates

		Keywords arguments: 
		val -- list of transposition values for every coordinates of the nodes
		"""

		if self._topo_graph is None:
			raise ValueError('Cannot transpose because no network was found.')

		else:

			G = self._topo_graph

			for n in G.nodes: 
				G.nodes[n]['coords'] = G.nodes[n]['coords'] + val

			for e in G.edges:
				for i in range(len(G.edges[e]['coords'])):
					G.edges[e]['coords'][i] = G.edges[e]['coords'][i] + val

		self.set_topo_graph(G)


	def set_minimim_radius(self, val):

		""" Sets the minimum radius value and modify the data graph consequently 
		val -- minimum radius value """

		if self._topo_graph is None:
			raise ValueError('Cannot scale because no network was found.')

		else:

			G = self._topo_graph

			for n in G.nodes: 
				if G.nodes[n]['coords'][3] < val:
					G.nodes[n]['coords'][3] = val

			for e in G.edges:
				for i in range(len(G.edges[e]['coords'])):
					if G.edges[e]['coords'][i][3] < val:
						G.edges[e]['coords'][i][3] = val

		self.set_topo_graph(G)


	def transform_radius(self): # TMP

		if self._topo_graph is None:
			raise ValueError('Cannot scale because no network was found.')

		else:
			# Get minimum and maximum values
			pos = nx.get_node_attributes(self._full_graph, 'coords')

			min_rad = np.inf
			max_rad = 0

			for key, val in pos.items():
				if val[3] < min_rad :
					min_rad = val[3]
				if val[3] > max_rad:
					max_rad = val[3]

			new_min = 0.5
			a = (max_rad - new_min) / (max_rad - min_rad)
			b = max_rad - max_rad*a

			G = self._topo_graph

			for n in G.nodes: 
				G.nodes[n]['coords'][3] = a*G.nodes[n]['coords'][3] + b

			for e in G.edges:
				for i in range(len(G.edges[e]['coords'])):
					G.edges[e]['coords'][i][3] = a * G.edges[e]['coords'][i][3] + b
						

		self.set_topo_graph(G)
			


	def scale(self, val):

		""" Rescales the centerline data points

		Keywords arguments: 
		val -- list of scaling values for every coordinates of the nodes
		"""

		if self._topo_graph is None:
			raise ValueError('Cannot scale because no network was found.')

		else:

			G = self._topo_graph

			for n in G.nodes: 
				G.nodes[n]['coords'] = G.nodes[n]['coords'] * val

			for e in G.edges:
				for i in range(len(G.edges[e]['coords'])):
					G.edges[e]['coords'][i] = G.edges[e]['coords'][i] * val


		self.set_topo_graph(G)


	def resample(self, p):

		""" Add resample the nodes of the initial centerlines.

		Keyword arguments:
		p -- percentage of points to keep
		"""
		nmax =  max(list(self._full_graph.nodes())) + 1

		for e in self._topo_graph.edges():

			pts = np.vstack((self._topo_graph.nodes[e[0]]['coords'], self._topo_graph.edges[e]['coords'], self._topo_graph.nodes[e[1]]['coords']))
			full_id = self._topo_graph.edges[e]['full_id']

			n = int(pts.shape[0]*p)
			if n <2:
				n = 2

			pts = resample(pts, num = n+2)[1:-1]

			if len(pts) <= len(full_id):
				full_id = full_id[:len(pts)]
				
			else:
				n_pt = len(pts) - len(full_id)
				for i in range(n_pt):
					full_id = full_id + [nmax]
					nmax+=1

			# Modify topo graph 
			self._topo_graph.edges[e]["coords"] = pts
			self._topo_graph.edges[e]["full_id"] = full_id

		# Change full graph
		self.topo_to_full()
		self.reset_model_mesh()


	def delete_data_point(self, n, apply = True):

		""" Delete a data point from the centerline and update topo graph. Can delete only regular points, to remove bifurcations, see remove_branch. 

		Keyword arguments :
		n -- node name (in full graph)
		apply -- whether to apply the modification directly to the topo graph or not"""

		if self._full_graph.in_degree(n) == 1 and self._full_graph.out_degree(n) == 1: # Regular node, can be removed

			prec = list(self._full_graph.predecessors(n))[0]
			succ = list(self._full_graph.successors(n))[0]
			self._full_graph.remove_node(n) # Remove node and edges
			self._full_graph.add_edge(prec, succ, coords = np.array([]).reshape(0,4)) # Add new edges



	def delete_data_edge(self, edg, apply = True):

		""" Delete an edge from the centerline. Update topo graph only if the full graph passes the validity check.

		Keyword arguments :
		edg -- edges name (in full graph)
		apply -- whether to apply the modification directly to the topo graph or not"""

		self._full_graph.remove_edge(edg[0], edg[1]) # Remove node and edges
		if apply:
			self.__set_topo_graph()



	def add_data_edge(self, n1, n2, apply = True):

		""" Add an edge to the centerline. The orientation is automatically defines by validity criterion or by the node order.
		Update topo graph only if the full graph passes the validity check. 

		Keyword arguments :
		n1, n2 -- node names (in full graph)
		apply -- whether to apply the modification directly to the topo graph or not"""

		edg = (n1, n2)
		if self._full_graph.out_degree(n2) == 0 and self._full_graph.in_degree(n1) == 0:
			edg = (n2, n1)

		self._full_graph.add_edge(edg[0], edg[1], coords = np.array([]).reshape(0,4))

		if apply:
			self.__set_topo_graph()



	def modify_control_point_coords(self, edg, i,  coords):

		""" Modify the coordinates of a control point of an edge spline of the model.

		Keyword arguments :
		edg -- edge name (in model graph)
		i -- index of the control point to modify
		coords -- new coordinates"""
		
		# Check if e corresponds to an edge
		if self._model_graph.nodes[edg[0]]!= "bif" and self._model_graph.nodes[edg[1]]!= "bif":
			P = self._model_graph.edges[edg]["spline"].get_control_points()
			P[i, :-1] = coords
			self._model_graph.edges[edg]["spline"].set_control_points(P.tolist())


	def modify_control_point_radius(self, edg, i, rad):

		""" Modify the radius of a control point of an edge spline of the model.

		Keyword arguments :
		edg -- edge name (in model graph)
		i -- index of the control point to modify
		rad -- new radius"""

		# Check if e corresponds to an edge
		if self._model_graph.nodes[edg[0]]!= "bif" and self._model_graph.nodes[edg[1]]!= "bif":
			P = self._model_graph.edges[edg]["spline"].get_control_points()
			P[i, -1] = rad
			self._model_graph.edges[edg]["spline"].set_control_points(P.tolist())


	def modify_data_point_coords(self, n, coords, apply = True):

		""" Modify the coordinates of a data point of the centerline.

		Keyword arguments :
		n -- node name (in full graph)
		coords -- new coordinates
		apply -- whether to apply the modification directly to the topo graph or not"""

		# Check if n in nodes
		if n in [elt for elt in self._full_graph.nodes()]:
			self._full_graph.nodes[n]['coords'][:-1] = coords

		if apply:
			self.__set_topo_graph()


	def modify_data_point_radius(self, n, rad, apply = True):

		""" Modify the radius of a data point of the centerline.

		Keyword arguments :
		n -- node name (in full graph)
		rad -- new radius
		apply -- whether to apply the modification directly to the topo graph or not"""
		
		# Check if n in nodes

		if n in [elt for elt in self._full_graph.nodes()]:
			self._full_graph.nodes[n]['coords'][-1] = rad

		if apply:
			self.__set_topo_graph()

	def modify_branch_radius(self, edg, eps, apply = True):

		""" Modify the radius of a data point of the centerline.

		Keyword arguments :
		edg -- edge name (in topo graph)
		eps -- the radius to add to every point of the branch
		apply -- whether to apply the modification directly to the topo graph or not"""
		
		full_id = self._topo_graph.edges[edg]["full_id"]
		for n in full_id:
			rad = self._full_graph.nodes[n]["coords"][-1] + eps
			self.modify_data_point_radius(n, rad, False)

		if apply:
			self.__set_topo_graph()



	def add_data_point(self, coords, idx = None, branch = False, apply = True):

		""" Add a data point to the centerlines and update topo graph. The connection with other nodes is determined by validity criterion. 

		Keyword arguments :
		coords -- coordinates of the new data point
		branch -- whether the point is a new branch or a regular point 
		apply -- whether to apply the modification directly to the topo graph or not"""

		# Get new point id
		nmax = max(list(self._full_graph.nodes())) + 1

		if idx is None:
			# Get the id of the closest data point
			pos = np.array(list(nx.get_node_attributes(self._full_graph, 'coords').values()))

			pt_ids = list(nx.get_node_attributes(self._full_graph, 'coords').keys())
			kdtree = KDTree(pos[:,:-1])
			d, idx = kdtree.query(coords[:-1])
			idx = pt_ids[idx]

		# Find connecting edges according to validity criteria
		if branch:
			self._full_graph.add_node(nmax, coords= coords)
			self._full_graph.add_edge(idx, nmax, coords = np.array([]).reshape(0,4))
		else:
			prec = list(self._full_graph.predecessors(idx))
			succ = list(self._full_graph.successors(idx))

			min_norm = norm(self._full_graph.nodes[prec[0]]['coords'][:-1] - coords[:-1])
			edg = (prec[0], idx)

			for n in prec:
				if norm(self._full_graph.nodes[n]['coords'][:-1] - coords[:-1]) < min_norm:
					min_norm = norm(self._full_graph.nodes[n]['coords'][:-1] - coords[:-1])
					edg = (n, idx)

			for n in succ:	
				if norm(self._full_graph.nodes[n]['coords'][:-1] - coords[:-1]) < min_norm:
						min_norm = norm(self._full_graph.nodes[n]['coords'][:-1] - coords[:-1])
						edg = (idx, n)

			self._full_graph.add_node(nmax, coords= coords)
			# Remove local edge and add new ones
			self._full_graph.remove_edge(edg[0], edg[1])
			# Build new edges
			self._full_graph.add_edge(edg[0], nmax, coords = np.array([]).reshape(0,4))
			self._full_graph.add_edge(nmax, edg[1], coords = np.array([]).reshape(0,4))

		if apply:
			self.__set_topo_graph()



	def rotate_branch(self, normal, edg, alpha):
		""" Rotate a branch and all the downstream branches
		
		Keyword arguments :
		normal -- rotation axis
		edg -- edge of the topo graph that is the center of rotation
		alpha -- rotation angle"""

		rot_center = self._topo_graph.nodes[edg[0]]["coords"][:-1]

		edg_list = [edg] + list(nx.dfs_edges(self._topo_graph, source=edg[1]))
		nds_list = list(nx.dfs_preorder_nodes(self._topo_graph, source=edg[1]))

		for e in edg_list:
			for i in range(self._topo_graph.edges[e]['coords'].shape[0]):
				
				coord = self._topo_graph.edges[e]['coords'][i, :-1]
					
				l = norm(coord - rot_center)
				v = (coord - rot_center) / l
							
				new_v = rotate_vector(v, normal, alpha)
				new_pos = rot_center + l* new_v
				self._topo_graph.edges[e]['coords'][i, :-1] = new_pos

		for n in nds_list:
			coord = self._topo_graph.nodes[n]['coords'][:-1]

			l = norm(coord - rot_center)
			v = (coord - rot_center) / l
						
			new_v = rotate_vector(v, normal, alpha)
			new_pos = rot_center + l* new_v
			self._topo_graph.nodes[n]['coords'][:-1] = new_pos

		self.reset_model_mesh() # Model and mesh needs to be recomputed completely



	def smooth_spline(self, edg, l, radius = False):

		""" Smooth or unsmooth the spline of a givn edge of the model graph.
		Keyword arguments :
		edg -- model graph edge
		l -- smoothing parameter
		radius -- smooth radius if true, trajectory is false"""

		if not radius:
			self._model_graph.edges[edg]["spline"]._set_lambda_model([l, None])
		else:
			self._model_graph.edges[edg]["spline"]._set_lambda_model([None, l])



	#####################################
	#############  ANALYSIS  ############
	#####################################

	def convert_edge_mode(self, edg, mode1 = "crsec", mode2 = "topo"):
		""" Find the correspondance between edges of mode1 and mode2. """
		matching_edg = []
		edg_list = [e for e in self._topo_graph.edges()]
		if mode1 == "crsec" and mode2 == "topo":
			for e in edg:
				if self._crsec_graph.nodes[e[0]]["type"] == "sep" and (self._crsec_graph.nodes[e[1]]["type"] == "sep" or self._crsec_graph.nodes[e[1]]["type"] == "end"):
				
					e0 = list(self._model_graph.predecessors(list(self._model_graph.predecessors(e[0]))[0]))[0]
					e1 = e[1]

				elif self._crsec_graph.nodes[e[0]]["type"] == "sep" and self._crsec_graph.nodes[e[1]]["type"] == "bif":
				
					e1 = e[0]
					connect = list(self._model_graph.in_edges(e[0])) + list(self._model_graph.out_edges(e[0]))
					connect.remove(e[1])
					e0 = connect[0]
				else:

					e0 = e[0]
					e1 = e[1]
				if (e0, e1) not in edg_list:
					print("Matching error!")
				matching_edg.append((e0, e1))
		else:
			print("The conversion you required is not supported yet.")
		return matching_edg
			


	def angle(self, distance=None, mode="topo"):

		""" Compute the angles between branches """

		if distance is not None:
				dist = distance

		if mode == "topo":

			G = self._topo_graph

			# Compute angles
			angles = []

			for n in G.nodes():
				if G.nodes[n]['type'] == "bif":
					bif_coord = G.nodes[n]['coords'][:-1]
					out_coord = []

					out_edg = list(G.out_edges(n))
					out_edg.sort()
					for e in out_edg:
						if distance is None:
							mean_radius = np.mean(G.edges[e]['coords'][:,-1])
							dist = mean_radius * 2

						l = length_polyline(G.edges[e]['coords'])
						i2 = np.argmax(l>dist)
						if i2 == 0:
							i2 = len(l)-1
						i1 = i2-1
						out_coord.append(G.edges[e]['coords'][i2])

					for i in range(len(out_coord)-1):

						v1 = out_coord[i][:-1] - bif_coord
						v2 =  out_coord[i+1][:-1] - bif_coord
						pos = bif_coord + ((v1/norm(v1) + v2/norm(v2)) / 2)*5

						a = angle(v1, v2) # Compute angle
						a = int(180 * a / pi)

						angles.append((n, pos, a))

		else:

			G = self._model_graph
			# Compute angles
			angles = []

			for n in G.nodes():
				if G.nodes[n]['type'] == "bif":

					bif = G.nodes[n]['bifurcation']
					angle_list, vectors = bif.get_angles()
					
					
					for i in range(len(angle_list)):

						pos = G.nodes[n]['coords'] + ((vectors[i][0]/norm(vectors[i][0]) + vectors[i][1]/norm(vectors[i][1])) / 2)*5
						angles.append((n, pos, angle_list[i]))

		return angles


	def count_nodes(self):

		""" Returns the number of nodes of each type in the graph.
		"""

		count = {'reg' : 0, 'bif' : 0, 'inlet' : 0, 'outlet' : 0, 'other' : 0}


		for n in self._full_graph.nodes():

			if self._full_graph.in_degree(n) == 1 and self._full_graph.out_degree(n) == 1:
				count['reg'] += 1
			elif self._full_graph.in_degree(n) == 0 and self._full_graph.out_degree(n) == 1:
				count['inlet'] += 1
			elif self._full_graph.in_degree(n) == 1 and self._full_graph.out_degree(n) == 0:
				count['outlet'] += 1
			elif self._full_graph.out_degree(n) > 1:
				count['bif'] += 1
			else: 
				count['other'] += 1

		return count


	def add_noise_centerline(self, std, normal = False):

		""" Add noise to the radius of the initial centerlines.

		Keyword arguments:
		std -- std deviations of the gaussian noise (mm) for radius
		"""

		for e in self._topo_graph.edges():

			pts = self._topo_graph.edges[e]['coords']

			for i in range(pts.shape[0]):
				if normal:
					# Random rotation angle
					alpha = np.random.uniform(low=0.0, high=2*pi)
					spl = self._model_graph.edges[e]['spline']
					t = spl.project_point_to_centerline(pts[i,:-1])
					tg = spl.tangent(t)
					n = cross(np.array([0,1,0]), tg)
					rand_dir = rotate_vector(n, tg, alpha)

				else:
					rand_dir = np.random.normal(0, 1, (1, 3))[0]

				rand_norm = np.random.normal(0, std, (1, 1))[0]
				pts[i, :-1] = pts[i,:-1] + rand_dir / norm(rand_dir) * pts[i,-1] * rand_norm

			#if std > 0.0:
			#	pts = order_points(pts)
				
			# Modify topo graph 
			self._topo_graph.add_edge(e[0], e[1], coords = pts)

			# Change full graph
			self.topo_to_full()
			self.reset_model_mesh()

	


	def add_noise_radius(self, std):

		""" Add noise to the radius of the initial centerlines.

		Keyword arguments:
		std -- std deviations of the gaussian noise (mm) for radius
		"""

		for e in self._topo_graph.edges():

			pts = self._topo_graph.edges[e]['coords']

			rand = np.hstack((np.zeros((pts.shape[0], 3)), np.random.normal(0, std, (pts.shape[0], 1))))
			pts += rand #pts * rand

			# Modify topo graph 
			self._topo_graph.add_edge(e[0], e[1], coords = pts)

			# Change full graph
			self.topo_to_full()
			self.reset_model_mesh()
		


	def low_sample(self, p, apply = True):

		""" Resample the nodes of the initial centerlines.

		Keyword arguments:
		p -- ratio of nodes to keep [0, 1]
		apply -- apply to topo graph directly
		"""

		# Change full graph
		self.topo_to_full()

		for e in self._topo_graph.edges():

			#pts = self._topo_graph.edges[e]['coords']
			full_id =  self._topo_graph.edges[e]['full_id']

			if p != 0 and len(full_id)!=0:
				# Resampling
				step = int(len(full_id)/(p*len(full_id)))

				if step > 0:
					#pts =  pts[:-1:step][1:]
					full_id_keep = full_id[:-1:step][1:]
				else:
					#pts = pts[int(pts.shape[0]/2)]
					full_id_keep = full_id[int(len(full_id)/2)]
			else:
				full_id_keep = full_id[:]


			# Modify full graph
			for  n in full_id:
				if n not in full_id_keep:
					self.delete_data_point(n)

		if apply:
			self.__set_topo_graph()
			#self._topo_graph.add_edge(e[0], e[1], coords = pts, full_id = full_id)





	def check_mesh(self, thres = 0, edg = [], mode = "crsec"):
		""" Compute quality metrics and statistics from the volume mesh

		Keyword arguments:
		thres -- scaled jacobian threshold for failure
		mode -- representation mode in "segment", "crsec"
		"""


		# Separate parts
		G = self._crsec_graph

		if self._surface_mesh is None:
			raise ValueError('Mesh not found.')

		if len(edg) == 0:
			edg = list(G.edges())
			nds = list(G.nodes())
		else:
			nds = []
			for e in edg:
				if e[0] not in nds:
					nds.append(e[0])
				if e[1] not in nds:
					nds.append(e[1])

		failed_edges = []
		failed_bifs = []

		failed_crsec = []

		color_field = np.zeros((len(self._surface_mesh[1]),), dtype = bool)
		
		for n in nds:
			if G.nodes[n]["type"] == "bif":
				print("Checking bif", n)
			
				edg_list = [e for e in G.in_edges(n)] + [e for e in G.out_edges(n)] 

				m = self.mesh_volume(edg=edg_list, link=True)
				link_vol = self.get_volume_link()
				m = m.compute_cell_quality('scaled_jacobian')	
				
				tab = m['CellQuality']
				del m
				gc.collect()

				print(np.min(tab))
				if np.min(tab) <= thres:
					failed_bifs.append(n) 


		for e in edg:
			if G.nodes[e[1]]["type"] != "bif" and G.nodes[e[0]]["type"] != "bif":
				print("Checking edg", e)
				
				m = self.mesh_volume(edg = [e], link=True)
				link_vol = self.get_volume_link()
				m = m.compute_cell_quality('scaled_jacobian')	
				tab = m['CellQuality']
				del m
				gc.collect()

				print(np.min(tab))
				if np.min(tab) <= thres:
					failed_edges.append(e)
					tab = tab <= thres
					crsec = []
					for i in range(len(tab)):
						if tab[i]:
							if link_vol[i, -1] not in crsec:
								crsec.append(link_vol[i, -1])
					failed_crsec.append(crsec)


		link_surf = self.get_surface_link()

		for n in failed_bifs:
			for e in [e for e in G.in_edges(n)] + [e for e in G.out_edges(n)]:

				add_field = (link_surf[:,0] == e[0]) & (link_surf[:,1] == e[1])
				color_field = color_field | add_field


		for i in range(len(failed_edges)):
			e = failed_edges[i]

			if mode == "crsec":

				for cr in failed_crsec[i]:

					add_field = (link_surf[:,0] == e[0]) & (link_surf[:,1] == e[1]) & (link_surf[:,2] == cr)
					color_field = color_field | add_field
			else :
				add_field = (link_surf[:,0] == e[0]) & (link_surf[:,1] == e[1])
				color_field = color_field | add_field


		return color_field.astype(int), failed_edges, failed_bifs



	#####################################
	##########  VISUALIZATION  ##########
	#####################################


	def show(self, points = False, centerline = False, control_points = False):

		""" Displays the centerline network in 3D viewer.

		Keywords arguments:
		points -- True to display centerline point data
		centerline -- True to display spline centerlines
		control_points -- True to display control polygon
		"""

		# 3D plot
		with plt.style.context(('ggplot')):
		
			fig = plt.figure(figsize=(10,7))
			ax = Axes3D(fig)
			ax.set_facecolor('white')

			
			pos = nx.get_node_attributes(self._topo_graph, 'coords')
			colors = {"end":'blue', "bif":'green', "reg":'orange', "sep":'purple', "sink":'red'}

			if not points and not centerline and not control_points:
				
				# Display topological nodes
				for key, value in pos.items():
					ax.scatter(value[0], value[1], value[2], c=colors[self._topo_graph.nodes[key]['type']], depthshade=False, s=40)
					ax.text(value[0], value[1], value[2], str(key))

				# Plot connecting lines
				pos = nx.get_node_attributes(self._full_graph, 'coords')
				for i,j in enumerate(self._full_graph.edges()):
					ax.plot(np.array((pos[j[0]][0], pos[j[1]][0])), np.array((pos[j[0]][1], pos[j[1]][1])), np.array((pos[j[0]][2], pos[j[1]][2])), c='black', alpha=0.5)

			if points: 

				# Display topological nodes
				for key, value in pos.items():
					ax.scatter(value[0], value[1], value[2], c=colors[self._topo_graph.nodes[key]['type']], depthshade=False, s=40)
					ax.text(value[0], value[1], value[2], str(key))
		
				coords = nx.get_edge_attributes(self._topo_graph, 'coords')

				# Display edge nodes
				for key, value in coords.items():
					if value.shape[0]!=0:
						ax.scatter(value[:,0], value[:,1], value[:,2], c='red', depthshade=False, s= 40)

			if centerline:

				pos = nx.get_node_attributes(self._model_graph, 'coords')

			
				# Display topological nodes
				for key, value in pos.items():
					ax.scatter(value[0], value[1], value[2], c=colors[self._model_graph.nodes[key]['type']], depthshade=False, s=40)
					ax.text(value[0], value[1], value[2], str(key))

				if self._model_graph is None:
					print('Please perform spline approximation first to display centerlines.')
				else: 
					spl = nx.get_edge_attributes(self._model_graph, 'spline')
					data = nx.get_edge_attributes(self._model_graph, 'coords')

					for key, value in spl.items():
						if value is not None:
							points = value.get_points()
							ax.plot(points[:,0], points[:,1], points[:,2],  c='black')

					#for key, value in data.items():
					#	if len(value)>0:
					#		ax.scatter(value[:,0], value[:,1], value[:,2],  c='red', s= 30)


					if control_points:
						for key, value in spl.items():
							if value is not None:
								points = value.get_control_points()
								ax.scatter(points[:,0], points[:,1], points[:,2],  c='grey', s=40)
			
			# Set the initial view
			ax.view_init(90, -90) # 0 is the initial angle

			# Hide the axes
			ax.set_axis_off()
			plt.show()




	#####################################
	##############  READ  ##############
	#####################################



	def __swc_to_graph(self, filename):

		"""Converts a swc centerline file to a graph and set full_graph attribute.

		Keyword arguments:
		filename -- path to centerline file
		"""

		file = np.loadtxt(filename, skiprows=0)
		G = nx.DiGraph()

		for i in range(0, file.shape[0]):

			# Brava database conversion to nii (x, ysize- z, zsize - y)
			#G.add_node(int(file[i, 0]), coords= np.array([file[i, 2],  198 - file[i, 4] , 115.9394 + file[i, 3], file[i, 5]]))
			G.add_node(int(file[i, 0]), coords= np.array([file[i, 2],  file[i, 3] , file[i, 4], file[i, 5]]))

			if file[i, 6] >= 0:
				G.add_edge(int(file[i, 6]), int(file[i, 0]), coords = np.array([]).reshape(0,4))

		return G

	def __edg_nds_to_graph(self, file_edg, file_nds):

		"""Converts a file of node coordinates a and a file of edges to a graph and set full_graph attribute.

		Keyword arguments:
		file_edg -- path to edges file
		file_nds -- path to nodes file
		"""

		fe = np.loadtxt(file_edg, skiprows=0)
		fn = np.loadtxt(file_nds, skiprows=0)

		G = nx.DiGraph()

		for i in range(0, fn.shape[0]):
			G.add_node(i, coords= np.array([float(fn[i, 0]), float(fn[i, 1]), float(fn[i, 2]), float(fn[i, 3])]))

		for i in range(0, fe.shape[0]):
			G.add_edge(int(fe[i, 0]), int(fe[i, 1]), coords = np.array([]).reshape(0,4))
		return G


	def __txt_to_graph(self, filename):

		"""Converts a txt centerline file to a graph and set full_graph attribute.

		Keyword arguments:
		filename -- path to centerline file
		"""

		file = np.loadtxt(filename, skiprows=0)
		G = nx.DiGraph()

		k = 1
		end_pts = {}
		for j in range(0, file.shape[0]):

			edg_id = int(file[j, 0])
			length = int(file[j, 1])
			radius = float(file[j, 2])
			nb_pts = int(file[j, 3])
			angle = (float(file[j, 4])* pi) / 180
			prec = int(file[j, 5])


			if prec == -1:
				start_pt = np.array([0,0,0])
				act_dir = np.array([1,0,0])
			else:
				prec_start_pts = G.nodes[end_pts[prec][0]]['coords'][:-1]
				prec_end_pts = G.nodes[end_pts[prec][1]]['coords'][:-1]

				start_pt = prec_end_pts

				prec_dir = prec_end_pts - prec_start_pts
				prec_dir = prec_dir / norm(prec_dir)

				act_dir = rotate_vector(prec_dir, np.array([0,0,1]), angle)
				

			end_pt = start_pt + length * act_dir
			pts = linear_interpolation(start_pt, end_pt, nb_pts)
			

			for i in range(len(pts)):
				pt = np.array([pts[i, 0], pts[i, 1], pts[i, 2], radius])
				if i == 0:
					if prec == -1:
						G.add_node(k, coords = pt)
						start_id = k
						k += 1
				elif i == 1:
					if prec != -1:
						G.add_node(k, coords = pt)
						G.add_edge(end_pts[prec][1], k, coords = np.array([]).reshape(0,4))
						start_id = end_pts[prec][1]
						k+=1
					else:
						G.add_node(k, coords = pt)
						G.add_edge(k-1, k, coords = np.array([]).reshape(0,4))
						k += 1
				else:
					G.add_node(k, coords = pt)
					G.add_edge(k-1, k, coords = np.array([]).reshape(0,4))
					k += 1


			end_pts[j] = (start_id, k-1)
		

		return G




	def __vtk_to_graph(self, filename):

		""" Converts vmtk centerline to a graph.

		Keyword arguments:
		filename -- path to vmtk .vtk or .vtp centerline file with branch extracted
		"""

		G = nx.DiGraph() # Create graph

		centerline = pv.read(filename)
		nb_centerlines = centerline.cell_data['CenterlineIds'].max() + 1

		cells = centerline.GetLines()
		cells.InitTraversal()
		idList = vtk.vtkIdList()

		radiusData = centerline.GetPointData().GetScalars('MaximumInscribedSphereRadius') 
		centerlineIds = centerline.GetCellData().GetScalars('CenterlineIds') 

		connect_list = []
		for i in range(nb_centerlines):
			connect_list.append([])

		g = 0
		while cells.GetNextCell(idList):

			if g > 0 and c_id != centerlineIds.GetValue(g):
				start_found = False
			
			c_id = centerlineIds.GetValue(g)

			connect = []
			if c_id == 0:
				
				for i in range(0, idList.GetNumberOfIds()):

					pId = idList.GetId(i)
					pt = centerline.GetPoint(pId)
					radius = radiusData.GetValue(pId)
					
					connect.append(pId)
					G.add_node(pId, coords=np.array([pt[0], pt[1], pt[2], radius]))

			else: 

				# Build kd tree of the points of the graph
				pt_coords = np.array(list(nx.get_node_attributes(G, 'coords').values()))
				pt_ids = list(nx.get_node_attributes(G, 'coords').keys())
				kdtree = KDTree(pt_coords[:,:-1])

				# For all successive points, find the minimum distance to the graph
				for i in range(0, idList.GetNumberOfIds()):

					pId = idList.GetId(i)
					pt = centerline.GetPoint(pId)
					radius = radiusData.GetValue(pId)

					if not start_found:
						d, idx = kdtree.query(pt)
						# If the distance exceeds a given threshold
						if d > 0.05:
							start_found = True
							connect.append(pt_ids[idx])
							connect.append(pId) # Write bifucation edge

							G.add_node(pId, coords=np.array([pt[0], pt[1], pt[2], radius])) # write point to graph
							start_found = True
					else:

						# Write nodes and edges
						connect.append(pId) 
						G.add_node(pId, coords=np.array([pt[0], pt[1], pt[2], radius])) 

			connect_list[c_id] += connect
			g+=1
	
		for connect in connect_list:
			# Add connecting edges
			for j in range(len(connect)-1):
				if connect[j] != connect[j+1]:
					G.add_edge(connect[j], connect[j+1], coords = np.array([]).reshape(0,4))

		return G


	def __tre_to_graph(self, filename):

		""" Converts vmtk centerline to a graph.

		Keyword arguments:
		filename -- path to vmtk .vtk or .vtp centerline file with branch extracted
		"""

		G = nx.DiGraph()

		with open(filename) as fp:

			line = fp.readline()
			k = 1

			while line:

				l = line.split()

				if l[0] == 'NPoints':
					N = int(l[2])

				if l[0] == 'Points':
					for i in range(N):
						line = fp.readline()
						l = line.split()

						c = np.array([float(l[0]), float(l[1]), float(l[2]), float(l[3])])
						G.add_node(k, coords = c)

						if i != 0:
							G.add_edge(k-1, k, coords = np.array([]).reshape(0,4))

						k = k + 1

				line = fp.readline()
		
		return G



	#####################################
	##############  WRITE  ##############
	#####################################

	def write_volume_mesh(self, output = "volume_mesh.vtk"):

		import meshio
		if self._volume_mesh is None:
			warnings.warn("Please compute the volume mesh first.")
		else:

			points = self._volume_mesh[2]
			cells = [("hexahedron", self._volume_mesh[0][:,1:])]
			mesh = meshio.Mesh(points, cells)
			mesh.write(output)


	def write_surface_mesh(self, output = "surface_mesh.vtk"):

		if self._surface_mesh is None:
			warnings.warn("Please compute the surface mesh first.")
		else:
			mesh = self.get_surface_mesh()
			mesh.save(output)



	def write_vtk(self, type, filename):

		""" Writes centerlines as VTK polyline file.

		Keywords arguments: 
		type -- "full", "topo"  or "spline"
		"""

		if type == "full":
			G = self._full_graph

		elif type == "topo":
			G = self._topo_graph

		elif type == "spline":
			G = self.model_to_full(replace=False)

		else: 
			raise ValueError("Wrong graph type.")


		v = np.zeros((G.number_of_nodes(), 4)) # Vertices
		f = [] # Connections
	
		iD = {} #id
		i = 0
		for n in G.nodes():
			v[i, :] = G.nodes[n]['coords']
			iD[n] = i
			i+=1

		for e in G.edges():
			f.append([2, iD[e[0]], iD[e[1]]])

		# Create VTK polyLine 
		poly = pv.PolyData()
		poly.points = v[:,:-1]
		poly.lines = np.array(f)

		# Add radius information
		#poly["radius"] = v[:,-1]
		#poly["id"] = np.array(list(iD.keys()))
		#poly_tube = poly.tube(radius = 0.6)

		poly.save(filename)


	def write_edg_nds(self, filename):

		""" Write swc Neurite Tracer file using depth fist search."""

		print('Writing  file...')


		if filename[-4:] == ".txt":
			filename = filename[:-4]

		file = open(filename + "_nds.txt", 'w') 

		node_names = {}
		k = 0
		# Find inlet
		for n in self._full_graph.nodes():
			node_names[n] = k
			coords = self._full_graph.nodes[n]["coords"]
			file.write(str(coords[0]) + '\t' + str(coords[1]) + '\t' + str(coords[2]) + '\t' + str(coords[3]) + '\n')
			k+=1
		file.close()

		file = open(filename + "_edg.txt", 'w')

		for e in self._full_graph.edges():
			file.write(str(node_names[e[0]]) + '\t' + str(node_names[e[1]]) + '\n')
		file.close()


	def write_swc(self, filename):

		""" Write swc Neurite Tracer file using depth fist search."""

		print('Writing swc file...')
		# Find inlet
		for n in self._full_graph.nodes():
			if self._full_graph.in_degree(n) == 0:
				source = n

		if filename[-4:] != ".swc":
			filename = filename + ".swc"		

		file = open(filename, 'w') 

		keys = list(nx.dfs_preorder_nodes(self._full_graph, source))
		values = range(1, len(keys) + 1)

		mapping = dict(zip(keys, values))

		for p in keys:

			c = self._full_graph.nodes[p]['coords']

			if self._full_graph.in_degree(p) == 1:
				n = mapping[list(self._full_graph.predecessors(p))[0]]
				i = 3

			else: 
				n = -1
				i = 1

			file.write(str(mapping[p]) + '\t' + str(i) + '\t' + str(c[0]) + '\t' + str(c[1]) + '\t' + str(c[2]) + '\t' + str(c[3]) + '\t' + str(n) + '\n')

		file.close()



	def write_data_as_circles(self):
		""" Writes the data points as circles on vtk format"""

		def circle_polyline(coord, normal):

			circle = pv.PolyData()
			num = 20

			v = np.zeros((num, 3))
			f = []

			angle = 2 * pi / num
			angle_list = angle * np.arange(num)

			ref = cross(normal, np.array([0,0,1]))
			ref = ref/norm(ref)
			for i in range(num):

				n = rotate_vector(ref, normal, angle_list[i])
				v[i, :] = coord[:-1] + coord[-1]*n
				f.append([2, i, i+1])

			f[-1][-1] = 0
			circle.points = v
			circle.lines = np.array(f)

			return circle

		circles = pv.MultiBlock()
		centers = pv.MultiBlock()
		for n in self._full_graph.nodes():

			coord = self._full_graph.nodes[n]['coords']
			# Get coordinates and normals of circle
			centers[str(n)]= pv.Sphere(radius=0.2, center=(coord[0], coord[1], coord[2]))

			normals = []
			for k in self._full_graph.predecessors(n):
				normal = coord[:-1] - self._full_graph.nodes[k]['coords'][:-1]
				normals.append(normal / norm(normal))
				

			for k in self._full_graph.successors(n):
				normal = coord[:-1] - self._full_graph.nodes[k]['coords'][:-1]
				normals.append(-normal / norm(normal))
				
			circles[str(n) + "," + str(k)] = circle_polyline(coord, sum(normals))

		return circles, centers









