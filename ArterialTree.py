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
			warnings.warn("No full graph found.")
			
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


	def get_volume_mesh(self):
		""" Return the volume mesh """

		if self._volume_mesh is None:
			warnings.warn("No volume mesh found.")
		else:	
			#return self.write_pyvista_mesh_from_vtk(self._volume_mesh[2], self._volume_mesh[0])
			cells = self._volume_mesh[0]
			cell_types = self._volume_mesh[1]
			points = self._volume_mesh[2]
			return pv.UnstructuredGrid(cells, cell_types, points)

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

	def set_full_graph(self, G):

		""" Set the full graph of the arterial tree."""

		self._full_graph = G
		self.__set_topo_graph()
		self._model_graph = None
		self._crsec_graph = None


	def set_topo_graph(self, G, replace = True):

		""" Set the topo graph of the arterial tree."""

		self._topo_graph = G
		if replace:
			self.topo_to_full(replace)
			self._model_graph = None
			self._crsec_graph = None


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
		
		if filename[-4:] == ".swc":
			print('Loading ' + filename[-3:] + ' file...')
			self._full_graph = self.__swc_to_graph(filename)

		elif filename[-4:] == ".vtk" or filename[-4:] == ".vtp":
			print('Loading ' + filename[-3:] + ' file...')
			self._full_graph = self.__vtk_to_graph(filename)

		elif filename[-4:] == ".txt":
			print('Loading ' + filename[-3:] + ' file...')
			self._full_graph = self.__txt_to_graph(filename)
		else:
			raise ValueError("The provided files must be in swc, vtp or vtk format.")



	def __set_topo_graph(self):

		""" Set the topo graph of the arterial tree. 
		All edges of but the terminal and bifurcation edges are collapsed.
		The coordinates of the collapsed regular points are stored as an edge attribute. The nodes are labelled (end, bif, reg) """


		self._topo_graph = self._full_graph.copy()
		nx.set_node_attributes(self._topo_graph, "reg", name="type")
		nx.set_node_attributes(self._topo_graph, None, name="full_id")
		nx.set_edge_attributes(self._topo_graph, None, name="full_id")


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
					self._topo_graph.nodes[n]['type'] = "end" 
				else: 
					self._topo_graph.nodes[n]['type'] = "bif" 

		# Store the matching of full graph node numbers
		for n in self._topo_graph.nodes():
			self._topo_graph.nodes[n]["full_id"] = n
		for e in self._topo_graph.edges():
			path = list(nx.all_simple_paths(self._full_graph, source=e[0], target=e[1]))[0]
			self._topo_graph.edges[e]["full_id"] = path[1:-1]
		

		# Relabel nodes
		self._topo_graph = nx.convert_node_labels_to_integers(self._topo_graph, first_label=1, ordering='default', label_attribute=None)






	#####################################
	##########  APPROXIMATION  ##########
	#####################################

	

	def model_network(self, radius_model = True, criterion="AIC", akaike=False, max_distance = np.inf):

		""" Create Nfurcation objects and approximate centerlines using splines. The network model is stored in the model_graph attribute."""

		print('Modeling the network...')

		# If there is a crsec_graph, remove it
		if self._crsec_graph != None:
			self._crsec_graph = None


		self._model_graph = self._topo_graph.copy()
		nx.set_node_attributes(self._model_graph, None, name='bifurcation')
		nx.set_node_attributes(self._model_graph, False, name='combine')
		nx.set_node_attributes(self._model_graph, None, name='tangent')
		nx.set_node_attributes(self._model_graph, None, name='ref')
		nx.set_edge_attributes(self._model_graph, None, name='spline')

		# dfs
		dfs = list(nx.dfs_successors(self._topo_graph, source=1).values())


		# Bifurcation models
		all_id = [n for n in self._topo_graph.nodes()]
		original_label = dict(zip(all_id, all_id))

		for l in dfs:
			for n in l:
				if n in [nds for nds in self._topo_graph.nodes()]:
					n = original_label[n]
					if self._topo_graph.nodes[n]['type'] == "bif":
						original_label = self.__model_furcation(n, original_label, criterion, akaike, max_distance)

		# Spline models
		for e in self._model_graph.edges():

			if self._model_graph.nodes[e[0]]['type'] == "end" and  self._model_graph.nodes[e[1]]['type'] :
				self.__model_vessel(e, criterion=criterion, akaike=akaike, radius_model=radius_model, max_distance = max_distance)

		# Add rotation attributes
		nx.set_edge_attributes(self._model_graph, None, name='alpha')
		nx.set_edge_attributes(self._model_graph, 0, name='connect')

		# Add rotation information on bifs and edges
		for e in self._model_graph.edges():
			self.__compute_rotations(e)


	def __model_furcation(self, n, original_label, criterion, akaike, max_distance):

		""" Extract bifurcation parameters from the data and modify the model graph to add the bifurcation object and edges.

		Keyword arguments:
		n -- bifurcation node (model graph)
		"""
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
				while nb_data_out < nb_min:
					if self._model_graph.out_degree(e_act[1])!= 0:
						e_next = (e_act[1], [e for e in self._model_graph.successors(e_act[1])][0]) # Get a successor edge		
						data =  np.vstack((data, self._model_graph.edges[e_next]['coords'], self._model_graph.nodes[e_next[1]]['coords'])) # Add data


						e_act = e_next
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


		if l_ap - splines[ind].radius(t_ap)*3 > 0.2: # We can cut!
			t_cut = splines[ind].length_to_time(l_ap - splines[ind].radius(t_ap)*3)

		
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

			# Compute apical and enc cross sections + new data
			C = [C0]
			AC = []

			spl_out = []
			D_out = []

			for i in range(len(splines)):

				t_ap = max(tAP[i])
				t_cut = splines[i].length_to_time(splines[i].radius(t_ap)*1 + splines[i].time_to_length(t_ap))

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


		# Show splines
		"""
		if not merge_next:
			with plt.style.context(('ggplot')):
				
				fig = plt.figure(figsize=(10,7))
				ax = Axes3D(fig)
				ax.set_facecolor('white')
				col = {0 : "red", 1 : "orange", 2 : "yellow", 3: "green"}

				for i in range(len(splines)):

					pts = splines[i].get_points()
					ax.plot(pts[:,0], pts[:,1], pts[:,2],  c=col[i])
					ax.scatter(D_in[:, 0], D_in[:, 1], D_in[:, 2],  c='blue', s = 40, depthshade=False)
					ax.scatter(D_out[i][:, 0], D_out[i][:, 1], D_out[i][:, 2],  c='red', s = 40, depthshade=False)

					data = splines[i].get_data()
					ax.scatter(data[:, 0],data[:, 1], data[:, 2],  c='black', s = 20, depthshade=False)


				for i in range(len(AP)):
					ax.scatter(AP[i][0], AP[i][1], AP[i][2],  c='red', s = 40, depthshade=False)


				ax.set_axis_off()
				plt.show()
				"""
			


		if merge_next:
			self.merge_branch(n_next) # Merge in topo graph and recompute
			self.merge_branch(n_next, mode = "model")
			
		
			# Re-model furcation 
			original_label = self.__model_furcation(n, original_label, criterion, akaike, max_distance)


		else: # Build bifurction and include it in graph
			original_label = original_label_modif
			bif = Nfurcation("crsec", [C, AC, AP, 0.6])
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


		pts = np.vstack((self._model_graph.nodes[e[0]]['coords'], self._model_graph.edges[e]['coords'], self._model_graph.nodes[e[1]]['coords']))
		if len(pts) <=4:
			pts = resample(pts, 4)

		values = np.zeros((4,4))
		constraint = [False] * 4

		if self._model_graph.nodes[e[0]]['type'] != "end":
			values[0,:] = self._model_graph.nodes[e[0]]['coords']
			constraint[0] = True
			values[1,:] = self._model_graph.nodes[e[0]]['tangent']
			constraint[1] = True
		
		if self._model_graph.nodes[e[1]]['type'] != "end":
			values[-1,:] =  self._model_graph.nodes[e[1]]['coords']
			constraint[-1] = True
			values[-2,:] = self._model_graph.nodes[e[1]]['tangent']
			constraint[-2] = True

		spl = Spline()
		spl.approximation(pts, constraint, values, False, criterion=criterion, akaike=akaike, radius_model=radius_model, max_distance = max_distance)
		
		self._model_graph.edges[e]['spline'] = spl
		self._model_graph.nodes[e[1]]['coords'] = spl.point(1.0, True)



	def __compute_rotations(self, e):

		""" Compute the rotation angle alpha and the connecting node for vessel of edge e

		Keyword arguments:
		e -- vessel (model graph)
		"""

		if self._model_graph.nodes[e[0]]['type'] == "end":
						
			sep_end = False
			# self._model_graphet path to the next sep node
			path = []
			for n in nx.dfs_successors(self._model_graph, source=e[0]):
				path.append(n)
				if self._model_graph.nodes[n]['type'] == "sep":
					sep_end = True
					break

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
				if len(path) == 1:
					path.append(e[1])
				
				spl = self._model_graph.edges[(path[0], path[1])]['spline']
				ref1 =  cross(spl.tangent(0), np.array([0,0,1])) 
				self._model_graph.nodes[path[0]]['ref'] = ref1
				

				# Transport reference along the path
				for i in range(len(path)-1):
					
					spl = self._model_graph.edges[(path[i], path[i+1])]['spline']
					# Transport ref vector 
					ref1 = spl.transport_vector(ref1, 0.0, 1.0)
					self._model_graph.nodes[path[i+1]]['ref'] = ref1


		if self._model_graph.nodes[e[0]]['type'] == "sep":

			# self._model_graph et path to the next sep node
			path = []
			end_bif = False
			for n in nx.dfs_successors(self._model_graph, source=e[0]):
				if self._model_graph.nodes[n]['type'] == "bif":
					end_bif = True
					break
				else: 
					path.append(n)

			if len(path) > 1:

				if end_bif:

					ref0 = self._model_graph.nodes[e[0]]['ref']
					in_edge = [edg for edg in self._model_graph.in_edges(path[0])][0]
					out_edge = [edg for edg in self._model_graph.out_edges(path[-1])][0]

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
					tg = -self._model_graph.edges[out_edge]['spline'].tangent(0.0)
					self._model_graph.nodes[path[-1]]['ref'] = rotate_vector(self._model_graph.nodes[path[-1]]['ref'], tg, coef[-1])

				else: 
						
					ref0 = self._model_graph.nodes[e[0]]['ref']

					# Transport original reference to the target reference
					for i in range(len(path) - 1):
						# Transport ref vector downstream
						spl = self._model_graph.edges[(path[i], path[i+1])]['spline']
						ref0 = spl.transport_vector(ref0, 0.0, 1.0)

						if self._model_graph.nodes[path[i+1]]['ref'] is None: # Reg node case
							self._model_graph.nodes[path[i+1]]['ref'] = ref0


	


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

		def connect_nodes(bif1, bif2, tspl, P0, P1, n):


			""" Compute the nodes connecting an end point to a separation point.

			Keyword arguments: 
			tind -- index of the trajectory spline
			ind -- index of the shape spline of reference 
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
					
				pt = bif2.send_to_surface(P, n, 1)
				pt = bif1.send_to_surface(pt, n, 0)
		

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
			nds[:,j,:] = connect_nodes(bif1, bif2, tspl, sep1[j], sep2[connect_sep[j]], num)


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

			if self._model_graph.nodes[e[0]]['type'] == "bif": #or self._model_graph.nodes[e[0]]['type'] == "end":
				self._crsec_graph.add_edge(e[1], e[0], crsec = None, connect = None)
			else:
				self._crsec_graph.add_edge(e[0], e[1], crsec = None, connect = None)


		if parallel:
			# Compute bifurcation sections (parallel)
			print('Meshing bifurcations.')
			args = []
			bif_id = []
			for  n in self._model_graph.nodes():
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
					self._crsec_graph.nodes[bif_id[i][j+1]]['crsec'] = end_crsec[j]
					self._crsec_graph.edges[(bif_id[i][j+1], bif_id[i][0])]['crsec'] = nds[j]
					self._crsec_graph.edges[(bif_id[i][j+1], bif_id[i][0])]['connect'] = connect_index[j]


			# Compute edges sections (parallel)
			print('Meshing edges.')
			args = []

			for e in self._model_graph.edges():
				if self._model_graph.nodes[e[0]]['type'] != "bif" and self._model_graph.nodes[e[1]]['type'] != "bif":
					
					spl = self._model_graph.edges[e]['spline']

					# Number of cross sections
					num = int(spl.length() / (spl.mean_radius()* d))
					if num <= 1:
						num = 2
					v0 = self._model_graph.nodes[e[0]]['ref']
					alpha = self._model_graph.edges[e]['alpha']

					args.append((spl, num, N, v0, alpha))

			p = Pool(cpu_count())
			crsec_list = p.starmap(segment_crsec, args)

			i = 0
			for e in self._model_graph.edges():
				if self._model_graph.nodes[e[0]]['type'] != "bif" and self._model_graph.nodes[e[1]]['type'] != "bif":

					connect_id = self._model_graph.edges[e]['connect'] * (N // 4)
					crsec = crsec_list[i]

					self._crsec_graph.edges[e]['crsec'] = crsec[1:-1, :, :]
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
					self.__furcation_cross_sections(n)
					

			# Compute edges sections
			print('Meshing edges.')
			for e in self._model_graph.edges():
				if self._model_graph.nodes[e[0]]['type'] != "bif" and self._model_graph.nodes[e[1]]['type'] != "bif":
					self.__vessel_cross_sections(e)

		#self.show(False, False, False)

		# Combine bifurcations
		for n in self._model_graph.nodes():
			if self._model_graph.nodes[n]['combine']:
				self.__combine_nfurcation(n)

		nx.set_node_attributes(self._crsec_graph, None, name='id') # Add id attribute for meshing


	def __furcation_cross_sections(self, n):

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
			self._crsec_graph.nodes[ids[j]]['crsec'] = end_crsec[j]
			self._crsec_graph.edges[(ids[j], n)]['crsec'] = nds[j]
			self._crsec_graph.edges[(ids[j], n)]['connect'] = ind[j]

	
	def __vessel_cross_sections(self, e):

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

		crsec = self.__segment_crsec(spl, num, self._N, v0, alpha)


		self._crsec_graph.edges[e]['crsec'] = crsec[1:-1, :, :]
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

		

		t = np.linspace(0.0, 1.0, num + 2) 

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

		return crsec




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
			G.add_node(k, coords = self._topo_graph.nodes[n]['coords'])
			#self._topo_graph.nodes[n]["full_id"] = k
			ndict[n] = k
			k  = k + 1

		for e in self._topo_graph.edges():
			pts = self._topo_graph.edges[e]['coords']

			if len(pts) == 0:

				G.add_edge(ndict[e[0]], ndict[e[1]], coords = np.array([]).reshape(0,4))
				#self._topo_graph.edges[e]["full_id"] = np.array([])

			else: 

				G.add_node(k, coords = pts[0])
				G.add_edge(ndict[e[0]], k, coords = np.array([]).reshape(0,4))
				#self._topo_graph.edges[e]["full_id"][0] = k

				k = k + 1

				for i in range(1, len(pts)):

					G.add_node(k, coords = pts[i])
					#self._topo_graph.edges[e]["full_id"][i] = k

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
				self._surface_mesh = None
				



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
			
			mesh = self.get_surface_mesh()
			mesh["SelectedPoints"] = selected_point
		
			inside = mesh.threshold(value = 0.5, scalars="SelectedPoints", preference = "point")
			inside.plot(show_edges = True)
			

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


	def topo_correction(self, threshold):

		""" Correct the topology by merging close branches to form (n+1)-furcations

		Keywork arguments:
		threshold -- distance threshold for merging """

		complete = False

		while not complete:
			complete = True
			for e in self._topo_graph.edges():
				D = np.vstack((self._topo_graph.nodes[e[0]]['coords'], self._topo_graph.edges[e]['coords'], self._topo_graph.nodes[e[1]]['coords']))
				l =  length_polyline(D[:,:-1])[-1]

				if l < threshold and self._topo_graph.nodes[e[0]]["type"] != "end":
					self.merge_branch(e[1])
					print("merging branch ", e[1])
					complete = False
					break


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
					self._topo_graph.add_edge(pred, e[1], coords = d)


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



	
	def remove_branch(self, e, preserve_shape = True):

		""" Cuts the branch at edge e and all the downstream branches. 
		If a cross section graph is already computed, recomputes the cross sections at the cut part.

		Keyword arguments: 
		e -- edge of the branch in topo graph as a tuple
		"""

		if self._topo_graph is None:
			raise ValueError('No network was found.')

		if e not in [e for e in self._topo_graph.edges()]:
			raise ValueError('Not a valid edge.')


		# Get all downstream nodes
		downstream_nodes =  list(nx.dfs_preorder_nodes(self._topo_graph, source=e[1]))

		# Remove them from topo graph
		for node in downstream_nodes:
			#self._full_graph.remove_node(self._topo_graph.nodes[node]["full_id"])
			self._topo_graph.remove_node(node)

		# Merge data and remove bifurcation point
		if self._topo_graph.out_degree(e[0]) == 1 and self._topo_graph.in_degree(e[0]) == 1:

			eprec = list(self._topo_graph.in_edges(e[0]))[0]
			esucc = list(self._topo_graph.out_edges(e[0]))[0]
 
			# Create tables of regular nodes data
			coords = np.vstack((self._topo_graph.edges[eprec]['coords'], self._topo_graph.nodes[e[0]]['coords'], self._topo_graph.edges[esucc]['coords']))
			ids = np.hstack((self._topo_graph.edges[eprec]['full_id'], self._topo_graph.nodes[e[0]]['full_id'], self._topo_graph.edges[esucc]['full_id']))
			
			# Create new edge by merging the 2 edges of regular point
			self._topo_graph.add_edge(eprec[0], esucc[1], coords = coords, full_id = ids)
			# Remove regular point
			self._topo_graph.remove_node(e[0])


			# Propagate to full_graph
			self.topo_to_full()


		
		if self._model_graph is not None:

			apply_crsec = False
			if self._crsec_graph is not None:
				apply_crsec = True


			# Get path between e[0] and e[1]
			path = list(nx.all_simple_paths(self._model_graph, source=e[0], target=e[1]))[0]
				
			if self._model_graph.nodes[path[1]]['type']=="bif": # Furcation branch
				if self._model_graph.nodes[path[1]]['bifurcation'].get_n() == 2: # Furcation is a bifucartion

					# Get all downstream nodes
					downstream_nodes = list(nx.dfs_preorder_nodes(self._model_graph, source=path[2]))

					if preserve_shape:

						# Get correct bifurcation shape spline number
						id_out = list(self._model_graph.successors(path[1]))
						if id_out[0] in path:
							id_spl = 1
						else:
							id_spl = 0

						spl = self._model_graph.nodes[path[1]]['bifurcation'].get_spl()[id_spl]

					# Remove downstream nodes from model graph
					for node in downstream_nodes:
						self._model_graph.remove_node(node)

						if apply_crsec:
							self._crsec_graph.remove_node(node)

					# Get in and out bifurcation edges
					n_in = e[0]
					n_out = list(self._model_graph.successors(path[1]))[0]

					n_first = n_in
					n_last = n_out

					# Get id of previous and next sep nodes 
					pred = self._model_graph.predecessors(n_in)
					for n in pred:
						if self._model_graph.nodes[n]['type'] == "sep" or self._model_graph.nodes[n]['type'] == "end":
							n_first = n
							break
						else: 
							pred = self._model_graph.predecessors(n)

					succ = self._model_graph.successors(n_out)
					for n in succ:
						if self._model_graph.nodes[n]['type'] == "sep" or self._model_graph.nodes[n]['type'] == "end":
							n_last = n
							break
						else:
							succ = self._model_graph.successors(n)

					self._model_graph.remove_node(path[1])

					if apply_crsec:
							self._crsec_graph.remove_node(path[1])

					if preserve_shape:
						
						self._model_graph.nodes[n_in]['type'] = "reg"
						self._model_graph.nodes[n_out]['type'] = "reg"

						self._model_graph.nodes[n_in]['ref'] = None
						self._model_graph.nodes[n_out]['ref'] = None

						# Connect them with bifurcation spline
						self._model_graph.add_edge(n_in, n_out, coords = np.array([]).reshape(0,4), spline = spl, alpha = None, connect = 0)

						reset_edges = list(nx.all_simple_edge_paths(self._model_graph, n_first, n_last))[0]

						for re in reset_edges:
							self._model_graph.edges[re]['connect'] = 0
							self._model_graph.edges[re]['alpha'] = None

						# Recompute rotation and connect values
						self.__compute_rotations(reset_edges[0])

						if apply_crsec:

							self._crsec_graph.add_edge(n_in, n_out, coords = np.array([]).reshape(0,4), crsec = None, connect = None)

							self._crsec_graph.nodes[n_in]['type'] = "reg"
							self._crsec_graph.nodes[n_out]['type'] = "reg"

							# Compute crsec
							for e in reset_edges:
								self.__vessel_cross_sections(e)

					else:

						# Get data coordinates between n_first and n_last
						coords = np.array([]).reshape(0,4)
						path_first = list(nx.all_simple_paths(self._model_graph, source=n_first, target=n_in))

						if len(path_first) > 0:
							path_first = path_first[0]
							for i in range(len(path_first) - 1):
								coords = np.vstack((coords, self._model_graph.edges[(path_first[i], path_first[i+1])]['coords']))
								coords = np.vstack((coords, self._model_graph.nodes[path_first[i+1]]['coords']))
						
							self._model_graph.remove_node(path_first[i+1])
							if apply_crsec:
								self._crsec_graph.remove_node(path_first[i+1])


						path_last = list(nx.all_simple_paths(self._model_graph, source=n_out, target=n_last))
						if len(path_last) > 0:
							path_last = path_last[0]
							for i in range(len(path_last) - 1):
								coords = np.vstack((coords, self._model_graph.nodes[path_last[i]]['coords']))
								coords = np.vstack((coords, self._model_graph.edges[(path_last[i], path_last[i+1])]['coords']))
						
							self._model_graph.remove_node(path_last[i])
							if apply_crsec:
								self._crsec_graph.remove_node(path_last[i])

						# Create edge between n_first and n_last
						self._model_graph.add_edge(n_first, n_last, coords = coords, spline = None, alpha = None, connect = 0)
						self.__model_vessel((n_first, n_last)) # Recompute spline
						self.__compute_rotations((n_first, n_last)) # Set rotation and connect values

						if apply_crsec:
							self._crsec_graph.add_edge(n_first, n_last, crsec = None, coord = None)
							self.__vessel_cross_sections((n_first, n_last))


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
						self.__furcation_cross_sections(path[1])



	def transpose(self, val):

		""" Cuts the original graph to a subgraph. 
		Remove any spline approximation of cross section computation performed on the previous graph.

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



	def scale(self, val):

		""" Cuts the original graph to a subgraph. 
		Remove any spline approximation of cross section computation performed on the previous graph.

		Keywords arguments: 
		val -- list of transposition values for every coordinates of the nodes
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

		""" Add noise and resample the nodes of the initial centerlines.

		Keyword arguments:
		p -- ratio of nodes to add (>1)
		"""

		for e in self._topo_graph.edges():

			pts = np.vstack((self._topo_graph.nodes[e[0]]['coords'], self._topo_graph.edges[e]['coords'], self._topo_graph.nodes[e[1]]['coords']))

			n = int(pts.shape[0]*p)
			pts =  resample(pts, num = n+2)

			# Modify topo graph 
			self._topo_graph.add_edge(e[0], e[1], coords = pts[1:-1])

			# Change full graph
			self.topo_to_full()

		




	#####################################
	#############  ANALYSIS  ############
	#####################################



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
		


	def low_sample(self, p):

		""" Resample the nodes of the initial centerlines.

		Keyword arguments:
		p -- ratio of nodes to keep [0, 1]
		"""

		for e in self._topo_graph.edges():

			pts = self._topo_graph.edges[e]['coords']

			if p != 0 and pts.shape[0]!=0:
				# Resampling
				step = int(pts.shape[0]/(p*pts.shape[0]))

				if step > 0:
					pts =  pts[:-1:step][1:]
				else:
					pts = pts[int(pts.shape[0]/2)]


			# Modify topo graph 
			self._topo_graph.add_edge(e[0], e[1], coords = pts)

			# Change full graph
			self.topo_to_full()



	def check_mesh(self, thres = 0):
		""" Compute quality metrics and statistics from the volume mesh

		Keyword arguments:
		thres -- scaled jacobian threshold for failure
		mode -- representation mode in "segment", "crsec"
		"""


		# Separate parts
		G = self._crsec_graph

		if self._surface_mesh is None:
			raise ValueError('Mesh not found.')

		failed_edges = []
		failed_bifs = []

		failed_crsec = []

		color_field = np.zeros((len(self._surface_mesh[1]),), dtype = bool)

		for n in G.nodes():
			if G.nodes[n]["type"] == "bif":
			
				edg = [e for e in G.in_edges(n)] + [e for e in G.out_edges(n)] 

				m = self.mesh_volume(edg=edg, link=True)
				link_vol = self.get_volume_link()
				m = m.compute_cell_quality('scaled_jacobian')	
				
				tab = m['CellQuality']
				del m
				gc.collect()

				if np.min(tab) <= thres:

					for e in edg:
						failed_bifs.append(n) 
						


		for e in G.edges():
			if G.nodes[e[1]]["type"] == "end" or G.nodes[e[0]]["type"] == "end":
				
				m = self.mesh_volume(edg = [e], link=True)
				link_vol = self.get_volume_link()
				m = m.compute_cell_quality('scaled_jacobian')	
				tab = m['CellQuality']
				del m
				gc.collect()

				
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

			for cr in failed_crsec[i]:

				add_field = (link_surf[:,0] == e[0]) & (link_surf[:,1] == e[1]) & (link_surf[:,2] == cr)
				color_field = color_field | add_field


		return color_field.astype(int), failed_edges, failed_bifs



	#####################################
	##########  VISUALIZATION  ##########
	#####################################


	def show(self, points = True, centerline = True, control_points = False):

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
			colors = {"end":'blue', "bif":'green', "reg":'orange', "sep":'purple'}


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
		type -- vtk centerline data type, either branch or group
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




	#####################################
	##############  WRITE  ##############
	#####################################


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



	def write_swc(self, filename):

		""" Write swc Neurite Tracer file using depth fist search."""

		print('Writing swc file...')

		if filename[-4:] != ".swc":
			filename = filename + "swc"		

		file = open(filename, 'w') 

		keys = list(nx.dfs_preorder_nodes(self._full_graph , 0))
		values = range(1, len(keys) + 1)

		mapping = dict(zip(keys, values))

		for p in keys:

			c = self._full_graph.nodes[p]['coords']

			if self._full_graph.in_degree(p) == 1:
				print(list(self._full_graph.predecessors(p)))
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









