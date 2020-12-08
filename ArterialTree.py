from multiprocessing import Pool, Process, cpu_count
import pyvista as pv # Meshing
import vtk
from scipy.spatial import KDTree

# Trigonometry functions
from math import pi
from numpy.linalg import norm 
from numpy import dot, cross

# Plot
import matplotlib.pyplot as plt # Tools for plots
from mpl_toolkits.mplot3d import Axes3D # 3D display

import networkx as nx

from utils import *
from Bifurcation import Bifurcation
from Spline import Spline





class ArterialTree:


	#####################################
	##########  CONSTRUCTOR  ############
	#####################################

	def __init__(self, patient_name, database_name, filename = None):

		# Check user parameters

		# Initiate attributes
		self.patient_name = patient_name
		self.database_name = database_name
		self._surface_mesh = None


		if filename == None:
			self._full_graph = None
			self._topo_graph = None
			self._spline_graph = None
			self._crsec_graph = None

		else:
			self.__load_file(filename)
			self.__set_topo_graph()
			self._spline_graph = None
			self._crsec_graph = None




	#####################################
	#############  GETTERS  #############
	#####################################

	def get_full_graph(self):

		if self._full_graph is not None:
			return self._full_graph
		else:
			raise AttributeError('Please set the centerline data points')

	def get_topo_graph(self):

		if self._topo_graph is not None:
			return self._topo_graph
		else:
			raise AttributeError('Please set the centerline data points')

	def get_spline_graph(self):

		if self._spline_graph is not None:
			return self._spline_graph
		else:
			raise AttributeError('Please perform spline approximation first.')

	def get_crsec_graph(self):

		if self._crsec_graph is not None:
			return self._crsec_graph
		else:
			raise AttributeError('Please perform meshing first.')




	#####################################
	#############  SETTERS  #############
	#####################################

	def set_full_graph(self, G):

		""" Set the full graph of the arterial tree."""

		self._full_graph = G
		self.__set_topo_graph()
		self._spline_graph = None
		self._crsec_graph = None


	def set_topo_graph(self, G):

		""" Set the topo graph of the arterial tree."""

		self._topo_graph = G
		self._full_graph = self.topo_to_full()
		self._spline_graph = None
		self._crsec_graph = None


	def set_spline_graph(self, G):

		""" Set the spline graph of the arterial tree."""

		self._spline_graph = G
		self._full_graph = self.spline_to_full()
		self.__set_topo_graph()
		self._crsec_graph = None


	def set_crsec_graph(self, G):

		""" Set the spline graph of the arterial tree."""

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
		else:
			raise ValueError("The provided files must be in swc, vtp or vtk format.")



	def __set_topo_graph(self):

		""" Set the topo graph of the arterial tree. 
		All edges of non end and non bifurcation nodes are collapsed to one.
		The coordinates of the collapsed regular points are stored as an edge attribute. """


		self._topo_graph = self._full_graph.copy()
		nx.set_node_attributes(self._topo_graph, "reg", name=type)


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

		# Relabel nodes
		self._topo_graph = nx.convert_node_labels_to_integers(self._topo_graph, first_label=1, ordering='default', label_attribute=None)



	#####################################
	##########  APPROXIMATION  ##########
	#####################################
	

	def spline_approximation(self):

		""" Approximate centerlines using splines and sets the attribute spline-graph.
		"""

		print('Modeling the network with splines...')

		G = self.__end_conditions()
		self._spline_graph = nx.DiGraph()

		for e in G.edges():

			pts = np.vstack((G.nodes[e[0]]['coords'], G.edges[e]['coords'], G.nodes[e[1]]['coords']))
			
			while len(pts) < 3:
				# Linearly interpolate data
				pts = np.array(linear_interpolation(pts, 1))

			values = np.zeros((4,4))
			constraint = [False] * 4

			if G.nodes[e[0]]['type'] == "bif":
				values[0,:] = pts[0]
				constraint[0] = True
				values[1,:] = G.nodes[e[0]]['tangent']
				constraint[1] = True
	
			if G.nodes[e[1]]['type'] == "bif":
				values[-1,:] =  pts[-1]
				constraint[-1] = True
				values[-2,:] = G.nodes[e[1]]['tangent']
				constraint[-2] = True

			spl = Spline()
			spl.approximation(pts, constraint, values, False)


			# Add edges with spline attributes
			self._spline_graph.add_node(e[0], coords = spl.point(0.0, True), type = G.nodes[e[0]]['type'])
			self._spline_graph.add_node(e[1], coords = spl.point(1.0, True), type = G.nodes[e[1]]['type'])
			self._spline_graph.add_edge(e[0], e[1], spline = spl, coords = G.edges[e]['coords']) 
			

	def __end_conditions(self):

		G = self._topo_graph.copy()

		for n in self._topo_graph.nodes():
			if self._topo_graph.nodes[n]['type'] == "bif":

				edges = list(self._topo_graph.in_edges(n)) + list(self._topo_graph.out_edges(n))
				pts, pt, tg = self.__bifurcation_point_estimation(edges, n)
			
				# Update bifurcation point information
				G.add_node(n, coords = pt, tangent = tg, type = "bif")

				G.add_edge(*edges[0], coords = pts[0])
				G.add_edge(*edges[1], coords = pts[1])
				G.add_edge(*edges[2], coords = pts[2])

		return G



	def __bifurcation_point_estimation(self, edges, node, pos = None):

		B = self._topo_graph.nodes[node]['coords'][:-1]

		pts0 = np.vstack((self._topo_graph.edges[edges[0]]['coords'], self._topo_graph.nodes[node]['coords']))
		pts1 = self._topo_graph.edges[edges[1]]['coords']
		pts2 = self._topo_graph.edges[edges[2]]['coords']


		# Fit splines from the main branch to the daughter branches
		D1 = np.vstack((self._topo_graph.nodes[edges[0][0]]['coords'], pts0, pts1, self._topo_graph.nodes[edges[1][1]]['coords']))
		spl1 = Spline()
		spl1.approximation(D1, [False, False, False, False], np.zeros((4,4)), False)

		D2 = np.vstack((self._topo_graph.nodes[edges[0][0]]['coords'], pts0, pts2, self._topo_graph.nodes[edges[2][1]]['coords']))
		spl2 = Spline()
		spl2.approximation(D2, [False, False, False, False], np.zeros((4,4)), False)
		
		# Find the separation point between the splines
		if pos is None:
			pos = np.mean(pts0[:,-1])

		t1 = spl1.project_point_to_centerline(B) 
		t2 = spl2.project_point_to_centerline(B)

		t1 = spl1.length_to_time(spl1.time_to_length(t1) - pos)
		t2 =  spl2.length_to_time(spl2.time_to_length(t2) - pos)	

		# Get bifurcation coordinates and tangent
		tg = (spl1.first_derivative(t1, True) + spl2.first_derivative(t2, True)) /2.0  #(spl1.tangent(t1, True) + spl2.tangent(t2, True)) / 2.0 
		pt = (spl1.point(t1, True) + spl2.point(t2, True)) / 2.0

		# Remove the points of the main branch further than the separation point
		for i in range(len(pts0)):
			t = spl1.project_point_to_centerline(pts0[-1][:-1]) 
					
			if t > t1:

				pts1 = np.vstack((pts0[-1], pts1))
				pts2 = np.vstack((pts0[-1], pts2))
				pts0 = pts0[:-1]
								
			else: break

		return [pts0, pts1, pts2], pt, tg 



	#####################################
	#########  MESHING METHODS  #########
	#####################################


	def compute_cross_sections(self, N, d, bifurcation_model = True):

		""" Splits the splines into segments and bifurcation parts and computes surface cross sections.

		Keyword arguments:

		N -- number of nodes in a transverse section (multiple of 4)
		d -- longitudinal density of nodes as a proportion of the radius
		bifurcation_model -- (true) bifurcation based on a model (false) bifurcation based on the data
		"""

		if self._spline_graph is None:

			print('Modeling the network with splines...')
			self.spline_approximation(0.05)
		
		self._N = N
		self._d = d	

		G = self._spline_graph.copy()
		#nx.set_node_attributes(G, None, name='id_crsec')

		nmax = max(list(G.nodes())) + 1
		count = 0

		print('Meshing bifurcations')

		bif_nds = []
		args = []

		# Bifurcation cross sections
		for n in self._spline_graph.nodes():

			if self._spline_graph.nodes[n]['type'] == "bif":

				spl0 =  self._spline_graph.edges[list(self._spline_graph.in_edges(n))[0]]['spline']

				S = []
				spl_bif = []

				spl_out = []
				for e in self._spline_graph.out_edges(n): # Out splines
					spl_out.append(self._spline_graph.edges[e]['spline'])

				# Find apex
				AP, times = spl_out[0].first_intersection(spl_out[1])

				i = 0
				for e in self._spline_graph.out_edges(n):

					# Cut spline
					spl = spl_out[i]
					splb, spls = spl.split_length(spl.time_to_length(times[i]) + spl.radius(times[i]))  

					spl_bif.append(splb)
					# Adding section
					S.append([splb.point(1.0, True), splb.tangent(1.0, True)])
					# Adding separation node
					G.add_node(nmax, coords = splb.tangent(1.0, True), type = "sep")
					# Adding segment edge
					G.add_edge(nmax, e[1], spline = spls)
					nmax = nmax + 1
					# Remove original edge
					G.remove_edge(*e)

					i+=1

				S0 = [spl.point(0.0, True), spl.tangent(0.0, True)]

				# Compute bifurcation object
				if bifurcation_model : 
					bif = Bifurcation(S0, S[0], S[1], 1, AP = AP)
				else: 
					bif = Bifurcation(S0, S[0], S[1], 1, spl = spl_bif, AP = AP)
				#bif.show(True)
				args.append((bif, N, d))
				bif_nds.append([n, nmax - 2, nmax - 1])

		# Return bifurcations
		p = Pool(cpu_count())
		liste_bifurcations = p.starmap(parallel_bif, args)	

		
		for i in range(len(liste_bifurcations)):	

			n = bif_nds[i][0]
			bif = liste_bifurcations[i]

			B = bif.get_B()
			tspl = bif.get_tspl()
			spl = bif.get_spl()
			end_crsec, bif_crsec, nds, connect_index = bif.get_crsec()
			S = bif.get_endsec()

			# Add separation nodes with cross sections
			G.add_node(n, coords = S[0][0], type = "sep", crsec = end_crsec[0], id = count)
			count += np.array(end_crsec[0]).shape[0]
			G.add_node(bif_nds[i][1], coords = S[1][0], type = "sep", crsec = end_crsec[1], id = count)
			count += np.array(end_crsec[1]).shape[0]
			G.add_node(bif_nds[i][2], coords = S[2][0], type = "sep", crsec = end_crsec[2], id = count)
			count += np.array(end_crsec[2]).shape[0]

			# Add bifurcation node
			G.add_node(nmax, coords = B, type = "bif", crsec = bif_crsec, id = count)
			count += len(bif_crsec)
				
			# Add bifurcation edges
			G.add_edge(n, nmax, spline = tspl[0], crsec = nds[0], connect = connect_index[0], id = count)
			count += np.array(nds[0]).shape[0] * np.array(nds[0]).shape[1]
			G.add_edge(bif_nds[i][1], nmax, spline = tspl[1], crsec = nds[1], connect = connect_index[1], id = count)
			count += np.array(nds[1]).shape[0] * np.array(nds[1]).shape[1]
			G.add_edge(bif_nds[i][2], nmax, spline = tspl[2], crsec = nds[2], connect = connect_index[2], id = count)
			count += np.array(nds[2]).shape[0] * np.array(nds[2]).shape[1]
				
			nmax = nmax + 1

		print('Meshing edges')
		# Connecting segments cross sections
		args = []
		list_connect = []
		for e in G.edges():

			spl = G.edges[e]['spline']

			if G.nodes[e[0]]['type'] != "bif" and G.nodes[e[1]]['type'] != "bif": # Connecting segments
				
				# Number of cross sections
				num = int(spl.length() / (spl.mean_radius()* d))

				if num <= 1:
					num = 2

				# Simple tube
				if G.nodes[e[0]]['type'] == "end" and G.nodes[e[1]]['type'] == "end": 

					args.append([spl, num, N, [], None])
					#crsec = self.__segment_crsec(spl, num, N)
					list_connect.append(np.arange(0, N).tolist())

				# Connecting tube
				elif G.nodes[e[0]]['type'] == "sep" and G.nodes[e[1]]['type'] == "sep":

					v0 = G.nodes[e[0]]['crsec'][0] - G.nodes[e[0]]['coords'][:-1]

					# Find the closest symmetric vector for v1
					crsec1 = G.nodes[e[1]]['crsec']
					tg1 = spl.tangent(1.0)
					v01 = spl.transport_vector(v0, 0, 1)

					min_a = 5.0
					for i in [0, int(N/4), int(N/2), int(3* N/4)]: #range(N)

						v = crsec1[i] - G.nodes[e[1]]['coords'][:-1]
						a = angle(v01, v, axis = tg1, signed =True)

						if abs(a) < abs(min_a):
							min_a = a
							min_ind = i
					
					args.append([spl, num, N, v0 / norm(v0), min_a])

					connect_index = []
					for i in range(N):

						if min_ind  == N:
							min_ind  = 0

						connect_index.append(min_ind)
						min_ind  = min_ind  + 1
				

					list_connect.append(connect_index)


				# Ending tube 
				else :
					
					if G.nodes[e[1]]['type'] == "end":
						v0 = G.nodes[e[0]]['crsec'][0] - G.nodes[e[0]]['coords'][:-1]
						args.append([spl, num, N, v0 / norm(v0)])
						#crsec = self.__segment_crsec(spl, num, N, v0 = v0 / norm(v0))
						list_connect.append(np.arange(0, N).tolist())

					else: 
						v0 = G.nodes[e[1]]['crsec'][0] - G.nodes[e[1]]['coords'][:-1]
						splr = spl.copy_reverse()
						args.append([splr, num, N, v0 / norm(v0)])
						#crsec = self.__segment_crsec(splr, num, N, v0 = v0 / norm(v0))
						#crsec = crsec[::-1, np.hstack((0, np.arange(N - 1,0,-1))), :]

						list_connect.append(np.arange(0, N).tolist())

		p = Pool(cpu_count())
		liste_crsec = p.starmap(segment_crsec, args)	


		i = 0
		for e in G.edges:

			if G.nodes[e[0]]['type'] != "bif" and G.nodes[e[1]]['type'] != "bif":

				crsec = liste_crsec[i]

				if G.nodes[e[0]]['type'] == "end":
					crsec = crsec[::-1, np.hstack((0, np.arange(N - 1,0,-1))), :]

				spl = G.edges[e]['spline']

				# Add cross sections
				G.add_edge(e[0], e[1], crsec = crsec[1:-1], spline = spl, connect = list_connect[i], id = count)
				count += crsec[1:-1].shape[0] * crsec[1:-1].shape[1]
					
				if G.nodes[e[0]]['type'] == "end":
					G.add_node(e[0], coords = G.nodes[e[0]]['coords'], crsec = crsec[0], type = "end", id = count)
					count += crsec[0].shape[0]

				if G.nodes[e[1]]['type'] == "end":
					G.add_node(e[1], coords = G.nodes[e[1]]['coords'], crsec = crsec[-1], type = "end", id = count)
					count += crsec[-1].shape[0]

				i +=1

		self._nb_nodes = count 
		self._crsec_graph = G



	def __segment_crsec(self, spl, num, N, v0 = [], alpha = None):

		""" Compute the cross section nodes along for a vessel segment.

		Keyword arguments:
		spl -- segment spline
		num -- number of cross sections
		N -- number of nodes in a cross section (multiple of 4)
		v0 -- reference vector
		alpha -- rotation angle
		"""

		

		t = np.linspace(0.0, 1.0, num + 2) #t = [0.0] + spl.resample_time(num) + [1.0]

		if len(v0) == 0:
			v0 = cross(spl.tangent(0), np.array([0,0,1])) # Random initialisation of the reference vector
			

		if alpha!=None:
			theta = np.hstack((0.0, np.linspace(0.0, alpha, num), alpha)) # Get rotation angles

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



	def __id_nodes(self, L, count):

		""" Write a list of indices for nodes in list L, starting from index count."""
		L = np.array(L)[...,0]

		with np.nditer(L, op_flags=['readwrite']) as it:
			for x in it:
				x[...] = count
				count = count + 1

		L = L.astype(int).tolist()		
	
		return L, count



	def mesh_surface(self):

		""" Meshes the surface of the arterial tree."""

		if self._crsec_graph is None:

			print('Computing cross sections with default parameters...')
			self.compute_cross_sections(24, 0.2, False) # Get cross section graph
		
		G = self._crsec_graph
		print('Meshing surface...')

		vertices = np.zeros((self._nb_nodes, 3))
		faces = np.zeros((self._nb_nodes, 5), dtype =int)
		nb_faces = 0

		for e in G.edges():		

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
				nb_faces += 1

			for k in range(G.edges[e]['crsec'].shape[0] -1): # Edge sections
				for i in range(G.edges[e]['crsec'].shape[1]):

					if i == G.edges[e]['crsec'].shape[1] - 1:
						j = 0
					else: 
						j = i + 1

					faces[nb_faces,:] = np.array([4, id_edge + (k * self._N) + i, id_edge + ((k + 1) * self._N) + i, id_edge + ((k + 1) * self._N) + j, id_edge + (k * self._N) + j])
					nb_faces += 1

			id_last_edge = id_edge + ((G.edges[e]['crsec'].shape[0] -1) * self._N)
			connect = G.edges[e]['connect']


			for i in range(G.edges[e]['crsec'].shape[1]):

				if i == G.edges[e]['crsec'].shape[1] - 1:
					j = 0
				else:
					j = i + 1

				faces[nb_faces,:] = np.array([4, id_last_edge + i, id_last + connect[i], id_last + connect[j], id_last_edge + j])
				nb_faces += 1
		
		faces = faces[:nb_faces]
		self._surface_mesh = pv.PolyData(vertices, faces)
		
		return self._surface_mesh



	def mesh_volume(self, layer_ratio, num_a, num_b):

		""" Meshes the volume of the arterial tree with O-grid pattern.

		Keyword arguments:

		layer_ratio -- radius ratio of the three O-grid parts [a, b, c] such as a+b+c = 1
		num_a, num_b -- number of layers in the parts a and b
		"""

		if self._crsec_graph is None:

			print('Computing cross sections with default parameters...')
			self.compute_cross_sections(24, 0.2, False) # Get cross section graph

		G = self._crsec_graph
		
		if self._N%8 != 0:
			raise ValueError('The number of cross section nodes must be a multiple of 8 for volume meshing.')

		print('Meshing volume...')

		# Compute node ids
		nx.set_node_attributes(G, None, name='id_volume')
		count = 0
		nb_nds_ogrid = int(self._N * (num_a + num_b + 3) + ((self._N - 4)/4)**2)
		nb_nds_bif_ogrid = int(nb_nds_ogrid +  (self._N/2 - 1) * (num_a + num_b + 3) + (self._N - 4)/4 * (((self._N - 4)/4 -1)/2))
	

		for n in G.nodes():
			G.nodes[n]['id_volume'] = count
			if G.nodes[n]['type'] == "bif":
				count +=  nb_nds_bif_ogrid
			else: 
				count += nb_nds_ogrid

		for e in G.edges():
			G.edges[e]['id_volume'] = count
			count += G.edges[e]['crsec'].shape[0] * nb_nds_ogrid

		# Compute faces
		f_ogrid = self.ogrid_pattern_faces(self._N, num_a, num_b)
		f_bif_ogrid = self.bif_ogrid_pattern_faces(self._N, num_a, num_b)


		# Add vertices and cells to the mesh
		vertices = np.zeros((count,3))
		cells = np.zeros((count,9), dtype=int)
		nb_cells = 0

		for e in G.edges():

			# Mesh the first cross section
			id_first = G.nodes[e[0]]['id_volume']
			id_edge = G.edges[e]['id_volume']
			id_last = G.nodes[e[1]]['id_volume']

			# Add vertices
			# First cross section
			crsec = G.nodes[e[0]]['crsec']
			center = G.nodes[e[0]]['coords'][:-1]
			v = self.ogrid_pattern_vertices(center, crsec, layer_ratio, num_a, num_b)

			vertices[id_first:id_first + nb_nds_ogrid, :] = v
			cells[nb_cells: nb_cells + f_ogrid.shape[0], :] = np.hstack((np.zeros((f_ogrid.shape[0], 1)) + 8, id_first + f_ogrid[:,1:], id_edge + f_ogrid[:,1:]))
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
				nb_cells += f_ogrid.shape[0]

			# Last cross section
			if G.nodes[e[1]]['type'] == "bif": # Bifurcation case 

				crsec = G.nodes[e[1]]['crsec']
				center = (np.array(crsec[0]) + crsec[1])/2.0

				# Mesh of the 3 half-sections
		
				v = self.bif_ogrid_pattern_vertices(center, np.array(crsec), layer_ratio, num_a, num_b)	
				vertices[id_last:id_last + nb_nds_bif_ogrid, :] = v

				# Get bifurcation half section id and orientation from connection information
				connect = G.edges[e]['connect']

				h = []
				for s in [1, int(self._N/2) + 1]:

					if connect[s] <= self._N/2:
						h1 = [0]
					elif connect[s] >= self._N:
						h1 = [2]
					else: 
						h1 = [1]
					if connect[s] < connect[s+1]:
						h1 += [0, 1]
					else: 
						h1 += [1, 0]
					h.append(h1)

				f_ogrid_reorder = self.__reorder_faces(h, f_bif_ogrid, self._N, num_a, num_b)
				cells[nb_cells: nb_cells + f_ogrid_reorder.shape[0], :] = np.hstack((np.zeros((f_ogrid_reorder.shape[0], 1)) + 8, id_edge + (nb_nds_ogrid * (G.edges[e]['crsec'].shape[0] - 1)) + np.array(f_ogrid)[:,1:], id_last + np.array(f_ogrid_reorder)[:,1:]))
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
				nb_cells += f_ogrid.shape[0]


		cells = cells[:nb_cells]
		cell_types = np.array([vtk.VTK_HEXAHEDRON] * cells.shape[0])

		# Return volume mesh
		self._volume_mesh = pv.UnstructuredGrid(np.array([0, 9]), cells, cell_types, vertices)
		return self._volume_mesh



	def bif_ogrid_pattern(self, N, center, crsec, layer_ratio, num_a, num_b):


		""" Computes the nodes of a O-grid pattern from the bifurcation separation nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array 
		"""
		vertices = self.bif_ogrid_pattern_vertices(center, crsec, layer_ratio, num_a, num_b) 
		faces = self.bif_ogrid_pattern_faces(N, num_a, num_b)


		return vertices, faces


	def bif_ogrid_pattern_faces(self, N, num_a, num_b):


		""" Computes the nodes of a O-grid pattern from the bifurcation separation nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array 
		"""

		# Get the suface nodes of each individual half section
		Nh = int((N - 2) / 2) 
		
		#nb = int((N * (num_a + num_b + 3) + ((N - 4)/4)**2)/ 2)*3
		nb = int(N/8) * (2 * (num_a + num_b + 2) + int(N/8))
		faces = np.zeros((3, 2, nb, 5), dtype = int)
			
		shared_edge2 = []
		count = 0

		for h in range(3):

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


	def bif_ogrid_pattern_vertices(self, center, crsec, layer_ratio, num_a, num_b):


		""" Computes the nodes of a O-grid pattern from the bifurcation separation nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array 
		"""


		if sum(layer_ratio) != 1.0:
			raise ValueError("The sum of the layer ratios must equal 1.")
			
		# Get the suface nodes of each individual half section
		Nh = int((crsec.shape[0] - 2) / 3)
		N = Nh * 2 + 2

		nb_nds_ogrid = int(N * (num_a + num_b + 3) + ((N - 4)/4)**2)
		nb_vertices = int(nb_nds_ogrid +  (N/2 - 1) * (num_a + num_b + 3) + (N - 4)/4 * (((N - 4)/4 -1)/2))

		half_crsec =  np.zeros((3, Nh + 2, 3))
		
		half_crsec[:, 0, :] = [crsec[0]] * 3
		half_crsec[:, -1, :] = [crsec[1]] * 3
		half_crsec[0, 1 : Nh + 1, :] = crsec[2:Nh + 2]
		half_crsec[1, 1 : Nh + 1, :] = crsec[Nh + 2:(Nh * 2) + 2]
		half_crsec[2, 1 : Nh + 1, :] = crsec[(Nh*2) + 2:(Nh * 3) + 2]


		vertices = np.zeros((nb_vertices, 3))
		count = 0
		for h in range(3):
		
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



	def ogrid_pattern(self, N, center, crsec, layer_ratio, num_a, num_b):

		""" Computes the nodes of a O-grid pattern from the cross section surface nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array
		layer_ratio, num_a, num_b -- parameters of the O-grid
		"""

		if sum(layer_ratio) != 1.0:
			raise ValueError("The sum of the layer ratios must equal 1.")
		
		vertices = self.ogrid_pattern_vertices(center, crsec, layer_ratio, num_a, num_b)
		faces = self.ogrid_pattern_faces(N, num_a, num_b)

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

	
	def topo_to_full(self):

		""" Converts topo_graph to a full graph.""" 
			
		G = nx.DiGraph()
		k = 1

		ndict = {}
		for n in self._topo_graph.nodes():
			G.add_node(k, coords = self._topo_graph.nodes[n]['coords'])
			ndict[n] = k
			k  = k + 1

		for e in self._topo_graph.edges():
			pts = self._topo_graph.edges[e]['coords']

			if len(pts) == 0:

				G.add_edge(ndict[e[0]], ndict[e[1]], coords = np.array([]).reshape(0,4))

			else: 

				G.add_node(k, coords = pts[0])
				G.add_edge(ndict[e[0]], k, coords = np.array([]).reshape(0,4))
				k = k + 1

				for i in range(1, len(pts)):

					G.add_node(k, coords = pts[i])
					G.add_edge(k - 1, k, coords = np.array([]).reshape(0,4))
					k = k + 1

				G.add_edge(k - 1, ndict[e[1]], coords = np.array([]).reshape(0,4))
	
		return G




	def spline_to_full(self):

		""" Converts spline_graph to a full graph."""

		G = nx.DiGraph()
		k = 1

		ndict = {}
		for n in self._spline_graph.nodes():
			G.add_node(k, coords = self._spline_graph.nodes[n]['coords'])
			ndict[n] = k
			k = k + 1

		for e in self._spline_graph.edges():
			spl = self._spline_graph.edges[e]['spline']

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

		return G


	#####################################
	############  OPERATIONS  ###########
	#####################################

	def deform(self, mesh):

		""" Deforms the original mesh to match a given surface mesh. 
		Overwrite the cross section graph.

		Keywords arguments: 
		mesh -- a surface mesh in vtk format
		"""


		for e in self._crsec_graph.edges():
 
			# Get the start cross section only if it is a terminating edge
			if self._crsec_graph.in_degree(e[0]) == 0:

				crsec = self._crsec_graph.nodes[e[0]]['crsec']
				center = (crsec[0] + crsec[int(crsec.shape[0]/2)])/2.0 # Compute the center of the section
			
				new_crsec = np.zeros([crsec.shape[0], 3])
				for i in range(crsec.shape[0]):
					new_crsec[i] = self.__intersection(mesh, center, crsec[i])
				self._crsec_graph.nodes[e[0]]['crsec'] = new_crsec

			# Get the connection cross sections
			crsec_array = self._crsec_graph.edges[e]['crsec']  # In edges, the sections are stored as an array of cross section arrays
			new_crsec = np.zeros([crsec_array.shape[0], crsec_array.shape[1], 3])

			for i in range(crsec_array.shape[0]):
					crsec = crsec_array[i] # Get indivisual cross section
					center = (crsec[0] + crsec[int(crsec.shape[0]/2)])/2.0
			
					for j in range(crsec.shape[0]):
						new_crsec[i, j, :] = self.__intersection(mesh, center, crsec[j])

			self._crsec_graph.edges[e]['crsec'] = new_crsec

			# Get the end cross section
			crsec = self._crsec_graph.nodes[e[1]]['crsec']
			if self._crsec_graph.nodes[e[1]]['type'] == "bif": # If bifurcation
				center = self._crsec_graph.nodes[e[1]]['coords']
			else:
				center = (crsec[0] + crsec[int(crsec.shape[0]/2)])/2.0

			new_crsec = np.zeros([crsec.shape[0], 3])

			for i in range(crsec.shape[0]):
				new_crsec[i, :] = self.__intersection(mesh, center, crsec[i])

			self._crsec_graph.nodes[e[1]]['crsec'] = new_crsec


	def __intersection(self, mesh, center, coord):

		search_dist = 2
		inter = coord
		points = []
		while len(points) == 0 and search_dist < 40:

			normal = coord - center
			normal = normal / norm(normal) # Normal=direction of the projection 
			p2 = center + normal * search_dist
			points, ind = mesh.ray_trace(center, p2)

			if len(points) > 0 :
				inter = points[0]
			else :
				search_dist += 1

		return inter


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
		




	#####################################
	#############  ANALYSIS  ############
	#####################################


	def count_nodes(self):

		""" Returns the number of nodes of each type in the graph.
		"""

		count = {'reg' : 0, 'bif' : 0, 'end' : 0}


		for n in self._full_graph.nodes():

			if self._full_graph.in_degree(n) == 1 and self._full_graph.out_degree(n) == 1:
				count['reg'] += 1
			elif self._topo_graph.out_degree(n) == 0:
				count['end'] += 1
			else:
				count['bif'] += 1

		return count




	def deteriorate_centerline(self, p, noise):

		""" Add noise and resample the nodes of the initial centerlines.

		Keyword arguments:
		p -- percentage of nodes to keep [0, 1]
		noise -- list of std deviations of the gaussian noise (mm) = 1 value for each dimension
		"""

		for e in self._topo_graph.edges():

			pts = self._topo_graph.edges[e]['coords']

			if p != 0:
				# Resampling
				step = int(pts.shape[0]/(p*pts.shape[0]))

				if step > 0:
					pts =  pts[:-1:step]
				else:
					pts = pts[int(pts.shape[0]/2)]
			else: 
				pts = pts[int(pts.shape[0]/2)]


			rand = np.hstack((np.random.normal(0, noise[0], (pts.shape[0], 1)), np.random.normal(0, noise[1], (pts.shape[0], 1)), np.random.normal(0, noise[2], (pts.shape[0], 1)), np.random.normal(0, noise[3], (pts.shape[0], 1))))
			pts += pts * rand

			# Modify topo graph 
			self._topo_graph.add_edge(e[0], e[1], coords = pts)

			# Change full graph
			self._full_graph = self.topo_to_full()



	def distance(self, ref_mesh, display=True):

		"""Compute mean node to node distance between the computed mesh and a reference mesh
		and display the distance map.

		Keyword arguments:
		ref_mesh -- path to the vtk file of the reference mesh
		display -- True to display the distance map
		"""

		if self._surface_mesh is None:
			raise AttributeError('Please perform meshing first.')

		ref = pv.read(ref_mesh)
		m = self._surface_mesh

		tree = KDTree(ref.points)
		d, idx = tree.query(m.points)

		tree2 = KDTree(m.points)
		d2, idx2 = tree2.query(ref.points)


		for i in range(len(idx2)):
			d[idx2[i]] = (d[idx2[i]] + d2[i]) / 2.0

		if display:
			m["distances"] = d
			p = pv.Plotter()
			p.add_mesh(m, scalars="distances")
			p.add_mesh(ref, color=True, opacity=0.25) # Reference mesh
			p.show()

		return np.mean(d)


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
			colors = {"end":'blue', "bif":'green'}


			if not points and not centerline and not control_points:
				
				# Display topological nodes
				for key, value in pos.items():
					ax.scatter(value[0], value[1], value[2], c=colors[self._topo_graph.nodes[key]['type']], depthshade=False, s=40)

				# Plot connecting lines
				for i,j in enumerate(self._topo_graph.edges()):
					ax.plot(np.array((pos[j[0]][0], pos[j[1]][0])), np.array((pos[j[0]][1], pos[j[1]][1])), np.array((pos[j[0]][2], pos[j[1]][2])), c='black', alpha=0.5)

			if points: 
				# Display topological nodes
				for key, value in pos.items():
					ax.scatter(value[0], value[1], value[2], c=colors[self._topo_graph.nodes[key]['type']], depthshade=False, s=40)

		
				coords = nx.get_edge_attributes(self._topo_graph, 'coords')
				# Display edge nodes
				for key, value in coords.items():
					if value.shape[0]!=0:
						ax.scatter(value[:,0], value[:,1], value[:,2], c='red', depthshade=False, s= 40)

			if centerline:

				if self._spline_graph is None:
					print('Please perform spline approximation first to display centerlines.')
				else: 
					spl = nx.get_edge_attributes(self._spline_graph, 'spline')

					for key, value in spl.items():
						points = value.get_points()
						ax.plot(points[:,0], points[:,1], points[:,2],  c='black')

					if control_points:
						for key, value in spl.items():
							points = value.get_control_points()
							ax.scatter(points[:,0], points[:,1], points[:,2],  c='grey', s=40)
			
			# Set the initial view
			ax.view_init(90, -90) # 0 is the initial angle

			# Hide the axes
			ax.set_axis_off()
			plt.show()


	def show_crsec_graph(self):

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

		
			pos = nx.get_node_attributes(self._crsec_graph, 'coords')
			colors = {"end":'blue', "bif":'green'}
				
			# Display topological nodes
			for key, value in pos.items():
				ax.scatter(value[0], value[1], value[2], c='red', depthshade=False, s=40)

			spl = nx.get_edge_attributes(self._crsec_graph, 'spline')

			for key, value in spl.items():
				points = value.get_points()
				ax.plot(points[:,0], points[:,1], points[:,2],  c='black')

			
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



	def __vtk_to_graph(self, filename):


		""" Converts vmtk centerline to a graph.

		Keyword arguments:
		filename -- path to vmtk .vtk or .vtp centerline file
		"""
		G = nx.DiGraph()
		c = pv.read(filename)

		pts = c.points.tolist()
		radius = c.point_arrays['MaximumInscribedSphereRadius']

		# Store the different centerlines
		CL = []
		p0 = pts[0] + [radius[0]]
		cl = [p0]
		
		for i in range(1,len(pts)):
			p = pts[i] + [radius[i]]

			if i>0 and norm(np.array(p) - np.array(pts[i-1] + [radius[i-1]])) > 20: #p == p0:
				CL.append(cl)
				cl = []
			else:
				cl.append(p) 

		CL.append(cl)

		# Write nodes and edges
		pts_mem = []

		# Write first centerline
		k = 0
		for pt in CL[0]:
			G.add_node(k, coords=pt)

			if k > 0:
				G.add_edge(k-1, k, coords = [])

			pts_mem.append(pt)
			k = k + 1

		# Write other centerlines
		for i in range(1, len(CL)):
			for j in range(len(CL[i])):
				p = CL[i][j]

				newcl = True
				for pts in pts_mem:
					if norm(np.array(p) - np.array(pts)) < 1:
						newcl = False

				if newcl:
					j = j - 10
					p = CL[i][j]
					break
					

			# Find closest point from p and write edge
			min_d = 1000
			for g in range(len(pts_mem)):
				d = norm(np.array(p)[:-1] - np.array(pts_mem[g])[:-1])
				if d < min_d:
					min_p = g
					min_d = d
				
			G.add_node(k, coords=p)
			pts_mem.append(p)
			G.add_edge(min_p, k, coords = [])
			k = k + 1

			# Write the rest of the centerline
			for h in range(j+1, len(CL[i])):
				p = CL[i][h]
				G.add_node(k, coords=p)
				pts_mem.append(p)
				G.add_edge(k-1, k, coords = [])
				k = k + 1	

		return G


	#####################################
	##############  WRITE  ##############
	#####################################


	def write_vtk(self, type, filename):

		""" Writes centerlines as VTK polyline file.

		Keywords arguments: 
		type -- "full" or "spline"
		"""

		if type == "full":
			G = self._full_graph

		elif type == "spline":
			G = self.spline_to_full()


		v = np.zeros((G.number_of_nodes(), 3)) # Vertices
		f = [] # Connections
		r = [] # Radius

		for p in G.nodes():

			v[p-1, :] = G.nodes[p]['coords'][:-1]
			r.append(G.nodes[p]['coords'][-1])

			if G.in_degree(p) == 1:

				n = list(G.predecessors(p))
				f.append([2, p-1, n[0]-1])

		# Create VTK polyLine 
		poly = pv.PolyData()
		poly.points = v
		poly.lines = np.array(f)

		# Add radius information
		poly["scalars"] = np.array(r)
		poly_tube = poly.tube(radius = 0.6)

		poly_tube.save(filename)




	def write_swc(self, filename):

		""" Write swc Neurite Tracer file using depth fist search."""

		print('Writing swc file...')

		if filename[-4:] != ".swc":
			filename = filename + "swc"		

		file = open(filename, 'w') 

		keys = list(nx.dfs_preorder_nodes(self._full_graph , 1))
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


