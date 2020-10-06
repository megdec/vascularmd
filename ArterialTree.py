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

		""" Set the full graph of the arterial tree."""

		self._topo_graph = G
		self._full_graph = self.topo_to_full()
		self._spline_graph = None
		self._crsec_graph = None


	def set_spline_graph(self, G):

		""" Set the full graph of the arterial tree."""

		self._spline_graph = G
		self._full_graph = self.spline_to_full()
		self.__set_topo_graph()
		self._crsec_graph = None




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
			print(n, self._topo_graph.in_degree(n), self._topo_graph.out_degree(n))
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

	def spline_approximation(self, dist):

		""" Approximate centerlines using splines and sets the attribute spline-graph.
		"""

		print('Modeling the network with splines...')

		G = self.__end_conditions(dist)
		self._spline_graph = nx.DiGraph()

		for e in G.edges():

			pts = np.vstack((G.nodes[e[0]]['coords'], G.edges[e]['coords'],G.nodes[e[1]]['coords']))
			
			while len(pts) < 3:
				# Linearly interpolate data
				pts = np.array(linear_interpolation(pts, 1))

			clip = [[], []]
			deriv = [[], []]

			if G.nodes[e[0]]['type'] == "bif":
				clip[0] = pts[0]
				deriv[0] = G.nodes[e[0]]['tangent']
 
			if G.nodes[e[1]]['type'] == "bif":
				clip[1] =  pts[-1]
				deriv[1] = G.nodes[e[1]]['tangent']

			spl = Spline()
			#spl.curvature_bounded_approximation(pts, 1, clip, deriv) 
			spl.distance_constraint_approximation(pts, dist, clip, deriv)

			if len(deriv[0]) != 0:
				print("alpha : ", ((spl.get_control_points()[1] - spl.get_control_points()[0]) / deriv[0])[0])

			if len(deriv[1]) != 0:
				print("beta : ", ((spl.get_control_points()[-2] - spl.get_control_points()[-1]) / deriv[1])[0])
			#spl.show(False, False, data = pts)

			# Add edges with spline attributes
			self._spline_graph.add_node(e[0], coords = spl.point(0.0, True), type = G.nodes[e[0]]['type'])
			self._spline_graph.add_node(e[1], coords = spl.point(1.0, True), type = G.nodes[e[1]]['type'])
			self._spline_graph.add_edge(e[0], e[1], spline = spl, coords = G.edges[e]['coords']) 



	def __end_conditions(self, dist):

		""" Approximates the coordinates and tangent of the bifurcation points.

		Keyword arguments:
		p -- spline order
		"""

		"""
		G = self._topo_graph.copy()

		for n in self._topo_graph.nodes():
			if self._topo_graph.nodes[n]['type'] == "bif":

				edges = list(self._topo_graph.in_edges(n)) + list(self._topo_graph.out_edges(n))

				best_pos = 1
				ratio_list = []
				mag_list = []
				for pos in [1.5, 1.8, 2, 2.2, 2.5, 3, 3.5]:
		
					pts, pt, tg = self.__bifurcation_point_estimation(dist, edges, n, pos)

					mag = []
					for i in range(3):
						if i == 0:
							clip = [[], pt]
							deriv = [[], tg]
						else: 
							clip = [pt, []]
							deriv = [tg, []]

						spl = Spline()
						spl.distance_constraint_approximation(pts[i], dist, clip, deriv)

						if len(deriv[0]) != 0:
							mag.append(abs(((spl.get_control_points()[1] - spl.get_control_points()[0]) / deriv[0])[0]))

						if len(deriv[1]) != 0:
							mag.append(abs(((spl.get_control_points()[-2] - spl.get_control_points()[-1]) / deriv[1])[0]))

					diff = min([mag[1]-mag[0], mag[2]-mag[0]])
					ratio_list.append(diff)
					mag_list.append(mag)

					if diff >= 0:
						best_ratio = diff
						best_pos = pos
						best_mag = mag
						break

				print(ratio_list, best_pos)
				
				pts, pt, tg = self.__bifurcation_point_estimation(dist, edges, n, best_pos)

				# Update bifurcation point information
				G.add_node(n, coords = pt, tangent = tg, type = "bif")

				G.add_edge(*edges[0], coords = pts[0])
				G.add_edge(*edges[1], coords = pts[1])
				G.add_edge(*edges[2], coords = pts[2])
		"""
		G = self._topo_graph.copy()

		for n in self._topo_graph.nodes():
			if self._topo_graph.nodes[n]['type'] == "bif":

				edges = list(self._topo_graph.in_edges(n)) + list(self._topo_graph.out_edges(n))
				pts, pt, tg = self.__bifurcation_point_estimation(dist, edges, n, 0)
			
				# Update bifurcation point information
				G.add_node(n, coords = pt, tangent = tg, type = "bif")

				G.add_edge(*edges[0], coords = pts[0])
				G.add_edge(*edges[1], coords = pts[1])
				G.add_edge(*edges[2], coords = pts[2])

		return G



	def __bifurcation_point_estimation(self, dist, edges, node, pos):

		B = self._topo_graph.nodes[node]['coords'][:-1]

		pts0 = self._topo_graph.edges[edges[0]]['coords']
		pts1 = self._topo_graph.edges[edges[1]]['coords']
		pts2 = self._topo_graph.edges[edges[2]]['coords']


		# Fit splines from the main branch to the daughter branches
		spl1 = Spline()
		#spl1.curvature_bounded_approximation(pts0 + pts1, 1) 
		spl1.distance_constraint_approximation(np.vstack((self._topo_graph.nodes[edges[0][0]]['coords'], pts0, pts1, self._topo_graph.nodes[edges[1][1]]['coords'])), dist)

		spl2 = Spline()
		#spl2.curvature_bounded_approximation(pts0 + pts2, 1) 
		spl2.distance_constraint_approximation(np.vstack((self._topo_graph.nodes[edges[0][0]]['coords'], pts0, pts2, self._topo_graph.nodes[edges[2][1]]['coords'])), dist)
		
			
		# Find the separation point between the splines
		r = np.mean(pts0[:,-1])

		t1 = spl1.project_point_to_centerline(B) 
		t2 = spl2.project_point_to_centerline(B)

		t1 = spl1.length_to_time(spl1.time_to_length(t1) - r)
		t2 =  spl2.length_to_time(spl2.time_to_length(t2) - r)	

		# Get bifurcation coordinates and tangent
		tg = (spl1.tangent(t1, True) + spl2.tangent(t2, True)) / 2.0 #(spl1.first_derivative(t1, True) + spl2.first_derivative(t2, True)) /2.0 
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
		print('Meshing bifurcations')
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
					#splb, spls = spl.split_length((spl.mean_radius() * 2 + spl0.mean_radius() * 2))

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
				
				# Find cross sections
				end_crsec, bif_crsec, nds, connect_index = bif.cross_sections(N, d)
				bif.show(nodes=True)
				B = bif.get_B()
				tspl = bif.get_tspl()

				# Add separation nodes with cross sections
				G.add_node(n, coords = spl.point(0.0, True), type = "sep", crsec = np.array(end_crsec[0]), id_crsec = None)
				G.add_node(nmax - 2, coords = S[0][0], type = "sep", crsec = np.array(end_crsec[1]), id_crsec = None)
				G.add_node(nmax - 1, coords = S[1][0], type = "sep", crsec = np.array(end_crsec[2]), id_crsec = None)

				# Add bifurcation node
				G.add_node(nmax, coords = B, type = "bif", crsec = bif_crsec, id_crsec = None)
				
				# Add bifurcation edges
				G.add_edge(n, nmax, spline = tspl[0], crsec = np.array(nds[0]), connect = connect_index[0])
				G.add_edge(nmax - 2, nmax, spline = tspl[1], crsec = np.array(nds[1]), connect = connect_index[1])
				G.add_edge(nmax - 1, nmax, spline = tspl[2], crsec = np.array(nds[2]), connect = connect_index[2])
				
				nmax = nmax + 1

		self._crsec_graph = G
		#self.show_crsec_graph()

		print('Meshing edges')
		# Connecting segments cross sections
		for e in G.edges():

			spl = G.edges[e]['spline']

			if G.nodes[e[0]]['type'] != "bif" and G.nodes[e[1]]['type'] != "bif": # Connecting segments
				
				# Number of cross sections
				num = int(spl.length() / (spl.mean_radius()* d))

				if num <= 1:
					num = 2

				# Simple tube
				if G.nodes[e[0]]['type'] == "end" and G.nodes[e[1]]['type'] == "end": 

					crsec = self.__segment_crsec(spl, num, N)
					connect_index = np.arange(0, N).tolist()

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
						a = directed_angle_negative(v01, v, tg1)

						if abs(a) < abs(min_a):
							min_a = a
							min_ind = i
					
					crsec = self.__segment_crsec(spl, num, N, v0 = v0 / norm(v0), alpha = min_a)
					
					connect_index = []
					for i in range(N):

						if min_ind  == N:
							min_ind  = 0

						connect_index.append(min_ind)
						min_ind  = min_ind  + 1
					

				# Ending tube 
				else :
					
					if G.nodes[e[1]]['type'] == "end":
						v0 = G.nodes[e[0]]['crsec'][0] - G.nodes[e[0]]['coords'][:-1]
						crsec = self.__segment_crsec(spl, num, N, v0 = v0 / norm(v0))
						connect_index = np.arange(0, N).tolist()

					else: 
						v0 = G.nodes[e[1]]['crsec'][0] - G.nodes[e[1]]['coords'][:-1]
						splr = spl.copy_reverse()
						crsec = self.__segment_crsec(splr, num, N, v0 = v0 / norm(v0))
						crsec = crsec[::-1, np.hstack((0, np.arange(N - 1,0,-1))), :]

						connect_index = np.arange(0, N).tolist()

				# Add cross sections
				G.add_edge(e[0], e[1], crsec = crsec[1:-1], spline = spl, connect = connect_index)

				
				if G.nodes[e[0]]['type'] == "end":
					G.add_node(e[0], coords = G.nodes[e[0]]['coords'], crsec = crsec[0], type = "end", id_crsec = None)

				if G.nodes[e[1]]['type'] == "end":
					G.add_node(e[1], coords = G.nodes[e[1]]['coords'], crsec = crsec[-1], type = "end", id_crsec = None)
				
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

		# Retirer l'attribut "count" du graphe
		vertices = []
		faces = []

		for e in G.edges():		

			# Mesh the first cross section
			if G.nodes[e[0]]['id_crsec'] is None:

				v_prec = G.nodes[e[0]]['crsec']
				ind_dep = len(vertices) 

				vertices += list(v_prec)
				G.nodes[e[0]]['id_crsec'] = ind_dep 

			else: 
				ind_dep = G.nodes[e[0]]['id_crsec']

			seg_crsec = G.edges[e]['crsec']

			# Mesh edges
			for i in range(len(seg_crsec)): 

				v_act = seg_crsec[i]
				for j in range(len(v_act)):

					if j == len(v_act) - 1:
						j2 = 0
					else:
						j2 = j + 1
					faces.append([4, ind_dep + j, len(vertices) + j,  len(vertices) + j2, ind_dep + j2])	

				vertices += list(v_act)
				v_prec = v_act
	
				ind_dep = len(vertices) - len(v_prec)

			# Mesh the last cross section
			
			if G.nodes[e[1]]['id_crsec'] is None:

				v_act = G.nodes[e[1]]['crsec']
	
				ind_dep = len(vertices)
				ind_prec = len(vertices) - len(v_prec)

				G.nodes[e[1]]['id_crsec'] = ind_dep
				vertices += list(v_act)

			else:
				ind_dep = G.nodes[e[1]]['id_crsec']
				ind_prec = len(vertices) - len(v_prec)
			
			# Get bifurcation half section id and orientation from connection information
			connect = G.edges[e]['connect']
								
			# Add faces
			for j in range(len(v_prec)): 
				if j == len(v_prec) - 1:
					j2 = 0
					j3 = connect[0]
				else:
					j2 = j + 1
					j3 = connect[j+1] 
			
				faces.append([4, ind_prec + j, ind_dep + connect[j],  ind_dep + j3, ind_prec + j2])
			
		# Effacer les id
		nx.set_node_attributes(G, None, 'id_crsec')
		
		return pv.PolyData(np.array(vertices), np.array(faces))



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

		vertices = []
		cells = []
		cell_types = []

		# Store the vertices and change the id of every cross section
		for e in G.edges():

			# Mesh the first cross section
			if G.nodes[e[0]]['id_crsec'] is None:

				crsec = np.array(G.nodes[e[0]]['crsec'])
				center = np.array(G.nodes[e[0]]['coords'])[:-1]
				v_prec, f_prec = self.ogrid_pattern((crsec[0] + crsec[int(self._N/2)])/2.0, crsec, layer_ratio, num_a, num_b)
			
				vertices += v_prec
				ind_dep = len(vertices) - len(v_prec)
				G.nodes[e[0]]['id_crsec'] = ind_dep
			

			else: 
				ind_dep = G.nodes[e[0]]['id_crsec']

			seg_crsec = np.array(G.edges[e]['crsec'])

			# Mesh edges
			for i in range(len(seg_crsec)): 

				crsec = seg_crsec[i]
				center = (crsec[0] + crsec[int(self._N/2)])/2.0
				v_act, f_act = self.ogrid_pattern(center, crsec, layer_ratio, num_a, num_b)	
				
				
				for j in range(len(f_act)):
			
					cells.append([8] + (np.array(f_prec[j])[1:] + ind_dep).tolist() + (np.array(f_act[j])[1:] + len(vertices)).tolist())
					cell_types += [vtk.VTK_HEXAHEDRON]

				vertices += v_act

				v_prec = v_act
				f_prec = f_act

				ind_dep = len(vertices) - len(v_prec)

			# Mesh the last cross section
			if G.nodes[e[1]]['type'] == "bif": # Bifurcation case 
				if G.nodes[e[1]]['id_crsec'] is None:

					crsec = G.nodes[e[1]]['crsec']
					center = (np.array(crsec[0]) + crsec[1])/2.0

					# Mesh of the 3 half-sections
					v, f = self.bif_ogrid_pattern(center, crsec, layer_ratio, num_a, num_b)				
							
					ind_dep = len(vertices)
					G.nodes[e[1]]['id_crsec'] = ind_dep
				
					vertices += v
					ind_prec = len(vertices) - len(v) -len(v_prec)


				else:
					ind_dep = G.nodes[e[1]]['id_crsec']
					ind_prec = len(vertices) - len(v_prec)

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
				
				
				f_act = []
				for s in range(2): # Both half-sections
					
					quarter = f[h[s][0]][h[s][1]]
					for ray in quarter: # Iterate on the rays
						for face in ray: # Iterate on the faces
							# Get bifurcation faces
							f_act.append(face) #[half][quarter][ray][face]

					# Reorder second quarter 
					quarter = f[h[s][0]][h[s][2]][::-1]
					for i in range(len(quarter)): # Iterate on the rays

						if i < len(quarter) / 2: # Long rays
							for j in range(len(quarter[i])): # Iterate on the faces
								face = quarter[i][j]
								f_act.append([4, face[2] , face[1], face[4], face[3]])

							for k in range(int(self._N/8)):
								face = quarter[-int(self._N/8)+k][-i -1]
								f_act.append([4, face[3] , face[2], face[1], face[4]])

						else:

							for j in range(len(quarter[i]) - int(self._N/8)): # Iterate on the faces
								face = quarter[i][j]
								f_act.append([4, face[2] , face[1], face[4], face[3]])

								
				# Add cells
				for k in range(len(f_act)): 
				
					cells.append([8] + (np.array(f_prec[k])[1:] + ind_prec).tolist() + (np.array(f_act[k])[1:] + ind_dep).tolist())
					cell_types += [vtk.VTK_HEXAHEDRON]


			# Rotated segment case
			else:

				if G.nodes[e[1]]['id_crsec'] is None:

					end_crsec = np.array(G.nodes[e[1]]['crsec'])
					center = (end_crsec[0] + end_crsec[int(self._N/2)])/2.0
					v_act, f_act = self.ogrid_pattern(center, end_crsec, layer_ratio, num_a, num_b)
					ind_dep = len(vertices)
					G.nodes[e[1]]['id_crsec'] = ind_dep
				
				else:
					ind_dep = G.nodes[e[1]]['id_crsec']

				j = int([0, int(self._N/4), int(self._N/2), int(3* self._N/4)].index(G.edges[e]['connect'][0]) * len(f_act) / 4)

				for k in range(len(f_act)):

					if j == len(f_act):
						j = 0
			
					cells.append([8] + (np.array(f_prec[j])[1:] + len(vertices) - len(v_prec)).tolist() + (np.array(f_act[j])[1:] + ind_dep).tolist())
					cell_types += [vtk.VTK_HEXAHEDRON]

					j +=1

				vertices += v_act
		
		# Return volume mesh
		return pv.UnstructuredGrid(np.array([0, 9]), np.array(cells), np.array(cell_types), np.array(vertices))




	def bif_ogrid_pattern(self, center, crsec, layer_ratio, num_a, num_b):


		""" Computes the nodes of a O-grid pattern from the bifurcation separation nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array 
		"""


		if sum(layer_ratio) != 1.0:
			raise ValueError("The sum of the layer ratios must equal 1.")
			
		# Get the suface nodes of each individual half section
		half_crsec =  []
		N = int((len(crsec) - 2) / 3)
		
		half_crsec.append([crsec[0]] + crsec[2:N + 2] + [crsec[1]])
		half_crsec.append([crsec[0]] + crsec[N + 2:(N * 2) + 2] + [crsec[1]])
		half_crsec.append([crsec[0]] + crsec[(N*2) + 2:(N * 3) + 2] + [crsec[1]])

		shared_edge2 = []
		nds_full = []
		vertices = []
		count = 0

		N = N*2 + 2
		for h in range(len(half_crsec)):

			nds_half = []
		
			# Separate in quarters
			quarters = np.array([half_crsec[h][:int(N/4)+1], half_crsec[h][int(N/4):][::-1]])

			for q in range(len(quarters)):

				# Computes the coordinates of the corners and the side nodes of the central square
				square_corners = []
				for n in [0, int(N/8), int(N/4)]:

					v = quarters[q][n] - center
					pt = (center + v / norm(v) * (layer_ratio[2] * norm(v))).tolist()
					square_corners.append(pt)
					

				square_sides1 = [lin_interp(square_corners[0], square_corners[1], N/8+1), lin_interp(center, square_corners[2], N/8+1)] # Horizontal square edges
				square_sides2 = [lin_interp(square_corners[0], center, N/8+1), lin_interp(square_corners[1], square_corners[2], N/8+1)] # Vertical square edges

				if q == 0:
					# Point of the horizontal shared line
					v = square_corners[2] - quarters[0][-1] # Direction of the ray
					pb = quarters[0][-1] + v / norm(v) * (layer_ratio[0] * norm(v)) # Starting point of layer b

					ray_vertices = lin_interp(quarters[0][-1], pb, num_a + 2)[:-1] + lin_interp(pb, square_corners[2], num_b + 2)[:-1] + lin_interp(square_corners[2], center, N/8 + 1)
					shared_edge1 = list(range(count, count + len(ray_vertices)))

					vertices += ray_vertices
					count += len(ray_vertices)	

				nds_quarter = []
				for i in range(len(quarters[q])-1):
					
					# First half points	
					if i <= (len(quarters[q])-1) / 2: 
						
						v = square_sides1[0][i] - quarters[q][i] # Direction of the ray
						pb = quarters[q][i] + v / norm(v) * (layer_ratio[0] * norm(v)) # Starting point of layer b

						if i == 0:
							if h == 0:
								ray_vertices = lin_interp(quarters[q][i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][i], num_b + 2)[:-1] + lin_interp(square_sides1[0][i], square_sides1[1][i], N/8 + 1)[:-1]
								ray_ind = list(range(count, count + len(ray_vertices))) + [shared_edge1[-i-1]]
								shared_edge2.append(ray_ind)
							else: 
								ray_vertices = []
								ray_ind = shared_edge2[q] # Common edge of the bifurcation plan

						else:
							ray_vertices = lin_interp(quarters[q][i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][i], num_b + 2)[:-1] + lin_interp(square_sides1[0][i], square_sides1[1][i], N/8 + 1)[:-1]
							ray_ind = list(range(count, count + len(ray_vertices))) + [shared_edge1[-i-1]]

						if i == (len(quarters[q])-1) / 2:
							shared_edge3 = ray_ind[-int(N/8):]
							

					else: # Second half points
						
						v = square_sides2[1][i-int(N/8)] - quarters[q][i] # Direction of the ray
						pb = quarters[q][i] + v / norm(v) * (layer_ratio[0] * norm(v)) # Starting point of layer b

						ray_vertices = lin_interp(quarters[q][i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides2[1][i-int(N/8)], num_b + 2)[:-1]
						ray_ind = list(range(count, count + len(ray_vertices))) + [shared_edge3[i-int(N/8)-1]]

					nds_quarter.append(ray_ind)
					vertices += ray_vertices
					count += len(ray_vertices)

				nds_quarter.append(shared_edge1)
				nds_half.append(nds_quarter)
			nds_full.append(nds_half)

		# Get faces list
		faces = []

		for nds_half in nds_full:
			faces_half = []

			for nds in nds_half:
				faces_quarter = []

				for i in range(len(nds) - 1):
					faces_ray = []
					for j in range(min([len(nds[i])-1, len(nds[i+1])-1])):
						faces_ray.append([4, nds[i][j], nds[i+1][j], nds[i+1][j+1], nds[i][j+1]])
					faces_quarter.append(faces_ray)

				faces_half.append(faces_quarter)
			faces.append(faces_half)
 

		return vertices, faces




	def ogrid_pattern(self, center, crsec, layer_ratio, num_a, num_b):

		""" Computes the nodes of a O-grid pattern from the cross section surface nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array
		layer_ratio, num_a, num_b -- parameters of the O-grid
		"""

		if sum(layer_ratio) != 1.0:
			raise ValueError("The sum of the layer ratios must equal 1.")
		
		count = 0

		vertices = []
		nds = []

		N = len(crsec)
	
		# Get the symmetric nodes of the pattern
		sym_nodes = np.array([0, int(N/8), int(N/4)])

		shared_edge1 = []
		
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
						ray_ind = list(range(count, count + len(ray_vertices))) + shared_edge1[::-1] # Common nodes of the previous quarter
						shared_edge1 = [ray_ind[-1]]

					elif s==3:
						ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][j], num_b + 2)[:-1] + lin_interp(square_sides1[0][j], square_sides1[1][j], N/8 + 1)[:-1]
						ray_ind = list(range(count, count + len(ray_vertices))) + [nds[0][-j - 1]] # Common nodes to close the circle
					else:
						ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][j], num_b + 2)[:-1] + lin_interp(square_sides1[0][j], square_sides1[1][j], N/8 + 1)
						ray_ind = list(range(count, count + len(ray_vertices)))
						shared_edge1.append(ray_ind[-1]) # Store the indices of the shared nodes

					if j == int(N/8):
						shared_edge2 = ray_ind[-int(N/8):]
						

				else: 
					v = square_sides2[1][j-int(N/8)] - crsec[i] # Direction of the ray
					pb = crsec[i] + v / norm(v) * (layer_ratio[0] * norm(v)) # Starting point of layer b

					ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides2[1][j-int(N/8)], num_b + 2)[:-1]
					ray_ind = list(range(count, count + len(ray_vertices))) + [shared_edge2[j-int(N/8)-1]]

				vertices += ray_vertices
				nds.append(ray_ind)
				count += len(ray_vertices)

				j = j + 1

			sym_nodes = sym_nodes + int(N/4)
			


		# Get faces list
		faces = []
		for i in range(len(nds)):

			if i != N - 1:
				i2 = i+1
			else:
				i2 = 0

			for j in range(min([len(nds[i])-1, len(nds[i2])-1])):
					faces.append([4, nds[i][j], nds[i2][j], nds[i2][j+1], nds[i][j+1]])
 

		return vertices, faces




	#####################################
	###########  CONVERSIONS  ###########
	#####################################

	
	def topo_to_full(self):

		""" Converts topo_graph to a full graph.""" 
			
		G = nx.DiGraph()
		k = 1

		for n in self._topo_graph.nodes():
		
			G.add_node(k, coords = self._topo_graph.nodes[n]['coords'])
			k  = k + 1

		for e in self._topo_graph.edges():
			pts = self._topo_graph.edges[e]['coords']

			if len(pts) == 0:

				G.add_edge(1, 2, coords = np.array([]).reshape(0,4))

			else: 

				G.add_node(k, coords = pts[0])
				G.add_edge(1, k, coords = np.array([]).reshape(0,4))
				k = k + 1

				for i in range(1, len(pts)):

					G.add_node(k, coords = pts[i])
					G.add_edge(k - 1, k, coords = np.array([]).reshape(0,4))
					k = k + 1

				G.add_edge(k - 1, 2, coords = np.array([]).reshape(0,4))
	
		return G




	def spline_to_full(self):

		""" Converts spline_graph to a full graph."""

		G = nx.DiGraph()
		k = 1

		for n in self._spline_graph.nodes():
			G.add_node(k, coords = self._spline_graph.nodes[n]['coords'])
			k = k + 1

		for e in self._spline_graph.edges():
			spl = self._spline_graph.edges[e]['spline']

			if spl != None:

				t = spl.resample_time(int(spl.length() * 10))

				pts = []
				for elt in t:
					pts.add(spl.point(t).tolist())

				G.add_node(k, coords = pts[0])
				G.add_edge(e[0], k, coords = [])
				k = k + 1

				for i in range(1, len(pts)):

					G.add_node(k, coords = pts[i])
					G.add_edge(k - 1, k, coords = [])
					k = k + 1

				G.add_edge(k - 1, e[1], coords = [])

			else:

				G.add_edge(e[0], e[1], coords = [])

		return G


	#####################################
	############  OPERATIONS  ###########
	#####################################


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



	def distance_mesh(self, ref_mesh, display=True):

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

