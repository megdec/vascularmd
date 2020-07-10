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

		else:
			self.__load_file(filename)
			self.__set_topo_graph()
			self._spline_graph = None




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
		if self._topo_graph is not None:
			return self._spline_graph
		else:
			raise AttributeError('Please perform spline approximation first.')

	def get_surface_mesh(self):
		if self._surface_mesh is not None:
			return self._surface_mesh
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


	def set_topo_graph(self, G):

		""" Set the full graph of the arterial tree."""

		self._topo_graph = G
		self._full_graph = self.topo_to_full()
		self._spline_graph = None


	def set_spline_graph(self, G):

		""" Set the full graph of the arterial tree."""

		self._spline_graph = G
		self._full_graph = self.spline_to_full()
		self.__set_topo_graph()




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


		for n in self._full_graph.nodes():

			# If regular nodes
			if self._full_graph.in_degree(n) == 1 and self._full_graph.out_degree(n) == 1:

				# Create tables of regular nodes data
				coords = list(self._topo_graph.in_edges(n, data=True))[0][2]['coords']
				coords += [self._topo_graph.nodes[n]['coords']]
				coords += list(self._topo_graph.out_edges(n, data=True))[0][2]['coords'] 

				# Create new edge by merging the 2 edges of regular point
				self._topo_graph.add_edge(list(self._topo_graph.predecessors(n))[0], list(self._topo_graph.successors(n))[0], coords = coords)

				# Remove regular point
				self._topo_graph.remove_node(n)

			else : 

				# Add node type attribute
				if self._topo_graph.out_degree(n) == 0 or self._topo_graph.in_degree(n) == 0:
					self._topo_graph.add_node(n, coords = self._topo_graph.nodes[n]['coords'], type = "end")
				else: 
					self._topo_graph.add_node(n, coords = self._topo_graph.nodes[n]['coords'], type = "bif")

		# Relabel nodes
		self._topo_graph = nx.convert_node_labels_to_integers(self._topo_graph, first_label=1, ordering='default', label_attribute=None)



	#####################################
	##########  APPROXIMATION  ##########
	#####################################

	def spline_approximation(self):

		""" Approximate centerlines using splines and sets the attribute spline-graph.
		"""

		p = 3 # Spline order
		G = self.__bifurcation_point_estimation(p)
		self._spline_graph = nx.DiGraph()

		for e in G.edges():

			pts = [G.nodes[e[0]]['coords']] + G.edges[e]['coords'] +  [G.nodes[e[1]]['coords']]
			
			while len(pts) < p:
				# Linearly interpolate data
				pts = linear_interpolation(pts, 1)

			clip = [[], []]
			deriv = [[], []]

			if G.nodes[e[0]]['type'] == "bif":
				clip[0] = pts[0]
				deriv[0] = G.nodes[e[0]]['tangent']
 
			if G.nodes[e[1]]['type'] == "bif":
				clip[1] =  pts[-1]
				deriv[1] = G.nodes[e[1]]['tangent']

			spl = Spline()
			spl.curvature_bounded_approximation(pts, 1, clip, deriv) 
			#spl.show(False, False, data = pts)

			# Add edges with spline attributes
			self._spline_graph.add_node(e[0], coords = spl.point(0.0, True), type = G.nodes[e[0]]['type'])
			self._spline_graph.add_node(e[1], coords = spl.point(1.0, True), type = G.nodes[e[1]]['type'])
			self._spline_graph.add_edge(e[0], e[1], spline = spl, coords = G.edges[e]['coords']) 



	def __bifurcation_point_estimation(self, p):

		""" Approximates the coordinates and tangent of the bifurcation points.

		Keyword arguments:
		p -- spline order
		"""

		G = self._topo_graph.copy()

		for n in G.nodes():
			if G.nodes[n]['type'] == "bif":

				B = G.nodes[n]['coords'][:-1]
				
				e0 = list(G.in_edges(n))[0]
				e1 = list(G.out_edges(n))[0]
				e2 = list(G.out_edges(n))[1]
				pts0 = G.edges[e0]['coords']
				pts1 = G.edges[e1]['coords']
				pts2 = G.edges[e2]['coords']

				# Fit splines from the main branch to the daughter branches
				spl1 = Spline()
				spl1.curvature_bounded_approximation(pts0 + pts1, 1) 

				spl2 = Spline()
				spl2.curvature_bounded_approximation(pts0 + pts2, 1) 
		
			
				# Find the separation point between the splines
				r = np.mean(np.array(pts0)[:,-1])

				t1 = spl1.project_point_to_centerline(B) 
				t2 = spl2.project_point_to_centerline(B)

				t1 = spl1.length_to_time(spl1.time_to_length(t1) - 2*r)
				t2 =  spl2.length_to_time(spl2.time_to_length(t2) - 2*r)	

				# Get bifurcation coordinates and tangent
				tg = (spl1.tangent(t1, True) + spl2.tangent(t2, True)) /2.0
				pt = (spl1.point(t1, True) + spl2.point(t2, True)) / 2.0

				# Remove the points of the main branch further than the separation point
				
				for i in range(len(pts0)):
					t = spl1.project_point_to_centerline(pts0[-1][:-1]) 
			
					if t > t1:

						pts1 = [pts0[-1]] + pts1
						pts2 = [pts0[-1]] + pts2
						pts0 = pts0[:-1]
						
					else: break	
			
				# Update bifurcation point information
				G.add_node(n, coords = pt.tolist(), tangent = tg.tolist(), type = "bif")

				G.add_edge(*e0, coords = pts0)
				G.add_edge(*e1, coords = pts1)
				G.add_edge(*e2, coords = pts2)

		return G




	#####################################
	#########  MESHING METHODS  #########
	#####################################


	def __cross_sections_graph(self, N, d, bifurcation_model = True):

		""" Splits the splines into segments and bifurcation parts and computes surface cross sections.

		Keyword arguments:

		N -- number of nodes in a transverse section (multiple of 4)
		d -- longitudinal density of nodes as a proportion of the radius
		"""

		if self._spline_graph is None:
			raise AttributeError('Please perform spline approximation first.')

		G = self._spline_graph.copy()
		nmax = max(list(G.nodes())) + 1
		count = 0
		
		# Bifurcation cross sections
		for n in self._spline_graph.nodes():

			if self._spline_graph.nodes[n]['type'] == "bif":				

				spl0 =  self._spline_graph.edges[list(self._spline_graph.in_edges(n))[0]]['spline']

				S = []
				spl_bif = []
				for e in self._spline_graph.out_edges(n): # Out splines

					#Cut spline
					spl = self._spline_graph.edges[e]['spline']
					splb, spls = spl.split_length((spl.mean_radius() * 2 + spl0.mean_radius() * 2.5))

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

				S0 = [spl.point(0.0, True), spl.tangent(0.0, True)]
			
				# Compute bifurcation object
				if bifurcation_model : 
					bif = Bifurcation(S0, S[0], S[1], 0.5)
				else: 
					bif = Bifurcation(S0, S[0], S[1], 0.5, spl = spl_bif)
					
				#bif.show(nodes = True)

				# Find cross sections
				end_crsec, bif_crsec, nds, connect_index = bif.cross_sections(N, d)
				B = bif.get_B()
				tspl = bif.get_tspl()

				
				# Add separation nodes with cross sections
				id_matrix, count = self.__id_nodes(end_crsec[0], count)
				G.add_node(n, coords = spl.point(0.0, True), type = "sep", crsec = end_crsec[0], id = id_matrix, id_crsec = None)

				id_matrix, count = self.__id_nodes(end_crsec[1], count)
				G.add_node(nmax - 2, coords = S[0][0], type = "sep", crsec = end_crsec[1], id = id_matrix, id_crsec = None)

				id_matrix, count = self.__id_nodes(end_crsec[2], count)
				G.add_node(nmax - 1, coords = S[1][0], type = "sep", crsec = end_crsec[2], id = id_matrix, id_crsec = None)

				# Add bifurcation node
				id_matrix, count = self.__id_nodes(bif_crsec, count)
				G.add_node(nmax, coords = B, type = "bif", crsec = bif_crsec, id = id_matrix, id_crsec = None)
				
				# Add bifurcation edges
				id_matrix, count = self.__id_nodes(nds[0], count)
				G.add_edge(n, nmax, spline = tspl[0], crsec = nds[0], connect = connect_index[0], id = id_matrix)

				id_matrix, count = self.__id_nodes(nds[1], count)
				G.add_edge(nmax - 2, nmax, spline = tspl[1], crsec = nds[1], connect = connect_index[1], id = id_matrix)

				id_matrix, count = self.__id_nodes(nds[2], count)
				G.add_edge(nmax - 1, nmax, spline = tspl[2], crsec = nds[2], connect = connect_index[2], id = id_matrix)
				
				nmax = nmax + 1

		self._crsec_graph = G
		self.show_crsec_graph()

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

					v0 = np.array(G.nodes[e[0]]['crsec'][0]) - np.array(G.nodes[e[0]]['coords'][:-1])

					# Find the closest symmetric vector for v1
					crsec1 = G.nodes[e[1]]['crsec']
					tg1 = spl.tangent(1.0)
					v01 = spl.transport_vector(v0, 0, 1)

					min_a = 5.0
					for i in [0, int(N/4), int(N/2), int(3* N/4)]: #range(N)

						v = np.array(crsec1[i]) - np.array(G.nodes[e[1]]['coords'][:-1])
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
						
						v0 = np.array(G.nodes[e[0]]['crsec'][0]) - np.array(G.nodes[e[0]]['coords'][:-1])
						crsec = self.__segment_crsec(spl, num, N, v0 = v0 / norm(v0))
						connect_index = np.arange(0, N).tolist()
					else: 

						v0 = np.array(G.nodes[e[1]]['crsec'][0]) - np.array(G.nodes[e[1]]['coords'][:-1])
						splr = spl.copy_reverse()
						crsecr = self.__segment_crsec(splr, num, N, v0 = v0 / norm(v0))[::-1]

						crsec = []
						for s in crsecr:
							crsec.append([s[0]] + s[1:][::-1])

						connect_index = np.arange(0, N).tolist()
					

				# Add cross sections
				id_matrix, count = self.__id_nodes(crsec[1:-1], count)
				G.add_edge(e[0], e[1], crsec = crsec[1:-1], spline = spl, connect = connect_index, id = id_matrix)

				if G.nodes[e[0]]['type'] == "end":
					id_matrix, count = self.__id_nodes(crsec[0], count)
					G.add_node(e[0], coords = G.nodes[e[0]]['coords'], crsec = crsec[0], type = "end", id = id_matrix, id_crsec = None)

				if G.nodes[e[1]]['type'] == "end":
					id_matrix, count = self.__id_nodes(crsec[-1], count)
					G.add_node(e[1], coords = G.nodes[e[1]]['coords'], crsec = crsec[-1], type = "end", id = id_matrix, id_crsec = None)
					

				self._crsec_graph = G

		return G, count



	def __segment_crsec(self, spl, num, N, v0 = [], alpha = None):

		""" Compute the cross section nodes along for a vessel segment.

		Keyword arguments:
		spl -- segment spline
		num -- number of cross sections
		N -- number of nodes in a cross section (multiple of 4)
		v0 -- reference vector
		alpha -- rotation angle
		"""

		crsec = []

		t = [0.0] + spl.resample_time(num) + [1.0]

		if len(v0) == 0:
			# Random initialisation of the reference vector
			v0 = cross(spl.tangent(0), np.array([0,0,1]))
		else:
			v0 = np.array(v0)
			

		if alpha!=None:
			# Get rotation angles
			theta = [0.0] + np.linspace(0.0, alpha, num).tolist() + [alpha]

		for i in range(num + 2):

			tg = spl.tangent(t[i])
			# Transports the reference vector to time t[i]
			v = spl.transport_vector(v0, 0, t[i]) 

			if alpha!=None: 
				# Rotation of the reference vector
				v = rotate_vector(v, tg, theta[i])

			crsec.append(self.__single_crsec(spl, t[i], v, N))

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

		nds = []
		for theta in angle_list:
			n = rotate_vector(v, tg, theta)
			nds.append(spl.project_time_to_surface(n, t).tolist())

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



	def mesh_surface(self, N, d, bifurcation_model = True):

		""" Meshes the arterial tree.

		Keyword arguments:

		N -- number of nodes in a transverse section (multiple of 4)
		d -- longitudinal density of nodes as a proportion of the radius
		"""

		if self._spline_graph is None:
			raise AttributeError('Please perform spline approximation first.')

		
		print('Computing cross sections...')
		G, count = self.__cross_sections_graph(N, d, bifurcation_model) # Get cross section graph

		print('Meshing surface...')

		vertices = np.zeros((count, 3))
		faces = []

		for e in G.edges():		

			# Meshing the edge cross sections
			for i in range(len(G.edges[e]['crsec'])-1):
				for j in range(N):

					# Append the vertex to vertices list
					id1 = G.edges[e]['id'][i][j]
					vertices[id1, :] =  G.edges[e]['crsec'][i][j]

					# Append face to face list (forward, counterclockwise)
					id2 = G.edges[e]['id'][i + 1][j]
					vertices[id2, :] =  G.edges[e]['crsec'][i + 1][j]

					if j == N - 1: 
						id3 = G.edges[e]['id'][i + 1][0]
						id4 = G.edges[e]['id'][i][0]
					else:
						id3 = G.edges[e]['id'][i + 1][j + 1]
						id4 = G.edges[e]['id'][i][j + 1]

					faces.append([4, id1, id2, id3, id4])


			# Meshing the connecting parts
			for l in range(N): # Starting cross section

				id1 = G.nodes[e[0]]['id'][l]
				vertices[id1, :] =  G.nodes[e[0]]['crsec'][l]


				# Add faces
				id2 = G.edges[e]['id'][0][l]

				if l == N - 1: 
					id3 = G.edges[e]['id'][0][0]
					id4 = G.nodes[e[0]]['id'][0]
				else:
					id3 = G.edges[e]['id'][0][l + 1]
					id4 = G.nodes[e[0]]['id'][l + 1]

				faces.append([4, id1, id2, id3, id4])


			for l in range(N): # Ending cross section 

				# Add faces
				id1 = G.edges[e]['id'][-1][l]
				id2 = G.nodes[e[1]]['id'][G.edges[e]['connect'][l]]


				if l == N - 1:

					id3 = G.nodes[e[1]]['id'][G.edges[e]['connect'][0]]
					id4 = G.edges[e]['id'][-1][0]
					vertices[id3, :] =  G.nodes[e[1]]['crsec'][G.edges[e]['connect'][0]]


				else:

					id3 = G.nodes[e[1]]['id'][G.edges[e]['connect'][l + 1]]
					id4 = G.edges[e]['id'][-1][l + 1]
					vertices[id3, :] =  G.nodes[e[1]]['crsec'][G.edges[e]['connect'][l + 1]]
			
				faces.append([4, id1, id2, id3, id4])

		self._surface_mesh = pv.PolyData(vertices, np.array(faces))

		self._surface_mesh.plot(show_edges=True)
		self._surface_mesh.save("Results/surf_mesh.vtk")
		



	def mesh_volume(self, N, d, layer_ratio, num_a, num_b, bifurcation_model=True):

		""" Meshes the inside of the 3D surface with 0-grid pattern."""

		if self._spline_graph is None:
			raise AttributeError('Please perform spline approximation first.')

		print('Computing cross sections...')

		G, count = self.__cross_sections_graph(N, d, bifurcation_model) # Get cross section graph
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
				v_prec, f_prec = self.ogrid_pattern((crsec[0] + crsec[int(N/2)])/2.0, crsec, layer_ratio, num_a, num_b)
			
				vertices += v_prec
				ind_dep = len(vertices) - len(v_prec)
				G.add_node(e[0], coords = G.nodes[e[0]]['coords'], crsec = G.nodes[e[0]]['crsec'], type = G.nodes[e[0]]['type'], id = G.nodes[e[0]]['id'], id_crsec = ind_dep)

			else: 
				ind_dep = G.nodes[e[0]]['id_crsec']

			seg_crsec = np.array(G.edges[e]['crsec'])

			# Mesh edges
			for i in range(len(seg_crsec)): 

				crsec = seg_crsec[i]
				center = (crsec[0] + crsec[int(N/2)])/2.0
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

					end_crsec = np.array(G.nodes[e[1]]['crsec'])
					center = (end_crsec[0] + end_crsec[int(N/2)])/2.0

					# Meshing the 3 halves
					v_act, f_act = self.ogrid_pattern(center, crsec, layer_ratio, num_a, num_b)
					ind_dep = len(vertices)

					G.add_node(e[1], coords = G.nodes[e[1]]['coords'], crsec = G.nodes[e[1]]['crsec'], type = G.nodes[e[1]]['type'], id = G.nodes[e[1]]['id'], id_crsec = ind_dep)

				else:
					ind_dep = G.nodes[e[1]]['id_crsec']
		

			# Rotated segment case
			else:

				if G.nodes[e[1]]['id_crsec'] is None:

					end_crsec = np.array(G.nodes[e[1]]['crsec'])
					center = (end_crsec[0] + end_crsec[int(N/2)])/2.0
					v_act, f_act = self.ogrid_pattern(center, end_crsec, layer_ratio, num_a, num_b)
					ind_dep = len(vertices)

					G.add_node(e[1], coords = G.nodes[e[1]]['coords'], crsec = G.nodes[e[1]]['crsec'], type = G.nodes[e[1]]['type'], id = G.nodes[e[1]]['id'], id_crsec = ind_dep)

				else:
					ind_dep = G.nodes[e[1]]['id_crsec']

				j = int([0, int(N/4), int(N/2), int(3* N/4)].index(G.edges[e]['connect'][0]) * len(f_act) / 4)

				for k in range(len(f_act)):

					if j == len(f_act):
						j = 0
			
					cells.append([8] + (np.array(f_prec[j])[1:] + len(vertices) - len(v_prec)).tolist() + (np.array(f_act[j])[1:] + ind_dep).tolist())
					cell_types += [vtk.VTK_HEXAHEDRON]

					j +=1

				vertices += v_act
		

		# Create volume mesh
		self._volume_mesh = pv.UnstructuredGrid(np.array([0, 9]), np.array(cells), np.array(cell_types), np.array(vertices))

		self._volume_mesh.plot(show_edges=True)
		self._volume_mesh.save("Results/hex_mesh.vtk")
		
		

	def ogrid_pattern(self, center, crsec, layer_ratio, num_a, num_b):

		""" Computes the nodes of a O-grid pattern from the cross section surface nodes.

		Keyword arguments: 
		center -- center point of the cross section as numpy array
		crsec -- list of cross section nodes as numpy array
		layer_ratio, num_a, num_b -- parameters of the O-grid
		"""

		if sum(layer_ratio) != 1.0:
			raise ValueError("The sum of the layer ratios must equal 1.")
			
		# Get the symmetric nodes of the pattern
		N = len(crsec)
		sym_nodes = np.array([0, int(N/8), int(N/4)])

		count = 0

		vertices = []
		nds = []

		for s in range(4):

			square_grid = []
			square_corners = []
			for n in sym_nodes:

				if n == N:
					n = 0
				v = crsec[n] - center
				pt = (center + v / norm(v) * (layer_ratio[2] * norm(v))).tolist()
				square_corners.append(pt)
				

			square_sides1 = [lin_interp(square_corners[0], square_corners[1], N/8+1), lin_interp(center, square_corners[2], N/8+1)]
			square_sides2 = [lin_interp(square_corners[0], center, N/8+1), lin_interp(square_corners[1], square_corners[2], N/8+1)]

			# First half points	
			j = 0
			for i in range(sym_nodes[0], sym_nodes[1]):
				
				v = square_sides1[0][j] - crsec[i]
				pb = crsec[i] + v / norm(v) * (layer_ratio[0] * norm(v))

				if s != 0 and i == sym_nodes[0]:

					ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][j], num_b + 2)
					vertices += ray_vertices
					nds.append(list(range(count, count + len(ray_vertices))) + square_edge)

					count += len(ray_vertices)
					
				elif s == 3 and i!= sym_nodes[0]:

					ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][j], num_b + 2)[:-1] + lin_interp(square_sides1[0][j], square_sides1[1][j], N/8 + 1)[:-1]
					
					vertices += ray_vertices
					nds.append(list(range(count, count + len(ray_vertices))) + [nds[0][-j - 1]])
				
					count += len(ray_vertices)

				else:
					ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][j], num_b + 2)[:-1] + lin_interp(square_sides1[0][j], square_sides1[1][j], N/8 + 1)
					vertices += ray_vertices
					nds.append(list(range(count, count + len(ray_vertices))))

					count += len(ray_vertices)
				
				square_grid.append(nds[-1][-int(N/8):])
				j+=1
		
			square_edge = np.array(square_grid)[::-1, -1].tolist()
			

			# Central points
			v = square_corners[1] - crsec[sym_nodes[1]]
			pb = crsec[sym_nodes[1]] + v / norm(v) * (layer_ratio[0] * norm(v))

			ray_vertices = lin_interp(crsec[sym_nodes[1]], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides1[0][-1], num_b + 2)
			vertices += ray_vertices 

			nds.append(list(range(count, count + len(ray_vertices))))
			count += len(ray_vertices)
			

			# Second half points
			j = 1
			for i in range(sym_nodes[1] + 1, sym_nodes[2]):

				v = square_sides2[1][j] - crsec[i]
				pb = crsec[i] + v / norm(v) * (layer_ratio[0] * norm(v))

				ray_vertices = lin_interp(crsec[i], pb, num_a + 2)[:-1] + lin_interp(pb, square_sides2[1][j], num_b + 2)
				vertices += ray_vertices

				nds.append(list(range(count, count + len(ray_vertices))) + np.array(square_grid)[::-1, j-1][0:1].tolist())
				count += len(ray_vertices)
				j += 1	

			sym_nodes = sym_nodes + int(N/4)


		# Get faces list
		faces = []
		joining = [int(N/8), int(N/8 + N/4), int(N/8 + 2*N/4), int(N/8 + 3*N/4)]
		
		for i in range(len(nds)):

			if (i not in joining) and (i+1 not in joining):
				for j in range(len(nds[i])-1):
					if i == N -1:
						faces.append([4, nds[i][j], nds[0][j], nds[0][j+1], nds[i][j+1]])
					else:
						faces.append([4, nds[i][j], nds[i+1][j], nds[i+1][j+1], nds[i][j+1]])

			if i in joining:
		
				l = len(nds[i])-1

				if i == N-1:
					k = 0
				else: 
					k = i+1
				
				for j in range(l):

					faces.append([4, nds[i-1][j], nds[i][j], nds[i][j+1], nds[i-1][j+1]])
					faces.append([4, nds[i][j], nds[k][j], nds[k][j+1], nds[i][j+1]])
					
				faces.append([4, nds[i][-1], nds[k][l], nds[k][l+1], nds[i-1][l]])
		
		#mesh = pv.PolyData(np.array(vertices), np.array(faces))
		#mesh.plot(show_edges=True)
		#mesh.save("Results/ogrid.ply")

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

				G.add_edge(e[0], e[1], coords = [], radius = [])

			else: 

				G.add_node(k, coords = pts[0])
				G.add_edge(e[0], k, coords = [])
				k = k + 1

				for i in range(1, len(pts)):

					G.add_node(k, coords = pts[i])
					G.add_edge(k - 1, k, coords = [])
					k = k + 1

				G.add_edge(k - 1, e[1], coords = [])
	
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




	def deteriorate_centerline(self, sample, noise):

		""" Add noise and resample the nodes of the initial centerlines.

		Keyword arguments:
		sample -- factor for resampling [0, 1]
		noise -- list of std deviations of the gaussian noise (mm) = 1 value for each dimension
		"""

		for e in self._topo_graph.edges():

			pts = self._topo_graph.edges[e]['coords']

			# Resampling
			pts = pts[:-1:int(len(pts)*sample)]
		
			# Adding noise
			for i in range(len(pts)): 
				for j in range(len(noise)):
					pts[i][j] = pts[i][j] + np.random.normal(0, noise[j], 1)[0]

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
					v = np.array(value)
					ax.scatter(v[:,0], v[:,1], v[:,2], c='red', depthshade=False, s= 40)

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
							#ax.plot(points[:,0], points[:,1], points[:,2],  c='grey')
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
			G.add_node(int(file[i, 0]), coords=[file[i, 2],  198 - file[i, 4] , 115.9394 + file[i, 3], file[i, 5]])

			if file[i, 6] >= 0:
				G.add_edge(int(file[i, 6]), int(file[i, 0]), coords = [])

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

