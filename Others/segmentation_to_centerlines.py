import numpy as np
from numpy.linalg import norm 
import pyvista as pv
import pandas
import networkx as nx
import pickle
from xml.dom import minidom
from skimage.morphology import skeletonize
from skimage import data, io
import sknw
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

from math import pi, sin, cos, tan, atan, acos, asin, sqrt
from numpy.linalg import norm 
from numpy import dot, cross, arctan2



#### Input / Output ####


def swc_to_networkx(swc_file):

	""" Skeletonize using sknw package.
	Keywords :
	swc_file -- path to the swc file"""

	file = np.loadtxt(swc_file, skiprows=0)
	G = nx.Graph()

	for i in range(0, file.shape[0]):

		G.add_node(int(file[i, 0]), coords= np.array([file[i, 2],  file[i, 3] , file[i, 4], file[i, 5]]))

		if file[i, 6] >= 0:
			G.add_edge(int(file[i, 6]), int(file[i, 0]))

	return G


def load_nii(img_file):

	# Importing the binary nifty image

	nii_image  = nib.load(img_file)
	pix_dim = nii_image.header['pixdim'][1:4] # voxel size in mm
	offset = [nii_image.header['qoffset_x'], nii_image.header['qoffset_y'], nii_image.header['qoffset_z']]
	img = np.array(nii_image.dataobj, dtype = float) # Image as numpy array
	axis = [1,1,1]
	if nii_image.header['srow_x'][0] < 0:
		axis[0] = -1
	if nii_image.header['srow_y'][1] < 0:
		axis[1] = -1
	if nii_image.header['srow_y'][2] < 0:
		axis[2] = -1

	return img, pix_dim, offset, axis


def write_centerline_nii(G, dim, outfile):

	""" Write a centerline to an nifti file for visualization purpose
	Keyword arguments :
		G --  Networkx graph containing the centerline (cf xxx_to_networkx functions)
		dim -- dimensions of the nifti image [x,y,z] 
		outfile -- name of the out nifti file "img.nii"
	"""

	img = np.zeros(dim)
	for n in G.nodes():
		img[int(G.nodes[n]["coords"][0]), int(G.nodes[n]["coords"][1]), int(G.nodes[n]["coords"][2])] = 1
	# Write nifti
	new_img = nib.nifti1.Nifti1Image(img, None, header=None)
	print("Saving nifti")
	nib.save(new_img, outfile)

def extract_radius_from_image(G, img, pix_dim):

	"""Extract radius from binary image and add radius information to the graph G"""
	if len(img.shape) == 2:
		distance_map = distance_transform_edt(img, sampling = pix_dim[:2])
	else:
		distance_map = distance_transform_edt(img, sampling = pix_dim)

	
	#new_img = nib.nifti1.Nifti1Image(distance_map, None, header=None)
	#nib.save(new_img, "test_distance.nii")
	for n in G.nodes():
		if len(img.shape) == 2:
			radius = distance_map[int(G.nodes[n]["coords"][0]), int(G.nodes[n]["coords"][1])]
		else:
			radius = distance_map[int(G.nodes[n]["coords"][0]), int(G.nodes[n]["coords"][1]), int(G.nodes[n]["coords"][2])]
		G.nodes[n]['coords'][3] = radius
	return G


def sknw_to_networkx(img_file):

	""" Skeletonize a segmented image using sknw package.
	Keywords :
	img_file -- path to the image file"""

	if img_file[-3:] == "jpg":
		img = io.imread(img_file)
		img = img>100
	else:
		data = nib.load(img_file)
		img = np.array(data.dataobj)

	ske = skeletonize(img, method = "lee").astype(np.uint16)
	#ske_img = nib.Nifti1Image(ske, data.affine, data.header)
	#nib.save(ske_img, 'ske_image.nii')

	# build graph from skeleton
	G = sknw.build_sknw(ske)
	# Change topo graph to the common format
	nx.set_node_attributes(G, None, name='coords')

	D = sknw_topo_to_undirected_graph(G)

	return D

def voreen_to_networkx(node_file, edge_file, xml_file=None):

	''' Convert Voreen csv files to a networkx undirected graph.
	Keyword arguments :
	node_file -- path to the node CSV file
	edge_file -- path to the edge CSV file
	'''

	G = nx.Graph()
	if xml_file is not None:

		# importing element tree
		import xml.etree.ElementTree as ET 

		# Pass the path of the xml document 
		tree = ET.parse(xml_file) 

		node_coords = []

		# get the parent tag 
		root = tree.getroot() 
		coords = []
		for segment in root.iter("item"):
			if "x" in segment.attrib.keys():
				coords.append([float(segment.attrib["x"]), float(segment.attrib["y"]), float(segment.attrib["z"])])
			else:
				node_coords.append(np.array(coords)[1:-1])
				coords = []
		node_coords = node_coords[1:]


	# Read files
	node_df = pandas.read_csv(node_file, sep=";")
	edge_df = pandas.read_csv(edge_file, sep=";")

	node_ids = node_df[["id", "pos_x", "pos_y", "pos_z"]].to_numpy()
	edge_ids = edge_df[["id", "node1id", "node2id", "avgRadiusAvg"]].to_numpy()
	if len(node_coords) != len(edge_ids)-1:
		raise ValueError("The number of edges in the xml file and csv file are not the same.")

	if xml_file is not None:
		# Add the key nodes to the graph
		for i in range(len(node_ids)):
			G.add_node(i, coords = node_ids[i, 1:])

		# Add the regular nodes and egdes to the graph
		k = len(node_ids)
		for i in range(len(node_coords)):
			if len(node_coords[i])>0:
				
				G.add_node(k, coords = np.array(node_coords[i][0, :].tolist() + [edge_ids[i, -1]]))
				G.add_edge(edge_ids[i, 1], k, radius = edge_ids[i, -1]) 
				k+=1
				
				for j in range(1, len(node_coords[i])): # Create path
					G.add_node(k, coords = np.array(node_coords[i][j, :].tolist() + [edge_ids[i, -1]]))
					G.add_edge(k-1, k, radius = edge_ids[i, -1]) 
					k+=1

				G.add_edge(k-1, edge_ids[i, 2], radius = edge_ids[i, -1]) 

			else:
				G.add_edge(edge_ids[i, 1], edge_ids[i, 2], radius = edge_ids[i, -1])
	else:
		# Add the key nodes to the graph
		for i in range(len(node_ids)):
			G.add_node(i, coords = node_ids[i][1:])

		for i in range(len(edge_ids)):
			G.add_edge(edge_ids[i, 1], edge_ids[i, 2], radius = edge_ids[i, -1])

	# Add radius to key nodes
	for n in G.nodes():
		if len(G.nodes[n]['coords']) == 3:
			radius = []
			for e in G.edges(n):
				radius.append(G.edges[e]["radius"]) 
			if len(radius)!= 0:
				G.nodes[n]['coords'] = np.array(G.nodes[n]['coords'].tolist() + [sum(radius)/len(radius)])
			else:
				G.nodes[n]['coords'] = np.array(G.nodes[n]['coords'].tolist() + [0])

	return G

def vesselvio_to_networkx(graph_file):
	''' Convert Vesselvio output file to a networkx undirected graph.
	Keyword arguments :
		graph_file -- pickle file of the networkx graph returned by vesselvio
	'''

	f = open(graph_file, 'rb')
	G = pickle.load(f)
	nx.set_node_attributes(G, None, name='coords')

	for n in G.nodes():
		G.nodes[n]['coords'] = np.array(G.nodes[n]['v_coords'].tolist() + [G.nodes[n]['v_radius']])

	return G


def register_coords(G, scaling = [1,1,1,1], translation = [0,0,0,0], order = [0,1,2,3]):
	""" Register the coordinate system in the image referential by applying translations / rotations / reordering"""
	for n in G.nodes():

		G.nodes[n]['coords'] = G.nodes[n]['coords'][order]
		G.nodes[n]['coords'] = G.nodes[n]['coords'] * np.array(scaling)
		G.nodes[n]['coords'] = G.nodes[n]['coords'] + np.array(translation)
	return G


def networkx_to_vtk(G, out_file, node_scalar = [], edg_scalar=[]):

	''' Convert networkx graph to vtk polyLine.
	Keyword arguments :
	G -- networx graph
	out_file -- path to output directory and output filename
	'''

	if max([n for n in G.nodes()]) >= G.number_of_nodes():
		# Rename nodes
		G = nx.convert_node_labels_to_integers(G, first_label=0)

	nodes = np.zeros((G.number_of_nodes(), 4))
	faces = np.zeros((G.number_of_edges(), 3), dtype = int) + 2

	edg_tab = []
	node_tab = []

	for scalar in edg_scalar:
		edg_tab.append(np.zeros((G.number_of_edges(), 1)))
	for scalar in node_scalar:
		node_tab.append(np.zeros((G.number_of_nodes(), 1)))
	
	for n in G.nodes():
		nodes[n, :3] =  G.nodes[n]['coords'][:3]
		nodes[n, -1] =  G.nodes[n]['coords'][-1]
		for i in range(len(node_scalar)):
			node_tab[i][n, 0] = G.nodes[n][node_scalar[i]]

	c = 0
	try:
		for e in G.edges(keys=True):
			faces[c, 1] = e[0]
			faces[c, 2] = e[1]
			for i in range(len(edg_scalar)):
				edg_tab[i][c, 0] = G.edges[e][edg_scalar[i]]

			c+=1
	except:
		for e in G.edges():
			faces[c, 1] = e[0]
			faces[c, 2] = e[1]
			for i in range(len(edg_scalar)):
				edg_tab[i][c, 0] = G.edges[e][edg_scalar[i]]
			c+=1

	# Create VTK polyLine 
	centerline = pv.PolyData()
	centerline.points = nodes[:, :-1]
	centerline.lines = faces

	# Add radius information
	centerline["node_radius"] = nodes[:, -1]
	for i in range(len(node_scalar)):
		centerline[node_scalar[i]] = node_tab[i]
	for i in range(len(edg_scalar)):
		centerline[edg_scalar[i]] = edg_tab[i]

	centerline = centerline.tube(radius = 0.3)
	print("Saving vtk file.")
	centerline.save(out_file)


def voreen_to_vtk(node_file, edge_file, out_file, xml_file=None, scaling = [1,1,1,1], translation = [1,1,1,1], order = [0,1,2,3], node_scalar = [], edg_scalar=[]):

	''' Convert Voreen csv files to vtk polyLine for visualization with paraview
	Keyword arguments :
	node_file -- path to the node CSV file
	edge_file -- path to the edge CSV file
	out_file -- path to output directory and output filename
	xml_file -- path to the xml file with the segment voxels (optional)
	pix_dim -- voxel dimension
	coord_order -- ordering of the x y z coordinates
	'''

	G = voreen_to_networkx(node_file=node_file, edge_file=edge_file, xml_file=xml_file)
	G = register_coords(G, scaling = scaling, translation = translation, order = order)
	graph_summary(G)
	networkx_to_vtk(G, out_file, node_scalar=node_scalar, edg_scalar=edg_scalar)



def vesselvio_to_vtk(graph_file, out_file, scaling = [1,1,1,1], translation = [0,0,0,0], order = [0,1,2,3], node_scalar = [], edg_scalar=[]):

	''' Convert vesselvio graph to vtk polyLine for visualization with paraview
	Keyword arguments :
	graph_file -- path to the graph object 
	out_file -- path to output directory and output filename
	pix_dim -- voxel dimension
	coord_order -- ordering of the x y z coordinates
	'''

	G = vesselvio_to_networkx(graph_file)
	G = register_coords(G, scaling = scaling, translation = translation, order = order)
	graph_summary(G)
	networkx_to_vtk(G, out_file, node_scalar=node_scalar, edg_scalar=edg_scalar)


def sknw_to_vtk(img_file, out_file, scaling = [1,1,1,1], translation = [0,0,0,0], order = [0,1,2,3], node_scalar = [], edg_scalar=[]):

	''' Convert sknw graph to vtk polyLine for visualization with paraview
	Keyword arguments :
	img_file -- path to the binary medical image
	out_file -- path to output directory and output filename
	pix_dim -- voxel dimension
	coord_order -- ordering of the x y z coordinates
	'''

	D = sknw_to_networkx(img_file)
	D = register_coords(D, scaling = scaling, translation = translation, order = order)
	graph_summary(D)
	networkx_to_vtk(D, out_file, node_scalar=node_scalar, edg_scalar=edg_scalar)


def networkx_to_swc(G, source, filename):

		""" Write swc Neurite Tracer file using depth fist search.
		Keyword arguments :
		G -- networkx graph
		source -- starting node for dfs
		pix_dim -- voxel dimension
		coord_order -- ordering of the x y z coordinates
		"""

		print('Writing swc file...')


		if filename[-4:] != ".swc":
			filename = filename + ".swc"		

		file = open(filename, 'w') 

		keys = list(nx.dfs_preorder_nodes(G, source))
		if len(keys) <2:
			raise ValueError("nothing to print")

		values = range(1, len(keys) + 1)

		mapping = dict(zip(keys, values))

		for p in keys:
			c = G.nodes[p]['coords'][:3]
			r = G.nodes[p]['coords'][-1]

			if G.in_degree(p) == 1:
				n = mapping[list(G.predecessors(p))[0]]
				i = 3

			else: 
				n = -1
				i = 1

			file.write(str(mapping[p]) + '\t' + str(i) + '\t' + str(c[0]) + '\t' + str(c[1]) + '\t' + str(c[2]) + '\t' + str(r) + '\t' + str(n) + '\n')

		file.close()

### Graph conversion ####

def sknw_topo_to_undirected_graph(G):

	""" Convert the topo graph provided by sknw to an undirected graph.
	Keyword arguments :
	G -- networkx graph as produced by sknw
	"""

	H = nx.Graph()
	k = G.number_of_nodes()

	for n in G.nodes():
		if len(G.nodes[n]['o'].tolist()) == 2:
			H.add_node(n, coords = np.array(G.nodes[n]['o'].tolist()+ [0, 0]))
		else:
			H.add_node(n, coords = np.array(G.nodes[n]['o'].tolist()+ [0]))
		
	for e in G.edges():
		
		pts_list = G.edges[e]['pts']
		
		if len(pts_list) > 1:
			# check order
			d1 = norm(pts_list[0] - G.nodes[e[0]]['o'])
			d2 = norm(pts_list[-1] - G.nodes[e[0]]['o'])
			if d2< d1:
				pts_list = pts_list[::-1, :]

		nprec = e[0]

		for pts in pts_list:
			if len(pts.tolist()) == 2:
				H.add_node(k, coords = np.array(pts.tolist() + [0, 0]))
			else:
				H.add_node(k, coords = np.array(pts.tolist() + [0]))
			H.add_edge(nprec, k)
			nprec = k
			k+=1

		H.add_edge(nprec, e[1])

	return H



def topo_to_undirected_graph(G, transfer_edg_att = [], collapse = False):

	""" Convert a topo graph back to an undirected graph.
	Keyword arguments :
	G -- networkx topo graph 
	"""
	G_nds_list = [n for n in G.nodes()]

	H = nx.Graph()
	for att in transfer_edg_att:
		nx.set_edge_attributes(H, 0, name=att)

	for n in G.nodes():
		if collapse:
			if G.nodes[n]["clique"] is not None:
				H.add_node(G.nodes[n]["clique"], coords = G.nodes[n]["coords"])
			else:
				H.add_node(n, coords = G.nodes[n]["coords"])
		else:
			H.add_node(n, coords = G.nodes[n]["coords"])
		
	for e in G.edges(keys=True):
		nds_list = G.edges[e]['nodes']
		edg_list = G.edges[e]['edges']
		coords = G.edges[e]['coords']

		for i in range(len(nds_list)):
			add = True
			if collapse:
				if nds_list[i] in G_nds_list:
					if G.nodes[nds_list[i]]["clique"] is not None:
						add = False
			if add:
				H.add_node(nds_list[i], coords = coords[i])

		for edg in edg_list:
			
			n1 = edg[0]
			n2 = edg[1]

			if collapse:
				if edg[0] in G_nds_list:
					if G.nodes[edg[0]]["clique"] is not None:
						n1 = G.nodes[edg[0]]["clique"]

				if edg[1] in G_nds_list:
					if G.nodes[edg[1]]["clique"] is not None:
						n2 = G.nodes[edg[1]]["clique"]
			

			if n1!= n2:
				H.add_edge(n1, n2)

			for att in transfer_edg_att:
				H.edges[(n1, n2)][att] = G.edges[e][att]

	return H


def topo_to_directed_graph(G, transfer_edg_att = []):

	""" Convert a topo graph back to an undirected graph.
	Keyword arguments :
	G -- networkx topo graph 
	"""

	H = nx.DiGraph()

	for n in G.nodes():
		H.add_node(n, coords = G.nodes[n]['coords'])
		
	for e in G.edges():
		nds_list = G.edges[e]['nodes']
		edg_list = G.edges[e]['edges']
		coords = G.edges[e]['coords']

		for i in range(len(nds_list)):
			H.add_node(nds_list[i], coords = coords[i])

		for edg in edg_list:
			H.add_edge(edg[0], edg[1])

	for att in transfer_edg_att:
		nx.set_edge_attributes(H, 0, name=att)

		for e in G.edges():
			edg_list = G.edges[e]['edges']
	
			for edg in edg_list:
				H.edges[edg][att] = G.edges[e][att]
		

	return H


def undirected_graph_to_topo(G, orientation = False):

	""" Convert an undirected graph to the corresponding topo graph.
	Keyword arguments :
	G -- networkx undirected Graph
	orientation -- add orientation for angle computation
	"""
	T = nx.MultiGraph()
	for n in G.nodes():
		T.add_node(n, coords = G.nodes[n]["coords"])
	for e in G.edges():
		T.add_edge(e[0], e[1])

	#T = G.copy()

	nx.set_edge_attributes(T, 0, name="length")
	nx.set_edge_attributes(T, [], name="nodes")
	nx.set_edge_attributes(T, [], name="edges")
	nx.set_edge_attributes(T, [], name="coords")

	G_nodes_list = [n for n in G.nodes()]
	for n in G_nodes_list:

		# If regular nodes
		if G.degree(n) == 2:
			nb_list = [nb for nb in T.neighbors(n)]
			edg = [e for e in T.edges(n, keys=True)]
		
			if len(nb_list) == 2:
				
				edges = T.edges[edg[0]]["edges"] + T.edges[edg[1]]["edges"]
				if len(T.edges[edg[0]]["edges"]) == 0:
					edges += [edg[0]]
				if len(T.edges[edg[1]]["edges"]) == 0:
					edges += [edg[1]]

				nodes = T.edges[edg[0]]["nodes"] + [n] + T.edges[edg[1]]["nodes"]
				coords = T.edges[edg[0]]["coords"] + [G.nodes[n]['coords']] + T.edges[edg[1]]["coords"] 
				
				# Create new edge by merging the 2 edges of regular point
				T.add_edge(nb_list[0], nb_list[1], coords = coords, edges = edges, nodes = nodes, length=len(coords)) #full_id = full_id
				# Remove regular point
				T.remove_node(n)

	# Add length
	T = add_length(T, directed = False)
	T = add_angle(T, directed = False, orientation = orientation)

	return T


def orient_tree(start_node, G):

	''' Build directed graph from tree-like network '''

	H = nx.create_empty_copy(G, with_data=True)
	H = H.to_directed()
	added = [start_node]
	order = list(nx.dfs_preorder_nodes(G, source=start_node))[1:]

	for n in order:
		for nb in G.neighbors(n):
			if nb in added:
				H.add_edge(nb, n)
				added.append(n)
				break
	return H


def directed_graph_to_topo(G, orientation = False):

	""" Convert a directed tree-like graph to the corresponding topo graph.
	Keyword arguments :
	G -- tree-like networkx Digraph
	orientation -- add orientation for angle computation
	"""

	T = nx.DiGraph()
	for n in G.nodes():
		T.add_node(n, coords = G.nodes[n]["coords"])
	for e in G.edges():
		T.add_edge(e[0], e[1])

	#T = G.copy()

	nx.set_edge_attributes(T, 0, name="length")
	nx.set_edge_attributes(T, [], name="nodes")
	nx.set_edge_attributes(T, [], name="edges")
	nx.set_edge_attributes(T, [], name="coords")

	G_nodes_list = [n for n in G.nodes()]
	for n in G_nodes_list:

		# If regular nodes
		if G.in_degree(n) + G.out_degree(n) == 2:
			nb_list = [e[0] for e in T.in_edges(n)] + [e[1] for e in T.out_edges(n)] 
			edg = [e for e in T.in_edges(n)] + [e for e in T.out_edges(n)] 
		
			if len(nb_list) == 2:
				
				edges = T.edges[edg[0]]["edges"] + T.edges[edg[1]]["edges"]
				if len(T.edges[edg[0]]["edges"]) == 0:
					edges += [edg[0]]
				if len(T.edges[edg[1]]["edges"]) == 0:
					edges += [edg[1]]

				nodes = T.edges[edg[0]]["nodes"] + [n] + T.edges[edg[1]]["nodes"]
				coords = T.edges[edg[0]]["coords"] + [G.nodes[n]['coords']] + T.edges[edg[1]]["coords"] 
				
				# Create new edge by merging the 2 edges of regular point
				T.add_edge(nb_list[0], nb_list[1], coords = coords, edges = edges, nodes = nodes, length=len(coords)) #full_id = full_id
				# Remove regular point
				T.remove_node(n)

	# Add length
	T = add_length(T, directed = True)
	T = add_angle(T, directed = True, orientation = orientation)
	return T


def get_ordered_nds(T, edg, start, nb):
	""" Get ordered nodes from topo graph """

	coords = []
	stop = False

	ext = edg[0]
	if edg[0] == start:
		ext = edg[1]

	n = start
	while len(coords) < nb and not stop:
		n1 = None
		for e in T.edges[edg]["edges"]:

			if e[1] == n:
				n1 = e[0]
				break
			if e[0] == n:
				n1 = e[1]
				break

		if n1 is None:
			stop = True 
		else:
			if n1 in T.edges[edg]["nodes"]:
				ind1 = T.edges[edg]["nodes"].index(n1)
				coords.append(T.edges[edg]["coords"][ind1][:-1])
			else:
				coords.append(T.nodes[ext]["coords"][:-1])
			n = n1

	return np.array(coords)


def add_clique_multifurcation(T, nb=2, orientation = False):

	""" Add clique at multifurcation to separate merged vessels by maximum path finding """
	nx.set_node_attributes(T, None, name="clique")
	nd_id = max_node_id(T)+1

	for n in [t for t in T.nodes()]:
		
		if T.degree(n) > 3: # If trifurcation or more
			bif_coord = T.nodes[n]["coords"][:3]
			clique_nodes = []
			clique_vectors = []
			orient = [] # true if in false if out
			# Add clique
			edg = [l for l in T.edges(n, keys=True)]
			
			for e in edg:
				if e[0] == n:
					ext = e[1]
				else:
					ext = e[0]
				if orientation:
					deg =  T.edges[e]["branch_degree"]
					if n == T.edges[e]["orientation"][0]:
						e_orient = False
						orient.append(e_orient)
					else:
						e_orient = True
						orient.append(e_orient)

				T.add_node(nd_id, coords = T.nodes[n]["coords"], clique = n)
				clique_nodes.append(nd_id)
				
				nodes = T.edges[e]["nodes"]
				coords = T.edges[e]["coords"]
				edges = T.edges[e]["edges"]
				length = T.edges[e]["length"]
				ang = T.edges[e]["angle"]

				pt = get_ordered_nds(T, e, n, nb=nb)

				v = []
				for i in range(len(pt)):
					vect = pt[i] - bif_coord
					v.append(vect / norm(vect))

				clique_vectors.append(sum(v)/len(v))
			
				for i in range(len(edges)):
					if edges[i][0] == n:
						edges[i] = (nd_id, edges[i][1])

					if edges[i][1] == n:
						edges[i] = (edges[i][0], nd_id)

				T.remove_edge(e[0], e[1], e[2])
				if orientation:
					if e_orient:
						T.add_edge(nd_id, ext, coords=coords, nodes = nodes, edges = edges, length = length, angle = ang, orientation = [ext, nd_id], branch_degree=deg)
					else:
						T.add_edge(nd_id, ext, coords=coords, nodes = nodes, edges = edges, length = length, angle = ang, orientation = [nd_id, ext], branch_degree=deg)

				else:
					T.add_edge(nd_id, ext, coords=coords, nodes = nodes, edges = edges, length = length, angle = ang)

				nd_id +=1

			T.remove_node(n)

			# Create clique AND add angle
			for i in range(len(clique_nodes)):
				for j in range(i+1, len(clique_nodes)):
					a = angle(clique_vectors[i], clique_vectors[j])

					if orientation:
						if orient[i] == orient[j]:
							a = max(abs(pi - a), a)
						else:
							a = abs(pi-a)
					else:
						a = min(abs(pi - a), a)

					T.add_edge(clique_nodes[i], clique_nodes[j], nodes = [], coords = [], edges = [(clique_nodes[i], clique_nodes[j])], length = 0, angle = a)
					
	return T


def max_node_id(T):
	""" Returns the maximum id of nodes in topo graph T"""

	max_id = 0
	for n in T.nodes():
		if n > max_id:
			max_id = n

	for e in T.edges(keys=True):
		nds = T.edges[e]["nodes"]
		for n in nds:
			if n > max_id:
				max_id = n

	return max_id

def add_length(T, directed = False):

	""" Add the length for each node in the topo graph T.
	Keywords arguments :
	T -- networkx UNdirected topo graph"""

	nx.set_edge_attributes(T, 0, name="length")

	if directed:
		# Add length
		for e in T.edges():

			if len(T.edges[e]['edges']) == 0:
				T.edges[e]['edges'] = [(e[0], e[1])]

			edg_list = T.edges[e]['edges']
			nds_list = T.edges[e]['nodes']
			coords = T.edges[e]['coords']
			length = 0

			for edg in edg_list:
				if edg[0] == e[0]:
					coord1 = T.nodes[e[0]]['coords'][:3]
				elif edg[0] == e[1]:
					coord1 = T.nodes[e[1]]['coords'][:3]
				else:
					id1 = nds_list.index(edg[0])
					coord1 = np.array(coords[id1])[:3]

				if edg[1] == e[0]:
					coord2 = T.nodes[e[0]]['coords'][:3]
				elif edg[1] == e[1]:
					coord2 = T.nodes[e[1]]['coords'][:3]
				else:
					id2 = nds_list.index(edg[1])
					coord2 = np.array(coords[id2])[:3]

				length += norm(coord1 - coord2)

			T.edges[e]['length'] = length
	else:
		# Add length
		for e in T.edges(keys=True):

			if len(T.edges[e]['edges']) == 0:
				T.edges[e]['edges'] = [(e[0], e[1])]

			edg_list = T.edges[e]['edges']
			nds_list = T.edges[e]['nodes']
			coords = T.edges[e]['coords']
			length = 0

			for edg in edg_list:
				if edg[0] == e[0]:
					coord1 = T.nodes[e[0]]['coords'][:3]
				elif edg[0] == e[1]:
					coord1 = T.nodes[e[1]]['coords'][:3]
				else:
					id1 = nds_list.index(edg[0])
					coord1 = np.array(coords[id1])[:3]

				if edg[1] == e[0]:
					coord2 = T.nodes[e[0]]['coords'][:3]
				elif edg[1] == e[1]:
					coord2 = T.nodes[e[1]]['coords'][:3]
				else:
					id2 = nds_list.index(edg[1])
					coord2 = np.array(coords[id2])[:3]

				length += norm(coord1 - coord2)

			T.edges[e]['length'] = length

	return T

def add_branch_id(T, directed = False):

	if directed:

		nx.set_edge_attributes(T, 0, name="id")
		max_e = len([e for e in T.edges() if T.edges[e]["length"] != 0])
		list_id = np.arange(0, max_e)
		np.random.shuffle(list_id)

		k = 0
		for e in T.edges():
			if T.edges[e]["length"] != 0:
				T.edges[e]["id"] = list_id[k]
				k+=1
	else:
		nx.set_edge_attributes(T, 0, name="id")
		max_e = len([e for e in T.edges(keys = True) if T.edges[e]["length"] != 0])
		list_id = np.arange(0, max_e)
		np.random.shuffle(list_id)

		k = 0
		for e in T.edges(keys = True):
			if T.edges[e]["length"] != 0:
				T.edges[e]["id"] = list_id[k]
				k+=1
	return T


def add_angle(T, directed = False, nb = 2, orientation = False):

	""" Add the mean angle value for each node in the topo graph T.
	Keywords arguments :
	T -- networkx UNdirected topo graph
	directed -- bool indicating if the topo graph is a multigraph or not
	nb -- number of points used to average the branch angle"""


	nx.set_edge_attributes(T, 0, name="angle")

	if directed:

		for e in T.edges():
			a_list = []
			for n, ext in [(e[0], e[1]), (e[1], e[0])]:
				bif_coord = T.nodes[n]["coords"][:-1]
				pt = get_ordered_nds(T, e, n, nb=nb)

				v = []
				for i in range(len(pt)):
					vect = pt[i] - bif_coord
					v.append((pt[i] - bif_coord) / norm(pt[i] - bif_coord))

				v1 = sum(v)/len(v)

				for edg in [e for e in T.in_edges(n)] + [e for e in T.out_edges(n)]:
					if edg[1] != ext: 
						
						pt = get_ordered_nds(T, edg, n, nb=nb)
						v = []
						for i in range(len(pt)):
							vect = pt[i] - bif_coord
							v.append((pt[i] - bif_coord) / norm(pt[i] - bif_coord))

						v2 = sum(v)/len(v)

						a = angle(v1, v2)

						if orientation:
							if n == T.edges[e][0]: # n is the vessel starting point
								if n == T.edges[edg][0]: # edg is oriented out
									a_list.append(a)
								else:
									a_list.append(abs(pi-a))

							else: # n is the vessel ending point
								if n == T.edges[edg][0]: # edg is oriented out
									a_list.append(abs(pi - a))
								else:
									a_list.append(a)
						else:
							a_list.append(min(abs(pi - a), a))

						a_list.append(min(abs(pi - a), a))

				if len(a_list) == 0 or np.isnan(sum(a_list) / len(a_list)):
					a_list = [0]

				T.edges[e]["angle"] = sum(a_list) / len(a_list)

	else:
		if orientation:
			T = add_orientation(T)

		for e in T.edges(keys = True):
			a_list = []
			for n, ext in [(e[0], e[1]), (e[1], e[0])]:
				bif_coord = T.nodes[n]["coords"][:-1]
				pt = get_ordered_nds(T, e, n, nb=nb)

				v = []
				for i in range(len(pt)):
					vect = pt[i] - bif_coord
					v.append((pt[i] - bif_coord) / norm(pt[i] - bif_coord))

				v1 = sum(v)/len(v)

				for edg in T.edges(n, keys=True):
					if edg[1] != ext: 
						
						pt = get_ordered_nds(T, edg, n, nb=nb)
						v = []
						for i in range(len(pt)):
							vect = pt[i] - bif_coord
							v.append((pt[i] - bif_coord) / norm(pt[i] - bif_coord))

						v2 = sum(v)/len(v)

						a = angle(v1, v2)

						if orientation:
							if n == T.edges[e]["orientation"][0]: # n is the vessel starting point
								if n == T.edges[edg]["orientation"][0]: # edg is oriented out
									a_list.append(a)
								else:
									a_list.append(abs(pi-a))

							else: # n is the vessel ending point
								if n == T.edges[edg]["orientation"][0]: # edg is oriented out
									a_list.append(abs(pi - a))
								else:
									a_list.append(a)
						else:
							a_list.append(min(abs(pi - a), a))


				if len(a_list) == 0 or np.isnan(sum(a_list) / len(a_list)):
					a_list = [0]
				T.edges[e]["angle"] = sum(a_list) / len(a_list)

	return T


def angle(v1, v2, axis = None, signed = False):

	if axis is not None: 
		# Compute angle on the plane of normal axis
		v1 = v1 - dot(v1, axis / norm(axis))
		v2 = v2 - dot(v2, axis / norm(axis))

	if signed:

		sign = np.sign(np.cross(v1, v2).dot(axis))
		# 0 means colinear: 0 or 180. Let's call that clockwise.
		if sign == 0:
			sign = 1
	else: 
		sign = 1

	x = dot(v1, v2) / (norm(v1) * norm(v2))

	if x > 1.0:
		x = 1.0

	if x < -1.0:
		x = -1.0

	return sign * acos(x)



def add_weight(T, coef=0.2):
	""" Add the weight value for each node in the topo graph T.
	Keywords arguments :
	T -- networkx UNdirected topo graph"""

	nx.set_edge_attributes(T, 0, name="weight")

	# Get max and min values of length and angle
	min_length = np.inf
	max_length = 0

	min_angle = np.inf 
	max_angle = 0

	for e in T.edges(keys=True):

		length = T.edges[e]["length"]
		angle = T.edges[e]["angle"]
		if length < min_length:
			min_length = length
		if length > max_length:
			max_length = length
		if angle < min_angle:
			min_angle = angle
		if angle > max_angle:
			max_angle = angle

	for e in T.edges(keys=True):
		w = (1-coef) * ((T.edges[e]["length"] - min_length)/ (max_length - min_length)) + coef * (1-((T.edges[e]["angle"] - min_angle)/ (max_angle - min_angle)))
		T.edges[e]["weight"] = w

	return T
	
				

def find_branch_degree(T):

	"""Add branching degree to the edges of a directed topo graph.
	Keywords arguments :
	T -- networkx directed topo graph"""

	nx.set_edge_attributes(T, 0, name="branch_degree")
	for n in T.nodes():
		if T.in_degree(n) == 0:
			prec = list(T.edges(n))
	
	propagate = True
	#prec = [origin]
	deg = 0
	while propagate:
		# Label the edges
		suc = []
		for e in prec:
			T.edges[e]["branch_degree"] = deg
			# Find the next edges
			suc += list(T.out_edges(e[1]))
		deg+=1
		prec = suc
		if len(prec) == 0:
			propagate = False 
			for e in prec:
				T.edges[e]["branch_degree"] = deg


	#for e in T.edges():
	#	path = list(nx.all_simple_paths(T, origin, e[1]))[0]
	#	T.edges[e]["branch_degree"] = len(path)

	return T

def add_orientation(T):

	"""Add branching degree to the edges of a directed topo graph.
	Keywords arguments :
	T -- networkx UNdirected topo graph"""


	nx.set_edge_attributes(T, [], name="orientation")
	nx.set_edge_attributes(T, 0, name="branch_degree")

	origin = [find_inlet(T)]
	#print("origins:", origin)
	prec = [list(T.edges(origin, keys=True))]

	propagate = True
	deg = 0
	while propagate:
		# Label the edges
		suc = []
		new_org = []
		for i in range(len(prec)):
			
			#print("prec", prec[i])
			for e in prec[i]:
				subsuc = []
				T.edges[e]["branch_degree"] = deg

				ext = e[0]
				if e[0] == origin[i]:
					ext = e[1]
				

				T.edges[e]["orientation"] = [origin[i], ext]
			
				# Find the next edges
				for elt in T.edges(ext, keys=True):
					#print("next edge", elt, T.edges[elt]["orientation"])
					if len(T.edges[elt]["orientation"]) == 0:
						subsuc.append(elt)
				if len(subsuc)>0:
					suc.append(subsuc)	
					new_org.append(ext)	
			
		deg+=1
		prec = suc
		#print("new edges: ", prec)
		origin = new_org
		#print("origins:", origin)

		if len(origin) == 0:
			propagate = False 
			for e in prec:
				ext = e[0]
				if e[0] == origin[i]:
					ext = e[1]

				new_org[i].append(ext)
				T.edges[e]["branch_degree"] = deg
				T.edges[e]["orientation"] = [origin[i], ext]

	return T


#### Cleaning algorithms ####

def undirected_graph_to_oriented_tree(G, min_size = None, remove_bulges = True, create_cliques = True, remove_multi = True, coef_weight = 0.2, orientation = True, dim_inlet = 2):
	''' Returns the maximum spanning tree of an input graph, starting with node start_node '''

	# Keep only the largest connected component(s)
	CC_graphs = keep_largest_cc(G, threshold = min_size)
	trees = []
	for G in CC_graphs:
		# Convert to topo graph
		T = undirected_graph_to_topo(G, orientation = orientation)

		if create_cliques:
			T = add_clique_multifurcation(T, orientation = orientation)

		# Maximum spanning tree
		T = add_weight(T, coef=coef_weight)
		MST = nx.maximum_spanning_tree(T, weight="weight")
		
		# Remove bulges
		if remove_bulges:
			MST = remove_small_end_branches(MST, it = 3, thres = 10)

		# Convert back to full graph and remove cliques
		if create_cliques:
			G = topo_to_undirected_graph(MST, collapse = False)
		else:
			G = topo_to_undirected_graph(MST)

		# Orient the full graph
		G = orient_tree(find_inlet(G, dim = dim_inlet), G)

		if remove_multi:
			T = directed_graph_to_topo(G)
			T = remove_multifurcations(T)
			G = topo_to_directed_graph(T)
			
		trees.append(G)
	return trees


def find_inlet(G, dim = 2):

	""" Find the inlet of graph G
	Keyword argments :
	G -- networkx undirected graph"""
	
	zmax = np.inf
	for n in G.nodes():
		if G.degree(n) == 1:
			if G.nodes[n]["coords"][dim] < zmax:
				zmax =  G.nodes[n]["coords"][dim]
				start_node = n

	return start_node



def crop_branch_degree(T, thres):

	""" Removes the extremity branches from a graph
	Keyword argments :
	G -- networkx topo graph with branch degree attribute
	thres -- degree threshold (maximum) """

	for e in T.edges():
		if T.edges[e][branch_id]> thres:
			T.remove_edge(e)
	return T


def remove_multifurcations(T, thres = 2):

	""" Removes the smallest branches of multifurcations 
	Keyword arguments :
	T -- networkx directed topo graph """

	
	to_remove = []
	for n in T.nodes():
		if T.out_degree(n) > thres:
			down_nodes = {}
			# Compute the length of branches
			for e in T.out_edges(n):
				downstream_nodes =  list(nx.dfs_preorder_nodes(T, source=e[1]))
				down_nodes[e] = downstream_nodes

			length = [len(l) for l in down_nodes.values()]
			length.sort()
			
			max_len = length[len(length)- thres-1]

			k = 0
			for e in T.out_edges(n):
				if len(down_nodes[e]) <= max_len and k < len(length)- thres:
					k+=1
					to_remove += down_nodes[e]		

			print("Removed ", k, "branches.")

	# Remove them from topo graph
	for n in to_remove:
		if n in list([n for n in T.nodes()]):
			T.remove_node(n)
	return T



def remove_small_end_branches(G, it = 3, thres = 3, directed = False):

	""" Removes small ending branches from the graph G.
	Keyword argments :
	G -- networkx topo graph
	thres -- length threshold (maximum) """
	if directed :
		for i in range(it):
			print("Pruning iteration ", i)
			c = 0
			for e in [e for e in G.edges()]:
				if G.out_degree(e[1]) +  G.in_degree(e[1]) == 1 or G.out_degree(e[0]) +  G.in_degree(e[0]) == 1 :

					length = G.edges[e]['length']
					if length < thres:
						c += 1
						# Remove edge
						if G.out_degree(e[0]) +  G.in_degree(e[0]) == 1:
							G.remove_node(e[0])
						else:
							G.remove_node(e[1])
					
			print(c, " branches were removed.")
	else:
		for i in range(it):
			print("Pruning iteration ", i)
			c = 0
			for e in [e for e in G.edges(keys=True)]:
				if G.degree(e[0]) == 1 or G.degree(e[1]) == 1:

					length = G.edges[e]['length']
					if length < thres:
						c += 1
						# Remove edge
						if G.degree(e[0]) == 1:
							G.remove_node(e[0])
						else:
							G.remove_node(e[1])
			print(c, " branches were removed.")
	return G



def remove_small_branches(G, it = 3, thres = 2):

	""" Removes small branches from the graph G while keeping connectivity.
	Keyword argments :
	G -- networkx topo graph
	thres -- length threshold (maximum) """

	for e in [elt for elt in G.edges()]:
		
		if G.edges[e]["length"] < thres:
			nodes = G.edges[e]["nodes"]
			length = G.edges[e]["length"]
			G.remove_edge(e[0], e[1])
			
			if not nx.is_connected(G):
				G.add_edge(e[0], e[1], nodes = nodes, length = length) # TMP ADD OTHER ARG
	return G




def keep_largest_cc(G, threshold = None):

	""" Create a subgraph from the largest connected component.
	Keyword argments :
	G -- networkx graph
	"""
	#largest_cc = max(nx.connected_components(G), key=len)
	if nx.is_connected(G):
		return [G]
	else:

		cc = sorted(nx.connected_components(G), key=len, reverse=True)
		cc_graphs = []

		if threshold is None:
			return [G.subgraph(cc[0]).copy()]
		else:

			for i in range(len(cc)):
				if len(cc[i]) > threshold:
					cc_graphs.append(G.subgraph(cc[i]).copy())
			if len(cc_graphs)==0:
				return [G.subgraph(cc[0]).copy()]
			else:
				return cc_graphs



#### Graph analysis ####

def graph_summary(G, verbose = True):
	""" Summarizes the features of graph G """
	T = undirected_graph_to_topo(G)

	if verbose:
		print("The graph has ", G.number_of_nodes(), "nodes and ", G.number_of_edges(), "edges.")
		print("The degree distribution of the nodes is :", degree_distribution(G))
		print("The graph has ", find_cycles(G, write_scalar=False), "cycles, including ", number_loop(G), " self loops.")
		print("The graph has ", len(find_connected_components(G)), " connected components, whose sizes are  :", find_connected_components(G))
		print("The graph has ", T.number_of_nodes(), "bifurcation / end nodes and ", T.number_of_edges(), "branches.")
		print("The graph has ", find_bulges(T, write_scalar=False) , "bulges.")
	else:
		cc = find_connected_components(G)
		if len(cc) == 0:
			cc = [G.number_of_nodes()]

		Tcc = keep_largest_cc(T)[0]
		nbranchmaxcc = Tcc.number_of_edges()

		deg = degree_distribution(G)
		distrib = [0,0,0,0,0]

		for i in range(5):
			if i in list(deg.keys()):
				distrib[i] = deg[i]
		others = 0
		for k in deg.keys():
			if k > 4:
				others+=deg[k]
		distrib.append(others)
		

		length = list(nx.get_edge_attributes(T, "length").values())

		names = ["nb_nds", "nb_edg", "nb_cc", "nb_nds_max_cc", "nb_cycles", "nb_branch", "bif_deg_0", "bif_deg_1", "bif_deg_2", "bif_deg_3", "bif_deg_4", "bif_deg_4+"]
		names += ["min_degree", "max_degree", "nb_bulges", "min_len", "avg_len", "max_len"]
		features = [G.number_of_nodes(), G.number_of_edges(), len(cc), nbranchmaxcc,  find_cycles(G, write_scalar=False), T.number_of_edges(), distrib[0], distrib[1], distrib[2], distrib[3], distrib[4], distrib[5]]
		features += [min(list(deg.keys())), max(list(deg.keys())), find_bulges(T, write_scalar=False), min(length), sum(length)/len(length), max(length)]
		return names, features



def find_cycles(G, write_scalar = True):
	""" Returns the number of cycles in graph G"""

	cycles = nx.cycle_basis(G)
	if write_scalar:
		nx.set_edge_attributes(G, 0, name = "cycle_id")
		for j in range(len(cycles)):
			c = cycles[j]
			for i in range(len(c)):
				i2 = i+1
				if i == len(c) - 1:
					i2 = 0
				G.edges[(c[i], c[i2])]["cycle_id"] = j

		return len(nx.cycle_basis(G)), G
	else:
		return len(nx.cycle_basis(G))

def degree_distribution(G):
	""" Returns the degree distribution of undirected graph G"""

	degree_dict = {}

	for n in G.nodes():
		d = G.degree(n)
		if d not in degree_dict.keys():
			degree_dict[d] = 1
		else:
			degree_dict[d] += 1

	return degree_dict


def degree_distribution_directed(G):
	""" Returns the undirected degree distribution of directed graph G"""

	degree_dict = {}

	for n in G.nodes():
		d_in = G.in_degree(n)
		d_out = G.out_degree(n)
		if (d_in, d_out) not in degree_dict.keys():
			degree_dict[(d_in, d_out)] = 1
		else:
			degree_dict[(d_in, d_out)] += 1

	return degree_dict


def number_loop(G):
	""" Returns the number of self loops in the graph"""

	c = 0
	for e in G.edges():
		if e[0] == e[1]:
			c+=1
	return c


def find_bulges(T, thres=3, write_scalar=True):

	""" Computes the number of bulges in the network. 
	A segment is considered to be a bulge if it is an ending segment and its length is lower that the given threshold.
	Keyword arguments:
	T -- networkx topology graph
	thres -- length threshold
	write_scalar -- if True, write the bulge position as edg attribute "bulge"
	"""

	bulges = []
	for e in T.edges(keys=True):
		if T.degree(e[0]) == 1 or T.degree(e[1]) == 1 and (not T.degree(e[0]) == 1 and T.degree(e[1]) == 1):
			length = T.edges[e]["length"]
			if length < thres:
					bulges.append(e)
					
	if write_scalar:
		nx.set_edge_attributes(T, 0, name = "bulge")
		for e in bulges:
			T.edges[e]["bulge"] = 1

		return len(bulges), T 

	else:
		return len(bulges)
	


def find_connected_components(G):
	"""
	Computes the number of connected components in a graph G and their size.
	"""

	if not nx.is_connected(G):
		length = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
	else: 
		length = []
	return length



### Conversion function from image + centerline to swc centerline ###

def write_swc(outfile, path_img, algo = "sknw", path_data = [], dim_inlet= 1, scaling = [], order = [], translation =[]):
	""" Convert a centerline + image to a swc centerline compatible with vascularmd.
		Keyword arguments:

		outfile -- Path to the of the out folder 
		path_img -- Path to the image in nifti or jpg format 
		algo -- Algorithm used to extract the centerlines (sknw, vesselvio or voreen)
		path_data -- Path to the centerline file returned by the extraction algorithm
			If sknw : the centerlines are extracted from the image directly so path_data = [path_to_img]
			If vesselvio : path_data contains the path to the pickle file of the networkx object returned by vesselvio
			If voreen : path_data contains the path to the two csv files +  veg file returned by Voreen
		dim_inlet -- Dimension of the image to determine the inlet position 
			(Ex : if the inlet of the network is located on the lowest values of the x dimension dim_inlet = 0)
		"""


	if path_img[-3:] == "jpg":
		pix_dim = [1,1,1]
		img = io.imread(path_img)
		img = img>100
	else:
		img, pix_dim, offset, axis = load_nii(path_img)
		

	if algo == "sknw":
		
		G = sknw_to_networkx(path_data[0])
		
		# If no registration parameters input, use the default parameters of the method
		if len(scaling)== 0: 
			scaling = [pix_dim[0], pix_dim[1], pix_dim[2], 1]
		if len(order)== 0: 
			order = [0,1,2,3]
		if len(translation)== 0: 
			translation = [0,0,0,0]

	elif algo == "vesselvio":
				
		G = vesselvio_to_networkx(path_data[0])
		if len(scaling)== 0: 
			scaling = [pix_dim[0], pix_dim[1], pix_dim[2], 1]
		if len(order)== 0: 
			order = [2,1,0,3]
		if len(translation)== 0: 
			translation = [-pix_dim[0]/2,-pix_dim[1]/2,-pix_dim[2]/2,0]
				
	else:
				
		G = voreen_to_networkx(path_data[0], path_data[1], path_data[2])

		if len(scaling)== 0:
			scaling = [-1, -1,1,1]
		if len(order)== 0:
			translation = [offset[0],-offset[1],-offset[2], 0]
		if len(translation)== 0: 
			order = [0,1,2,3]

	# Register and extract radius
	G = register_coords(G, scaling = scaling, translation = translation, order = order)

	G = register_coords(G, scaling = [1/pix_dim[0],1/pix_dim[1],1/pix_dim[2],1]) # Convert to voxels
	G = extract_radius_from_image(G, img, pix_dim) # Extract radius from image
	G = register_coords(G, scaling = [pix_dim[0],pix_dim[1],pix_dim[2],1]) # Back to mm

	# Clean graph
	G_list = undirected_graph_to_oriented_tree(G,  min_size = 200, remove_multi = True, create_cliques = True, coef_weight = 0.2, orientation = True, dim_inlet = dim_inlet)
	for k in range(len(G_list)):
		G = G_list[k]
		#G = register_coords(G, scaling = [0.1,0.1,0.1,0.1]) #RETINA

		# Checking features 
		print("Degree distribution : ", degree_distribution_directed(G))

		# Writing file
		source = 0
		for n in G.nodes():
			if G.in_degree(n) == 0:
				source = n
		networkx_to_swc(G, source, outfile + "_" + str(k) + ".swc")

		# Add branch id and degree
		T = directed_graph_to_topo(G)
		T = find_branch_degree(T)
		T = add_branch_id(T, directed = True)
		G = topo_to_directed_graph(T, transfer_edg_att= ["branch_degree", "id"])

		networkx_to_vtk(G, outfile + "_centerline_" + str(k) + ".vtk", edg_scalar = ["branch_degree", "id"])


## CODE EXAMPLE

#  Write the swc file compatible with our framework from a segmented image
path_to_img = "home/Images/img.nii"
path_to_outfolder = "home/Results/"

write_swc(path_to_outfolder, path_to_img, algo = "sknw", path_data = [path_to_img])


