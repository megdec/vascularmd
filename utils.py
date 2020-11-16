# Python 3

import numpy as np # Tools for matrices

# Trigonometry functions
from math import pi, sin, cos, tan, atan, acos, asin, sqrt
from numpy.linalg import norm 
from numpy import dot, cross, arctan2

import pyvista as pv


#####################################
############# GEOMETRY ##############
#####################################


def rotate_vector(v, axis, theta):

	"""
	Return the rotation matrix associated with counterclockwise rotation about
	the given axis by theta radians.

	Keywords arguments:
	v -- vector to rotate
	axis -- axis of rotation
	theta -- angle
	"""
	
	axis = axis / norm(axis)
	a = cos(theta / 2.0)
	b, c, d = -axis * sin(theta / 2.0)
	aa, bb, cc, dd = a * a, b * b, c * c, d * d
	bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

	R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
					 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
					 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


	return dot(R, v)



def directed_angle(v1, v2, u):

	"""
	Return the directed angle between v1 and v2 using spatial reference u.

	Keywords arguments:
	v1, v2 -- vectors
	u -- reference vector 
	"""

	v1 = np.array(v1)
	v2 = np.array(v2)
	u = np.array(u)

	x = dot(v1, v2) / (norm(v1) * norm(v2))

	if x > 1.0:
		x = 1.0

	if x < -1.0:
		x = -1.0

	return acos(x)



def directed_angle_negative(v1, v2, u):

	"""
	Return the directed angle between v1 and v2 using spatial reference u.

	Keywords arguments:
	v1, v2 -- vectors
	u -- reference vector 
	"""

	v1 = np.array(v1)
	v2 = np.array(v2)
	u = np.array(u)

	x = dot(v1, v2) / (norm(v1) * norm(v2))

	if x > 1.0:
		x = 1.0

	if x < -1.0:
		x = -1.0

	if np.linalg.det(np.array([[v1[0], v2[0], u[0]], [v1[1], v2[1], u[1]], [v1[2], v2[2], u[0]]])) >= 0:
		return acos(x)
	else: 
		return - acos(x)



def angle(v1, v2):

	"""
	Return the angle between v1 and v2.

	Keywords arguments:
	v1, v2 -- vectors
	"""

	if dot(v1, v2) / (norm(v1) * norm(v2)) < -1:
		theta = acos(-1)
		
	else:
		theta = acos(dot(v1, v2) / (norm(v1) * norm(v2)))

	return  theta



def linear_interpolation(nodes, num):

	""" Linearly interpolate num nodes between each given 4D node.

	Keyword arguments:
	nodes -- list of node coordinates
	num -- number of interpolation points
	"""

	interp_nodes = []
	for i in range(len(nodes) - 1):

		interp_nodes.append(nodes[i])
		x = np.linspace(nodes[i][0], nodes[i + 1][0], num + 2)
		y = np.linspace(nodes[i][1], nodes[i + 1][1], num + 2)
		z = np.linspace(nodes[i][2], nodes[i + 1][2], num + 2)
		r = np.linspace(nodes[i][3], nodes[i + 1][3], num + 2)

		for j in range(num):
			interp_nodes.append([x[j+1], y[j+1], z[j+1], r[j+1]])

	interp_nodes.append(nodes[-1])

	return interp_nodes


def lin_interp(p0, p1, num):

	num = int(num)
	x = len(p0)
	points = np.zeros((num, x))

	for i in range(x):
		points[:,i] = np.linspace(p0[i], p1[i], num).tolist()

	return points.tolist()



#####################################
############ VALIDATION  ############
#####################################

def distance(mesh1, mesh2, display=False):

	mesh1.compute_implicit_distance(mesh2, inplace=True)

	if display: 
		p = pv.Plotter()
		p.add_mesh(mesh1, scalars = 'implicit_distance')
		p.add_mesh(mesh2, color=True, opacity=0.25) # Reference mesh
		p.show()

	tab = mesh1['implicit_distance']
	return np.mean(tab), np.min(tab), np.max(tab)




def quality(mesh, display=False, metric='scaled_jacobian'):

	""" Compute the quality metric form the cells of a surface mesh. 
	Returns the mean, min and max values.

	Keyword arguments:
	metric -- name of the metric (see pyvista doc)
	display -- true to display the mesh with quality values
	"""
	quality = mesh.compute_cell_quality(metric)

	if display: 
		quality.plot(show_edges = True, scalars = 'CellQuality')

	tab = np.absolute(quality['CellQuality'])
	return np.mean(tab), np.min(tab), np.max(tab)
	



#####################################
##########  MULTIPROCESSING  ########
#####################################


def parallel_bif(bif, N, d):

	# Find cross sections
	bif.cross_sections(N, d)

	return bif


def segment_crsec(spl, num, N, v0 = [], alpha = None):

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

		crsec[i,:,:] = single_crsec(spl, t[i], v, N)

	return crsec


def single_crsec(spl, t, v, N):


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