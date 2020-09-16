# Python 3

import numpy as np # Tools for matrices

# Trigonometry functions
from math import pi, sin, cos, tan, atan, acos, asin, sqrt
from numpy.linalg import norm 
from numpy import dot, cross, arctan2


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

	points = np.zeros((num, 3))
	points[:,0] = np.linspace(p0[0], p1[0], num).tolist()
	points[:,1] = np.linspace(p0[1], p1[1], num).tolist()
	points[:,2] = np.linspace(p0[2], p1[2], num).tolist()

	return points.tolist()



#####################################
############ VALIDATION  ############
#####################################

def distance(mesh1, mesh2, display=False):

	mesh1.compute_implicit_distance(mesh2, inplace=True)

	if display: 
		mesh1.plot(show_edges = True)

	tab = mesh1['implicit_distance']
	return np.mean(tab), np.min(tab), np.max(tab)




def quality(mesh, metric='scaled_jacobian', display=False):

	""" Compute the quality metric form the cells of a surface mesh. 
	Returns the mean, min and max values.

	Keyword arguments:
	metric -- name of the metric (see pyvista doc)
	display -- true to display the mesh with quality values
	"""
	quality = mesh.compute_cell_quality(metric)

	if display: 
		mesh.plot(show_edges = True)

	tab = quality['CellQuality']
	return np.mean(tab), np.min(tab), np.max(tab)
	
