####################################################################################################
# Author: Meghane Decroocq
#
# This file is part of vascularmd project (https://github.com/megdec/vascularmd)
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3 of the License.
#
####################################################################################################


# Python 3

import numpy as np # Tools for matrices
# Trigonometry functions
from math import pi, sin, cos, tan, atan, acos, asin, sqrt
from numpy.linalg import norm 
from numpy import dot, cross, arctan2

import pyvista as pv
import vtk

import matplotlib.pyplot as plt # Tools for plots
from scipy.spatial import KDTree
#from Spline import Spline
import networkx as nx




#####################################
############# GEOMETRY ##############
#####################################

def cart2pol(x, y):

	rho = sqrt(x**2 + y**2)
	phi = arctan2(y, x)

	return np.array([rho, phi])

def pol2cart(rho, phi):

	x = rho * cos(phi)
	y = rho * sin(phi)

	return np.array([x, y])

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


"""
def linear_interpolation(nodes, num):

	Linearly interpolate num nodes between each given 4D node.

	Keyword arguments:
	nodes -- list of node coordinates
	num -- number of interpolation points
	

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
"""

def linear_interpolation(start_pt, end_pt, num):

	""" Linearly interpolate num nodes between each given 4D node.

	Keyword arguments:
	nodes -- list of node coordinates
	num -- number of interpolation points
	"""

	interp_nodes = np.zeros((num, len(start_pt)))
	for i in range(len(start_pt)):
		interp_nodes[:, i] = np.linspace(start_pt[i], end_pt[i], num)

	return interp_nodes


def lin_interp(p0, p1, num):

	num = int(num)
	x = len(p0)
	points = np.zeros((num, x))

	for i in range(x):
		points[:,i] = np.linspace(p0[i], p1[i], num).tolist()

	return points.tolist()



def length_polyline(D):

	""" Return the length of a polyLine

	Keyword arguments:
	D -- list of node coordinates
	"""
	if D.shape[0] == 0:
		length = [0.0]
	else:
		length = np.zeros((D.shape[0],))
		length[0] = 0.0

	for i in range(1, len(D)):
		length[i] = length[i-1] + norm(D[i] - D[i-1])

	return length



def resample(D, num = 0):

	""" Resamples a polyline with equally spaced points

	Keyword arguments:
	D -- list of node coordinates
	num -- number of resampled points
	"""

	if num == 0:
		num = len(D)

	length = np.array(length_polyline(D))
	spacing = np.linspace(0, length[-1], num) 

	D_resamp = []
	for l in spacing:
		ind = np.argmax(length > l) -1
		d = l - length[ind]

		orient = D[ind + 1] - D[ind]
		if norm(orient) == 0.0:
			D_resamp.append(D[ind].tolist())
		else:
			D_resamp.append((D[ind] + orient/norm(orient)*d).tolist())


	return np.array(D_resamp)


def averaging_knot(data, p):

		""" Returns a B-spline averaging knot vector."""
		p = 3
		n = len(data)
		t = chord_length_parametrization(data)

		knot = [0.0] * p # First knot of multiplicity p

		for i in range(p, n):
			knot.append((1.0 / (p - 1.0)) * sum(t[i-3+1:i]))

		knot = knot + [1.0] * p

		return knot


def optimal_knot(nb_max, data):
	from Spline import Spline

	""" Search of dominant points in data data. """
	# Approximation of the curvature values

	data  = order_points(data)

	n = 4
	spl = Spline()
	spl.approximation(data, [0,0,0,0], np.zeros((4,4)), False, n = n)

	ASE_act = spl.ASE(data)
	ASE_prec = (ASE_act[0] + 1, ASE_act[1] + 1)

	
	thres = 10**(-2)
	while n <= len(data) and (abs(ASE_prec[0] - ASE_act[0])> thres or abs(ASE_prec[1] - ASE_act[1])> thres): # Stability
		n += 1

		spl = Spline()
		spl.approximation(data, [0,0,0,0], np.zeros((4,4)), False, n = n)

		ASE_prec = ASE_act[:]
		ASE_act = spl.ASE(data)

	spl = Spline()
	spl.approximation(data, [0,0,0,0], np.zeros((4,4)), False, n = n)
	spl.show(data=data)
		
	print(n)

	# Plot curvature curve
	times, dist = spl.distance(data)
	curv = spl.curvature(times)
	plt.plot(times, curv, color='black')
	

	# Find dominant points
	dominant = data[0]
	ind = [0]
	for i in range(1, len(data) -1):
		if (curv[i-1] > curv[i] and curv[i + 1] > curv[i]) or (curv[i-1] < curv[i] and curv[i + 1] < curv[i]):
			dominant = np.vstack((dominant, data[i]))
			ind.append(i)

	dominant = np.vstack((dominant, data[-1]))
	ind.append(len(data)-1)

	plt.scatter(times[ind], curv[ind])
	plt.scatter(times[ind], np.zeros((1, len(ind))), color = 'black')
	
	print(len(dominant))
	knot = averaging_knot(dominant, 3)
	plt.scatter(knot, np.zeros((1, len(knot))), color = 'black', s=10)
	plt.show()

	spl = Spline()
	spl.approximation(data, [0,0,0,0], np.zeros((4,4)), False, n = len(dominant))
	spl.show(data=data)
	print(spl.ASE(data))

	spl = Spline()
	spl.approximation(data, [0,0,0,0], np.zeros((4,4)), False, n = len(dominant), knot = knot)
	spl.show(data=data)
	print(spl.ASE(data))

	return knot


def order_points(points):
	''' Reorder points to form the shortest possible polyline '''

	# Create kneighbor graph
	kdtree = KDTree(points[:,:-1])
	G = nx.Graph()
	G.add_nodes_from(range(len(points)))
	nb = 5

	for i in range(len(points)):
		d, idx = kdtree.query(points[i, :-1], k=nb)
		for j in range(nb):
			G.add_edge(i, idx[j], distance = d[j])

	path = dict(nx.all_pairs_dijkstra_path(G, weight = 'distance'))

	points = points[list(path.keys()), :]

	return points



#####################################
######## MESHING GEOMETRY ###########
#####################################


def intersection_segment_segment(coefs, data, normals, radius,i, j):

	a = coefs[i, 0]
	b = coefs[i, 1]
	c = coefs[j, 0]
	d = coefs[j, 1]
	inter2 = []
	inter1 = []
	center = []
	found = False
	if a != c:

		x = (b-d)/(c-a)
		y = a*x + b
		center = np.array([x, y])

		inter1 = center - normals[j, :] * radius
		inter2 = center - normals[i, :] * radius

			
		# Search if the intersections lies in the segments
		if (norm(data[j, :] - inter1) <= norm(data[j, :] - data[j + 1, :]) and norm(data[j+1, :] - inter1) < norm(data[j, :] - data[j + 1, :])):
			if (norm(data[i, :] - inter2) <= norm(data[i, :] - data[i + 1, :]) and norm(data[i+1, :] - inter2) < norm(data[i, :] - data[i + 1, :])):
				found = True
				project = [inter2]

				for k in range(i+1, j+1):
					project.append(data[k,:])
					
				project.append(inter1)

				angles = [0.0]
				for k in range(len(project)-1):
					angles.append(angles[-1] + norm(project[k] - project[k+1]))

				# Compute angle between both intersections
				angles = angles / max(angles)

				v0 = np.array((project[0] - center).tolist()  + [0.0])
				v1 = np.array((project[-1] - center).tolist()  + [0.0])

				ref = cross(v0, v1)
				angles = angles * (angle(v0, v1, axis = ref, signed = True))

				count = i + 1
				for k in range(1, len(angles) - 1):
					data[count] = center + rotate_vector(v0, ref, angles[k])[:-1]
					count+=1

	return data, found, inter1, inter2, center


def intersection_cercle_cercle(data, normals, radius,i, j):

	found = False
	xi = data[i][0]
	yi = data[i][1]
	xj = data[j][0]
	yj = data[j][1]
	inter2 = []
	inter1 = []
	center = []
	alpha = -(yi-yj)/(xi-xj)
	gamma = (xi + xj)/2 + ((yi + yj)*(yi - yj))/(2*(xi-xj))
	a1  = alpha**2 + 1
	b1 = 2*alpha*(gamma - xi) - 2*yi
	c1 = (gamma-xi)**2 + yi**2 - radius**2
	if b1**2 - 4*a1*c1 >= 0:
		y1 = (-b1 - sqrt(b1**2 - 4*a1*c1))/(2*a1)
		x1 = alpha*y1 + gamma
		y2 = (-b1 + sqrt(b1**2 - 4*a1*c1))/(2*a1)
		x2 = alpha*y2 + gamma
		v0 = [x1-xi, y1-yi]

		v3 = [x1-xj, y1-yj]

		v1 = normals[i-1][0], normals[i-1][1]
		v2 = normals[i][0], normals[i][1]
		v4 = normals[j-1][0], normals[j-1][1]
		v5 = normals[j][0], normals[j][1]
		

		if ((angle(v1, v2) > 0 and angle(v0, v2) < angle(v1,v2)) or (angle(v1, v2) < 0 and angle(v0, v2) > angle(v1,v2))) and ((angle(v4, v5) > 0 and angle(v3, v5) < angle(v4,v5)) or (angle(v4, v5) < 0 and angle(v3, v5) > angle(v4,v5))):
			center = np.array([x1, y1])
			inter2 = data[i]
			inter1 = data[j]
			found = True
	   

		v0 = [x2-xi, y2-yi]
		r, phi0 = cart2pol(v0[0], v0[1])
		v3 = [x2-xj, y2-yj]
		r, phi3 = cart2pol(v1[0], v1[1])
		if ((angle(v1, v2) > 0 and angle(v0, v2) < angle(v1,v2)) or (angle(v1, v2) < 0 and angle(v0, v2) > angle(v1,v2))) and ((angle(v4, v5) > 0 and angle(v3, v5) < angle(v4,v5)) or (angle(v4, v5) < 0 and angle(v3, v5) > angle(v4,v5))):
			center = np.array([x2, y2])
			inter2 = data[i]
			inter1 = data[j]
			found = True
		
	if found :
		project = [inter2]
		for k in range(i+1, j+1):
			project.append(data[k,:])
			
		project.append(inter1)

		angles = [0.0]
		for k in range(len(project)-1):
			angles.append(angles[-1] + norm(project[k] - project[k+1]))

		# Compute angle between both intersections
		angles = angles / max(angles)

		v0 = np.array((project[0] - center).tolist()  + [0.0])
		v1 = np.array((project[-1] - center).tolist()  + [0.0])

		ref = cross(v0, v1)
		angles = angles * (angle(v0, v1, axis = ref, signed = True))

		count = i + 1
		for k in range(1, len(angles) - 1):
			data[count] = center + rotate_vector(v0, ref, angles[k])[:-1]
			count+=1

	
	return data, found, inter1, inter2, center


def intersection_segment_cercle(coefs, data, normals, radius,i, j):

	found =False
	xi = data[i][0]
	yi = data[i][1]
	inter2 = []
	inter1 = []
	center = []
	coef_dir =  coefs[j,0]
	ord_org = coefs[j,1]
	v = np.array([coef_dir, ord_org])

	A = v[0]**2 + 1
	B = -2*xi + 2*v[0]*(v[1] - yi)
	C = xi**2 + (v[1] - yi)**2 - radius**2

	Delta = B**2 - 4*A*C
	

	if Delta >= 0 and abs(i-j)>3:

		if Delta == 0:
			x = -B / (2*A)
			y = v[0] * x + v[1]
			sol = [np.array([x, y])]

		if Delta > 0:

			x1 = (-B + sqrt(Delta))/ (2*A)
			y1 =  v[0] * x1 + v[1]
			sol = [np.array([x1, y1])]

			x2 = (-B - sqrt(Delta))/ (2*A)
			y2 =  v[0] * x2 + v[1]
			sol.append(np.array([x2, y2]))
		
		v1 = normals[i-1][0], normals[i-1][1]
		v2 = normals[i][0], normals[i][1]

		for k in range(len(sol)):
			if not found:
				v0 = [sol[k][0]-xi, sol[k][1]-yi]
				inter1 = sol[k] - normals[j, :] * radius


			if (norm(data[j, :] - inter1) <= norm(data[j, :] - data[j + 1, :]) and norm(data[j+1, :] - inter1) < norm(data[j, :] - data[j + 1, :])) and ((angle(v1, v2) > 0 and angle(v0, v2) < angle(v1,v2)) or (angle(v1, v2) < 0 and angle(v0, v2) > angle(v1,v2))): # The intersection is on the segment
				center = np.array([sol[k][0], sol[k][1]])
				inter2 = [xi, yi]
				found = True
				if i > j:
					inter2 = sol[k] - normals[j, :] * radius
					inter1 = [xi, yi]

		if found:
			project = [inter2]

			if i<j:
				for k in range(i+1, j):
					project.append(data[k,:])
			else:
				for k in range(j+1, i):
					project.append(data[k,:])  
					
			project.append(inter1)
			angles = [0.0]

			for k in range(len(project)-1):
				angles.append(angles[-1] + norm(project[k] - project[k+1]))

			# Compute angle between both intersections
			angles = angles / max(angles)

			v0 = np.array((project[0] - center).tolist()  + [0.0])
			v1 = np.array((project[-1] - center).tolist()  + [0.0])

			ref = cross(v0, v1)
			angles = angles * (angle(v0, v1, axis = ref, signed = True))

			if i>j:
				count = j + 1
			else:
				count = i + 1
			
			for k in range(1, len(angles) - 1):
				data[count] = center + rotate_vector(v0, ref, angles[k])[:-1]
				count+=1


	return data, found, inter1, inter2, center



def smooth_polyline(data, radius, show=False):
	""" Smoothes a polyline using the inscribed circle method """

	def display(dataorg, normals, data, inter1, inter2, center, radius):



		figure, axes = plt.subplots(2, 1)
		ax1 = axes[0]
		ax2 = axes[1]

		ax1.set_aspect(1)
		ax2.set_aspect(1)

		colorline = 'red'
		colorcircle = 'grey'

		ylim = [dataorg[1:-1, 1].min() - 0.1, dataorg[1:-1, 1].max() + 0.1]
		xlim = [dataorg[1:-1, 0].min() - 0.1, dataorg[1:-1, 0].max() + 0.1]
		ylim[1] = max(ylim[1], center[1] + radius + 0.1)
		ax1.set_ylim(ylim)

		ax2.sharex(ax1)
		ax2.sharey(ax1)

		plt.xticks([])
		plt.yticks([])

		ax1.plot(dataorg[1:-1,0], dataorg[1:-1,1], color = colorline, linewidth=2)
		ax1.scatter(dataorg[1:-1,0], dataorg[1:-1,1], color = colorline, s=30)

		ax1.scatter(center[0], center[1], color = 'black', s = 20)

		circle = plt.Circle(center, radius=radius, alpha=0.4, color=colorcircle)
		ax1.add_artist(circle)

		circle2 = plt.Circle(center, radius=radius, alpha=0.4, color=colorcircle)
		ax2.add_artist(circle2)
		ax2.scatter(center[0], center[1], color = 'black', s = 20)

		ax2.scatter(data[1:-1,0], data[1:-1,1], color = colorline, s=30)		
		ax2.plot(data[1:-1,0], data[1:-1,1], color = colorline,linewidth=2)	
		plt.show()


	ext_length = 2

	pstart = data[0] + ((data[0] - data[1]) / norm(data[0] - data[1])) * ext_length
	pend = data[-1] + ((data[-1] - data[-2]) / norm(data[-1] - data[-2])) * ext_length
	data = np.vstack((pstart, data, pend))


	dataorg = data.copy()

	found = True

	start = 0
	while found: # As long as we find an intersection

		coefs = np.zeros((data.shape[0]-1, 2))
		normals =  np.zeros((data.shape[0]-1, 2))

		orient = 1

		# Get parallel line and normals for every point of the polyline
		for i in range(data.shape[0] - 1):
			
			coefs[i, 0] =  (data[i+1, 1] - data[i, 1])/(data[i+1, 0] - data[i, 0]) 
			n = (data[i+1] - data[i])[::-1]
			
			n = (data[i+1] - data[i])[::-1]
			n = n * orient * np.array([1,-1])

			n = n / norm(n)
			normals[i, :] = n
			pt = data[i] + n * radius

			coefs[i, 1] = pt[1] - coefs[i, 0] * pt[0]
			

		# Search for intersections

		for i in range(start, data.shape[0]-2):

			for j in range(i+2, data.shape[0]-1):

				data, found, inter1, inter2, center = intersection_cercle_cercle(data, normals, radius,i, j)
				if found:
					break

				data, found, inter1, inter2, center = intersection_segment_cercle(coefs, data, normals, radius,j, i)
				if found:
					break
				
				data, found, inter1, inter2, center = intersection_segment_cercle(coefs, data, normals, radius,i, j)
				if found:
					break
				

				data, found, inter1, inter2, center = intersection_segment_segment(coefs, data, normals, radius,i, j)
				if found:
					break		
						
			if found:
				start = j
				if start > data.shape[0]-4:
					found = False
					break
				if show:
					display(dataorg, normals, data, inter1, inter2, center, radius)
				break

	return data[1:-1,:]



def neighbor_faces(vertices, faces):
	""" Create a numpy array of the neighbor cells of every vertices"""

	faces = faces[:, 1:]
	adj_faces = np.zeros((vertices.shape[0], 7), dtype=int) - 1

	for i in range(vertices.shape[0]):

		face_ids = np.where(faces == i)[0]
		adj_faces[i, 0] = face_ids.shape[0]

		if face_ids.shape[0] == 4:

			if np.intersect1d(faces[face_ids[1]], faces[face_ids[2]]).shape[0] == 2:
				adj_faces[i, 1:1 + face_ids.shape[0]] = face_ids[[0,1,2,3]]
			else: 
				adj_faces[i, 1:1 + face_ids.shape[0]] = face_ids[[0,1,3,2]]

		elif face_ids.shape[0] == 6:

			if np.intersect1d(faces[face_ids[1]], faces[face_ids[3]]).shape[0] == 2:
				adj_faces[i, 1:1 + 6] = face_ids[[0,1,3,2,5,4]]

			elif np.intersect1d(faces[face_ids[1]], faces[face_ids[5]]).shape[0] == 2:
				adj_faces[i, 1:1 + 6] = face_ids[[0,1,5,4,3,2]]

			else: 
				print("PROBLEM")
		else: 
			adj_faces[i, 1:1 + face_ids.shape[0]] = face_ids

		"""		
		# Teste l'ordre des faces
			for j in range(face_ids.shape[0]):
				j2 = j + 1
				if j2 == face_ids.shape[0]:
					j2 = 0
				if np.intersect1d(faces[adj_faces[i, 1 + j]], faces[adj_faces[i, 1 + j2]]).shape[0] != 2:
					print("PROBLEME!")
					print(np.intersect1d(faces[adj_faces[i, 1 + j]], faces[adj_faces[i, 1 + j2]]).shape[0])
		"""

	return adj_faces


def neighbor_faces_id(vertices, faces, id_vertice):

	adj_faces = np.zeros((vertices.shape[0], 7), dtype=int) - 1
	face_ids = np.where(faces[:, 1:] == id_vertice)[0]

	return face_ids


def neighbor_faces_normals(normals, adj_faces, id_vertice):

	adj_normals = np.zeros((adj_faces[id_vertice, 0], 3))
	adj = adj_faces[id_vertice, 1:1 + adj_faces[id_vertice, 0]]

	for i in range(adj_faces[id_vertice, 0]):
		adj_normals[i, :] = normals[adj[i], :]

	return adj_normals


def neighbor_vertices_id(adj_faces, faces, id_vertice):
	""" Returns a list of the neighbor vertices of a given vertice"""

	adj = []
	for i in adj_faces[id_vertice, 1:1 + adj_faces[id_vertice, 0]]:
		face = faces[i, 1:].tolist()
		ind = face.index(id_vertice)

		if ind - 1 < 0:
			adj.append(face[-1])
		else:
			adj.append(face[ind - 1])

		if ind + 1 > 3:
			adj.append(face[0])
		else:
			adj.append(face[ind + 1])

	adj = np.unique(adj)

	return adj


def neighbor_vertices_coords(adj_faces, faces, vertices, id_vertice):

	adj_vertices = neighbor_vertices_id(adj_faces, faces, id_vertice)

	coords = np.zeros((len(adj_vertices), 3))
	for i in range(len(adj_vertices)):
		coords[i, :] = vertices[adj_vertices[i], :]

	return coords


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



def split_tubes(path_in, file_in, path_out, bifurcations = True):
	""" Take vmtk file, split the network in tubes and write swc files """
			

	centerline = pv.read(path_in + file_in)
	nb_centerlines = centerline.cell_arrays['CenterlineIds'].max() + 1

	cells = centerline.GetLines()
	cells.InitTraversal()
	idList = vtk.vtkIdList()

	radiusData = centerline.GetPointData().GetScalars('MaximumInscribedSphereRadius') 
	centerlineIds = centerline.GetCellData().GetScalars('CenterlineIds') 

	centerlines = []
	# Write centerline points
	for i in range(nb_centerlines):
		centerlines.append(np.array([]).reshape(0,4))

	g = 0
	pos = 0

	while cells.GetNextCell(idList):

		if g > 0 and c_id != centerlineIds.GetValue(g):
			pos += 1
			
		c_id = centerlineIds.GetValue(g)

		for i in range(0, idList.GetNumberOfIds()):

			pId = idList.GetId(i)
			pt = centerline.GetPoint(pId)
			radius = radiusData.GetValue(pId)

			centerlines[pos] = np.vstack((centerlines[pos], np.array([pt[0], pt[1], pt[2], radius])))
		g += 1

	# Write swc file
	for i in range(nb_centerlines):

		file_out = file_in[:-4] + "_tube_" + str(i) + ".swc" 
		file = open(path_out + file_out, 'w') 

		for j in range(centerlines[i].shape[0]):

			if j == 0:
				n = -1
				m = 1
			else:
				n = j - 1
				m = 3

			c = centerlines[i][j]

			file.write(str(j) + '\t' + str(m) + '\t' + str(c[0]) + '\t' + str(c[1]) + '\t' + str(c[2]) + '\t' + str(c[3]) + '\t' + str(n) + '\n')

		file.close()

		
def evaluate_model(spl1, data, spl2):
	""" Returns evaluation measures of the tube model : length, curvature, radius derivative, square distance to original spatial data, square distance to original radius"""

	# Compute length
	l1 = spl1.length()
	l2 = spl2.length()

	# Compute curvature
	t1 = spl1.get_times()
	curv1 = spl1.curvature(t1)

	t2 = spl2.get_times()
	curv2 = spl2.curvature(t2)

	# Radius derivative
	rad_der1 = spl1.first_derivative(t1)
	rad_der2 = spl2.first_derivative(t2)

	# Distance to data
	times, dist_spatial = spl2.distance(data)
	rad_estim = spl2.point(times, True)
	dist_rad = abs(data[:,-1] - rad_estim[:,-1])

	return l1, l2, curv1, curv2, rad_der1, rad_der2, dist_spatial, dist_rad


def chord_length_parametrization(D):

	""" Returns the chord length parametrization for data D.

	Keyword arguments:
	D -- data points
	"""
		
	t = [0.0]
	for i in range(1, len(D)):
		t.append(t[i-1] + np.linalg.norm(D[i] - D[i-1]))
	t = [time / max(t) for time in t]

	return t

#####################################
##########  MULTIPROCESSING  ########
#####################################


def parallel_bif(bif, N, d, end_ref):

	bif.compute_cross_sections(N, d, end_ref)
	return bif


def parallel_apex(spl):

	AP = []
	tAP = []

	for i in range(len(spl)):
		tAP.append([])

	# Find apex
	for i in range(len(spl) - 1):
		apex, time = spl[i].first_intersection(spl[i+1])
		AP.append(apex)
		tAP[i].append(time[0])
		tAP[i+1].append(time[1])

	return AP, tAP


def segment_crsec(spl, num, N, v0 = [], alpha = None):

	""" Compute the cross section nodes along for a vessel segment.

	Keyword arguments:
	spl -- segment spline
	num -- number of cross sections
	N -- number of nodes in a cross section (multiple of 4)
	v0 -- reference vector
	alpha -- rotation angle
	"""

	#t = np.linspace(0.0, 1.0, num + 2) #t = [0.0] + spl.resample_time(num) + [1.0]
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


def intersection(self, spl, v0, t0, t1):

	""" Returns the intersection point and time between two splines models, given a initial vector v0.

	Keywords arguments: 
	spl -- Spline object
	v0 -- reference vector for the search
	t0, t1 -- Times in between the search occurs
	"""

	def is_inside(t, v, spl):

		pt = self.project_time_to_surface(v, t) 
			
		t2 = spl.project_point_to_centerline(pt)
		pt2 = spl.point(t2, True)

		if norm(pt - pt2[:-1]) <= pt2[-1]:
			return True
		else: 
			return False


	tinit = t1
	# Check search interval
	if is_inside(t1, v0, spl):
		print("t1 is not set correctly.")

	v = self.transport_vector(v0, tinit, t0)
	if not is_inside(t0, v, spl):
		pass
		#print("t0 is not set correctly.")

	while abs(t1 - t0) > 10**(-6):

		t = (t1 + t0) / 2.
		v = self.transport_vector(v0, tinit, t)
		#pt = self.project_time_to_surface(v, t) 
			
		#t2 = spl.project_point_to_centerline(pt)
		#pt2 = spl.point(t2, True)

		if is_inside(t, v, spl): #norm(pt - pt2[:-1]) <= pt2[-1]:
			t0 = t
		else: 
			t1 = t

	t = (t0 + t1) / 2.
	v = self.transport_vector(v0, tinit, t)
	pt = self.project_time_to_surface(v, t) 
	t2 = spl.project_point_to_centerline(pt)	

	return pt, [t, t2]
