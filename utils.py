# Python 3

import numpy as np # Tools for matrices

# Trigonometry functions
from math import pi, sin, cos, tan, atan, acos, asin, sqrt
from numpy.linalg import norm 
from numpy import dot, cross, arctan2

import pyvista as pv

import matplotlib.pyplot as plt # Tools for plots
from scipy.spatial import KDTree


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



def length_polyline(D):

	""" Return the distance to origin for each point in a polyLine

	Keyword arguments:
	D -- list of node coordinates
	"""

	length = [0.0]

	for i in range(1, len(D)):
		length.append(length[i-1] + norm(D[i] - D[i-1]))

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
		D_resamp.append((D[ind] + orient/norm(orient)*d).tolist())


	return np.array(D_resamp)




#####################################
######## MESHING GEOMETRY ###########
#####################################

def search_intersection(data, center, radius, i):

	intersections = []
	indices = []

	for j in range(i, data.shape[0]-1):
		coef_dir =  (data[j+1, 1] - data[j, 1])/(data[j+1, 0] - data[j, 0]) 
		ord_org = data[j,1] - coef_dir * data[j, 0]
		v = np.array([coef_dir, ord_org])

		A = v[0]**2 + 1
		B = -2*center[0] + 2*v[0]*(v[1] - center[1])
		C = center[0]**2 + (v[1]- center[1])**2 - radius**2

		Delta = B**2 - 4*A*C

		if Delta >= 0:

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
			

			for k in range(len(sol)):
				if sol[k][0] >= data[j,0] and sol[k][0] <= data[j+1,0] and sol[k][1] >= data[j,1] and sol[k][1] <= data[j+1,1]: # The intersection is on the segment
					intersections.append(sol[k])
					indices.append(j)

	return intersections, indices



def smooth_polyline(data, radius):
	""" Smoothes a polyline using the inscribed circle method """

	figure, axes = plt.subplots()
	axes.set_aspect(1)

	axes.plot(data[:,0], data[:,1])
	axes.scatter(data[:,0], data[:,1])

	coefs = np.zeros((data.shape[0]-1, 2))
	normals =  np.zeros((data.shape[0]-1, 2))

	# Get parallel line for every point
	for i in range(data.shape[0] - 1):

		n = (data[i+1] - data[i])[::-1]
		if False: #n[1] <= 0:
			n = n * np.array([1,-1])
		else: 
			n = n * np.array([-1, 1])

		n = n / norm(n)
		normals[i, :] = n
		pt = data[i] + n * radius

		coefs[i, 0] =  (data[i+1, 1] - data[i, 1])/(data[i+1, 0] - data[i, 0]) 
		coefs[i, 1] = pt[1] - coefs[i, 0] * pt[0]
		
	coefs = np.around(coefs, 3)
	normals = np.around(normals, 3)
	axes.plot([data[:-1,0], data[:-1,0] + normals[:,0] * radius] , [data[:-1,1], data[:-1,1] + normals[:,1] * radius])


	# Search for intersections
	for i in range(data.shape[0]-1):
		print("search point", i)
		a = coefs[i, 0]
		b = coefs[i, 1]

		for j in range(i+2, data.shape[0]-1):

			c = coefs[j, 0]
			d = coefs[j, 1]

			if a != c:

				x = (b-d)/(c-a)
				y = a*x + b
				center = np.array([x, y])

				inter1 = center - normals[j, :] * radius
				inter2 = center - normals[i, :] * radius

					
				# Search if the intersections lies in the segments
				if (norm(data[j, :] - inter1) <= norm(data[j, :] - data[j + 1, :]) and norm(data[j+1, :] - inter1) < norm(data[j, :] - data[j + 1, :])):
					if (norm(data[i, :] - inter2) <= norm(data[i, :] - data[i + 1, :]) and norm(data[i+1, :] - inter2) < norm(data[i, :] - data[i + 1, :])):

						axes.scatter(inter1[0], inter1[1])	
						axes.scatter(inter2[0], inter2[1])	
						axes.scatter(center[0], center[1])

						circle = plt.Circle(center, radius=radius, alpha=0.3, color='red')
						axes.add_artist(circle)
							

						project = [inter2]

						for k in range(i, j+1):
							project.append(data[k,:])
							
						project.append(inter1)

						angles = [0.0]
						for k in range(len(project)-1):
							angles.append(angles[-1] + norm(project[k] - project[k+1]))

						# Compute angle between both intersections
						angles = angles / max(angles)
						v0 = project[0]- center
						v1 = project[-1] - center
						r, phi0 =cart2pol(v0[0], v0[1])
						r, phi1 =cart2pol(v1[0], v1[1])
						angles = angles * (phi1 - phi0)

						count = i
						for k in range(1, len(angles) - 1):
							data[count] = pol2cart(radius, phi0 + angles[k]) + center
							count+=1

						break

	plt.scatter(data[:,0], data[:,1])		
	plt.show()

	return data


def smooth_polylineorg(data, radius):
	""" Smoothes a polyline using the inscribed circle method """

	figure, axes = plt.subplots()
	axes.set_aspect(1)

	axes.plot(data[:,0], data[:,1])
	axes.scatter(data[:,0], data[:,1])



	# Get normals for every point

	for i in range(data.shape[0]):

		if i - 1 < 0:
			n = (data[i+1] - data[i])[::-1]

			if n[1] <= 0:
				n = n * np.array([1,-1])
			else: 
				n = n * np.array([-1, 1])

		elif i + 1 == data.shape[0]:
			n = (data[i] - data[i-1])[::-1]

			if n[1] <= 0:
				n = n * np.array([1,-1])
			else: 
				n = n * np.array([-1, 1])
		else:

			n = (data[i+1] - data[i])[::-1]

			if n[1] <= 0:
				n = n * np.array([1,-1])
			else: 
				n = n * np.array([-1, 1])


			n1 = (data[i+1] - data[i])[::-1]

			if n1[1] <= 0:
				n1 = n1 * np.array([1,-1])
			else: 
				n1 = n1 * np.array([-1, 1])

			n2 = (data[i] - data[i-1])[::-1]

			if n2[1] <= 0:
				n2 = n2 * np.array([1,-1])
			else: 
				n2 = n2 * np.array([-1, 1])

			n = (n1 / norm(n1) + n2 / norm(n2)) / 2.0

		n = n / norm(n)
		axes.plot([data[i,0], data[i,0] + n[0]], [data[i,1], data[i,1] + n[1]])

		center = data[i] + n * radius

		# Search for intersections
		intersections = search_intersection(data, center, radius, i+1)[0]
		
	
		if len(intersections)>0:

			print("collision")

			# Dichotomie
			b0 = data[i-1] + n * radius
			b1 = data[i] + n * radius

			while norm(b1-b0) > 10e-3:
				b = (b0 + b1) / 2.0
				nb_inter = len(search_intersection(data, b, radius, i+1)[0])

				if nb_inter == 1:
					b0 = b
					b1 = b
				if nb_inter > 1:
					b1 = b
				if nb_inter < 1:
					b0 = b

			new_c = plt.Circle(b1, radius=radius, alpha=0.3, color='red')
			axes.add_artist(new_c)
			intersection, indices = search_intersection(data, b1, radius, i+1)
			
			project = [b1 - n*radius]

			for k in range(i, indices[0]+1):
				project.append(data[k,:])
				
			project.append(intersection[0])

			coefs = [0.0]
			for k in range(len(project)-1):
				coefs.append(coefs[-1] + norm(project[k] - project[k+1]))

			coefs = coefs / max(coefs)
			v0 = project[0]- b1
			v1 = project[-1] - b1
			r, phi0 =cart2pol(v0[0], v0[1])
			r, phi1 =cart2pol(v1[0], v1[1])
			coefs = coefs * (phi1 - phi0)

			count = i
			for k in range(1, len(coefs) - 1):
				data[count] = pol2cart(radius, phi0 + coefs[k]) + b1
				count+=1


			"""

			c = (data[i+1, 1] - data[i, 1])/(data[i+1, 0] - data[i, 0]) 
			d = center[1] - c * center[0]	
			axes.plot([1, 10], [c*1+d, c*10+d])		

			for j in segment:
		
				a = (data[j+1, 1] - data[j, 1])/(data[j+1, 0] - data[j, 0]) 
				b = data[j,1] - a * data[j, 0]
				
				A = -4*((a+c)**2)
				B = 8*(d-b)*(a-c)
				C = -4*((b-d)**2) + 4*(a**2 +1)*(radius**2)

				Delta = B**2 - 4*A*C
				print("Delta", Delta, "B", B, "A", A)

				if Delta >= 0:

					if Delta == 0:
						x = -B / (2*A)
						y = c * x + d
						sol = [np.array([x, y])]

					if Delta > 0:

						x1 = (-B + sqrt(Delta))/ (2*A)
						y1 =  c * x1 + d
						sol = [np.array([x1, y1])]

						x2 = (-B - sqrt(Delta))/ (2*A)
						y2 =  c * x2 + d
						sol.append(np.array([x2, y2]))
					print(sol)

					for k in range(len(sol)):
						if sol[k][0] <= data[j,0] and sol[k][0] <= data[j+1,0] and sol[k][1] <= data[j,1] and sol[k][1] <= data[j+1,1]: # The intersection is on the segment
							circle = plt.Circle(sol, radius=radius, alpha=0.3)
							axes.add_artist(circle)
							"""

			break


			# Find the last fitting circle by dichotomie

			# Project points to the circle 

	axes.scatter(data[:,0], data[:,1])		
	plt.show()

	return data





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
	



#####################################
##########  MULTIPROCESSING  ########
#####################################


def parallel_bif(bif, N, d, end_ref =[None, None, None]):

	# Find cross sections
	bif.cross_sections(N, d, end_ref)
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

	t = np.linspace(0.0, 1.0, num + 2) #t = [0.0] + spl.resample_time(num) + [1.0]

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