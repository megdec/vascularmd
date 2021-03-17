# Python 3
import numpy as np # Tools for matrices
import pyvista as pv # Meshing

# Trigonometry functions
from math import pi
from numpy.linalg import norm 
from numpy import dot, cross

import matplotlib.pyplot as plt # Tools for plots
from mpl_toolkits.mplot3d import Axes3D # 3D display

from utils import *
from Spline import Spline


class Nfurcation:

	""" Generalization of the bifurcation code to coplanar multifurcations with n branches """

	#####################################
	##########  CONSTRUCTOR  ############
	#####################################

	def __init__(self, model, args):

		""" Keyword arguments:
		method -- furcation model used (spline or crsec)
		if "spline" : args = [[spl1, spl2, ...], [AP1, AP2,...], R]
		if "crsec" : args = [[crsec1, crsec2, ...], [AP_crsec1, AP_crsec2...], AP, R]
		if "angle" : args = [[crsec1, crsec2, ...], [a1, a2...], R]
		"""

		self.model = model
		
		# Set user paramaters
		self.R = args[2]  # Minimum curvature for rounded apex

		if model == "spline":

			# Set end sections
			self._spl = args[0]
			self._AP = args[1]

			self._endsec = []

			for i in range(len(args[0]) + 1):
				self._endsec.append([])

			for i in range(len(args[0])):
				self._endsec[0] = np.vstack((self._spl[i].point(0.0, True), self._spl[i].tangent(0.0, True)))
				self._endsec[i+1] = np.vstack((self._spl[i].point(1.0, True), self._spl[i].tangent(1.0, True)))

		elif model == "crsec":

			# Set end sections
			self._endsec = args[0]
			self._AP = args[2]
			self.__set_spl(args[1])

		self.n = len(self._endsec) - 1


		self.__set_tAP() # Times at apex

		self.__set_key_pts() # Set key points times
		self.__set_B() # Geometric center (np array)
		self.__set_SP() # Separation points (np array)
		self.__set_CP() # Common points (np array)

		self.__set_tspl() # Trajectory splines

	

	#####################################
	#############  GETTERS  #############
	#####################################

	def get_spl(self):
		return self._spl

	def get_tspl(self):
		return self._tspl

	def get_endsec(self):
		return self._endsec

	def get_B(self):
		return self._B

	def get_crsec(self):
		if self._crsec != None:
			return self._crsec

	def get_AP(self):
		return self._AP

	def get_tAP(self):
		return self._tAP

	def get_reference_vectors(self):

		""" Returns the reference vectors for the three end sections of the bifurcation """
		ref_list = []

		key_pts = self._SP[::-1] + self._AP[:-1] + [self._SP[1]]
		spl = [self._tspl[0]] + self._spl

		for i in range(len(spl)):
			
			t = 1.0
			if i == 0:
				t = 0.0
			
			# Reference vector
			tS = spl[i].project_point_to_centerline(key_pts[i])
			ptS = spl[i].point(tS)
			nS = spl[i].transport_vector(key_pts[i] - ptS, tS, t)

			if i == 0 or i == len(spl)- 1:
				ref = cross(self._endsec[i][1][:-1], nS)
			else:
				ref = cross(nS, self._endsec[i][1][:-1])

			ref_list.append(ref / norm(ref))
			
		return ref_list
	

	def get_crsec_normals(self):

		mesh = self.mesh_surface()
		mesh.compute_normals(cell_normals=False, inplace=True)
		normals = mesh['Normals']

		pts = self._crsec

		# Get points and re-order them in the original structure
		bif_normals = normals[:len(pts[1])] # Fill bifurcation
		end_normals = []
		nds_normals = []
		N = len(pts[0][0])

		j = len(pts[1])
		for s in range(len(self._tspl)):
			end_normals.append(mesh.points[j:j+N]) # Fill end_sections
			j += N

			n = []
			for i in range(len(pts[2][s])):
				n.append(mesh.points[j:j+N].tolist()) # Fill connecting nodes
				j += N
			nds_normals.append(np.array(n))

		return bif_normals, end_normals, nds_normals	


	def get_curves(self):
		""" Get all the curves along bifurcation """

		curve_set = []
		normals_set = []
		bif_normals, end_normals, nds_normals = self.get_crsec_normals()
		end_crsec, bif_crsec, nds_crsec, connect = self._crsec
		connect = np.array(connect)

		for i in range(2, len(bif_crsec)):


			ind = np.where(connect == i)
			ind = list(zip(ind[0], ind[1]))

			curve_set.append(np.vstack((end_crsec[ind[0][0]][ind[0][1]], nds_crsec[ind[0][0]][:, ind[0][1]], bif_crsec[i],  nds_crsec[ind[1][0]][:, ind[1][1]][::-1], end_crsec[ind[1][0]][ind[1][1]])))
			normals_set.append(np.vstack((end_normals[ind[0][0]][ind[0][1]], nds_normals[ind[0][0]][:, ind[0][1]], bif_normals[i],  nds_normals[ind[1][0]][:, ind[1][1]][::-1], end_normals[ind[1][0]][ind[1][1]])))

		return curve_set, normals_set


	def set_curves(self, curve_set):
		""" Set all the curves along bifurcation is crsec format """
		pass





	#####################################
	###### SET SHAPE PARAMETERS  ########
	#####################################


	def set_R(self, R):#(?)

		""" Set the R parameter of the bifurcation."""

		self.R = R


	def __set_spl(self, AP_crsec): 

		""" Set the shape splines of the bifurcation.

		Keyword arguments:
			AP_crsec -- crsec sections at the apical region
			r -- norm of the end tangents. 
		"""

		
		# Compute the shape splines from cross sections
		self._spl = []

		# Correct the radius 
		if len(AP_crsec) < 0:
			D0, D1, D2 = self._endsec[0][0][-1]*2, self._endsec[1][0][-1]*2, self._endsec[2][0][-1]*2
			AP_crsec[0][0][-1] = ((sqrt(2) * D0*D1)/(sqrt(D1**2 + D2**2))) / 2.
			AP_crsec[1][0][-1] = ((sqrt(2) * D0*D2)/(sqrt(D1**2 + D2**2))) / 2.
	

		for i in range(1, len(self._endsec)):

			C0, C1, C2 = self._endsec[0][0], AP_crsec[i-1][0], self._endsec[i][0] 
			T0, T1, T2 = self._endsec[0][1], AP_crsec[i-1][1], self._endsec[i][1] 

			p0 = C0 + 0.25 * norm(C1 - C0) * T0
			p0[-1] = C0[-1] + 0.25 * abs((C1[-1] - C0[-1]))
			p1 = C1 - 0.25 * norm(C1 - C0) * T1
			p1[-1] = C1[-1] - 0.25 * abs((C1[-1] - C0[-1]))

			p2 = C1 + 0.25 * norm(C2 - C1) * T1
			p2[-1] = C1[-1] + 0.25 * abs((C2[-1] - C1[-1]))
			p3 = C2 - 0.25 * norm(C2 - C1) * T2
			p3[-1] = C2[-1] - 0.25 * abs((C2[-1] - C1[-1]))

			# Fit splines
			spl1 = Spline()
			spl1.approximation(np.vstack((C0,p0,p1,C1)), [1,1,1,1], np.vstack((C0, T0, T1, C1)), False, n = 4, radius_model=False, criterion= "None")
			P1 = spl1.get_control_points()

			spl2 = Spline()
			spl2.approximation(np.vstack((C1,p2,p3,C2)), [1,1,1,1], np.vstack((C1, T1, T2, C2)), False, n = 4, radius_model=False, criterion= "None")
			P2 = spl2.get_control_points()

			P = np.vstack((P1[:-1], P2))

			knot = [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0]

			self._spl.append(Spline(control_points = P, knot = knot))

		# Check angles of the apex cross sections and apply rotation correction if necessary
		#for i in range(self.n):
		#	vec = (AP_crsec[i-1][0][:-1] - self._AP[0])
		#	OP = AP_crsec[i-1][0][:-1] + AP_crsec[i-1][0][-1] * vec / norm(vec)

			# Test if OP is at the surface of all splines

			# If not, rotation


	def __set_tAP(self): 

		""" Set the times at apex. """

		self._tAP = []
		for i in range(len(self._spl)):
			self._tAP.append([])

		for i in range(len(self._AP)):
			self._tAP[i].append(self._spl[i].project_point_to_centerline(self._AP[i]))
			self._tAP[i + 1].append(self._spl[i+1].project_point_to_centerline(self._AP[i]))



	#####################################
	####### SET MESH PARAMETERS  ########
	#####################################


	def __set_key_pts(self):

		""" Compute VMTK key points """
		t_list = []
		self._key_pts = []

		for i in range(self.n):

			self._key_pts.append([])
			spl_list = np.arange(self.n).tolist()
			spl_list.remove(i)
			times = []

			for j in spl_list:
				times.append(self._spl[i].first_intersection_centerline(self._spl[j], t0=0, t1=1)[1][0])

			t1 = max(times)
			t0 = 0.0

			L1 = self._spl[i].time_to_length(t1)

			while abs(t0-t1) < 10e-3:
				t = (t0 + t1) / 2.

				length = L1 - self._spl[i].time_to_length(t)
				radius = self._spl[i].radius(t)

				if length == radius:
					t0 = t
					t1 = t

				elif length < radius:
					t0 = t
				else: 
					t1 = t

			self._key_pts[i] = [max(times),  (t0 + t1)/2.]
		



	def __set_B(self): 

		""" Set the coordinates of the geometric center of the nfurcation. """

		# Antiga 2015
		a = 0
		b = 0
		for i in range(self.n):
			for j in range(len(self._key_pts[i])):
				a += self._spl[i].point(self._key_pts[i][j]) * self._spl[i].radius(self._key_pts[i][j])**2
				b += self._spl[i].radius(self._key_pts[i][j])**2

		self._B = a / b



	def __set_SP(self):

		""" Set the coordinates of the two separation points S of the nfurcation. """

		# Project X to the spline to get t
		SP = []
		spl_ind = [0, len(self._spl) - 1]

		for ind in [0, -1]:
			# Project X to the spline to get t
			t = self._spl[ind].project_point_to_centerline(self._B)

			ptAP = self._spl[ind].point(self._tAP[ind][0])
			nAP = self._AP[ind] - ptAP

			nAP0 = self._spl[ind].transport_vector(nAP, self._tAP[ind][0], 1.0) 
			nS = self._spl[ind].transport_vector(-nAP0, 1.0, t) 
		
			ptS = self._spl[ind].point(t)  
			SP.append(self.__send_to_surface(ptS, nS / norm(nS), spl_ind[ind]))	

		self._SP = SP




	def __set_CP(self): 

		""" Set the coordinates of the common points CP of the bifurcation. """
		normals = []

		for i in range(len(self._AP)):
			n = cross(self._AP[i] - self._SP[0], self._AP[i] - self._SP[1])
			normals.append(n/norm(n))

		n = sum(normals) / len(normals)

		self._CP = [self.__send_to_surface(self._B, n, 0), self.__send_to_surface(self._B, -n, 0)]



	def __set_tspl(self, r=6):

		""" Set the coordinates of the trajectory splines. """

		tspl = []
		key_pts = self._SP[::-1] + self._AP

		for i in range(len(key_pts)):

			i2 = i+1
			if i == len(key_pts) - 1:
				i2 = 0

			tg0 = -self._endsec[i][1][:-1]

			if i == 0:
				tg0 = -tg0

			tg1 = ((key_pts[i] - self._B) + (key_pts[i2] - self._B)) / 2.0

			p0 = self._endsec[i][0][:-1]
			p1 = self._B


			d = norm(p0 - p1)

			pint0 = np.array(p0) + tg0 * (d/r) / norm(tg0)
			pint1 = np.array(p1) +  tg1 * (d/r) / norm(tg1)

			P = np.vstack([p0, pint0,  pint1, p1])
			P = np.hstack((P, np.zeros((4,1))))

			tspl.append(Spline(P))

		self._tspl = tspl



	def set_crsec(self, mesh):

		pts = self._crsec

		# Get points and re-order them in the original structure
		self._crsec[1] = mesh.points[:len(pts[1])] # Fill bifurcation
		end_crsec = []
		nds = []
		N = len(pts[0][0])

		j = len(pts[1])
		for s in range(len(self._tspl)):
			end_crsec.append(mesh.points[j:j+N]) # Fill end_sections
			j += N

			n = []
			for i in range(len(pts[2][s])):
				n.append(mesh.points[j:j+N].tolist()) # Fill connecting nodes
				j += N
			nds.append(np.array(n))
	
		self._crsec[0] = end_crsec
		self._crsec[2] = nds




	#####################################
	##########  VISUALIZATION  ##########
	#####################################
	

	def show(self, nodes = False):

		""" Display the bifurcation key points and modeling splines."""

		# 3D plot
		with plt.style.context(('ggplot')):
				
			fig = plt.figure(figsize=(10,7))
			ax = Axes3D(fig)
			ax.set_facecolor('white')
				
			# Plot the shape splines 
			for s in self._spl:
				points = s.get_points()
				ax.plot(points[:,0], points[:,1], points[:,2])

			# Plot the trajectory splines 
			for s in self._tspl:
				points = s.get_points()
				ax.plot(points[:,0], points[:,1], points[:,2],  c='black')

			# Plot control points
			ax.scatter(self._B[0], self._B[1], self._B[2], c='black', s = 40)
			for pt in self._SP:
				ax.scatter(pt[0], pt[1], pt[2], c='green', s = 40)
			for pt in self._CP:
				ax.scatter(pt[0], pt[1], pt[2], c='blue', s = 40)
			for pt in self._AP:
				ax.scatter(pt[0], pt[1], pt[2], c='red', s = 40)

			# Plot key times
			for i in range(self.n):
				pt1 = self._spl[i].point(self._key_pts[i][0])
				pt2 = self._spl[i].point(self._key_pts[i][1])

				ax.scatter(pt1[0], pt1[1], pt1[2], c='black', s = 20)
				ax.scatter(pt2[0], pt2[1], pt2[2], c='black', s = 20)


			if nodes: 

				N = 24
				# Separation sections
				nds = self.__separation_section(N)
				ax.scatter(nds[:,0], nds[:,1], nds[:,2],  c='black')

				# End sections
				nds = self.__end_sections(24)

				for i in range(len(nds)):
					ax.scatter(nds[i][:,0], nds[i][:,1], nds[i][:,2],  c='black')



		# Set the initial view
		ax.view_init(90, -90) # 0 is the initial angle

		# Hide the axes
		ax.set_axis_off()
		plt.show()



	#####################################
	#########  MESHING METHODS  #########
	#####################################


	def mesh_surface(self, N=24, d=0.2):

		""" Returns the surface mesh of the bifurcation

		Keyword arguments:
		N -- number of nodes in a cross section
		d -- density of cross section along the vessel (proportional to the radius)
		If not given, the cross section are computed with default parameters.
		"""

		if self._crsec == None:
			end_crsec, bif_crsec, nds, ind = self.cross_sections(N, d)
		else: 
			end_crsec, bif_crsec, nds, ind = self._crsec

		vertices = []
		faces = []

		vertices += bif_crsec.tolist()
		 
		for e in range(len(self._tspl)):

			# Mesh the end cross section
			v_prec = end_crsec[e]
			ind_dep = len(vertices)
			vertices += v_prec.tolist()
			
			seg_crsec = nds[e]
			# Mesh edges
			for i in range(len(seg_crsec)): 

				v_act = seg_crsec[i]
				for j in range(len(v_act)):

					if j == len(v_act) - 1:
						j2 = 0
					else:
						j2 = j + 1
			
					if e != 0: 
						faces.append([4, ind_dep + j, len(vertices) + j,  len(vertices) + j2, ind_dep + j2])
					else: # Flip normals
						faces.append([4, ind_dep + j, ind_dep + j2, len(vertices) + j2, len(vertices) + j])

				ind_dep = len(vertices)
				vertices += v_act.tolist()
				v_prec = v_act
	
			# Mesh the bifurcation
			ind_dep = len(vertices) - len(v_prec)
			connect = ind[e]
			for j in range(len(v_prec)): 
				if j == len(v_prec) - 1:
					j2 = 0
					j3 = connect[0]
				else:
					j2 = j + 1
					j3 = connect[j+1] 
				
				if e != 0:
					faces.append([4, ind_dep + j, connect[j], j3, ind_dep + j2])
				else: # Flip normals
					faces.append([4, ind_dep + j, ind_dep + j2, j3, connect[j]])

		mesh = pv.PolyData(np.array(vertices), np.array(faces))
		#mesh.plot(show_edges = True)

		return mesh
		
		


	def cross_sections(self, N, d, end_ref = None):


		""" Returns the nodes of the surface mesh ordered by transverse sections
		for a bifurcation defined by three end cross sections.

		Keyword arguments:

		N -- number of nodes in a transverse section (multiple of 4)
		d -- longitudinal density of nodes as a proportion of the radius

		Output: 

		end_crsec -- list of coordinates of the end cross sections
		bif_crsec -- list of the coordinates of the separation plan
		connect_index -- connectivity indices of the end cross sections and the separation plan
		nds -- list of connecting nodes
		"""

		# Get separation section nodes and end section nodes
		bif_crsec = self.__separation_section(N)
		end_crsec = self.__end_sections(N, end_ref)

		# Write connectivity index
		connect_index = []

		nodes_num = np.arange(2, len(bif_crsec)).tolist()
		l = len(nodes_num) // len(self._tspl)

		ind = []
		for i in range(len(self._tspl)):
			ind.append(nodes_num[i*l:(i+1)*l])

		connect_index.append([1] + ind[0][::-1] + [0] + ind[-1])

		for i in range(len(self._spl)):
			connect_index.append([1] + ind[i][::-1] + [0] + ind[i + 1])

		# Compute connecting nodes
		nds = []
		ind = [0] + np.arange(0, len(self._spl)).tolist() 

		for i in range(len(self._tspl)):

			num = int(self._tspl[i].length() / (self._endsec[i][0][-1] * d))

			# Minimum value for the number of cross sections
			if num == 0:
				num = 2

			nds_seg = np.zeros((num, N, 3))
		
			for j in range(N):
				nds_seg[:, j, :] = self.__bifurcation_connect(i, ind[i], end_crsec[i][j], bif_crsec[connect_index[i][j]], num)  

			nds.append(nds_seg)

		self._crsec = [end_crsec, bif_crsec, nds, connect_index]
		self.resample()
		self.smooth(1)
		return self._crsec



	def __bifurcation_connect(self, tind, ind, P0, P1, n):


		""" Compute the nodes connecting an end point to a separation point.

		Keyword arguments: 
		tind -- index of the trajectory spline
		ind -- index of the shape spline of reference 
		P0, P1 -- end points
		n -- number of nodes
		"""

		P0 = np.array(P0)
		P1 = np.array(P1)

		# Method 1 using surface splines
		d = norm(P0 - P1)

		tg0 = self._tspl[tind].tangent(0.0)
		tg1 = -self._tspl[tind].tangent(1.0)

		pint0 = P0 +  tg0 * (d/6) / norm(tg0)
		pint1 = P1 +  tg1 * (d/6) / norm(tg1)
		P = np.vstack([P0, pint0,  pint1, P1])
		P = np.hstack((P, np.zeros((4,1))))
		
		tsplsurf = Spline(P)
		tsamp = tsplsurf.resample_time(n)
		pts = []
		for t in tsamp:
			pts.append(tsplsurf.point(t))
		pts = np.array(pts)

		# Method 2 using linear interpolation
		#pts = []
		#for i in range(len(P0)):
		#	pts.append(np.linspace(P0[i], P1[i], n + 2)[1:-1].tolist())
		#pts = np.array(pts).transpose()

		nds = np.zeros((n,3))
		for i in range(n):

			t = self._tspl[tind].project_point_to_centerline(pts[i])
			P = self._tspl[tind].point(t)
			n =  pts[i] - P
			n = n / norm(n)
			
			pt = self.__send_to_surface(P, n, ind)
			nds[i] = pt
		
		return nds



	def __separation_section(self, N): 


		""" Returns the nodes of the separation plan.
	
		Keyword arguments: 
		N -- number of nodes in a section (multiple of 4)
		"""

		key_pts = [self._SP[0]]  + self._AP + [self._SP[1]]

		nds = np.zeros((2 + (N//2 - 1) * len(key_pts),3))
		nds[:2] = self._CP
	
		n = N//4 - 1

		j = 2
		for i in range(len(key_pts)):

			nds[j: j + n] = self.__separation_segment(self._CP[0], key_pts[i], n)
			nds[j + n] = key_pts[i]
			nds[j+n+1: j + 2*n +1] = self.__separation_segment(key_pts[i], self._CP[1], n) 
			j += 2*n + 1

		return nds



	def __separation_segment(self, P1, P2, n): 


		""" Returns the nodes of the half section between p1 and p2

		Keyword arguments:
		p1, p2 -- end points
		n --  number of required nodes for the segment

		"""

		v1 = P1 - self._B
		v2 = P2 - self._B

		theta = (directed_angle(v1, v2, cross(v1,v2)) / (n + 1)) * np.arange(1, n + 1) 

		# Computing nodes using t and theta parameters
		nds = np.zeros((n, 3))
		for i in range(n):

			n = rotate_vector(v1, cross(v1, v2), theta[i])
			nds[i, :] = self.__send_to_surface(self._B, n, 0)

		return nds



	def __end_sections(self, N, end_ref = None):


		""" Returns the nodes of end sections.

		Keyword arguments: 
		N -- number of nodes in a section (multiple of 4)
		"""
		
		nds = []

		key_pts = self._SP[::-1] + self._AP[:-1] + [self._SP[1]]
		spl = [self._tspl[0]] + self._spl

		for i in range(len(spl)):

			sec = np.zeros((N,3))

			if end_ref is not None: 
				ref = end_ref[i]
			else: 

				t = 1.0
				if i == 0:
					t = 0.0
				
				# Reference vector
				tS = spl[i].project_point_to_centerline(key_pts[i])
				ptS = spl[i].point(tS)
				nS = spl[i].transport_vector(key_pts[i] - ptS, tS, t)

				if i == 0 or i == len(spl)- 1:
					ref = cross(self._endsec[i][1][:-1], nS)
				else:
					ref = cross(nS, self._endsec[i][1][:-1])

			ref = ref / norm(ref)
			angle_list = (2 * pi / N) * np.arange(N)

			for j in range(N):
				n = np.array(rotate_vector(ref, self._endsec[i][1][:-1], angle_list [j]))
				sec[j, :] = self._endsec[i][0][:-1] + n * self._endsec[i][0][-1] / norm(n)

			nds.append(sec)

		return nds


	#####################################
	########## POST PROCESSING  #########
	#####################################

	def resample(self):
		""" Resamples the cells of the bifurcation """

		init_mesh = self.mesh_surface()
		self.smooth(5)
		self.deform(init_mesh)


	def smooth(self, n_iter):

		""" Smoothes the bifurcation.

		Keyword arguments:
		n_iter : number of iteration for the smoothing
		"""

		if self._crsec == None:
			raise ValueError('Please first perform cross section computation.')
		else: 
			pts = self._crsec

		mesh = self.mesh_surface()
		mesh = mesh.smooth(n_iter, boundary_smoothing=False, relaxation_factor=0.8) # Laplacian smooth
		self.set_crsec(mesh)




	def deform(self, mesh):

		""" Deforms the original bifurcation mesh to match a given surface mesh. 
		Overwrite the bifurcation nodes.

		Keywords arguments: 
		mesh -- a surface mesh in vtk format
		"""

		if self._crsec is None:
			raise ValueError("Please compute bifurcation cross sections")

		end_crsec = self._crsec[0]
		bif_crsec = self._crsec[1]
		nds = self._crsec[2] 

		# Project end cross sections
		for i in range(len(end_crsec)):
			center = (np.array(end_crsec[i][0]) + end_crsec[i][int(len(end_crsec[i])/2)])/2.0 # Compute the center of the section
			for j in range(len(end_crsec[i])):
				end_crsec[i][j] = self.__intersection(mesh, center, end_crsec[i][j])

		# Project connecting nodes
		for i in range(len(nds)):
			for j in range(len(nds[i])):
				center = (np.array(nds[i][j][0]) + nds[i][j][int(len(nds[i][j])/2)])/2.0 # Compute the center of the section
				for k in range(len(nds[i][j])):
					nds[i][j, k] = self.__intersection(mesh, center, nds[i][j][k])

		# Project bifurcation plan
		for i in range(len(bif_crsec)):
			bif_crsec[i] = self.__intersection(mesh, self._B, bif_crsec[i])




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


	#####################################
	##############  UTILS  ##############
	#####################################


	def __send_to_surface(self, O, n, ind):

		""" Sends a point to the surface defined by all shape splines according to direction n """

		# Project to main ind
		pt = self.__projection(O, n, 0, 2.5, ind)

		ind_list = np.arange(0, len(self._spl)).tolist()
		ind_list.remove(ind)

		# Check distance to other splines
		for i in ind_list:
			t = self._spl[i].project_point_to_centerline(pt)
			pt2 = self._spl[i].point(t, True)

			# Check distance
			if norm(pt - pt2[:-1]) < pt2[-1]:
				pt = self.__projection(pt, n, 0.0, 2.0, i)

		return pt



	def __projection(self, O, n, c0, c1, ind):

		""" Sends a point to the surface defined by shape spline of ind ind according to direction n.

		Keyword arguments: 
		O -- origin point as numpy array
		n -- direction (3D vector) as numpy array
		c0, c1 -- search limits
		ind -- index of the spline
		"""

		n = n / norm(n)

		# Checking inital born
		if c1<c0:
			raise ValueError("We must have c1>c0.")
		
		t = self._spl[ind].project_point_to_centerline(O + c1 * n)
		pt = self._spl[ind].point(t, True)

		if False: #norm(pt[:-1] - (O + c1 * n)) < pt[-1]: # Incorrect initial born
			return self.__projection(O, n, c0, c1 - 0.01, ind)

		else:

			while abs(c0 - c1)> 10**(-3):

				c = (c0 + c1) / 2.0

				t = self._spl[ind].project_point_to_centerline(O + c * n)
				pt = self._spl[ind].point(t, True)

				if norm(pt[:-1] - (O + c * n)) < pt[-1]:
					c0 = c
				else:
					c1 = c

			pt = O + ((c0 + c1) / 2.0) * n

			return pt












