# Python 3
import numpy as np # Tools for matrices
import pyvista as pv # Meshing
import pickle

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
		if "spline" : args = [[spl1, spl2, ...], R]
		if "spline" : args = [[spl1, spl2, ...], [[AP1], [AP2],...], R]
		if "crsec" : args = [[crsec1, crsec2, ...], [[AP_crsec1], [AP_crsec2]...], [AP], R]
		if "angle" : args = [[crsec1, crsec2, ...], [a1, a2...], R]
		"""
	
		self.model = model
		
		# Set user paramaters
		self.R = args[-1]  # Minimum curvature for rounded apex

		if model == "spline":

			# Set end sections
			self._spl = args[0]
			if len(args) < 3:
				self.__set_AP()
			else:
				self._AP = args[1]

			self._endsec = []

			for i in range(len(args[0]) + 1):
				self._endsec.append([])

			for i in range(len(args[0])):
				self._endsec[0] = np.vstack((self._spl[i].point(0.0, True), self._spl[i].tangent(0.0, True)))
				self._endsec[i+1] = np.vstack((self._spl[i].point(1.0, True), self._spl[i].tangent(1.0, True)))

			self.n = len(self._endsec) - 1
			self.__set_tAP() # Times at apex
			self.__set_apexsec()

		elif model == "crsec":

			# Set end sections
			self._endsec = args[0]
			self._apexsec = args[1]
			self.n = len(self._endsec) - 1

			self._AP = args[2]
			self.__set_spl()
			self.__set_tAP() # Times at apex


		self.__set_key_pts() # Set key points times
		self.__set_X() # Geometric center (np array)
		self.__set_SP() # Separation points (np array)
		self.__set_CT() # Common points (np array)

		self.__set_tspl() # Trajectory splines
		self._crsec = None
		self._N = 24 # Number of nodes in a cross section
		self._d = 0.2 # Ratio of density of cross section

	

	#####################################
	#############  GETTERS  #############
	#####################################

	def get_spl(self):
		return self._spl

	def get_tspl(self):
		return self._tspl

	def get_endsec(self):
		return self._endsec

	def get_apexsec(self):
		return self._apexsec

	def get_X(self):
		return self._X

	def get_crsec(self):
		if self._crsec != None:
			return self._crsec

	def get_AP(self):
		return self._AP

	def get_tAP(self):
		return self._tAP

	def get_N(self):
		return self._N

	def get_d(self):
		return self.d

	def get_R(self):
		return self.R

	def get_n(self):
		return self.n 

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

		""" Returns the mesh normals for the nodes of the bifurcation in crsec format """

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

		""" Get all the curves along bifurcation and the projection referential """

		curve_set = []
		referential_set = []
		bif_normals, end_normals, nds_normals = self.get_crsec_normals()
		end_crsec, bif_crsec, nds_crsec, connect = self._crsec
		connect = np.array(connect)

		N = (len(bif_crsec) - 2) // len(nds_crsec)

		count = 0
		for i in range(2, len(bif_crsec)):

			if count == N: 
				count = 0

			ind = np.where(connect == i)
			ind = list(zip(ind[0], ind[1]))

			curve_set.append(np.vstack((end_crsec[ind[0][0]][ind[0][1]], nds_crsec[ind[0][0]][:, ind[0][1]], bif_crsec[i],  nds_crsec[ind[1][0]][:, ind[1][1]][::-1], end_crsec[ind[1][0]][ind[1][1]])))

			normal = bif_normals[i]
			normal = normal / norm(normal)
			if count == 0:
				t1 = bif_crsec[0] - bif_crsec[i]
				t2 = bif_crsec[i] - bif_crsec[i + 1] 
			elif count == N-1:
				t1 = bif_crsec[i-1] - bif_crsec[i]
				t2 = bif_crsec[i] - bif_crsec[1] 
			else: 
				t1 = bif_crsec[i-1] - bif_crsec[i]
				t2 = bif_crsec[i] - bif_crsec[i + 1] 

			tangent = (t1 + t2) / 2.
			tangent = tangent / norm(tangent)			

			binormal = cross(tangent, normal)
			binormal = binormal / norm(binormal)

			tangent = cross(normal, binormal)
			tangent = tangent / norm(tangent)	

			if count > (N - 1)//2:
				tangent = -tangent

			origin = bif_crsec[i]


			referential_set.append(np.vstack((binormal, normal, tangent, origin)))
			count += 1

		return curve_set, referential_set


	def curves_to_crsec(self, curve_set):

		""" Set all the curves along bifurcation is crsec format """

		end_crsec, bif_crsec, nds_crsec, connect = self._crsec
		connect = np.array(connect)

		k = 0
		for i in range(2, len(bif_crsec)):

			ind = np.where(connect == i)
			ind = list(zip(ind[0], ind[1]))

			j = 0
			end_crsec[ind[0][0]][ind[0][1]] = curve_set[k][j]
			nds_crsec[ind[0][0]][:, ind[0][1]] = curve_set[k][j+1:j+1 + nds_crsec[ind[0][0]].shape[0]]
			j = j + 1 + nds_crsec[ind[0][0]].shape[0]

			bif_crsec[i] = curve_set[k][j]
			nds_crsec[ind[1][0]][:, ind[1][1]] = curve_set[k][j+1:-1][::-1]
			end_crsec[ind[1][0]][ind[1][1]] = curve_set[k][-1]

			k += 1

		self._crsec = [end_crsec, bif_crsec, nds_crsec, connect]


		


	#####################################
	###### SET SHAPE PARAMETERS  ########
	#####################################

	def set_crsec(self, crsec):
		self._crsec = crsec

	def set_R(self, R):

		""" Set the R parameter of the bifurcation."""

		self.R = R

	def __set_apexsec(self):

		self._apexsec = []
		for i in range(len(self._spl)):
			self._apexsec.append(np.vstack((self._spl[i].point(self._tAP[i][0], True), self._spl[i].tangent(self._tAP[i][0], True))))


	def set_apexsec_radius(self, radius, branch_id):

		""" Change the radius of the apex cross section of index i but keep the apex position. Only works for bifurcations!
		Keyword arguments:
		radius -- Value of the radius to apply
		branch_id -- id of the apex section to modify """

		v = self._apexsec[branch_id][0][:-1] - self._AP[0]
		v = v / norm(v)
		
		self._apexsec[branch_id][0][:-1] = self._AP[0] + v * radius
		self._apexsec[branch_id][0][-1] = radius

		# Recompute the shape splines
		self.__set_spl()
		self.__set_tAP() # Times at apex

		self.__set_key_pts() # Set key points times
		self.__set_X() # Geometric center (np array)
		self.__set_SP() # Separation points (np array)
		self.__set_CT() # Common points (np array)

		self.__set_tspl() # Trajectory splines
	


	def __set_spl(self): 

		""" Set the shape splines of the bifurcation.
		"""

		relax = 0.25

		# Compute the shape splines from cross sections
		self._spl = []

		# Correct the radius 
		"""
		if len(self._apexsec) < 0:
			print('correct radius')
			D0, D1, D2 = self._endsec[0][0][-1]*2, self._endsec[1][0][-1]*2, self._endsec[2][0][-1]*2
			self._apexsec[0][0][-1] = ((sqrt(2) * D0*D1)/(sqrt(D1**2 + D2**2))) / 2.
			self._apexsec[1][0][-1] = ((sqrt(2) * D0*D2)/(sqrt(D1**2 + D2**2))) / 2.
		"""
	
		# Check distance between apexsec coordinates
		for i in range(len(self._apexsec)):
			for j in range(len(self._apexsec[i]) - 1):
				if norm(self._apexsec[i][j + 1][:-1] - self._apexsec[i][j][:-1]) < 10*(-1):
					self._apexsec.pop(j)

		for i in range(1, len(self._endsec)):
			sec_list = [self._endsec[0]] + self._apexsec[i-1] + [self._endsec[i]] 

			l = []
			for j in range(len(sec_list) - 1):

				p0 = sec_list[j][0] + relax * norm(sec_list[j+1][0] - sec_list[j][0]) * sec_list[j][1] 
				p0[-1] =  sec_list[j][0][-1] + relax * abs(( sec_list[j+1][0][-1] -  sec_list[j][0][-1]))
				
				p1 = sec_list[j+1][0] - relax * norm(sec_list[j+1][0] - sec_list[j][0]) * sec_list[j + 1][1]
				p1[-1] = sec_list[j+1][0][-1] - relax * abs(( sec_list[j+1][0][-1] -  sec_list[j][0][-1]))

				cp_list = np.vstack((sec_list[j][0],p0,p1,sec_list[j+1][0]))

				spl = Spline()
				spl.approximation(cp_list, [1,1,1,1], np.vstack((sec_list[j][0],sec_list[j][1], sec_list[j+1][1], sec_list[j+1][0])), False, n = 4, radius_model=False, criterion= "None")

				if j == 0:
					P = spl.get_control_points()
					l.append(spl.length())
				else:
					P = np.vstack((P[:-1], spl.get_control_points()))
					l.append(spl.length() + l[-1])


			l = l/max(l)
			#l = np.linspace(0.0, 1.0, 3)[1:].tolist()
			knot = [0.0, 0.0, 0.0]
			for k in range(len(l) - 1):
				knot += [(l[k] + knot[-1])/2, l[k], l[k]]
			knot += [(1.0 + knot[-1])/2, 1.0, 1.0, 1.0]
		

			self._spl.append(Spline(control_points = P, knot = knot))

		# Check angles of the apex cross sections and apply rotation correction if necessary
		"""
		for i in range(self.n):
			vec = (self._apexsec[i-1][0][:-1] - self._AP[0])
			OP = self._apexsec[i-1][0][:-1] + self._apexsec[i-1][0][-1] * vec / norm(vec)

			# Test if OP is at the surface of all other splines
			ind_list = np.arange(0, self.n).tolist()
			ind_list.remove(i)

			for ind in ind_list:

				t = self._spl[ind].project_point_to_centerline(OP)
				pt = self._spl[ind].point(t, True)

				if norm(pt[:-1] - OP) < pt[-1]:
					print("Error crsec")

					# If not, rotation(?)
		"""

	def __set_AP(self):
		""" Set AP point """

		for i in range(len(self._spl)-1):

			t_merge = [0.0, 0.0]
			intersec = [None, None]

			for ind in [0, 1]:

				#Plot distance as a function of time
				times, dist = splines[i+1-ind].distance(splines[i+ind].get_points())
				radius = splines[i+1-ind].radius(times)
				idx = np.argwhere(np.diff(np.sign(radius - dist))).flatten()

				if len(idx)> 0:
					intersec[ind] = splines[i+ind].get_points()[idx[-1], :-1] 
					t_merge[ind] = splines[i+ind].project_point_to_centerline(intersec[ind])

			
			if t_merge[0] > 0.0 and t_merge[1] > 0.0:

				# Apex direction 
				v = intersec[1] - intersec[0]
				v = v / norm(v)
				ap, t = splines[i].intersection(splines[i+1], v, t_merge[0], 1.0)
				AP.append(ap)

		self._AP = AP


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
		



	def __set_X(self): 

		""" Set the coordinates of the geometric center of the nfurcation. """

		# Antiga 2015
		'''
		a = 0
		b = 0
		for i in range(self.n):
			for j in range(len(self._key_pts[i])):
				a += self._spl[i].point(self._key_pts[i][j]) * 1/(self._spl[i].radius(self._key_pts[i][j]))
				b += 1/(self._spl[i].radius(self._key_pts[i][j]))

		self._X = a / b
		'''
	
		pts = self._AP.copy()
		for ind in [0, -1]:

			t = self._key_pts[ind][0]

			ptAP = self._spl[ind].point(self._tAP[ind][0])
			nAP = self._AP[ind] - ptAP

			nS = self._spl[ind].transport_vector(-nAP, self._tAP[ind][0], t) 
		 
			pt = self._spl[ind].project_time_to_surface(nS / norm(nS), t)
			pts += [pt]
		
		self._X = sum(pts) / len(pts)
	



	def __set_SP(self):

		""" Set the coordinates of the two separation points S of the nfurcation. """

		# Project X to the spline to get t
		SP = []
		spl_ind = [0, len(self._spl) - 1]

		for ind in [0, -1]:
			# Project X to the spline to get t
			t = self._key_pts[ind][0]
			t = self._spl[ind].project_point_to_centerline(self._X)

			ptAP = self._spl[ind].point(self._tAP[ind][0])
			nAP = self._AP[ind] - ptAP

			nS = self._spl[ind].transport_vector(-nAP, self._tAP[ind][0], t) 
		
			ptS = self._spl[ind].point(t)  
			SP.append(self.send_to_surface(ptS, nS / norm(nS), spl_ind[ind]))	

		self._SP = SP




	def __set_CT(self): 

		""" Set the coordinates of the common points CT of the bifurcation. """
		normals = []

		for i in range(len(self._AP)):
			n = cross(self._AP[i] - self._SP[0], self._AP[i] - self._SP[1])
			normals.append(n/norm(n))

		n = sum(normals) / len(normals)

		self._CT = [self.send_to_surface(self._X, n, 0), self.send_to_surface(self._X, -n, 0)]
		self._X = sum(self._CT) / 2


	def __set_tspl(self):

		""" Set the coordinates of the trajectory splines. """

		relax = 0.15

		tspl = []
		key_pts = self._SP[::-1] + self._AP

		for i in range(len(key_pts)):

			i2 = i+1
			if i == len(key_pts) - 1:
				i2 = 0

			tg0 = -self._endsec[i][1][:-1]

			if i == 0:
				tg0 = -tg0

			tg1 = ((key_pts[i] - self._X) + (key_pts[i2] - self._X)) / 2.0

			tg0 = tg0 / norm(tg0)
			tg1 = tg1 / norm(tg1)

			p0 = self._endsec[i][0][:-1]
			p1 = self._X

			pint0 = p0 + tg0 * norm(p0 - p1) * relax
			pint1 = p1 +  tg1 * norm(p0 - p1) * relax

			# Fit spline
			spl = Spline()
			spl.approximation(np.vstack((p0, pint0,  pint1, p1)), [1,1,1,1], np.vstack((p0, tg0, tg1, p1)), False, n = 4, radius_model=False, criterion= "None")

			P = spl.get_control_points()
			P = np.hstack((P, np.zeros((4,1))))

			tspl.append(Spline(P))

		self._tspl = tspl





	def mesh_to_crsec(self, mesh):

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
	
		#self._crsec[0] = end_crsec
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
			ax.scatter(self._X[0], self._X[1], self._X[2], c='black', s = 40)
			for pt in self._SP:
				ax.scatter(pt[0], pt[1], pt[2], c='green', s = 40)
			for pt in self._CT:
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

				# Apex sections
				for i in range(len(self._apexsec)):

					nds = np.zeros((N, 3))
					tg = self._apexsec[i][0][1][:-1]
					ref = cross(tg, np.array([0,0,1]))
					ref = ref / norm(ref)

					angle_list = (2 * pi / N) * np.arange(N)

					for j in range(N):
						n = np.array(rotate_vector(ref, tg, angle_list [j]))
						nds[j] = self._apexsec[i][0][0][:-1] + n * self._apexsec[i][0][0][-1] / norm(n)

					ax.scatter(nds[:,0], nds[:,1], nds[:,2],  c='black')

					

		# Set the initial view
		ax.view_init(90, -90) # 0 is the initial angle

		# Hide the axes
		ax.set_axis_off()
		plt.show()



	#####################################
	#########  MESHING METHODS  #########
	#####################################


	def mesh_surface(self):

		""" Returns the surface mesh of the bifurcation

		Keyword arguments:
		N -- number of nodes in a cross section
		d -- density of cross section along the vessel (proportional to the radius)
		If not given, the cross section are computed with default parameters.
		"""

		if self._crsec == None:
			end_crsec, bif_crsec, nds, ind = self.compute_cross_sections(self._N, self._d)
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



	def compute_cross_sections(self, N, d, end_ref = None):


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
		self._N = N
		self._d = d

		self.relaxation(5)

		if self.R> 0:
			self.smooth_apex(self.R)

		return self._crsec



	def __bifurcation_connect(self, tind, ind, P0, P1, n):


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

		tg0 = self._tspl[tind].tangent(0.0)
		tg1 = P0 - P1

		tg0 = tg0 / norm(tg0)
		tg1 = tg1 / norm(tg1)

		pint0 = P0 +  tg0 * norm(P0 - P1) * relax
		pint1 = P1 +  tg1 * norm(P0 - P1) * relax

		# Fit spline
		spl = Spline()
		spl.approximation(np.vstack((P0, pint0, pint1, P1)), [1,1,1,1], np.vstack((P0, tg0, tg1, P1)), False, n = 4, radius_model=False, criterion= "None")

		P = spl.get_control_points()
		P = np.hstack((P, np.zeros((4,1))))
		
		trajectory = Spline(P)
		times = np.linspace(0, 1, n+2)[1:-1]
		#times = trajectory.resample_time(n)
		
		for i in range(n):
			pts[i, :] = trajectory.point(times[i])
		

		# Method 2 using linear interpolation
		#for i in range(3):
		#	pts[:, i] = np.linspace(P0[i], P1[i], n + 2)[1:-1]	

		
		nds = np.zeros((n,3))
		for i in range(n):

			t = self._tspl[tind].project_point_to_centerline(pts[i])
			P = self._tspl[tind].point(t)
			n =  pts[i] - P
			n = n / norm(n)
			
			pt = self.send_to_surface(P, n, ind)
			nds[i] = pt
		
		return nds



	def __separation_section(self, N): 


		""" Returns the nodes of the separation plan.
	
		Keyword arguments: 
		N -- number of nodes in a section (multiple of 4)
		"""

		key_pts = [self._SP[0]]  + self._AP + [self._SP[1]]

		nds = np.zeros((2 + (N//2 - 1) * len(key_pts),3))
		nds[:2] = self._CT
	
		n = N//4 - 1

		j = 2
		for i in range(len(key_pts)):

			nds[j: j + n] = self.__separation_segment(self._CT[0], key_pts[i], n)
			nds[j + n] = key_pts[i]
			nds[j+n+1: j + 2*n +1] = self.__separation_segment(key_pts[i], self._CT[1], n) 
			j += 2*n + 1

		return nds



	def __separation_segment(self, P1, P2, n): 


		""" Returns the nodes of the half section between p1 and p2

		Keyword arguments:
		p1, p2 -- end points
		n --  number of required nodes for the segment

		"""

		v1 = P1 - self._X
		v2 = P2 - self._X

		theta = (directed_angle(v1, v2, cross(v1,v2)) / (n + 1)) * np.arange(1, n + 1) 

		# Computing nodes using t and theta parameters
		nds = np.zeros((n, 3))
		for i in range(n):

			n = rotate_vector(v1, cross(v1, v2), theta[i])
			nds[i, :] = self.send_to_surface(self._X, n, 0)

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

	def relaxation(self, n_iter = 5):
		""" Resamples the cells of the bifurcation """

		for i in range(n_iter):

			init_mesh = self.mesh_surface()
			self.smooth(1)
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
		self.mesh_to_crsec(mesh)


	def smooth_apex(self, radius):

		if self._crsec == None:
			raise ValueError('Please first perform cross section computation.')

		curve_set, referential_set = self.get_curves()

		# Set the curve in the projection referential
		curve_set_referential = []
		for i in range(len(curve_set)):
			curve = curve_set[i]
			curve_referential = np.zeros(curve.shape)

			for j in range(len(curve)):
				v1, v2, v3, origin = referential_set[i][0], referential_set[i][1], referential_set[i][2], referential_set[i][3]
				curve_referential[j] = np.array([dot(curve[j], v1), dot(curve[j], v2), dot(curve[j], v3)])

			curve_set_referential.append(curve_referential)

		# Smooth apex curves
		start_curve = len(curve_set) // (self.n + 1) 
		end_curve = len(curve_set) - start_curve 

		for i in range(start_curve, end_curve): #range(len(curve_set_referential)): To smooth all the curves
			data = curve_set_referential[i][:,:-1]
			data_smooth = smooth_polyline(data, radius)
			curve_set_referential[i][:,:-1] = data_smooth

		# Project back in original referential
		curve_set_back = []
		for i in range(len(curve_set)):
			curve = curve_set_referential[i]
			curve_back = np.zeros(curve.shape)

			for j in range(len(curve)):
				v1, v2, v3, origin = referential_set[i][0], referential_set[i][1], referential_set[i][2], referential_set[i][3]
				# Write original referential to the projection referential
				v1, v2, v3 = np.array([dot(np.array([1,0,0]), v1), dot(np.array([1,0,0]), v2), dot(np.array([1,0,0]), v3)]), np.array([dot(np.array([0,1,0]), v1), dot(np.array([0,1,0]), v2), dot(np.array([0,1,0]), v3)]), np.array([dot(np.array([0,0,1]), v1), dot(np.array([0,0,1]), v2), dot(np.array([0,0,1]), v3)])

				curve_back[j] = np.array([dot(curve[j], v1), dot(curve[j], v2), dot(curve[j], v3)])

			curve_set_back.append(curve_back)
		
		# Set curves
		self.curves_to_crsec(curve_set_back)



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
		#for i in range(len(end_crsec)):
		#	center = (np.array(end_crsec[i][0]) + end_crsec[i][int(len(end_crsec[i])/2)])/2.0 # Compute the center of the section
		#	for j in range(len(end_crsec[i])):
		#		end_crsec[i][j] = self.__intersection(mesh, center, end_crsec[i][j])

		# Project connecting nodes
		for i in range(len(nds)):
			for j in range(len(nds[i])):
				center = (np.array(nds[i][j][0]) + nds[i][j][int(len(nds[i][j])/2)])/2.0 # Compute the center of the section
				for k in range(len(nds[i][j])):
					nds[i][j, k] = self.__intersection(mesh, center, nds[i][j][k])

		# Project bifurcation plan
		for i in range(len(bif_crsec)):
			bif_crsec[i] = self.__intersection(mesh, self._X, bif_crsec[i])




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


	def send_to_surface(self, O, n, ind):

		""" Sends a point to the surface defined by all shape splines according to direction n """

		# Project to main ind
		pt = self.__projection(O, n, 0.0, 2.5, ind)

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

		t = self._spl[ind].project_point_to_centerline(O)
		if False: #t == 1.0 or t == 0.0:
			# Out of the spline, don't project!
			return O
			
		else:
		
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












