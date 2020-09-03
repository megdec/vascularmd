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


class Bifurcation:


	#####################################
	##########  CONSTRUCTOR  ############
	#####################################

	def __init__(self, S0, S1, S2, R, spl = []):

		# Check user parameters 

		
		# Set user paramaters

		self._endsec = np.array([S0, S1, S2]) # Cross sections of main and daughter branches
		self.R = R  # Minimum curvature for rounded apex

		# Set modeling variables
		if len(spl)!= 0:
			self._spl = spl # Shape splines

		else:
			self.__set_spl()
		
		self.__set_AP() # Apex point
		self.__set_SP() # Separation points
		self.__set_B() # Geometric center
		self.__set_CP() # Common points

		self.__set_tspl()# Trajectory splines
	

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



	#####################################
	#############  SETTERS  #############
	#####################################


	def set_R(self, R):

		""" Set the R parameter of the bifurcation."""

		self.R = R



	def set_spl(self, spl):

		""" Public setter for the bifurcation splines."""

		self._spl = spl

		self._AP = self.__set_AP() 
		self._B = self.__set_B() 
		self._SP = self.__set_SP() 
		self._CP = self.__set_CP()

		self._tspl = self.__set_tspl()



	def __set_spl(self, r=1.5):


		""" Set the shape splines of the bifurcation.

		Keyword arguments:
			r -- norm of the end tangents. 
		"""

		p0 = self._endsec[0][0] + r * self._endsec[0][1]
		p1 = self._endsec[1][0] - r * self._endsec[1][1]
		p2 = self._endsec[2][0] - r * self._endsec[2][1]

		spl1 = Spline([self._endsec[0][0].tolist(), p0.tolist(), p1.tolist(), self._endsec[1][0].tolist()])
		spl2 = Spline([self._endsec[0][0].tolist(), p0.tolist(), p2.tolist(), self._endsec[2][0].tolist()])

		self._spl = [spl1, spl2]



	def __set_AP(self):

		""" Set the coordinates of the apex point parameter and the times at apex. """

		v = cross(self._endsec[1][1][:-1], np.array([1,0,0]))
		v = v / norm(v)


		tmax = 0.0
		angle = np.linspace(0,2*pi, 100)

		for a in angle:

			vrot = rotate_vector(v, self._endsec[1][1][:-1], a)
			ap, times = self.__find_intersection(vrot)

			if times[1] > tmax:
				tmax = times[1]

				AP = ap
				tAP = times

		self._AP = np.array(AP)
		self._tAP = tAP




	def __find_intersection(self, v0):

		""" Returns the intersection time between the shape splines, given a initial vector v0.

		Keywords arguments: 
		v0 -- reference vector for the search
		"""

		t0 = 0.0
		t1 = 1.0

		while abs(t1 - t0) > 10**(-6):

			t = (t1 + t0) / 2.
			
			v = self._spl[0].transport_vector(v0, 1.0, t)
			pt = self._spl[0].project_time_to_surface(v, t) 
			
			t2 = self._spl[1].project_point_to_centerline(pt)
			pt2 = self._spl[1].point(t2, True)

			if norm(pt - pt2[:-1]) <= pt2[-1]:
				t0 = t
			else: 
				t1 = t

		return pt, [t, t2]




	def __set_SP(self):

		""" Set the coordinates of the separation points SP of the bifurcation. """

		SP = []
		for ind in [[0, 1], [1, 0]]:
		
			t = self._spl[ind[0]].length_to_time(self._spl[ind[0]].length() / 2.0)

			ptAP = self._spl[ind[0]].point(self._tAP[ind[0]])
			nAP = self._AP - ptAP

			nAP0 = self._spl[ind[0]].transport_vector(nAP, self._tAP[ind[0]], 1.0) 
			nS = self._spl[ind[0]].transport_vector(-nAP0, 1.0, t) 
		

			ptS = self._spl[ind[0]].point(t)  

			SP.append(self.__send_to_surface(ptS, nS / norm(nS), ind))

		self._SP = np.array(SP)

 


	def __set_B(self):

		""" Set the coordinates of the geometric center of the bifurcation. """

		self._B = (self._SP[0] + self._SP[1] + self._AP) / 3.0



	def __set_CP(self):

		""" Set the coordinates of the common points CP of the bifurcation. """

		n = cross(self._AP - self._SP[0], self._AP - self._SP[1])
		n = n / norm(n)

		self._CP = np.array([self.__send_to_surface(self._B, n, [0, 1]), self.__send_to_surface(self._B, -n, [0, 1])])




	def __set_tspl(self, r=6):

		""" Set the coordinates of the trajectory splines. """

		tspl = []
		for i in range(3):

			if i == 0:
				tg1 = (cross(self._CP[0] - self._B, self._SP[0] - self._B) + cross(self._SP[1] - self._B, self._CP[0]  - self._B)) / 2.0
				tg0 = self._endsec[i][1][:-1]

			elif i == 2:

				tg1 = (cross(self._CP[0] - self._B, self._SP[1] - self._B) + cross(self._AP - self._B, self._CP[0]  - self._B)) / 2.0
				tg0 = -self._endsec[i][1][:-1]

			else: 
				tg1 = (cross(self._CP[0] - self._B, self._AP - self._B) + cross(self._SP[0] - self._B, self._CP[0]  - self._B)) / 2.0
				tg0 =  -self._endsec[i][1][:-1]
				

			p0 = self._endsec[i][0][:-1]
			p1 = self._B


			d = norm(p0 - p1)

			pint0 = np.array(p0) + tg0 * (d/r) / norm(tg0)
			pint1 = np.array(p1) +  tg1 * (d/r) / norm(tg1)
			P = [p0.tolist() + [0.0], pint0.tolist() + [0.0],  pint1.tolist() + [0.0], p1.tolist() + [0.0]]
		
			tspl.append(Spline(P))

		self._tspl = tspl





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

			# Plot key points 
			ax.scatter(self._AP[0], self._AP[1], self._AP[2], c='red')
			ax.scatter(self._B[0], self._B[1], self._B[2], c='black')
			ax.scatter(self._CP[:, 0], self._CP[:, 1], self._CP[:, 2], c='blue')
			ax.scatter(self._SP[:, 0], self._SP[:, 1], self._SP[:, 2], c='green')

			if nodes: 

				N = 24
				# Separation section
				nds = np.array(self.__separation_section(N))
				ax.scatter(nds[:,0], nds[:,1], nds[:,2],  c='black')

				# End sections
				nds = np.array(self.__end_sections(24))
				ax.scatter(nds[0,:,0], nds[0,:,1], nds[0,:,2],  c='black')
				ax.scatter(nds[1,:,0], nds[1,:,1], nds[1,:,2],  c='black')
				ax.scatter(nds[2,:,0], nds[2,:,1], nds[2,:,2],  c='black')

				ax.scatter(nds[0,0,0], nds[0,0,1], nds[0,0,2],  c='blue')
				ax.scatter(nds[0,-1,0], nds[0,-1,1], nds[0,-1,2],  c='red')

				#end_crsec, bif_crsec, nds, ind = self.cross_sections(N, 0.1)
				
				
				#for i in range(3):
				#	ndscr = np.array(nds[i])
				#	for j in range(N):
				#		ax.plot(ndscr[:, j, 0], ndscr[:,j, 1], ndscr[:, j, 2], c='black')

				#	for k in range(len(ndscr)):
				#		ax.plot(ndscr[k, :, 0], ndscr[k, :, 1], ndscr[k, :, 2], c='black')


		# Set the initial view
		ax.view_init(90, -90) # 0 is the initial angle

		# Hide the axes
		ax.set_axis_off()
		plt.show()



	#####################################
	#########  MESHING METHODS  #########
	#####################################



	def mesh(self, N, d):

		""" Writes the mesh of a bifurcation defined by three end cross sections.

		Keyword arguments:

		S0, S1, S2 -- downstream and upstram cross sections 
		N -- number of nodes in a transverse section (multiple of 4)
		d -- longitudinal density of nodes as a proportion of the radius """

		end_crsec, bif_crsec, nds, ind = self.cross_sections(N, d)

		vertices = []
		faces = []

		# Attribute a id to every nodes
		count = 0

		id_end = np.zeros(np.array(end_crsec).shape[:-1])
		for i in range(id_end.shape[0]):
			for j in range(id_end.shape[1]):
					id_end[i, j] = count
					vertices.append(end_crsec[i][j])
					count += 1
		id_end = id_end.tolist()

		id_bif = np.zeros(np.array(bif_crsec).shape[:-1])
		for i in range(id_bif.shape[0]):
			id_bif[i] = count
			vertices.append(bif_crsec[i])
			count += 1
		id_bif = id_bif.tolist()

		id_nds = [] 
		for i in range(len(nds)):
			tab = np.zeros(np.array(nds[i]).shape[:-1]) 
			for j in range(tab.shape[0]):
				for k in range(tab.shape[1]):
					tab[j, k] = count
					vertices.append(nds[i][j][k])
					count += 1
			id_nds.append(tab.tolist())

		# Connect end sections
		for s in range(3):
			for i in range(N):

				if i == N - 1:
					i2 = 0
				else: 
					i2 = i + 1

				faces.append([4, id_end[s][i], id_end[s][i2], id_nds[s][0][i2],  id_nds[s][0][i]])

		# Connect bif section
		for s in range(3):
			for i in range(N):

				if i == N - 1:
					i2 = 0
				else: 
					i2 = i + 1

				faces.append([4, id_nds[s][-1][i], id_nds[s][-1][i2], id_bif[ind[s][i2]],  id_bif[ind[s][i]]])


		# Connect other nodes
		for s in range(3):
			for i in range(len(nds[s])-1):
				for j in range(N):

					if j == N - 1:
						j2 = 0
					else: 
						j2 = j + 1

					faces.append([4, id_nds[s][i][j] , id_nds[s][i][j2], id_nds[s][i + 1][j2],  id_nds[s][i + 1][j]])

		mesh = pv.PolyData(np.array(vertices), np.asarray(faces, dtype=int))
		return mesh


	def __smooth_apex(self, mesh):

		""" Smoothes the bifurcation apex and separation sections."""

		return mesh


	def cross_sections(self, N, d):


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
		end_crsec = self.__end_sections(N)

		# Write connectivity index
		connect_index = []

		nodes_num = np.arange(2, len(bif_crsec)).tolist()
		l = int(len(nodes_num) / 3)
		ind = [nodes_num[:l], nodes_num[l:2*l], nodes_num[2*l:3*l]]

		connect_index.append([1] + ind[0][::-1] + [0] + ind[2])
		connect_index.append([1] + ind[0][::-1] + [0] + ind[1])
		connect_index.append([1] + ind[1][::-1] + [0] + ind[2])
		
	
		# Compute connecting nodes
		nds = []

		for i in range(3):

			if i == 0:
				ind = [0, 1]

			elif i == 1:
				ind = [0, 1]

			else:
				ind = [1, 0]

			num = int(self._tspl[i].length() / (self._endsec[i][0][-1] * d))

			# Minimum value for the number of cross sections
			if num == 0:
				num = 2

			nds_seg = np.zeros((num, N, 3))
		
			for j in range(N):
				nds_seg[:, j, :] = self.__bifurcation_connect(i, ind, end_crsec[i][j], bif_crsec[connect_index[i][j]], num)  

			nds.append(nds_seg.tolist())

		return end_crsec, bif_crsec, nds, connect_index


	def __bifurcation_connect(self, tind, ind, P0, P1, n):


		""" Compute the nodes connecting an end point to a separation point.

		Keyword arguments: 
		ind -- indice of the trajectory spline
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
		P = [P0.tolist() + [0.0], pint0.tolist() + [0.0],  pint1.tolist() + [0.0], P1.tolist() + [0.0]]
		
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

		nds = []
		for i in range(n):

			t = self._tspl[tind].project_point_to_centerline(pts[i])
			P = self._tspl[tind].point(t)
			n =  pts[i] - P
			n = n / norm(n)
			
			pt = self.__send_to_surface(P, n, ind)
			nds.append(pt)
		
		return nds



	def __separation_section(self, N):


		""" Returns the nodes of the separation plan.
	
		Keyword arguments: 
		N -- number of nodes in a section (multiple of 4)
		"""

		nds = [self._CP[0].tolist(), self._CP[1].tolist()]
		n = N/4 - 1

		for i in range(3):

			if i == 0:
				P = self._SP[0]
			elif i == 1:
				P = self._AP
			else: 
				P = self._SP[1]

			nds = nds + self.__separation_segment(self._CP[0], P, n) + [P.tolist()] + self.__separation_segment(P, self._CP[1], n) 

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
		nds = []
		for i in range(int(n)):

			n = rotate_vector(v1, cross(v1, v2), theta[i])
			nds.append(self.__send_to_surface(self._B, n, [0,1]))

		return nds



	def __end_sections(self, N):


		""" Returns the nodes of end sections.

		Keyword arguments: 
		N -- number of nodes in a section (multiple of 4)
		"""
		
		nds = []

		for i in range(3):

			crsec_nds = []

			if i == 0:
				
				# Reference vector
				tS = self._tspl[0].project_point_to_centerline(self._SP[1])
				ptS = self._tspl[0].point(tS)
				nS = self._tspl[0].transport_vector(self._SP[1] - ptS, tS, 0.0)
				ref = cross(self._endsec[0][1][:-1], nS)

			elif i == 1:
				
				# Reference vector
				tS = self._spl[0].project_point_to_centerline(self._SP[0])
				ptS = self._spl[0].point(tS)
				nS = self._spl[0].transport_vector(self._SP[0] - ptS, tS, 1.0)
				ref = cross(nS, self._endsec[1][1][:-1])

			else:
				
				# Reference vector
				tS = self._spl[1].project_point_to_centerline(self._SP[1])
				ptS = self._spl[1].point(tS)
				nS = self._spl[1].transport_vector(self._SP[1] - ptS, tS, 1.0)
				ref = cross(self._endsec[2][1][:-1], nS)

			ref = ref / norm(ref)
			angle_list = (2 * pi / N) * np.arange(N)

			for theta in angle_list:
				n = np.array(rotate_vector(ref, self._endsec[i][1][:-1], theta))
				crsec_nds.append((self._endsec[i][0][:-1] + n * self._endsec[i][0][-1] / norm(n)).tolist())

			nds.append(crsec_nds)
		
		
		return nds




	#####################################
	##############  UTILS  ##############
	#####################################


	def __send_to_surface(self, O, n, ind):

		pt = self.__projection(O, n, 0, 5.0, ind[0])

		# Check distance to spl2
		t = self._spl[ind[1]].project_point_to_centerline(pt)
		pt2 = self._spl[ind[1]].point(t, True)
		
		if norm(pt - pt2[:-1]) < pt2[-1]:
			pt = self.__projection(pt, n, 0.0, 5.0, ind[1])

		return pt



	def __projection(self, O, n, c0, c1, ind):

		""" Sends a point to the surface defined by shape splines according to direction n.

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

		if norm(pt[:-1] - (O + c1 * n)) < pt[-1]: # Incorrect initial born
			return self.__projection(O, n, c0, c1 - 0.1, ind)

		else:

			while abs(c0 - c1)> 10**(-10):

				c = (c0 + c1) / 2.0

				t = self._spl[ind].project_point_to_centerline(O + c * n)
				pt = self._spl[ind].point(t, True)

				if norm(pt[:-1] - (O + c * n)) < pt[-1]:
					c0 = c
				else:
					c1 = c

			pt = O + ((c0 + c1) / 2.0) * n

			return pt










