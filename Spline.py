# Python 3

import numpy as np # Tools for matrices
from geomdl import BSpline, operations, helpers # Spline storage and evaluation

from numpy.linalg import norm 
from numpy import dot, cross
import matplotlib.pyplot as plt # Tools for plots

from utils import *


class Spline:


	#####################################
	##########  CONSTRUCTOR  ############
	#####################################

	def __init__(self, control_points = None, knot = None, order = None):

		self._spl = BSpline.Curve()
		self._spl.order = 3		

		if control_points is not None: 

			if len(control_points[0]) != 3 and len(control_points[0]) != 4:
				raise ValueError ('The spline control points must have dimension 3 (x, y, z) or  (x, y, z, r).')

			self._spl.ctrlpts = control_points

			if order is not None:
				self._spl.order = order

			if knot is None:
				self._spl.knotvector = self.__uniform_knot(self._spl.order, self._spl.ctrlpts_size)
			else: 
				self._spl.knotvector = knot

			self.__set_length_tab()




	#####################################
	#############  GETTERS  #############
	#####################################

	def get_knot(self):
		return self._spl.knotvector

	def get_control_points(self):
		return np.array(self._spl.ctrlpts)

	def get_points(self):
		return np.array(self._spl.evalpts)

	def get_length(self):
		return self._length_tab



	#####################################
	#############  SETTERS  #############
	#####################################

	def set_spl(self, spl):

		""" Sets spline given as Bspline object."""

		self._spl = spl
		self.__set_length_tab()


	def set_control_points(self, P):

		""" Sets spline control points using a list table of points."""

		self._spl.ctrlpts = P
		self._spl.knotvector = self.__uniform_knot()
		self.__set_length_tab()



	def __set_length_tab(self):

		""" Set arc length estimation of spline."""

		if self._spl.dimension == 3:
			pts = np.array(self._spl.evalpts)
		else:
			pts = np.array(self._spl.evalpts)[:, :-1]


		length = [0.0]
		for i in range(1, len(pts)):
			length.append(length[i-1] + norm(pts[i] - pts[i-1]))

		self._length_tab = length



	#####################################
	###########  EVALUATION  ############
	#####################################


	def first_derivative(self, t, radius = False):

		""" Returns the unit first derivative of spline at time t as a numpy array."""

		der = self._spl.derivatives(t, order=1)[1]

		if self._spl.dimension == 3:
			der = np.array(der)
		else:
			if radius:
				der = np.array(der)
			else:
				der = np.array(der)[:-1]

		return der
		

	def second_derivative(self, t, radius = False):

		""" Returns the unit second derivative of spline at time t as a numpy array."""

		der = self._spl.derivatives(t, order=2)[2]

		if self._spl.dimension == 3:
			der = np.array(der)
		else:
			if radius:
				der = np.array(der)
			else:
				der = np.array(der)[:-1]

		return der 
		

	def tangent(self, t, radius = False):

		""" Returns the unit tangent of spline at time t as a numpy array."""

		tg = self._spl.tangent(t)[1]

		if self._spl.dimension == 3:
			tg = np.array(tg)
		else:
			if radius:
				tg = np.array(tg)
			else:
				tg = np.array(tg)[:-1]

		return tg / norm(tg)


	def point(self, t, radius = False):

		""" Returns evaluation point of spline at time t as a numpy array."""

		pt = self._spl.evaluate_single(t)

		if self._spl.dimension == 3:
			pt = np.array(pt)
		else:
			if radius:
				pt = np.array(pt)
			else:
				pt = np.array(pt)[:-1]
	
		return pt


	#####################################
	#############  UTILS  ###############
	#####################################


	def __uniform_knot(self, p, n):

		""" Returns a B-spline uniform knot vector."""

		knot = []

		for i in range(p + n):
			if i < p:
				knot.append(0.0)
			elif p <= i <= n-1:
				knot.append(float(i-p+1))
			else:
				knot.append(float(n-p+1))

		return (np.array(knot) / knot[-1]).tolist()



	def __averagingKnot(self, t, p, n):

		""" Returns a B-spline averaging knot vector.

		Keyword arguments:
		t -- time parametrization vector
		"""

		knot = [0.0] * p # First knot of multiplicity p

		for i in range(p, n):
			knot.append((1.0 / (p - 1.0)) * sum(t[i-p+1:i]))

		knot = knot + [1.0] * p

		return knot


	def __chord_length_parametrization(self, D):

		""" Returns the chord length parametrization for data D.

		Keyword arguments:
		D -- data points
		"""

		D = np.array(D)
		
		t = [0.0]
		for i in range(1, len(D)):
			t.append(t[i-1] + np.linalg.norm(D[i] - D[i-1]))
		t = [time / max(t) for time in t]

		return t


	def __basis_functions(self, knot, t, p, n):


		"""Computes the value of B-spline basis functions evaluated at t

		Keyword arguments:
		knot -- knot vector
		t -- time parameter 
		p -- B-spline degree
		n -- number of control points

		"""

		N = [0.0]*n # list of basis function values 

		# Handle special cases for t 
		if t == knot[0]:
			N[0] = 1.0

		elif t == knot[-1]:
			N[-1] = 1.0
		else:

			# Find the bounding knots for t
			k = 0
			for kn in range(len(knot)-1):
				if knot[kn] <= t < knot[kn+1]:
					k = kn
		
			N[k] = 1.0 # Basis function of order 0
			
			# Compute basis functions = recurrence??!!
			for d in range(1, p): 

				if knot[k + 1] == knot[k- d + 1]:
					N[k-d] = 0
				else:
					N[k-d] = (knot[k + 1] - t) / (knot[k + 1] - knot[k- d + 1]) * N[k- d + 1]

				for i in range(k-d + 1, k):

					if knot[i+d] == knot[i]:
						c1 = 0
					else:
						c1 = (t - knot[i]) / (knot[i+d] - knot[i]) * N[i]

					if knot[i + d + 1] == knot[i + 1]:
						c2 = 0
					else:
						c2 = (knot[i + d + 1] - t) / (knot[i + d + 1] - knot[i + 1]) * N[i + 1]

					N[i] =  c1 + c2

				if knot[k+d] == knot[k]:
					N[k] = 0
				else:
					N[k] = (t - knot[k]) / (knot[k+d] - knot[k]) * N[k]
			

		# Return array of n basis function values at t 
		return N


	#####################################
	########## APPROXIMATION  ###########
	#####################################

	def curvature_bounded_approximation(self, D, ratio, clip = [[],[]], deriv=[[],[]]): 


		"""Approximate data points using a spline of degree p with n control_points, 
		using constraint on the error of the fitting to find the optimal number of control points.

		Keyword arguments:
		D -- numpy array of coordinates for data points
		ratio -- curvature radius / radius minimum ratio
		clip -- end points if clipped ends
		deriv -- end tangents to apply if constrainted tangents
		"""

		n = int(len(D) / 2)
		t = self. __chord_length_parametrization(D)
		knot =  self.__uniform_knot(3, n)
		
		search = True
		lbd = 0

		while (search and lbd < 1):
			
			self._spl = self.__solve_system(D, 3, n, knot, t, lbd, clip, deriv) 
			rad_curv = self.curvature_radius(np.linspace(0, 1, self.spl.)).tolist()
			rad = self._spl.evalpts[:,-1].tolist()

			if all(rad_curv > rad/ratio):
				search = False

			lbd = lbd + 0.001

		self.set_spl(self._spl)



	def __solve_system(self, D, p, n, knot, t, lbd, clip = [[], []], tangent = [[], []]):

		"""Approximate data points using a spline of degree p with n control_points.
		Returns a geomdl.Bspline object

		Keyword arguments:
		D -- numpy array of coordinates for data points
		p -- degree
		n -- number of control points
		knot -- knot vector
		t -- time parametrization vector
		lbd -- lambda coefficient that balances smoothness and accuracy
		clip -- list of end points 
		deriv -- list of end derivatives
		"""

		D = np.array(D)
		m = D.shape[0]
		x = D.shape[1]

		D = D.reshape((m * x, 1))
		
		

		if (len(tangent[0]) != 0 and len(clip[0]) == 0) or (len(tangent[1]) != 0 and len(clip[1]) == 0):
			raise ValueError("Please use clip ends to add tangent constraint.")

		if len(tangent[0]) != 0 and len(tangent[1]) != 0 and n<4:
			n = 4


		# Definition of light matrix N
		Nl = []
		for time in t:
			Nl.append(self.__basis_functions(knot, time, p, n))
		Nl = np.array(Nl)
		

		# Definition of full matrix N
		N = []
		for i in range(Nl.shape[0]):

			for k in range(x):
				line = []
				for j in range(Nl.shape[1]):
					elt = [0] * x
					elt [k] = Nl[i, j]
					line += elt
				N.append(line)

		N = np.array(N)
		
		

		P1 = np.array([0]*x)
		P2 = np.array([0]*x)
		P3 = np.array([0]*x)
		P4 = np.array([0]*x)

		h = [0, 0]

		# Get fixed control points, if necessary
		if len(clip[0]) != 0:
			P1 = np.array(clip[0])
			h[0] = h[0] + x

		if len(clip[1]) != 0:
			P4 = np.array(clip[1])
			h[1] = h[1] + x

		if len(tangent[0]) != 0:
			P2 = np.array(clip[0])
			h[0] = h[0] + x - 1

		if len(tangent[1]) != 0:
			P3 = np.array(clip[1])
			h[1] = h[1] + x - 1

		# Definition of matrix Q1
		Q1 = []
		for i in range(0, m):
			Q1.append(list(Nl[i, 0] * P1 + Nl[i, 1] * P2 + Nl[i, -2] * P3 + Nl[i, -1] * P4))
		Q1 = np.array(Q1)

		Q1 = Q1.reshape((Q1.shape[0] * Q1.shape[1], 1))



		# Resizing N if clipped ends and tangents
		if h[1] == 0:
			N = N[:, h[0]:]
		else: 
			N = N[:, h[0]:-h[1]]

		# Fill first and last columns if tangents
		if len(tangent[0]) != 0:
			j = 0
			for i in range(N.shape[0]):
				if j == 4:
					j = 0

				N[i, 0] = Nl[int(i/4), 1] * tangent[0][j]
			
				j +=1

		if len(tangent[1]) != 0:
			j = 0
			for i in range(N.shape[0]):
				if j == 4:
					j = 0

				N[i, -1] = Nl[int(i/4), -2] * tangent[1][j]
				j = j + 1




		# Definition of the smoothing matrix Delta
		U = []
		for i in range(2*x, n*x):
			u = [0]*n*x

			u[i] = 1.0
			u[i - 1*x] = -2.0
			u[i - 2*x] = 1.0

			U.append(u)
		U = np.array(U)


		Delta = np.dot(U.transpose(), U)

		

		# Definition of matrix Q2
		Q2 = []
		for i in range(h[0], Delta.shape[0]-h[1]):
			q2 = 0
			for j in range(x):
				q2 += Delta[i, j]* P1[j]
				q2 += Delta[i, x + j]* P2[j]
				q2 += Delta[i, - 2*x + j]* P3[j]
				q2 += Delta[i, - x + j]* P4[j]

			Q2.append(q2)
		Q2 = np.array(Q2)
		Q2 = Q2.reshape((len(Q2), 1))



		# Resize Delta if clipped ends and 
		if h[1] == 0:
			Delta = Delta[h[0]:, h[0]:]
		else: 
			Delta = Delta[h[0]:-h[1], h[0]:-h[1]]

		# Write columns if tangent constraint
		if len(tangent[0])!= 0:
			Delta[0, 0] = 5 * dot(np.array(tangent[0]), np.array(tangent[0]))

			for i in range(1, Delta.shape[0]):
				if i<=4:
					Delta[i, 0] = -4 * tangent[0][i -1]

				elif i>4 and i<=8:
					Delta[i, 0] = 1 * tangent[0][i - x -1]

				else: 
					Delta[i, 0] = 0

			#for i in range(1, Delta.shape[1]):
			#	if i<=4:
			#		Delta[0, i] = -4

			#	elif i>4 and i<=8:
			#		Delta[0, i] = 1

			#	else: 
			#		Delta[0, i] = 0

					


		if len(tangent[1])!= 0:

			Delta[-1, -1] = 5 * dot(np.array(tangent[1]), np.array(tangent[1]))

			for i in range(1, Delta.shape[1]):

				if i<=4:
					Delta[Delta.shape[1] - i - 1, -1] = -4 * tangent[1][i-1]

				elif i>4 and i<=8:
					Delta[Delta.shape[1] - i - 1, -1] = 1 * tangent[1][i - x -1]

				else: 
					Delta[Delta.shape[1] - i - 1, -1] = 0

			#for i in range(1, Delta.shape[1]):

			#	if i<=4:
			#		Delta[-1, Delta.shape[1] - i - 1] = -4

			#	elif i>4 and i<=8:
			#		Delta[-1, Delta.shape[1] - i - 1] = 1

			#	else: 
			#		Delta[-1, Delta.shape[1] - i - 1]  = 0

			if len(tangent[0])!= 0:
				Delta[0, -1] = 0
				Delta[-1, 0] = 0


		
		#print("Delta: ", Delta)
		

		# Write matrix M1 = NtN + lbd * Delta
		M1 = (1-lbd) * np.dot(N.transpose(), N) + lbd * Delta

		# Write matrix M2 
		M2 = (1-lbd) * np.dot(N.transpose(), D - Q1) - lbd * Q2
	     

		# Solve the system
		P = np.dot(np.linalg.pinv(M1), M2)


		if len(tangent[0]) != 0:
			alpha = P[0]
			P = P[1:]
			

		if len(tangent[1]) != 0:
			beta = P[-1]
			P = P[:-1]

		P = P.reshape((int(len(P) / x), x))
		P = P.tolist()



		# Add tangent points to the list
		if len(tangent[0]) != 0:
			P = [(clip[0] + alpha * np.array(tangent[0])).tolist()] + P
		if len(tangent[1]) != 0:
			P = P + [(clip[1] + beta * np.array(tangent[1])).tolist()]


		# Add fixed points to the list
		if len(clip[0]) != 0:
			P = [clip[0]] + P	
		if len(clip[1]) != 0:
			P = P + [clip[1]]	
		

		# Create spline
		spl = BSpline.Curve()
		spl.order = p
		spl.ctrlpts = P
		spl.knotvector = knot


		return spl



	#####################################
	###########  GEOMETRY  ##############
	#####################################


	def split_time(self, t):

		""" Splits the spline into two splines at time t."""

		spla, splb = operations.split_curve(self._spl, t)

		spl1 = Spline()
		spl1.set_spl(spla)

		spl2 = Spline()
		spl2.set_spl(splb)

		return spl1, spl2



	def split_length(self, l):

		""" Splits the spline into two splines at length l."""

		t = self.length_to_time(l)
		return self.split_time(t)


	def reverse(self):

		""" Reverses the spline."""

		self._spl.reverse()
		self.__set_length_tab()


	def copy_reverse(self):

		""" Returns the reversed spline."""

		splr = self._spl
		splr.reverse()

		spl = Spline()
		spl.set_spl(splr) 


		return spl



	def length(self):

		return self._length_tab[-1]


	def mean_radius(self):

		return np.mean(self.get_points()[:, -1])


	def curvature(self, T):

		""" Returns the curvature value(s) of spline at time(s) T.

		Keyword arguments:	
		T -- curve times (list or single float value)
		"""

		if type(T) == list:

			C = []
			for t in T:
				der = [self.first_derivative(t), self.second_derivative(t)]
				C.append(norm(cross(der[0], der[1])) / (norm(der[0]))**3)
		else:

			der = [self.first_derivative(T), self.second_derivative(T)]
			C = (norm(cross(der[0], der[1])) / (norm(der[0]))**3)

		return np.array(C)

		


	def curvature_radius(self, T):

		""" Returns the radius of curvature of spline at time(s) T.

		Keyword arguments:	
		T -- curve times (list or single float value)
		"""

		return 1.0 / self.curvature(T)




	def length_to_time(self, L):

		""" Return the list (resp. float) of time(s) for which the length of the spline equals L.
		The precision of the approximation depends on the attribute delta of spline spl.

		Keyword arguments:
		L -- curve length (list or scalar)
		"""

		length = np.array(self._length_tab)

		if type(L) == list:

			T = []
			for i in range(len(L)):

				i1 = np.argmax(length >= L[i])

				if i1 == len(length) - 1 or (i1 == 0 and length[0]< L[i]):
					T.append(1.0)
				else: 
					i2 = i1 + 1
					T.append(i1 * self._spl.delta - (i1 * self._spl.delta - i2 * self._spl.delta) * ((L[i] - length[i1]) / (length[i2] - length[i1])))

		else:
			i1 = np.argmax(length >= L)

			if i1 == len(length) - 1 or (i1 == 0 and length[0]< L):
				T = 1.0
			else: 
				i2 = i1 + 1
				T = i1 * self._spl.delta - (i1 * self._spl.delta - i2 * self._spl.delta) * ((L - length[i1]) / (length[i2] - length[i1])) 
		
		return T
			 


	def time_to_length(self, T):

		""" Return arc length value for a vector (resp. float) of time(s) parameter.
		The precision of the approximation depends on the attribute delta of spline spl.

		Keyword arguments:
		T -- curve times (list or single float value)
		"""

		length = np.array(self._length_tab)

		if type(T) == list:

			L = []
			for i in range(len(T)):

				i1 = int(np.floor(T[i] / self._spl.delta))

				if i1 == len(length) - 1:
					L.append(length[-1])
				else:
					i2 = i1 + 1
					L.append(length[i1] - (length[i1] - length[i2]) * ((T[i] / self._spl.delta - i1) / (i2 - i1)))

		else:

			i1 = int(np.floor(T/ self._spl.delta))

			if i1 == len(length) - 1:
					L = length[-1]
			else: 
				i2 = i1 + 1
				L = length[i1] - (length[i1] - length[i2]) * ((T / self._spl.delta - i1) / (i2 - i1))

		return L
		


	def resample_time(self, n, t0 = 0.0, t1 = 1.0):

		""" Return a vector of n times with equal spacing on spline spl.

		Keyword arguments:
		n -- number of times required
		t0, t1 -- first and last times
		"""

		l1 = self.time_to_length(t1)
		l0 = self.time_to_length(t0)

		if n == 0:
			raise ValueError('n must be a positive int')

		L = np.linspace(l0, l1, n+2)[1:-1].tolist() 
		T = self.length_to_time(L)

		return T



	def transport_vector(self, v, t0, t1):

		""" Smoothly transport a vector from t0 to t1 along the spline. 

		Keyword arguments:
		v -- vector to transport
		t0, t1 -- transport times
		"""

		distance = abs(self.time_to_length(t1) - self.time_to_length(t0))
		times = np.linspace(t0, t1, int(distance) + 2)

		tg = self.tangent(t0)

		if dot(v, tg) > 0.001:
			print("Warning : The vector to transport was not normal to the spline.", dot(v, tg))

		v = cross(tg, cross(v, tg)) # Make sure than v is normal to the spline
		v = v / norm(v)

		# Reference vector
		ref = cross(tg, v) 

		for i in range(1, len(times)):

			tg = self.tangent(times[i]) 

			v = cross(ref, tg)
			v = v / norm(v)
			ref = cross(tg, v)
			

		return v

	


	def project_time_to_surface(self, v, t):

		""" Project a point defined by a time value to the surface of the spline.

		Keyword arguments:
		v -- unit projection vector as numpy array
		t -- time scalar [0, 1]
		"""

		if self._spl.dimension == 3:
			raise AttributeError("This function can not be used with a 3D spline.")

		if t < 0 or t > 1:
			raise ValueError("Time value must be a number between 0 and 1.")

		tg = self.tangent(t)

		if dot(v, tg) > 0.001:
			print("Warning : The projection vector was not normal to the spline.", dot(v, tg))

		v = cross(tg, cross(v, tg)) # Make sure than v is normal to the spline
		v = v / norm(v)
		pt = self.point(t, True)

		return pt[:-1] + v * pt[-1]



	def project_point_to_surface(self, pt):

		""" Project a point to the nearest surface of the spline.

		Keyword arguments:
		pt -- 3D point to be projected
		"""

		if self._spl.dimension == 3:
			raise AttributeError("This function can not be used with a 3D spline.")

		t = self.project_point_to_centerline(pt)
		pt2 = self.point(t)
		v = pt2 - pt

		return self.project_time_to_surface(v, t)



	def project_point_to_centerline(self, pt):

		""" Smoothly transport a vector from t0 to t1 along the spline.

		Keyword arguments:
		pt -- 3D point to be projected as numpy array
		"""

		# Point table uniform t
		pts = self.get_points()

		if self._spl.dimension != 3:
			pts = pts[:, :-1]
	

		# Distance table
		dist = np.array([norm(pt - pt2) for pt2 in pts]) 
	
		# Minimum distance point
		i1 = np.argmin(dist) 

		# Find the closest segment
		if i1 == 0:
			i2 = 1
		elif i1 == len(dist) - 1:
			i2 = len(dist) - 2
		else:
			if dist[i1 - 1] > dist[i1 + 1]:
				i2 = i1 + 1
			else: 
				i2 = i1 - 1

		# Find the closest point on the segment
		u = pts[i2] - pts[i1]
		A = pts[i1]
	
		k = (u[0]*(pt[0] - A[0]) + u[1]*(pt[1] - A[1]) + u[2]*(pt[2] -A[2])) / (u[0]**2 + u[1]**2 + u[2]**2)
		P = np.array([k*u[0] + A[0], k*u[1] + A[1], k*u[2] + A[2]])

		# Convert it to t parameter linearly 
		t = i1 * self._spl.delta - (i1 * self._spl.delta - i2 * self._spl.delta) * (norm(P - pts[i1]) / norm(pts[i2] - pts[i1])) 
		
		if t < 0.0:
			t = 0.0
			print("Warning : The point was beyond the spline ends.")

		if t > 1.0:
			t = 1.0
			print("Warning : The point was beyond the spline ends.")

		return t


	#####################################
	#########  VISUALIZATION  ###########
	#####################################

	def show(self, knot = True, control_points = True, data = []):

		""" Displays the spline in 3D viewer.

		Keywords arguments:
		knot -- True to display the knots position
		control_points -- True to display control polygon
		"""

		# 3D plot
		with plt.style.context(('ggplot')):
		
			fig = plt.figure(figsize=(10,7))
			ax = Axes3D(fig)
			ax.set_facecolor('white')

			points = self.get_points()
			ax.plot(points[:,0], points[:,1], points[:,2],  c='black')

			if knot:
				knots = self.get_knot()
				for k in knots:
					pt = self.point(k)
					ax.scatter(pt[0], pt[1], pt[2],  c='black', s = 20)

			if control_points:
				points = self.get_control_points()
				ax.plot(points[:,0], points[:,1], points[:,2],  c='grey')
				ax.scatter(points[:,0], points[:,1], points[:,2],  c='grey', s = 40)

			if len(data) != 0:
				data = np.array(data)
				ax.scatter(data[:,0], data[:,1], data[:,2],  c='blue', s = 40)

		# Set the initial view
		ax.view_init(90, -90) # 0 is the initial angle

		# Hide the axes
		ax.set_axis_off()
		plt.show()

	