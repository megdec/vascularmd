# Python 3

import numpy as np # Tools for matrices
from geomdl import BSpline, operations, helpers # Spline storage and evaluation

# Trigonometry functions
from math import pi, sin, cos, tan, atan, acos, asin, sqrt
from numpy.linalg import norm 
from numpy import dot, cross
import matplotlib.pyplot as plt # Tools for plots

from VTgeom import *
from VTsignal import *
from VTvisu import *


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
				self._spl.knotvector = self.__uniform_knot()
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


	def __uniform_knot(self):

		""" Returns a B-spline uniform knot vector."""

		knot = []
		p = self._spl.order
		n = self._spl.ctrlpts_size

		for i in range(p + n):
			if i < p:
				knot.append(0.0)
			elif p <= i <= n-1:
				knot.append(float(i-p+1))
			else:
				knot.append(float(n-p+1))

		return (np.array(knot) / knot[-1]).tolist()



	def __averagingKnot(self, t):

		""" Returns a B-spline averaging knot vector.

		Keyword arguments:
		t -- time parametrization vector
		"""

		p = self._spl.order
		n = self._spl.ctrlpts_size

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


	#####################################
	########## APPROXIMATION  ###########
	#####################################

	

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

	