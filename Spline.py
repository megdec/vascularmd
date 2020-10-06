# Python 3

import numpy as np # Tools for matrices
from geomdl import BSpline, operations, helpers # Spline storage and evaluation

from numpy.linalg import norm 
from numpy import dot, cross
import matplotlib.pyplot as plt # Tools for plots
from mpl_toolkits.mplot3d import Axes3D # 3D display

from utils import *
from termcolor import colored


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

		# tg = self._spl.tangent(t)[1]
		tg = operations.tangent(self._spl, t)[1]

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


	def radius(self, t):

		""" Returns evaluation point of spline at time t as a numpy array."""

		pt = self._spl.evaluate_single(t)

		if self._spl.dimension == 4:
			radius = pt[-1]
		else:
			raise ValueError('The dimension of the spline must be 4.')
	
		return radius


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



	def __averaging_knot(self, t, p, n):

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



	def __basis_functions_derivative(self, knot, p, n, t):


		""" Computes the value of the first derivative of a B-spline basis functions.

		Keyword arguments:
		knot -- knot vector
		t -- time parameter
		n -- number of control points
		i -- index of the basis function
		p -- spline degree

		"""

		# Compute the basis function at time t for degree p - 1

		N = self.__basis_functions(knot, t, p - 1, n)


		# Compute the derivative of basis function i at time t (The NURBS Book p.62)
		#derN = []
		#for i in range(0, n-1):
		#	derN.append(((p / (knot[i + p] - knot[i])) * N[i]) - ((p / (knot[i + p + 1] - knot[i + 1])) * N[i + 1]))

		derN = []
		for i in range(n):
			derN.append(helpers.basis_function_ders_one(2, knot, i, t, 2)[1])

		return derN


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
		if n < 4:
			n = 4
		
		t = self.__chord_length_parametrization(D)
		knot =  self.__uniform_knot(3, n)
		
		search = True
		lbd = 0

		while (search and lbd < 1):
			
			self._spl = self.__solve_system(D, 3, n, knot, t, lbd, clip, deriv) 
			rad_curv = self.curvature_radius(np.arange(0, 1, self._spl.delta).tolist()) # + [1.0]
			rad = (self.get_points()[:,-1] / ratio).tolist()

			if all(rad_curv > rad):
				search = False
			else:
				lbd = lbd + 0.001

		self.set_spl(self.__solve_system(D, 3, n, knot, t, lbd, clip, deriv))



	def distance_constraint_approximation(self, D, dist, clip = [[],[]], deriv=[[],[]]): 


		"""Approximate data points using a spline of degree p with n control_points, 
		using constraint on the error of the fitting to find the optimal number of control points.

		Keyword arguments:
		D -- numpy array of coordinates for data points
		dist -- The maximum distance allowed from the original points
		clip -- end points if clipped ends
		deriv -- end tangents to apply if constrainted tangents
		"""

		# Number of control points
		n = len(D)
		if n < 4: # Minimum of 4 control points
			n = 4
		
		t = self.__chord_length_parametrization(D)
		knot =  self.__uniform_knot(3, n)
		
		search = True
		lbd1 = 0
		lbd2 = 1

		self.pspline_approximation(D, n, 0.0, clip=clip, deriv=deriv)
		mean_dist = np.mean(self.distance(t, D))
		max_dist = np.max(self.distance(t, D))

		if mean_dist > dist:
			print('Not enough control points')
			self.pspline_approximation(D, n, 0.0, clip=clip, deriv=deriv)
		else:

			# Dichotomy on lambda
			while abs(lbd2-lbd1) > 0.001 :
				
				self.pspline_approximation(D, n, (lbd2 + lbd1)/2, clip=clip, deriv=deriv)
				mean_dist = np.mean(self.distance(t, D))
				max_dist = np.max(self.distance(t, D))

				if mean_dist > dist:
					lbd2 =  (lbd2 + lbd1)/2
				else: 
					lbd1 =  (lbd2 + lbd1)/2
	
			self.pspline_approximation(D, n, (lbd2 + lbd1)/2, clip=clip, deriv=deriv)

			if len(deriv[0]) != 0:
				print(self.get_control_points()[1], deriv[0])
				print("alpha : ", ((self.get_control_points()[1] - self.get_control_points()[0]) / deriv[0])[0])

			if len(deriv[1]) != 0:
				print("beta : ", ((self.get_control_points()[-2] - self.get_control_points()[-1]) / deriv[1])[0])

		

	def pspline_approximation(self, D, n, lbd, knot=None, t=None, clip = [[], []], deriv = [[], []], derivative=False):


		"""Approximate data points using a spline with n control_points, 
		using constraint on the error of the fitting to find the optimal number of control points.

		Keyword arguments:
		D -- numpy array of coordinates for data points
		n -- number of control points
		clip -- end points if clipped ends
		deriv -- end tangents to apply if constrainted tangents
		derivatives -- True for fixed derivatives at the end, False for fixed tangent
		"""

		p = self._spl.order

		if knot is None:
			knot = self.__uniform_knot(p, n)

		if t is None:
			t = self.__chord_length_parametrization(D)


		dim = len(D[0])
		D = np.array(D)

		if type(lbd) == list:
			if len(lbd) != 2:
				raise ValueError('The vector of smoothing parameters must have two values.')
			else: 

				P = np.zeros((n, dim))

				# Spatial coordinates system

				c = [[], []]
				if len(clip[0])!= 0:
					c[0] = clip[0][:-1]
				if len(clip[1])!= 0:
					c[1] = clip[1][:-1]

				d = [[],[]]
				if len(deriv[0])!= 0:
					d[0] = deriv[0][:-1]
				if len(deriv[1])!= 0:
					d[1] = deriv[1][:-1]

				if derivative:
					P[:,:-1] = np.array(self.__solve_system_derivative(D[:,:-1], n, knot, t, lbd[0], c, d))
				else: 
					P[:,:-1] = np.array(self.__solve_system_tangent(D[:,:-1], n, knot, t, lbd[0], c, d))

				# Radius system
				c = [[], []]
				if len(clip[0])!= 0:
					c[0] = np.array([clip[0][-1]]) 
				if len(clip[1])!= 0:
					c[1] = np.array([clip[1][-1]])

				d = [[],[]]
				if len(deriv[0])!= 0:
					d[0] = np.array([deriv[0][-1]]) 
				if len(deriv[1])!= 0:
					d[1] = np.array([deriv[1][-1]])

				if derivative:
					P[:,-1] = np.reshape(self.__solve_system_derivative(np.reshape(D[:,-1], (len(D),1)), n, knot, t, lbd[1], c, d), n)
				else: 
					P[:,-1] = np.reshape(self.__solve_system_tangent(np.reshape(D[:,-1], (len(D),1)), n, knot, t, lbd[1], c, d), n)

		else:
			if derivative:
				P = self.__solve_system_derivative(D, n, knot, t, lbd, clip, deriv)
			else: 
				P = self.__solve_system_tangent(D, n, knot, t, lbd, clip, deriv)

		# Create spline
		spl = BSpline.Curve()
		spl.order = p
		spl.ctrlpts = P.tolist()
		spl.knotvector = knot

		self.set_spl(spl)
		#print("tg 0", deriv[0], self.tangent(0, True))
		#print("tg 1", deriv[1], self.tangent(1, True))



	def __solve_system_derivative(self, D, n, knot, t, lbd, clip = [[], []], deriv = [[], []]):

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

		p = self._spl.order
		m, x = D.shape

		if (len(deriv[0]) != 0 and len(clip[0]) == 0) or (len(deriv[1]) != 0 and len(clip[1]) == 0):
			raise ValueError("Please use clip ends to add tangent constraint.")

		if len(deriv[0]) != 0 and len(deriv[1]) != 0 and n<4:
			n = 4

		# Definition of matrix N
		N = np.zeros((len(t), n))
		for i in range(len(t)):
			N[i, :] = self.__basis_functions(knot, t[i], p, n)
		
		Pt = np.zeros((4, x))
		d = [0, n]

		# Get fixed control points at the ends
		if len(clip[0]) != 0:
			Pt[0,:] = clip[0]
			d[0] += 1

		if len(clip[1]) != 0:
			Pt[3,:] = clip[1]
			d[1] -= 1

		if len(deriv[0]) != 0:
			der0 = self.__basis_functions_derivative(knot, p, n, 0.0)
			Pt[1,:] = (1.0 / der0[1]) * (deriv[0] - (der0[0] * clip[0]))
			d[0] += 1

		if len(deriv[1]) != 0:
			#der1 = self.__basis_functions_derivative(knot, p, n, 1.0)
			der0 = self.__basis_functions_derivative(knot, p, n, 0.0)
			Pt[2,:] = (1.0 / -der0[1]) * (deriv[1] - (-der0[0] * clip[1]))
			d[1] -= 1


		# Definition of matrix Q1
		Q1 = np.zeros((m, x))
		for i in range(m):
			Q1[i, :] = N[i,0] * Pt[0,:] + N[i,1] * Pt[1,:] + N[i, -2] * Pt[2,:] + N[i, -1] * Pt[3,:]

		# Resizing N if clipped ends
		N = N[:, d[0]:d[1]]	


		# Get matrix Delta = UtU of difference operator
		U = np.zeros((n-2, n))
		for i in range(n-2):
			U[i, i:i+3] = [1.0, -2.0, 1.0]

		Delta = np.dot(U.transpose(), U)

		Q2 = np.zeros((d[1] - d[0], x))
		for i in range(d[0], d[1]):
			Q2[i - d[0], :] = Delta[i,0] * Pt[0,:] + Delta[i,1] * Pt[1,:] + Delta[i, -2] * Pt[2,:] + Delta[i, -1] * Pt[3,:]

		Delta = Delta[d[0]:d[1], d[0]:d[1]]
		
		# Write matrix M1 = NtN + lbd * Delta
		M1 = (1-lbd) * np.dot(N.transpose(), N) + lbd * Delta

		# Write matrix M2 
		M2 = (1-lbd) * np.dot(N.transpose(), D - Q1) - lbd * Q2

		# Solve the system
		P = np.dot(np.linalg.pinv(M1), M2)

		# Add fixed points to the list
		if len(deriv[0]) != 0:
			P = np.concatenate([np.transpose(np.expand_dims(Pt[1,:], axis=1)), P])
		if len(clip[0]) != 0:
			P = np.concatenate([np.transpose(np.expand_dims(Pt[0,:], axis=1)), P])
		if len(deriv[1]) != 0:
			P = np.concatenate([P, np.transpose(np.expand_dims(Pt[2,:], axis=1))])
		if len(clip[1]) != 0:
			P = np.concatenate([P, np.transpose(np.expand_dims(Pt[3,:], axis=1))])

		return P



	def __solve_system_tangent(self, D, n, knot, t, lbd, clip = [[], []], tangent = [[], []]):

		"""Approximate data points using a spline of degree p with n control_points.

		Keyword arguments:
		D -- numpy array of coordinates for data points
		n -- number of control points
		knot -- knot vector
		t -- time parametrization vector
		lbd -- lambda coefficient that balances smoothness and accuracy
		clip -- list of end points as np arrays
		tangent-- list of end tangents as np arrays
		"""

		p = self._spl.order
		m, x = D.shape
		
		D = D.reshape((m * x, 1)) 
				
		if (len(tangent[0]) != 0 and len(clip[0]) == 0) or (len(tangent[1]) != 0 and len(clip[1]) == 0):
			raise ValueError("Please use clip ends to add tangent constraint.")

		if len(tangent[0]) != 0 and len(tangent[1]) != 0 and n<4:
			n = 4

		# Definition of the basis function matrix
		Nl = np.zeros((len(t), n))
		for i in range(len(t)):
			Nl[i, :] = self.__basis_functions(knot, t[i], p, n)

		# Definition of matrix N
		N = np.zeros((len(t) * x, n * x))
		for i in range(len(t)):
			for j in range(x):
				N[i*x + j, j::x] = Nl[i]

		# Definition of the smoothing matrix Delta
		U = np.zeros((x*(n-2), n*x))
		for i in range(x*(n-2)):
			U[i, i:i+(x*2) +1:x] = [1.0, -2.0, 1.0]

		Delta = np.dot(U.transpose(), U)

		Q1 = np.zeros((x*m,))
		d = [0, x*n]
	
		# Get fixed control points and adjust matrix N
		if len(clip[0]) != 0:
			Q1 += np.sum(N[:, :x], axis=1) * np.concatenate([clip[0]] * m)
			N = N[:, x:]
			d[0] += x

		if len(tangent[0]) != 0:
			Q1 += np.sum(N[:, :x], axis=1) * np.concatenate([clip[0]] * m)
			N0 = np.sum(N[:, :x], axis = 1) * np.concatenate([tangent[0]]*len(t))
			N = N[:, x-1:]
			N[:,0] = N0
			d[0] += x - 1
			
		if len(clip[1]) != 0:
			Q1 += np.sum(N[:, -x:], axis=1) * np.concatenate([clip[1]] * m)
			N = N[:, :-x]
			d[1] -= x

		if len(tangent[1]) != 0:
			Q1 += np.sum(N[:, -x:], axis=1) * np.concatenate([clip[1]] * m)
			N0 = np.sum(N[:, -x:], axis = 1) * np.concatenate([tangent[1]]*len(t))
			if x-1 != 0:
				N = N[:, :-x+1]
				
			N[:,-1] = N0
			d[1] -= x - 1

		Q1 = np.expand_dims(Q1, axis=1)

		# Store the sum of line and columns
		Sc = np.zeros((Delta.shape[0], 4))
		Sc[:,0] = np.sum(Delta[:, :x], axis=1)
		Sc[:,1] = np.sum(Delta[:, x:2*x], axis=1)
		Sc[:,2] = np.sum(Delta[:, -2*x:-x], axis=1)
		Sc[:,3] = np.sum(Delta[:, -x:], axis=1)

		Sl = np.zeros((2, Delta.shape[1]))
		Sl[0,:] = np.sum(Delta[x:2*x, :], axis = 0)
		Sl[1,:] = np.sum(Delta[-2*x:-x, :], axis = 0)

		# Resize Delta
		Delta = Delta[d[0]:d[1], d[0]:d[1]]

		# Fill Delta if tangents
		if len(tangent[0])!= 0:
			Delta[0,:] = (Sl[0,:] * np.concatenate([tangent[0]]*n))[d[0]:d[1]]
			Delta[:,0] = (Sc[:,1] * np.concatenate([tangent[0]]*n))[d[0]:d[1]]
			Delta[0,0] = Sc[d[0],1]*np.dot(tangent[0], tangent[0])

		if len(tangent[1])!=0:
			Delta[-1,:] = (Sl[1,:] * np.concatenate([tangent[1]]*n))[d[0]:d[1]]
			Delta[:,-1] = (Sc[:,2] * np.concatenate([tangent[1]]*n))[d[0]:d[1]]
			Delta[-1,-1] = Sc[d[1],2]*np.dot(tangent[1], tangent[1])

			if len(tangent[0])!= 0:
				Delta[0, -1] = Sc[d[0],2] * np.dot(tangent[0], tangent[1])
				Delta[-1, 0] = Sc[d[1],0] * np.dot(tangent[0], tangent[1])

		# Define Q2 
		Q2 = np.zeros((x*n,))

		if len(clip[0])!=0:
			Q2 += Sc[:,0] * np.concatenate([clip[0]] * n) 
		if len(clip[1])!=0:
			Q2 += Sc[:,3] * np.concatenate([clip[1]] * n)
		if len(tangent[0])!= 0:
			Q2 += Sc[:,1] * np.concatenate([clip[0]] * n) 
		if len(tangent[1])!= 0:
			Q2 += Sc[:,2] * np.concatenate([clip[1]] * n) 

		Q2 = Q2[d[0]:d[1]]

		# Fell Q2 if tangents
		if len(tangent[0])!= 0:
			Q2[0] = (Sc[d[0],0] + Sc[d[0],1]) * np.dot(clip[0], tangent[0])
			if len(tangent[1])!= 0:
				Q2[-1] = (Sc[d[1],0] + Sc[d[1],1]) * np.dot(clip[0], tangent[1])

		if len(tangent[1])!= 0:
			if len(tangent[0])!= 0:
				Q2[-1] += (Sc[d[1],2] + Sc[d[1],3]) * np.dot(clip[1], tangent[1])
				Q2[0] += (Sc[d[0],2] + Sc[d[0],3]) * np.dot(clip[1], tangent[0])
			else: 
				Q2[-1] = (Sc[d[1],2] + Sc[d[1],3]) * np.dot(clip[1], tangent[1])

		Q2 = np.expand_dims(Q2, axis=1)
		
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

		# Add tangent points to the list
		if len(tangent[0]) != 0:
			P = np.concatenate([np.transpose(np.expand_dims(clip[0] + alpha * tangent[0], axis=1)), P])

		if len(tangent[1]) != 0:
			P = np.concatenate([P, np.transpose(np.expand_dims(clip[1] + beta * tangent[1], axis=1))])

		# Add fixed points to the list
		if len(clip[0]) != 0:
			P = np.concatenate([np.transpose(np.expand_dims(clip[0], axis=1)), P])
		if len(clip[1]) != 0:
			P = np.concatenate([P, np.transpose(np.expand_dims(clip[1], axis=1))])

		return P


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
		if t >= 1.0:
			t = 0.9 # WORKAROUND !!!
		elif t<= 0.0:
			t = 0.1
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

		""" Returns the estimated length of the spline."""
		return self._length_tab[-1]


	def mean_radius(self):

		""" Return the estimated mean radius of the spline."""
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

				i1 = np.argmax(length > L[i])

				if i1 == 0:
					if length[0] > L[i]:
						T.append(0.0) # length is < 0
					else: 
						T.append(1.0) # length is > length of the spline
				
				else: 
					i1 = i1 - 1
					i2 = i1 + 1
					T.append((i1 * self._spl.delta) + (self._spl.delta * ((L[i] - length[i1]) / (length[i2] - length[i1]))))


		else:

			i1 = np.argmax(length > L)

			if i1 == 0:
				if length[0] > L:
					T = 0.0 # length is < 0
				else: 
					T = 1.0 # length is > length of the spline
				
			else: 
				i1 = i1 - 1
				i2 = i1 + 1
				T = (i1 * self._spl.delta) + (self._spl.delta * ((L - length[i1]) / (length[i2] - length[i1])))

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

			if i1 >= len(length) - 1:
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
			print(colored('Warning : The vector to transport was not normal to the spline (', 'red'), colored(dot(v, tg), 'red'), colored(')', 'red'))

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
			print(colored('Warning : The vector to transport was not normal to the spline (', 'red'), colored(dot(v, tg), 'red'), colored(')', 'red'))

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

		""" Project a point to the spline.

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
			#print(colored('Warning : The point was beyond the spline ends.', 'red'))

		if t > 1.0:
			t = 1.0
			#print(colored('Warning : The point was beyond the spline ends.', 'red'))

		return t



	def distance(self, t, D):


		""" Compute the minimum distance between the spline and a list of points.

		Keyword arguments: 
			D -- numpy array of data points
			t -- numpy array of times
		"""

		if len(D.shape) == 1:
			proj = self.point(t)
			dist =  norm(D[:-1] - proj)

		else: 

			dist = np.zeros((D.shape[0],))

			for i in range(D.shape[0]):
				proj = self.point(t[i])
				dist[i] = norm(D[i, :-1] - proj)

		return dist


	def estimated_distance(self, D):

		""" Compute the minimum distance between the spline and a list of points.

		Keyword arguments: 
			D -- numpy array of data points
		"""

		if len(D.shape) == 1:
			proj = self.point(self.project_point_to_centerline(D))
			dist =  norm(D[:-1] - proj)

		else: 

			dist = np.zeros((D.shape[0],))

			for i in range(D.shape[0]):
				proj = self.point(self.project_point_to_centerline(D[i,:-1]))
				dist[i] = norm(D[i, :-1] - proj)

		return dist



	def first_intersection(self, spl, t0=0.0, t1=1.0):

		""" Returns the coordinates and time of the furthest intersection point

		Keyword arguments: 
		spl -- Spline object
		t0, t1 -- Times in between the search occurs
		"""
		
		tg1 = self.tangent(t1)
		v = cross(tg1, np.array([1,0,0]))
		v = v / norm(v)


		tmax = 0.0
		angle = np.linspace(0,2*pi, 100)

		for a in angle:

			vrot = rotate_vector(v, tg1, a)
			ap, times = self.intersection(spl, vrot, t0, t1)

			if max(times) > tmax:
				tmax = max(times)

				AP = ap
				tAP = times


		return np.array(AP), tAP
	



	def intersection(self, spl, v0, t0, t1):

		""" Returns the intersection point and time between two splines, given a initial vector v0.

		Keywords arguments: 
		spl -- Spline object
		v0 -- reference vector for the search
		t0, t1 -- Times in between the search occurs
		"""

		tinit = t1
		while abs(t1 - t0) > 10**(-6):

			t = (t1 + t0) / 2.
			
			v = self.transport_vector(v0, tinit, t)
			pt = self.project_time_to_surface(v, t) 
			
			t2 = spl.project_point_to_centerline(pt)
			pt2 = spl.point(t2, True)

			if norm(pt - pt2[:-1]) <= pt2[-1]:
				t0 = t
			else: 
				t1 = t
				

		return pt, [t, t2]


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

	