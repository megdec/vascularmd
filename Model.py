# Python 3

import numpy as np # Tools for matrices
from numpy.linalg import norm 
from geomdl import BSpline, operations, helpers # Spline storage and evaluation

from Spline import Spline
from utils import *
import math

class Model:

	#####################################
	##########  CONSTRUCTOR  ############
	#####################################

	def __init__(self, D, n, p, end_constraint, end_values, derivatives, lbd, knot = None, t = None):

		""" Create a penalized spline model.

		Keyword arguments:
		D -- numpy array of coordinates for data points
		p -- degree
		n -- number of control points
		knot -- knot vector
		t -- time parametrization vector
		lbd -- lambda coefficient that balances smoothness and accuracy
		end_constraint -- list of booleans for end points and tangent constraints
		end_values -- np array of values for end points and tangent constraints
		derivatives -- true to contraint derivatives and false to constraint tangents
		"""

		self._D = D
		self._p = p
		self._n = n

		if t is None:
			self._t = self.chord_length_parametrization()
		else: 
			self._t = t

		if knot is None: 
			self._knot = self.uniform_knot() #self.averaging_knot()
		else: 
			self._knot = knot

		self._end_constraint = end_constraint
		self._end_values = end_values
		self._derivatives = derivatives
		self._lbd = lbd

		if self._n <= 3:
			self._smoothing_order = n-1
		else:
			self._smoothing_order = 3

		self._N, self._Q1, self._Delta, self._Q2, self._Pt = self.__compute_matrices()
		self.P = self.__solve_system()
	
		self.spl = Spline(self.P, self._knot, self._p)
		#self.spl.show(True, True, data=self._D)
		

	def get_t(self):
		return self._t
		
	def get_order(self):
		return self._p

	def get_knot(self):
		return self._knot

	def get_lbd(self):
		return self._lbd

	def set_lambda(self, lbd):

		""" Change the smoothing parameter of the model """

		self._lbd = lbd
		self.P = self.__solve_system()
		self.spl = Spline(self.P, self._knot, self._p)



	def get_magnitude(self):

		"""Return the magnitudes alpha and beta of the tangents"""

		tg0 = self.spl.first_derivative(0) 
		tg1 = self.spl.first_derivative(1) 

		sign0 = 1
		sign1 = 1

		if self._end_constraint[1]:
			if tg0[0] * self._end_values[1,0] < 0:
				sign0 = -1

		if self._end_constraint[2]:
			if tg1[0] * self._end_values[2,0] < 0:
				sign1 = -1

		return sign0 * norm(tg0), sign1 * norm(tg1)
		

	def quality(self, criterion="CV"):

		""" Returns the smoothing criterion value (AIC, AICC, SBC, CV, GCV) for the given data.

		Keyword arguments:
		criterion -- string of the chosen criterion ("AIC", "AICC", "SBC", "CV", "GCV", "SSE")
		"""

		m, x = self._D.shape

		De = np.zeros((m, x))

		for i in range(m):
			De[i, :] = self.spl.point(self._t[i], True)

		# Hat matrix H
		M = np.linalg.pinv(np.dot(self._N.transpose(), self._N) + self._lbd * self._Delta)
		H = np.dot(np.dot(self._N, M), self._N.transpose())
		t = np.trace(H)


		if criterion == "CV":
			res = 0
			for i in range(m):
				res += (norm((self._D[i] - De[i])) / (1 - H[i, i]))**2

		elif criterion == "GCV":
			res = 0
			for i in range(m):
				res += (norm(self._D[i] - De[i]) / (m - t))**2

		elif criterion == "AIC":
			sse = 0
			for i in range(m):
				sse += norm(self._D[i] - De[i])**2
			if self._lbd != 0.0:
				res = m * math.log(sse/m) + 2 * t
			else:
				res = m * math.log(sse) + 2*(4*self._n + self._p)

		elif criterion == "AICC":
			sse = 0
			for i in range(m):
				sse += norm(self._D[i] - De[i])**2
			if self._lbd != 0.0:
				res = 1 + math.log(sse/m) + (2*(t+1))/(m - t - 2)
			else:
				K = 4*self._n + self._p
				if m == K + 1:
					res = m * math.log(sse) + 2*K + ((2*K*(K+1)) / 1)
				else:
					res = m * math.log(sse) + 2*K + ((2*K*(K+1)) / (m-K-1))

		elif criterion == "SBC":

			sse = 0
			for i in range(m):
				sse += norm(self._D[i] - De[i])**2
			res = m * math.log(sse/m) + math.log(m)*t

		elif criterion == "SSE":
			res = 0
			for i in range(m):
				res += norm(self._D[i] - De[i])**2

		elif criterion == "ASE":
			
			SE = np.sum((self._D - De)**2, axis = 1)
			ASE_spatial = np.sum(SE[:-1]) / len(self._D)
			ASE_radius = SE[-1] / len(self._D)

			res = [ASE_spatial, ASE_radius]
			

		elif criterion == "ASEder":
			# Estimation of the first derivative
			length = length_polyline(self._D)
			data_der = np.zeros((self._D.shape[0]-2, self._D.shape[1]))

			for i in range(1, len(self._D)-1):
				data_der[i-1] = (self._D[i+1] - self._D[i-1]) / (length[i+1] - length[i-1])


			# ASEder computation
			estim = self.spl.tangent(self._t, True)

			SE = np.sum((data_der - estim)**2, axis = 1)
			ASEder_spatial = np.sum(SE[:-1]) / len(data_der)
			ASEder_radius = SE[-1] / len(data_der)
			res = [ASEder_spatial, ASEder_radius]


		else: 
			raise ValueError('Invalid criterion name')

		return res


	def __solve_system(self):

		# Write matrix M1 = NtN + lbd * Delta
		#M1 = (1-lbd) * np.dot(N.transpose(), N) + lbd * Delta

		M1 = np.dot(self._N.transpose(), self._N) + self._lbd * self._Delta
		D = self._D
		if not self._derivatives: 

			m, x = self._D.shape
			D = self._D.reshape((m * x, 1))

		
		# Write matrix M2 
		#M2 = (1-lbd) * np.dot(N.transpose(), D - Q1) - lbd * Q2
		M2 = np.dot(self._N.transpose(), D - self._Q1) - self._lbd * self._Q2


		# Solve the system
		P = np.dot(np.linalg.pinv(M1), M2)
		Pt = np.copy(self._Pt)

		if not self._derivatives: 

			if self._end_constraint[1]:
				Pt[1, :] = self._Pt[0, :] + P[0] * self._Pt[1, :]
				P = P[1:]

			if self._end_constraint[-2]:
				Pt[2, :] = self._Pt[3, :] + P[-1] * self._Pt[2, :]
				P = P[:-1]

			P = P.reshape((int(len(P) / x), x))	
	
	
		if self._end_constraint[1]:
			P = np.concatenate([np.transpose(np.expand_dims(Pt[1,:], axis=1)), P])
		if self._end_constraint[0]:
			P = np.concatenate([np.transpose(np.expand_dims(Pt[0,:], axis=1)), P])
		if self._end_constraint[-2]:
			P = np.concatenate([P, np.transpose(np.expand_dims(Pt[2,:], axis=1))])
		if self._end_constraint[-1]:
			P = np.concatenate([P, np.transpose(np.expand_dims(Pt[3,:], axis=1))])

		return P


	def __compute_matrices(self):

		"""Compute the necessary matrices to approximate data points."""

		m, x = self._D.shape
		n = self._n

		if (self._end_constraint[1] and not self._end_constraint[0]) or (self._end_constraint[2] and not self._end_constraint[3]):
			raise ValueError("Please use clip ends to add tangent constraint.")

		if self._end_constraint[1] and self._end_constraint[-2] and n<4:
			n = 4

		if self._derivatives:

			# Definition of matrix N
			N = np.zeros((len(self._t), n))
			for i in range(len(self._t)):
				N[i, :] = self.__basis_functions(self._t[i])
			
			Pt = np.zeros((4, x))
			d = [0, n]

			# Get fixed control points at the ends
			if self._end_constraint[0]:
				Pt[0,:] = self._end_values[0,:]
				d[0] += 1

			if self._end_constraint[-1]:
				Pt[-1,:] = self._end_values[-1,:]
				d[1] -= 1

			if self._end_constraint[1]:
				der0 = self.__basis_functions_derivative(0.0)
				Pt[1,:] = (1.0 / der0[1]) * (self._end_values[1,:] - (der0[0] * self._end_values[0,:]))
				d[0] += 1

			if self._end_constraint[-2]:
				#der1 = self.__basis_functions_derivative(knot, p, n, 1.0)
				der0 = self.__basis_functions_derivative(0.0)
				Pt[2,:] = (1.0 / -der0[1]) * (self._end_values[-2,:]  - (-der0[0] * self._end_values[-1,:]))
				d[1] -= 1


			# Definition of matrix Q1
			Q1 = np.zeros((m, x))
			for i in range(m):
				Q1[i, :] = N[i,0] * Pt[0,:] + N[i,1] * Pt[1,:] + N[i, -2] * Pt[2,:] + N[i, -1] * Pt[3,:]

			# Resizing N if clipped ends
			N = N[:, d[0]:d[1]]	


			# Get matrix Delta = UtU of difference operator
			coefs = [[1.0, -2.0, 1.0], [1.0, -3.0, 3.0, -1.0], [1.0, -4.0, 6.0, -4.0, 1.0]]

			U = np.zeros((n-self._smoothing_order, n))
			for i in range(n-self._smoothing_order):
				U[i, i:i+self._smoothing_order + 1] = coefs[self._smoothing_order-2]


			Delta = np.dot(U.transpose(), U)

			Q2 = np.zeros((d[1] - d[0], x))
			for i in range(d[0], d[1]):
				Q2[i - d[0], :] = Delta[i,0] * Pt[0,:] + Delta[i,1] * Pt[1,:] + Delta[i, -2] * Pt[2,:] + Delta[i, -1] * Pt[3,:]

			Delta = Delta[d[0]:d[1], d[0]:d[1]]


		else: 

			D = self._D.reshape((m * x, 1))

			# Definition of the basis function matrix
			Nl = np.zeros((len(self._t), n))
			for i in range(len(self._t)):
				Nl[i, :] = self.__basis_functions(self._t[i])

			# Definition of matrix N
			N = np.zeros((len(self._t) * x, self._n * x))
			for i in range(len(self._t)):
				for j in range(x):
					N[i*x + j, j::x] = Nl[i]

			# Definition of the smoothing matrix Delta
			coefs = [[1.0, -2.0, 1.0], [1.0, -3.0, 3.0, -1.0], [1.0, -4.0, 6.0, -4.0, 1.0]]

			U = np.zeros((x*(n-self._smoothing_order), n*x))
			for i in range(x*(n-self._smoothing_order)):
				U[i, i:i+(x*self._smoothing_order) +1:x] = coefs[self._smoothing_order - 2]

			#U = np.zeros((x*(n-2), n*x))
			#for i in range(x*(n-2)):
			#	U[i, i:i+(x*2) +1:x] = [1.0, -2.0, 1.0]

			Delta = np.dot(U.transpose(), U)

			Q1 = np.zeros((x*m,))
			d = [0, x*n]
			Pt = np.zeros((4, x)) 

			# Get fixed control points and adjust matrix N
			if self._end_constraint[0]:
				Q1 += np.sum(N[:, :x], axis=1) * np.concatenate([self._end_values[0,:]] * m)
				N = N[:, x:]
				Pt[0,:] = self._end_values[0,:]
				d[0] += x

			if self._end_constraint[1]:
				Q1 += np.sum(N[:, :x], axis=1) * np.concatenate([self._end_values[0,:]] * m)
				N0 = np.sum(N[:, :x], axis = 1) * np.concatenate([self._end_values[1,:]]*len(self._t))
				N = N[:, x-1:]
				N[:,0] = N0
				Pt[1,:] = self._end_values[1,:]
				d[0] += x - 1
				
			if self._end_constraint[-1]:
				Q1 += np.sum(N[:, -x:], axis=1) * np.concatenate([self._end_values[-1,:]] * m)
				N = N[:, :-x]
				Pt[3,:] = self._end_values[-1,:]
				d[1] -= x

			if self._end_constraint[-2]:
				Q1 += np.sum(N[:, -x:], axis=1) * np.concatenate([self._end_values[-1,:]] * m)
				N0 = np.sum(N[:, -x:], axis = 1) * np.concatenate([self._end_values[-2,:]]*len(self._t))
				if x-1 != 0:
					N = N[:, :-x+1]
					
				N[:,-1] = N0
				Pt[2,:] = self._end_values[-2,:]
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
			if self._end_constraint[1]:
				Delta[0,:] = (Sl[0,:] * np.concatenate([self._end_values[1,:]]*n))[d[0]:d[1]]
				Delta[:,0] = (Sc[:,1] * np.concatenate([self._end_values[1,:]]*n))[d[0]:d[1]]
				Delta[0,0] = Sc[d[0],1]*np.dot(self._end_values[1,:], self._end_values[1,:])

			if self._end_constraint[-2]:
				Delta[-1,:] = (Sl[1,:] * np.concatenate([self._end_values[-2,:]]*n))[d[0]:d[1]]
				Delta[:,-1] = (Sc[:,2] * np.concatenate([self._end_values[-2,:]]*n))[d[0]:d[1]]
				Delta[-1,-1] = Sc[d[1],2]*np.dot(self._end_values[-2,:], self._end_values[-2,:])

				if self._end_constraint[1]:
					Delta[0, -1] = Sc[d[0],2] * np.dot(self._end_values[1,:], self._end_values[-2,:])
					Delta[-1, 0] = Sc[d[1],0] * np.dot(self._end_values[1,:], self._end_values[-2,:])

			# Define Q2 
			Q2 = np.zeros((x*n,))

			if self._end_constraint[0]:
				Q2 += Sc[:,0] * np.concatenate([self._end_values[0,:]] * n) 
			if self._end_constraint[-1]:
				Q2 += Sc[:,3] * np.concatenate([self._end_values[-1,:]] * n)
			if self._end_constraint[1]:
				Q2 += Sc[:,1] * np.concatenate([self._end_values[0,:]] * n) 
			if self._end_constraint[-2]:
				Q2 += Sc[:,2] * np.concatenate([self._end_values[-1,:]] * n) 

			Q2 = Q2[d[0]:d[1]]

			# Fill Q2 if tangents
			if self._end_constraint[1]:
				Q2[0] = (Sc[d[0],0] + Sc[d[0],1]) * np.dot(self._end_values[0,:], self._end_values[1,:])
				if self._end_constraint[-2]:
					Q2[-1] = (Sc[d[1],0] + Sc[d[1],1]) * np.dot(self._end_values[0,:], self._end_values[-2,:])

			if self._end_constraint[-2]:
				if self._end_constraint[1]:
					Q2[-1] += (Sc[d[1],2] + Sc[d[1],3]) * np.dot(self._end_values[-1,:], self._end_values[-2,:])
					Q2[0] += (Sc[d[0],2] + Sc[d[0],3]) * np.dot(self._end_values[-1,:], self._end_values[1,:])
				else: 
					Q2[-1] = (Sc[d[1],2] + Sc[d[1],3]) * np.dot(self._end_values[-1,:], self._end_values[-2,:])

			Q2 = np.expand_dims(Q2, axis=1)


		return N, Q1, Delta, Q2, Pt



	def uniform_knot(self):

		""" Returns a B-spline uniform knot vector."""

		knot = []

		for i in range(self._p + self._n):
			if i < self._p:
				knot.append(0.0)
			elif self._p <= i <= self._n-1:
				knot.append(float(i-self._p+1))
			else:
				knot.append(float(self._n-self._p+1))

		return (np.array(knot) / knot[-1]).tolist()


	def uniform_averaging_knot(self):

		""" Returns a uniform knot vector based on the position of the data """

		# Choose n points equally spread along the data
		indices = np.arange(len(self._t)).tolist()

		# Compute point distances
		L = length_polyline(self._D)
		dist = np.vstack((abs(L[1:-1] - L[:-2]), abs(L[2:] - L[1:-1])))

		for i in range(len(self._t) - self._n):
			# Remove the point with minimum distance to others
			ind_min = np.argmin(np.sum(dist, axis=0))

			j = indices.index(ind_min+1)
			ind_bef = indices[j - 1]
			ind_aft = indices[j + 1]

			indices.remove(ind_min + 1)

			if ind_bef > 0:
				dist[1, ind_bef] += dist[1, ind_min]
			if ind_aft + 1 < dist.shape[1]:
				dist[0, ind_aft] += dist[0, ind_min]

			dist[0, ind_min] = np.inf
			dist[1, ind_min] = np.inf

			
		# Find averaging_knot
		knot = self.averaging_knot(self._t[indices])
		return knot



	def averaging_knot(self, t=None):

		""" Returns a B-spline averaging knot vector."""
		if t is None:
			t = self._t

		knot = [0.0] * self._p # First knot of multiplicity p

		for i in range(self._p, self._n):
			knot.append((1.0 / (self._p - 1.0)) * sum(t[i-self._p+1:i]))

		knot = knot + [1.0] * self._p

		return knot



	def chord_length_parametrization(self):

		""" Returns the chord length parametrization for data D.

		Keyword arguments:
		D -- data points
		"""
		
		t = [0.0]
		for i in range(1, len(self._D)):
			t.append(t[i-1] + np.linalg.norm(self._D[i] - self._D[i-1]))
		t = [time / max(t) for time in t]

		return np.array(t)


	def __basis_functions(self, t):


		"""Computes the value of B-spline basis functions evaluated at t

		Keyword arguments:
		t -- time parameter 
		"""

		N = [0.0]*self._n # list of basis function values 

		# Handle special cases for t 
		if t == self._knot[0]:
			N[0] = 1.0

		elif t == self._knot[-1]:
			N[-1] = 1.0
		else:

			# Find the bounding knots for t
			k = 0
			for kn in range(len(self._knot)-1):
				if self._knot[kn] <= t < self._knot[kn+1]:
					k = kn
		
			N[k] = 1.0 # Basis function of order 0
			
			# Compute basis functions = recurrence??!!
			for d in range(1, self._p): 

				if self._knot[k + 1] == self._knot[k- d + 1]:
					N[k-d] = 0
				else:
					N[k-d] = (self._knot[k + 1] - t) / (self._knot[k + 1] - self._knot[k- d + 1]) * N[k- d + 1]

				for i in range(k-d + 1, k):

					if self._knot[i+d] == self._knot[i]:
						c1 = 0
					else:
						c1 = (t - self._knot[i]) / (self._knot[i+d] - self._knot[i]) * N[i]

					if self._knot[i + d + 1] == self._knot[i + 1]:
						c2 = 0
					else:
						c2 = (self._knot[i + d + 1] - t) / (self._knot[i + d + 1] - self._knot[i + 1]) * N[i + 1]

					N[i] =  c1 + c2

				if self._knot[k+d] == self._knot[k]:
					N[k] = 0
				else:
					N[k] = (t - self._knot[k]) / (self._knot[k+d] - self._knot[k]) * N[k]
			
		# Return array of n basis function values at t 
		return N



	def __basis_functions_derivative(self, t):


		""" Computes the value of the first derivative of a B-spline basis functions.

		Keyword arguments:
		knot -- knot vector
		t -- time parameter
		n -- number of control points
		i -- index of the basis function
		p -- spline degree

		"""

		derN = []
		for i in range(self._n):
			derN.append(helpers.basis_function_ders_one(2, self._knot, i, t, 2)[1])

		return derN