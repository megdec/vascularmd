# Python 3

import numpy as np # Tools for matrices
from geomdl import BSpline, operations, helpers, fitting # Spline storage and evaluation

from numpy.linalg import norm 
from numpy import dot, cross
import matplotlib.pyplot as plt # Tools for plots
from mpl_toolkits.mplot3d import Axes3D # 3D display
import math

from utils import *
from termcolor import colored


class Spline:


	#####################################
	##########  CONSTRUCTOR  ############
	#####################################

	def __init__(self, control_points = None, knot = None, order = None):

		self._spl = BSpline.Curve()
		self._spl.order = 3	
		self._length_tab = None	

		if control_points is not None: 

			#if len(control_points[0]) != 3 and len(control_points[0]) != 4:
			#	raise ValueError ('The spline control points must have dimension 3 (x, y, z) or  (x, y, z, r).')

			self._spl.ctrlpts = control_points.tolist()

			if order is not None:
				self._spl.order = order

			if knot is None:
				self._spl.knotvector = self.__uniform_knot(self._spl.order, self._spl.ctrlpts_size)
			else: 
				self._spl.knotvector = knot

		self._model = [None, None]





	#####################################
	#############  GETTERS  #############
	#####################################

	def get_spl(self):
		return self._spl

	def get_knot(self):
		return self._spl.knotvector

	def get_control_points(self):
		return np.array(self._spl.ctrlpts)

	def get_points(self):
		if self._length_tab is None:
			self.__set_length_tab()
		return np.array(self._spl.evalpts)

	def get_times(self):
		if self._length_tab is None:
			self.__set_length_tab()
		return np.linspace(0, 1, len(self.get_points())) #np.arange(0, 1, self._spl.delta) 

	def get_length(self):

		if self._length_tab is None:
			self.__set_length_tab()

		return self._length_tab

	def get_nb_control_points(self):
		return len(self._spl.ctrlpts)

	def get_model(self):
		return self._model

	def get_data(self):

		D = None
		if self._model[0] is not None and self._model[1] is None:
			D = self._model[0].get_data()

		if self._model[0] is not None and self._model[1] is not None:
			D_spatial = self._model[0].get_data()
			D_radius = self._model[1].get_data()
			
			D = np.hstack((D_spatial, D_radius[:,1].reshape((-1, 1))))

		return D


	def get_lbd(self):
		
		lbd = []
		
		for m in self._model:
			if m is not None:
				lbd.append(m.get_lbd())
		return lbd



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
		self._spl.knotvector = self.__uniform_knot(self._spl.order, self._spl.ctrlpts_size)
		if self._length_tab is not None:
			self.__set_length_tab()



	def __set_length_tab(self):

		""" Set arc length estimation of spline."""
		length = operations.length_curve(self._spl)
		
		if int(length * 20) > 1:
			self._spl.delta = 1 / int(length * 20)
		else:
			self._spl.delta = 0.01

		if self._spl.dimension == 3:
			pts = np.array(self._spl.evalpts)
		else:
			pts = np.array(self._spl.evalpts)[:, :-1]

		self._kdtree = KDTree(pts)

		length = [0.0]
		for i in range(1, len(pts)):
			length.append(length[i-1] + norm(pts[i] - pts[i-1]))

		self._length_tab = np.array(length)


	def _set_lambda_model(self, lbd=[None, None]):
		""" Modify the lambda values of the approximation models """
		if lbd[0] != None:
			self._model[0].set_lambda(lbd[0])

		if lbd[1] != None:
			self._model[1].set_lambda(lbd[1])

		if len(self._model) == 2:

			P =  np.hstack((self._model[0].P, np.reshape(self._model[1].P[:,-1], (self._model[1].P.shape[0],1))))
			self._spl.ctrlpts = P.tolist()
			self._spl.knotvector = self._model[0].get_knot()

		else:
			self._spl.ctrlpts = self_model[0].P.tolist()
			self._spl.knotvector = self._model[0].get_knot()






	#####################################
	###########  EVALUATION  ############
	#####################################


	def first_derivative(self, t, radius = False):

		""" Returns the unit first derivative of spline at time t as a numpy array."""
		if type(t) == list or type(t) == np.ndarray:
			der = np.zeros((len(t), self._spl.dimension))

			for i in range(len(t)):
				der[i, :] = self._spl.derivatives(t[i], order=1)[1]

			if self._spl.dimension > 3 and not radius:	
				der = der[:, :-1]

		else:

			der = self._spl.derivatives(t, order=1)[1]

			if self._spl.dimension <= 3:
				der = np.array(der)
			else:
				if radius:
					der = np.array(der)
				else:
					der = np.array(der)[:-1]

		return der
		

	def second_derivative(self, t, radius = False):

		""" Returns the unit second derivative of spline at time t as a numpy array."""

		if type(t) == list or type(t) == np.ndarray:

			der = np.zeros((len(t), self._spl.dimension))

			for i in range(len(t)):
				der[i, :] = self._spl.derivatives(t[i], order=2)[2]

			if self._spl.dimension > 3 and not radius:	
				der = der[:, :-1]
		else:

			der = self._spl.derivatives(t, order=2)[2]

			if self._spl.dimension <= 3:
				der = np.array(der)
			else:
				if radius:
					der = np.array(der)
				else:
					der = np.array(der)[:-1]

		return der 
		

	def tangent(self, t, radius = False):

		""" Returns the unit tangent of spline at time t as a numpy array."""

		if type(t) == list or type(t) == np.ndarray:

			tg = np.zeros((len(t), self._spl.dimension))

			for i in range(len(t)):
				tg[i, :] = operations.tangent(self._spl, t[i])[1]

			if self._spl.dimension > 3 and not radius:	
				tg = tg[:, :-1]

			# Normalize
			for i in range(len(t)):
				tg[i, :] = tg[i, :] / norm(tg[i, :])

		else:

			# tg = self._spl.tangent(t)[1]
			tg = operations.tangent(self._spl, t)[1]

			if self._spl.dimension <= 3:
				tg = np.array(tg)
			else:
				if radius:
					tg = np.array(tg)
				else:
					tg = np.array(tg)[:-1]
			tg = tg / norm(tg)

		return tg 
		


	def point(self, t, radius = False):

		""" Returns evaluation point of spline at time t as a numpy array."""

		if type(t) == list or type(t) == np.ndarray:

			pt = np.array(self._spl.evaluate_list(t))

			if self._spl.dimension > 3 and not radius:
				pt = pt[:,:-1]


		else:

			pt = self._spl.evaluate_single(t)

			if self._spl.dimension <= 3:
				pt = np.array(pt)
			else:
				if radius:
					pt = np.array(pt)
				else:
					pt = np.array(pt)[:-1]
		
		return pt


	def radius(self, t):

		""" Returns evaluation point of spline at time t as a numpy array."""

		if type(t) == list or type(t) == np.ndarray:

			pt = np.array(self._spl.evaluate_list(list(t)))

			if self._spl.dimension == 4:
				radius = pt[:,-1]
			else:
				raise ValueError('The dimension of the spline must be 4.')
			
		else: 
			pt = self._spl.evaluate_single(t)

			if self._spl.dimension == 4:
				radius = pt[-1]
			else:
				raise ValueError('The dimension of the spline must be 4.')


	
		return radius





	#####################################
	########## APPROXIMATION  ###########
	#####################################
	

	def approximation(self, D, end_constraint, end_values, derivatives, radius_model=True, curvature=False, akaike=False, min_tangent=False, n=None, knot=None, lbd=0.0, criterion="None"): 

		"""Approximate data points using a spline with given end constraints.

		Keyword arguments:
		D -- numpy array of coordinates for data points
		end_constraint -- list of booleans for end points and tangent constraints
		end_values -- np array of values for end points and tangent constraints
		derivatives -- True for fixed derivatives at the end, False for fixed tangent
		criterion -- smoothing criterion
		"""
	
		from Model import Model
		
		if n == None:
			if akaike:
				n = self.__nb_control_points_akaike(D)
			else:
				n = self.__nb_control_points_RMSE_thres(D, [0.5, 0.5*10**(-2)])
				#n = self.__nb_control_points_max_dist_thres(D, [10**(-1), 10**(-3)])
				#n = self.__nb_control_points_stability(D, 10**(-4))


		if radius_model: 

			# Spatial model
			spatial_model = Model(D[:,:-1], n, 3, end_constraint, end_values[:,:-1], derivatives, lbd, knot= knot)
			spatial_model = self.__optimize_model(spatial_model, criterion)

			# Radius model
			t = spatial_model.get_t()
			data = np.transpose(np.vstack((t, D[:,-1])))
			radius_model = Model(data, n, 3, end_constraint, np.vstack((data[0], [1,0], [1,0], data[-1])), False, lbd, knot = spatial_model.get_knot(), t = spatial_model.get_t())
			radius_model = self.__optimize_model(radius_model, criterion)

			
			if min_tangent:

				alpha, beta = spatial_model.get_magnitude()
				thres = 5

				if (alpha < thres or beta < thres) and not derivatives:
					thres = 10

					#print(alpha, beta)
					#print(end_constraint[1], end_constraint[-2])
					if end_constraint[1]:
						if alpha >= thres:
							end_values[1,:] = end_values[1,:] / norm(end_values[1,:]) * alpha
						else:
							end_values[1,:] = end_values[1,:] / norm(end_values[1,:]) * thres

					if end_constraint[-2]:
						if beta >= thres:
							end_values[-2,:] = end_values[-2,:] / norm(end_values[-2,:]) * beta
						else: 
							end_values[-2,:] = end_values[-2,:] / norm(end_values[-2,:]) * thres


					spatial_model = Model(D[:,:-1], n, 3, end_constraint, end_values[:,:-1], True, lbd)
					spatial_model = self.__optimize_model(spatial_model, criterion)
					#print(spatial_model.get_magnitude())

			# Curvature optimization 
			if curvature: 
				spatial_model = self.__constraint_curvature(spatial_model, radius_model)


			P =  np.hstack((spatial_model.P, np.reshape(radius_model.P[:,-1], (radius_model.P.shape[0],1))))

			self._model[0] = spatial_model
			self._model[1] = radius_model
			self._spl.ctrlpts = P.tolist()
			self._spl.knotvector = spatial_model.get_knot()

			


		else:
			global_model = Model(D, n, 3, end_constraint, end_values, derivatives, lbd, knot=knot)
			global_model = self.__optimize_model(global_model, criterion)


			if min_tangent:

				alpha, beta = global_model.get_magnitude()
				thres = 15

				if (alpha < thres or beta < thres) and not derivatives:
					

					end_values[-2,:] = end_values[-2,:] / norm(end_values[-2,:]) * beta
					end_values[1,:] = end_values[1,:] / norm(end_values[1,:]) * alpha

					if alpha < thres:
						end_values[1,:] = end_values[1,:] / norm(end_values[1,:]) * thres
					if beta < thres:
						end_values[-2,:] = end_values[-2,:] / norm(end_values[-2,:]) * thres


					global_model = Model(D, n, 3, end_constraint, end_values, True, lbd)
					global_model = self.__optimize_model(global_model, criterion)
					alpha, beta = global_model.get_magnitude()

			self._model[0] = global_model
			self._spl.ctrlpts = global_model.P.tolist()
			self._spl.knotvector = global_model.get_knot()
		

			
			
		
			#self.show(data = D)
		
	def __nb_control_points_stability(self, data, thres):

		""" Find the optimal number of control points for a given RMSE threshold """

		from Model import Model

		
		n = 4
		model = Model(data, n, 3, [0,0,0,0], np.zeros((4,4)), False, 0.0)

		RMSE_act = model.quality("RMSE")
		RMSE_prec = (RMSE_act[0] + thres + 1, RMSE_act[1] + thres + 1)

		while n < len(data) and n < 200 and (abs(RMSE_prec[0] - RMSE_act[0])> thres or abs(RMSE_prec[1] - RMSE_act[1])> thres): # Stability
			n += 1
			model = Model(data, n, 3, [0,0,0,0], np.zeros((4,4)), False, 0.0)

			RMSE_prec = RMSE_act[:]
			RMSE_act = model.quality("RMSE")

		return n

	def __nb_control_points_RMSE_thres(self, data, thres):

		""" Find the optimal number of control points for a given RMSE threshold """

		from Model import Model

		
		n = 4
		model = Model(data, n, 3, [0,0,0,0], np.zeros((4,4)), False, 0.0)

		RMSE = model.quality("RMSE")

		while n < len(data) and n < 200 and (RMSE[0] > thres[0] or RMSE[1] > thres[1]): # Stability
			n += 1
			model = Model(data, n, 3, [0,0,0,0], np.zeros((4,4)), False, 0.0)

			RMSE = model.quality("RMSE")


		return n



	def __nb_control_points_max_dist_thres(self, data, thres):

		""" Find the optimal number of control points for a given RMSE threshold """

		from Model import Model
		
		n = 4
		model = Model(data, n, 3, [0,0,0,0], np.zeros((4,4)), False, 0.0)

		min_dist = model.quality("max_dist")

		while n < len(data) and n < 200 and (min_dist[0] > thres[0] or min_dist[1] > thres[1]): # Stability
			n += 1
			model = Model(data, n, 3, [0,0,0,0], np.zeros((4,4)), False, 0.0)

			min_dist = model.quality("max_dist")

		return n



	def __nb_control_points_akaike(self, data):

		""" Find the optimal number of control points using Akaike criterion """

		from Model import Model

		n = 4
		model = Model(data, n, 3, [0,0,0,0], np.zeros((4,4)), False, 0.0)

		#AIC_list = [AIC_act]
		#n_list = [n]

		AIC_min = model.quality("AICC")

		while n < len(data) and n < 200:# and (AIC_prec > AIC_act): 
			n += 1
			model = Model(data, n, 3, [0,0,0,0], np.zeros((4,4)), False, 0.0)

			AIC = model.quality("AICC")
			#AIC_list.append(AIC_act)
			#n_list.append(n)
			if AIC < AIC_min:
				n_min = n
				AIC_min = AIC

		#plt.plot(n_list, AIC_list)
		#plt.show()
	

		return n_min




	def __optimize_model(self, model, criterion):

		""" Optimise the value of smoothing parameter lambda 
		for a given model according to a smoothing criterion."""

		if criterion != "None":

			# Smoothing criterion
			gr = (math.sqrt(5) + 1) / 2
			a = 10**(-6)
			b = 100000

			# Golden-section search to find the minimum

			c = b - (b - a) / gr
			d = a + (b - a) / gr
			while abs(b - a) > 10**(-2):

				model.set_lambda(c)
				fc = model.quality(criterion)
				model.set_lambda(d)
				fd = model.quality(criterion)
			
				if fc < fd:
					b = d
				else:
					a = c

				# We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
				c = b - (b - a) / gr
				d = a + (b - a) / gr

			opt_lbd = (b + a) / 2
			model.set_lambda(opt_lbd)
			#print("optimized lambda : ", opt_lbd)

		return model


	def __constraint_curvature(self, spatial_model, radius_model):

		""" Check curvature and smoothes the spline is the constraint 
		radius of curvature > radius is not respected """

		num = 100
		tol = 10**(-6)
		t = np.linspace(0, 1, num)[50:]

		curv_rad = spatial_model.spl.curvature_radius(t)
		rad = radius_model.spl.point(t)[:, -1]
			
		if not all((curv_rad - rad) > tol):

			# Find lbd by dichotomy
			a = spatial_model.get_lbd()
			b = a + 1000000

			spatial_model.set_lambda(b)
			curv_rad = spatial_model.spl.curvature_radius(t)

			if not all((curv_rad - rad) > tol):
				#spatial_model.set_lambda(a)
				print("Could not correct curvature")
			else:

				while abs(a-b) >  10**(-2):

					spatial_model.set_lambda((a + b) / 2)
					curv_rad = spatial_model.spl.curvature_radius(t)

					if all((curv_rad - rad) > tol):
						b = (a + b) / 2
					else: 
						a = (a + b) / 2


				spatial_model.set_lambda((a + b) / 2)
				print("optimized lambda after curvature correction : ", (a + b) / 2)

		return spatial_model



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


	#####################################
	###########  GEOMETRY  ##############
	#####################################


	def split_time(self, t):

		""" Splits the spline into two splines at time t."""

		if t >= 1.0:
			t = 0.9# WORKAROUND !!!
		elif t<= 10**(-2):
			t = 0.1


		spla, splb = operations.split_curve(self._spl, t)

		spl1 = Spline()
		spl1.set_spl(spla)

		spl2 = Spline()
		spl2.set_spl(splb)

		# Keep model if the original spline had one
		if self._model[0] is not None and self._model[1] is not None:
			from Model import Model

			# Separate data points 
			T = self._model[0].get_t()
			thres = np.argmax(T>t)
			D_spatial = self._model[0].get_data()
			D_radius = self._model[1].get_data()
			
			D_spatial = [D_spatial[:thres,:], D_spatial[thres:,:]]
			D_radius = [D_radius[:thres, 1], D_radius[thres:, 1]]

			spl = [spl1, spl2]
			# Set the spline models
			for i in range(2):
				if len(D_spatial[i])> 1:

					if spl[i].get_nb_control_points() > 3:

						values = np.zeros((4,4))
						constraint = [True, True, True, True]

						if constraint[0]:
							values[0,:] = spl[i].point(0, True)
						if constraint[1]:
							values[1,:] = spl[i].tangent(0, True)

						if constraint[-1]:
							values[-1,:] = spl[i].point(1, True)
						if constraint[-2]:
							values[-2,:] = spl[i].tangent(1, True)
					
						spatial_model = Model(D_spatial[i], spl[i].get_nb_control_points(), 3, constraint, values[:,:-1], False, lbd = 0)
						times = spatial_model.get_t()
						data_radius = np.transpose(np.vstack((times, D_radius[i])))
						radius_model = Model(data_radius, spl[i].get_nb_control_points(), 3, constraint, np.vstack((data_radius[0], [1,0], [1,0], data_radius[-1])), False, lbd = 0, knot = spatial_model.get_knot(), t = times)

						if i == 0:
							spl1._model = [spatial_model, radius_model]
						else:
							spl2._model = [spatial_model, radius_model]

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
		return self.get_length()[-1]


	def mean_radius(self):

		""" Return the estimated mean radius of the spline."""
		return np.mean(self.get_points()[:, -1])


	def curvature(self, t):

		""" Returns the curvature value(s) of spline at time(s) T.

		Keyword arguments:	
		t -- curve times (list or single float value)
		"""
		
		if type(t) == list or type(t) == np.ndarray:

			C = np.zeros(len(t))
			for i in range(len(t)):
				der = [self.first_derivative(t[i]), self.second_derivative(t[i])]
				C[i] = (norm(cross(der[0], der[1])) / (norm(der[0]))**3)
			
		else:
			der = [self.first_derivative(t), self.second_derivative(t)]
			C = (norm(cross(der[0], der[1])) / (norm(der[0]))**3)

		return C

		


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

		length = self.get_length() #np.array(self._length_tab)

		if type(L) == list or type(L) == np.ndarray:

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

		length = self.get_length() #np.array(self._length_tab)

		if type(T) == list or type(T) == np.ndarray:

			L = []
			for i in range(len(T)):

				i1 = int(np.floor(T[i] / self._spl.delta))

				if i1 >= len(length) - 1:
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

		#if dot(v, tg) > 0.001:
		#	print(colored('Warning : The vector to transport was not normal to the spline (', 'red'), colored(dot(v, tg), 'red'), colored(')', 'red'))
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

		#if dot(v, tg) > 0.001:
		#	print(colored('Warning : The vector to transport was not normal to the spline (', 'red'), colored(dot(v, tg), 'red'), colored(')', 'red'))

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
		length = self.get_length()
		pts = self.get_points()
		
		if self._spl.dimension != 3:
			pts = pts[:, :-1]
	
		"""
		# Distance table
		dist = np.array([norm(pt - pt2) for pt2 in pts]) 
	
		# Minimum distance point
		i1 = np.argmin(dist) 
		"""
		d, i1 = self._kdtree.query(pt)

		# Find the closest segment
		if i1 == 0:
			i2 = 1
		elif i1 == len(pts) - 1:
			i2 = len(pts) - 2
		else:
			if norm(pt - pts[i1 - 1]) > norm(pt - pts[i1 + 1]): #dist[i1 - 1] > dist[i1 + 1]:
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



	def distance(self, D):


		""" Compute the minimum distance between the spline and a list of points.

		Keyword arguments: 
			D -- numpy array of data points
		"""

		dist = np.zeros((D.shape[0],))
		times = np.zeros((D.shape[0],))

		for i in range(D.shape[0]):
			t = self.project_point_to_centerline(D[i, :-1])
			times[i] = t
			dist[i] = norm(D[i, :-1] - self.point(t))

		return times, dist


	def estimated_point(self, D):
		"""  Point data estimated by the spline.

		Keyword arguments: 
			D -- numpy array of data points """

		P = np.zeros((D.shape))

		for i in range(D.shape[0]):
			t = self.project_point_to_centerline(D[i, :-1])
			P[i, :] = self.point(t, True)
		

		return P


	def estimated_first_derivative(self, D):

		""" First derivative of data estimated by the spline.

		Keyword arguments: 
			D -- numpy array of data points """

		P = np.zeros((D.shape))

		for i in range(D.shape[0]):
			t = self.project_point_to_centerline(D[i, :-1])
			P[i, :] = self.first_derivative(t, True)
		

		return P

	def estimated_tangent(self, D):

		""" Tangent of data estimated by the spline.

		Keyword arguments: 
			D -- numpy array of data points """

		P = np.zeros((D.shape))

		for i in range(D.shape[0]):
			t = self.project_point_to_centerline(D[i, :-1])
			P[i, :] = self.tangent(t, True)

		return P


	def estimated_curvature(self, D):

		""" Tangent of data estimated by the spline.

		Keyword arguments: 
			D -- numpy array of data points """

		t = []
		for i in range(D.shape[0]):
			t.append(self.project_point_to_centerline(D[i, :-1]))

		return self.curvature(t)



	def RMSE(self, data):
		""" Returns the average squared error between the spline estimation and a set of data points.
		Spatial and radius RMSE are returned separatly."""

		estim = self.estimated_point(data)

		MSE_spatial = np.sum(norm(data[:, :-1] - estim[:, :-1], axis=1)**2) / len(data)
		RMSE_spatial = np.sqrt(MSE_spatial)
		
		MSE_radius = np.sum((data[:, -1] - estim[:, -1])**2) / len(data)
		RMSE_radius = np.sqrt(MSE_radius)

		return [RMSE_spatial, RMSE_radius]


	def RMSEder(self, data, data_der=None):
		""" Returns the average squared error between the spline derivative estimation and a set of data points.
		Spatial and radius RMSE are returned separatly."""

		# Estimation of the first derivative

		if data_der is None:
			length = length_polyline(data)
			data_der = np.zeros((data.shape[0]-2, data.shape[1]))

			for i in range(1, len(data)-1):
				data_der[i-1] = (data[i+1] - data[i-1]) / (length[i+1] - length[i-1])

			estim = self.estimated_tangent(data[1:-1])


		# RMSEder computation
		estim = self.estimated_tangent(data)

		MSEder_spatial = np.sum(norm(data_der[:, :-1] - estim[:, :-1], axis=1)**2) / len(data)
		RMSEder_spatial = np.sqrt(MSEder_spatial)
		
		MSEder_radius = np.sum((data_der[:, -1] - estim[:, -1])**2) / len(data)
		RMSEder_radius = np.sqrt(MSEder_radius)

		return [RMSEder_spatial, RMSEder_radius]


	def RMSEcurv(self, data, data_curv):

		""" Returns the average squared error between the spline curvature estimation and a set of data points.
		Spatial and radius RMSE are returned separatly."""

		# Estimation of the curvature

		# RMSEcurv computation
		estim = self.estimated_curvature(data)

		MSEcurv = np.sum((data_curv - estim)**2) / len(data)
		RMSEcurv = np.sqrt(MSEcurv)

		return RMSEcurv



	def first_intersection(self, spl, t0=0.0, t1=1.0):

		""" Returns the coordinates and time of the furthest intersection point

		Keyword arguments: 
		spl -- Spline object
		t0, t1 -- Times in between the search occurs
		"""

		#t1 = self.length_to_time(self.radius(0) * 10)
		tg0 = self.tangent(t0)

		v = cross(tg0, np.array([1,0,0]))
		v = v / norm(v)
		
		# Search angles
		n_angles = 60
		angles = np.linspace(0,2*pi, n_angles)
		res = np.zeros((n_angles, 5))

		
		for i in range(n_angles):

			vrot = rotate_vector(v, tg0, angles[i])
			ap, times = self.intersection(spl, vrot, t0, t1)

			res[i, :2] = times
			res[i, 2:] = ap

		# Analyze the results
		measure = res[:, 0] / max(res[:,0]) + res[:, 1] / max(res[:,1])
		ind = np.argmax(measure)

		return res[ind, 2:], res[ind, :2]


	def first_intersectionv2(self, spl):
		""" Returns the coordinates and time of the furthest intersection point

		Keyword arguments: 
		spl -- Spline object
		"""
		inter_pt1, inter_t1 = self.first_intersection_centerlinev2(spl)
		inter_pt2, inter_t2 = spl. first_intersection_centerlinev2(self)
	
		if inter_t1 > 0.0 and inter_t2 > 0.0:

			# Apex direction 
			v = inter_pt2 - inter_pt1
			v = v / norm(v)
				
			# Compute apex value
			AP, tAP = self.intersection(spl, v, inter_t1, 1.0)
				
	
		else:
			AP = spl.point(0)
			tAP = [0, 0]


		return AP, tAP


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


		tinit = t0
		'''
		# Check search interval
		if is_inside(t1, v0, spl):
			print("t1 is not set correctly.")

		v = self.transport_vector(v0, tinit, t0)
		if not is_inside(t0, v, spl):
			pass
			#print("t0 is not set correctly.")
			'''

		while abs(t1 - t0) > 10**(-6):

			t = (t1 + t0) / 2.
			v = self.transport_vector(v0, tinit, t)
			pt = self.project_time_to_surface(v, t) 
			
			t2 = spl.project_point_to_centerline(pt)
			pt2 = spl.point(t2, True)

			if norm(pt - pt2[:-1]) <= pt2[-1]: #norm(pt - pt2[:-1]) <= pt2[-1]:
				t0 = t
			else: 
				t1 = t

		'''

		t = (t0 + t1) / 2.

		v = self.transport_vector(v0, tinit, t)
		pt = self.project_time_to_surface(v, t) 
		t2 = spl.project_point_to_centerline(pt)	
		'''

		return pt, [t, t2]


	def first_intersection_centerlinev2(self, spl):

		""" Returns the intersection point and time between a the spline and a spline model

		Keywords arguments: 
		spl -- Spline object
		"""

		#Plot distance as a function of time
		times, dist = spl.distance(self.get_points())
		radius = spl.radius(times)
		sign = np.sign(radius - dist)
		idx = np.argwhere(np.diff(sign)).flatten()

		# Get point and time where distance and radius intersects
		if len(idx)> 0:
			inter_pt = self.get_points()[idx[-1], :-1] 
			inter_t = self.project_point_to_centerline(inter_pt)
		else:
			if sign[-1] > 0:
				# Splines are included in each other
				inter_pt = self.point(1.0)
				inter_t = 1.0

			else:
				# Splines are totally separated 
				inter_pt = self.point(0.0)
				inter_t = 0.0

		return inter_pt, inter_t



	def first_intersection_centerline(self, spl, t0=0, t1=1):

		""" Returns the intersection point and time between a the spline and a spline model

		Keywords arguments: 
		spl -- Spline object
		v0 -- reference vector for the search
		t0, t1 -- Times in between the search occurs
		"""

		tinit = t1
		while abs(t1 - t0) > 10**(-6):

			t = (t1 + t0) / 2.
			pt = self.point(t)
			
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


	def show(self, knot = False, control_points = True, data = []):

		""" Displays the spline in 3D viewer.

		Keywords arguments:
		knot -- True to display the knots position
		control_points -- True to display control polygon
		"""
		points = self.get_points()

		if points.shape[1] <3:

			plt.plot(points[:,0], points[:,1],  c='black')

			if knot:
				knots = self.get_knot()
				for k in knots:
					pt = self.point(k)
					plt.scatter(pt[0], pt[1],  c='black', s = 20)

			if control_points:
				points = self.get_control_points()
				plt.plot(points[:,0], points[:,1],  c='grey')
				plt.scatter(points[:,0], points[:,1],  c='grey', s = 40)

			if len(data) != 0:
				data = np.array(data)
				plt.scatter(data[:,0], data[:,1],  c='red', s = 40)

			#plt.show()

		else:
			# 3D plot
			with plt.style.context(('ggplot')):
			
				fig = plt.figure(figsize=(10,7))
				ax = Axes3D(fig)
				ax.set_facecolor('white')

				
				ax.plot(points[:,0], points[:,1], points[:,2],  c='black')

				if knot:
					knots = self.get_knot()
					for k in knots:
						pt = self.point(k)
						ax.scatter(pt[0], pt[1], pt[2],  c='black', s = 20, depthshade=False)

				if control_points:
					points = self.get_control_points()
					ax.plot(points[:,0], points[:,1], points[:,2],  c='grey')
					ax.scatter(points[:,0], points[:,1], points[:,2],  c='grey', s = 40, depthshade=False)

				if len(data) != 0:
					data = np.array(data)
					ax.scatter(data[:,0], data[:,1], data[:,2],  c='red', s = 40, depthshade=False)
					#ax.plot(data[:,0], data[:,1], data[:,2],  c='red')

			# Set the initial view
			ax.view_init(90, -90) # 0 is the initial angle

			# Hide the axes
			ax.set_axis_off()
			plt.show()


