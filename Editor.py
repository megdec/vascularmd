import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D display
import pickle
import pyvista as pv
from scipy.spatial import ckdtree
from vpython import *
import os
from numpy.linalg import norm

from Nfurcation import Nfurcation
from ArterialTree import ArterialTree
from Spline import Spline
from Model import Model
from utils import *

class Editor:

	""" Class to edit vascular networks with a vpython GUI """

	def __init__(self, tree):

		"""
		Keyword argument:
		tree -- Arterial Tree object 
		"""

		self.tree = tree
		self.barycenter = self.barycenter_finder()

		# Scene set up
		scene.caption = "\nZoom in with a mouse or by pinching the screen and rotate by right clicking and moving the mouse\n"

		scene.background = color.white
		scene.width = 2000
		scene.height = 1000
		#scene.range = 75
		scene.center = vector(self.barycenter[0], self.barycenter[1], self.barycenter[2])

		slider_length = 300
		slider_width = 5
		slider_right_margin = 90
		self.scene = scene

		self.elements = {'full' : {}, 'topo' : {}, 'model' : {}, 'mesh' : {}} 

		self.full_nodes = None
		self.full_edges = None

		self.topo_nodes = None
		self.topo_edges = None

		self.model_nodes = None
		self.model_edges = None

		self.mesh = None
		self.mesh_edges = None

		self.visibility_state = {'full' : True, 'topo' : False, 'model' : False, 'mesh' : False}

		scene.append_to_caption('\n\n')

		# Check boxes
		self.checkboxes = {'full' : checkbox(text= "Full Graph\t\t\t\t\t\t\t\t", bind=self.update_visibility_state, checked=True)}
		self.checkboxes['topo'] = checkbox(text= "Topo Graph\t\t\t\t\t\t\t\t", bind=self.update_visibility_state, checked = False)
		self.checkboxes['model'] = checkbox(text= "Model Graph\t\t\t\t\t\t\t\t", bind=self.update_visibility_state)
		self.checkboxes['mesh'] = checkbox(text= "Mesh", bind=self.update_visibility_state)

		scene.append_to_caption('\n')
		scene.append_to_caption("\nOpacity\t\t\t\t\t\t\t\t\t" + "Opacity\n")

		# Transparency slides
		self.opacity_sliders  = {'full' : slider(bind = self.update_opacity_state, value = 1, length = slider_length, width = slider_width, right = slider_right_margin, disabled = True)}
		self.opacity_sliders['topo'] = slider(bind = self.update_opacity_state, value = 1, length = slider_length, width = slider_width, right = slider_right_margin, disabled = True)

		self.opacity_value = {'full' : 1, 'topo' : 1, 'model' : 1, 'mesh' : 1}

		scene.append_to_caption('\nEdge radius\t\t\t\t\t\t\t\tEdge radius\t\t\t\t\t\t\t\tEdge radius\t\t\t\t\t\t\t\tEdge radius\n')

		# Size sliders
		
		self.edge_size_sliders = {'full' :  slider(bind = self.update_edge_size, value = 0.2, min=0, max = 0.5, length=slider_length, width = slider_width, right = slider_right_margin, disabled = True)}
		self.edge_size_sliders['topo'] = slider(bind = self.update_edge_size, value = 0.2, min=0, max = 0.5, length=slider_length, width = slider_width, right = slider_right_margin, disabled = True)
		self.edge_size_sliders['model'] = slider(bind = self.update_edge_size, value = 0.2, min=0, max = 0.5, length=slider_length, width = slider_width, right = slider_right_margin, disabled = True)
		self.edge_size_sliders['mesh'] = slider(bind = self.update_edge_size, value = 0.05, min=0, max = 0.2, length=slider_length, width = slider_width, right = slider_right_margin, disabled = True)

		self.edge_size = {'full' : 0.2, 'topo' : 0.2, 'model': 0.2, 'mesh' : 0.05}
		
		scene.append_to_caption('\n\t\t\t\t\t\t\t\t\t\tNode radius\t\t\t\t\t\t\t\tNode radius\n')

		self.node_size_sliders  = {'topo' :  slider(bind = self.update_node_size, value = 0.5, min=0, max = 1, length=slider_length, width = slider_width, left= slider_length + slider_right_margin -10, right = slider_right_margin -10, disabled = True)}
		self.node_size_sliders['model'] = slider(bind = self.update_node_size, value = 0.5, min=0, max = 1, length=slider_length, width = slider_width, right = slider_right_margin, disabled = True)

		self.node_size = {'topo' : 0.5, 'model' : 0.5}

		# Generate full graph

		self.create_elements('full')
		


	def update_visibility_state(self):

		for mode in self.checkboxes.keys():
			if self.checkboxes[mode].checked != self.visibility_state[mode]:
				if self.checkboxes[mode].checked:
					self.show(mode)
				else:
					self.hide(mode)


	def update_edge_size(self):

		def set_edge_size(elt, args):
			if type(elt) == curve:
				for n in range(elt.npoints):
					elt.modify(n, radius=args[0])
			else:
				elt.radius = args[0]

		for mode in self.edge_size_sliders.keys():
			if self.edge_size_sliders[mode].value != self.edge_size[mode]:
	
				if mode == "mesh":
					categories = ['section_edges', 'connecting_edges']
				else:
					categories = ['edges']

				self.apply_function(mode, func=set_edge_size, args=[self.edge_size_sliders[mode].value], categories = categories)
				self.edge_size[mode] = self.edge_size_sliders[mode].value



	def update_node_size(self):

		def set_node_size(elt, args):
			elt.radius = args[0]

		for mode in self.node_size_sliders.keys():
			if self.node_size_sliders[mode].value != self.node_size[mode]:

				categories = ['nodes']

				self.apply_function(mode, func=set_node_size, args=[self.node_size_sliders[mode].value], categories = categories)
				self.node_size[mode] = self.node_size_sliders[mode].value




	def barycenter_finder(self):

		coords = list(nx.get_node_attributes(self.tree.get_full_graph(), 'coords').values())
		barycenter = sum(coords) / len(coords)
		return barycenter[:3]




	def apply_function(self, mode, func, args, categories=[]):


		self.disable_all(True)
		if len(categories) == 0:
			categories = list(self.elements[mode].keys())

		for category in categories:
			obj_cat = self.elements[mode][category]
	

			if type(obj_cat) == dict:
				for elt in obj_cat.values():
					if type(elt) == list:
						for e in elt:
							func(e, args)
					else:
						func(elt, args)

			else:
				for elt in obj_cat:
					func(elt, args)

		self.disable_all(False)


	def update_opacity_state(self):

		def set_opacity(elt, args):
			elt.opacity = args[0]

		for mode in self.opacity_sliders.keys():
			if self.opacity_sliders[mode].value != self.opacity_value[mode]:
				self.apply_function(mode, func=set_opacity, args=[self.opacity_sliders[mode].value])
				self.opacity_value[mode] = self.opacity_sliders[mode].value

	def show(self, mode):

		def set_visible(elt, args):
			elt.visible = args[0]

		if len(self.elements[mode]) == 0:
			# Creation of the mode objects
			self.create_elements(mode)
		else:
			self.apply_function(mode, func = set_visible, args=[True])

		self.visibility_state[mode] = True



	def hide(self, mode):

		def set_visible(elt, args):
			elt.visible = args[0]

		self.apply_function(mode, func = set_visible, args=[False])
		self.visibility_state[mode] = False
		

	def create_elements(self, mode):

		if mode == 'full':

			# Import full graph
			G = self.tree.get_full_graph()

			# plot data points by iterating over the nodes
			nodes = {}
			for n in G.nodes():
				pos = vector((G.nodes[n]['coords'][0]), (G.nodes[n]['coords'][1]), (G.nodes[n]['coords'][2]))
				radius = G.nodes[n]['coords'][3]
				ball = sphere(pos=pos, color=color.red, radius=radius)
				nodes[n] = ball
		
			# plot edges
			edges = {}
			for e in G.edges():
				pos = vector((G.nodes[e[0]]['coords'][0]), (G.nodes[e[0]]['coords'][1]), (G.nodes[e[0]]['coords'][2]))
				axis = G.nodes[e[1]]['coords'][:-1] - G.nodes[e[0]]['coords'][:-1]
				length = norm(axis)
				direction = axis / length
				c = cylinder(pos=pos, axis=vector(direction[0], direction[1], direction[2]), length=length, radius=0.2, color=color.black)
				edges[e] = c

			self.elements['full']['nodes'] = nodes
			self.elements['full']['edges'] = edges


		elif mode == 'topo':

			G = self.tree.get_topo_graph()
			
			nodes = {}
			col = {'end': color.blue, 'bif' : color.red, 'reg' : color.green}

			for n in G.nodes():
				pos = vector((G.nodes[n]['coords'][0]), (G.nodes[n]['coords'][1]), (G.nodes[n]['coords'][2]))
				pt_type = G.nodes[n]['type']
				ball = sphere(pos=pos, color=col[pt_type], radius=0.5)
				nodes[n] = ball

			edges = {}
			for e in G.edges():
				coords = np.vstack((G.nodes[e[0]]['coords'], G.edges[e]['coords'], G.nodes[e[1]]['coords']))

				c_list = []
				for i in range(len(coords) - 1):
					pos = vector((coords[i][0]), (coords[i][1]), (coords[i][2]))
					axis = coords[i+1][:-1] - coords[i][:-1]
					length = norm(axis)
					direction = axis / length
					c_list.append(cylinder(pos=pos, axis=vector(direction[0], direction[1], direction[2]), length=length, radius=0.2, color=color.black))

					edges[e] = c_list


				'''
				pos = vector((G.nodes[e[0]]['coords'][0]), (G.nodes[e[0]]['coords'][1]), (G.nodes[e[0]]['coords'][2]))
				axis = G.nodes[e[1]]['coords'][:-1] - G.nodes[e[0]]['coords'][:-1]
				length = norm(axis)
				direction = axis / length
				c = cylinder(pos=pos, axis=vector(direction[0], direction[1], direction[2]), length=length, radius=0.2, color=color.black)
				edges[e] = c
				'''

			self.elements['topo']['nodes'] = nodes
			self.elements['topo']['edges'] = edges

		elif mode == 'model':

			G = self.tree.get_model_graph()
			if G is None:
				self.disable_checkboxes(True)
				self.tree.model_network()
				G = self.tree.get_model_graph()
				self.disable_checkboxes(False)

			# plot data points by iterating over the nodes
			nodes = {}
			col = {'end': color.blue, 'bif' : color.red, 'reg' : color.green, 'sep': color.purple}

			for n in G.nodes():
				pos = vector((G.nodes[n]['coords'][0]), (G.nodes[n]['coords'][1]), (G.nodes[n]['coords'][2]))
				pt_type = G.nodes[n]['type']

				if pt_type != "sep":
					ball = sphere(pos=pos, color=col[pt_type], radius=0.5)
					nodes[n] = ball

			edges = {}
			for e in G.edges():
				spl = G.edges[e]['spline']
				coords = spl.get_points()
				pos = vector(coords[0][0], coords[0][1], coords[0][2])
				c = curve(pos=pos, color = color.black, radius = 0.2)
				for i in range(1,len(coords)):
					c.append(vector(coords[i][0], coords[i][1], coords[i][2]))
				edges[e] = c

			self.elements['model']['nodes'] = nodes
			self.elements['model']['edges'] = edges

		elif mode == 'mesh':

			mesh = self.tree.get_surface_mesh()
			if mesh is None:
				self.disable_checkboxes(True)
				mesh = self.tree.mesh_surface()
				self.disable_checkboxes(False)

			vertices = mesh.points
			faces = mesh.faces.reshape((-1, 5))
			faces.astype('int32')

			v_obj = []
			for v in vertices:
				v_obj.append(vertex(pos=vector(v[0], v[1], v[2]), color=color.gray(0.75)))

			
			quads = []
			for f in faces:
				q = quad(v0=v_obj[f[1]], v1=v_obj[f[2]], v2=v_obj[f[3]], v3=v_obj[f[4]])
				quads.append(q)

			curves = {}
			lines = []
			G = self.tree.get_crsec_graph()
			for n in G.nodes():
				crsec = G.nodes[n]['crsec']
				if G.nodes[n]['type'] == "bif":
					N = (self.tree._N // 2) - 1
					curve_list = []
					s = 2
					for i in range(len(crsec) // N):
						c = curve(pos = vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]), radius=0.05, color = color.black)
						for k in range(N):
							c.append(vector(crsec[s+k, 0], crsec[s+k, 1], crsec[s+k, 2]))
						s += k + 1
						c.append(vector(crsec[1, 0], crsec[1, 1], crsec[1, 2]))
						curve_list.append(c)
					curves[n] = curve_list


				else:
					c = curve(radius=0.05, color = color.black)
					for i in range(len(crsec)):
						c.append(vector(crsec[i, 0], crsec[i, 1], crsec[i, 2]))
					c.append(vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
					curves[n] = [c]

			for e in G.edges():
				crsec_list = G.edges[e]['crsec']
				# Longitudinal curves
				for i in range(crsec_list.shape[1]):
					c = curve(radius=0.05, color = color.black)
					# Starting point
					start_sec = G.nodes[e[0]]['crsec']
					c.append(vector(start_sec[i, 0], start_sec[i, 1], start_sec[i, 2]))
					for j in range(crsec_list.shape[0]):
						c.append(vector(crsec_list[j, i, 0], crsec_list[j, i, 1], crsec_list[j, i, 2]))
					# Ending point
					end_sec = G.nodes[e[1]]['crsec']
					l = G.edges[e]['connect'][i]
					c.append(vector(end_sec[l][0], end_sec[l][1], end_sec[l][2]))

					lines.append(c)

				# Cross section curves
				curve_list = []
				for i in range(len(crsec_list)):
					crsec = crsec_list[i]
					c = curve(radius=0.05, color = color.black)
					for j in range(len(crsec)):
						c.append(vector(crsec[j, 0], crsec[j, 1], crsec[j, 2]))
					c.append(vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
					curve_list.append(c)
				curves[e] = curve_list


			self.elements['mesh']['surface'] = quads
			self.elements['mesh']['section_edges'] = curves
			self.elements['mesh']['connecting_edges'] = lines

		else:
			print("Unknown mode.")

		if mode in self.edge_size_sliders.keys():
			self.edge_size_sliders[mode].disabled = False
		if mode in self.node_size_sliders.keys():
			self.node_size_sliders[mode].disabled = False
		if mode in self.opacity_sliders.keys():
			self.opacity_sliders[mode].disabled = False



	def disable_all(self, disabled=True):
		self.disable_checkboxes(disabled)
		self.disable_opacity_sliders(disabled)

	def disable_checkboxes(self, disabled=True):

		for c in self.checkboxes.values():
			c.disabled = disabled

	def disable_opacity_sliders(self, disabled=True):

		for c in self.opacity_sliders.values():
			c.disabled = disabled

