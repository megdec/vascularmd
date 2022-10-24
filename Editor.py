
####################################################################################################
# Author: Meghane Decroocq
#
# This file is part of vascularmd project (https://github.com/megdec/vascularmd)
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3 of the License.
#
####################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imt

import pickle
import pyvista as pv

from vpython import *
from numpy.linalg import norm
import copy
import nibabel as nib
import os

from Nfurcation import Nfurcation
from ArterialTree import ArterialTree
from Spline import Spline
from Model import Model
from utils import *



class Editor:

	""" Class to edit vascular networks with a vpython GUI """
	def __init__(self, tree, width=1200, height=600):

		"""
		Keyword argument:
		tree -- Arterial Tree object 
		"""

		self.tree = tree
		self.barycenter = self.barycenter_finder()

		# Scene set up
		scene.caption = ""
		self.text_output = wtext(text="")
		self.output_message("Zoom in using mouse middle button, rotate by right clicking and moving the mouse and translate by pressing the shift key.")

		scene.background = color.white
		scene.width = width
		scene.height = height
		#scene.range = 75
		scene.center = vector(self.barycenter[0], self.barycenter[1], self.barycenter[2])

		slider_length = 300
		slider_width = 5
		slider_right_margin = 50

		self.scene = scene

		self.elements = {'full' : {}, 'topo' : {}, 'model' : {}, 'mesh' : {}, 'pathology' : {}} 

		scene.append_to_caption('\n\nEdit  ')
		self.edition_menu = menu(choices = ['off', 'data', 'topo', 'model', 'crop', 'mesh', 'pathology'], selected = 'off', index=0, bind = self.update_edition_mode)
		self.edition_mode = 'off'
		self.mesh_selection = []
		self.crop_selection = []

		scene.append_to_caption('\nImport centerline  ')

		self.centerline_file_winput = winput(text="", bind = self.reset_scene, width=200)

		scene.append_to_caption('\nImport image  ')
		self.slice_button = button(text= "Cut slice  " , bind=self.compute_slice, disabled = True)
		self.cursor_checkbox = checkbox(text= "Show origin  " , bind=self.update_visibility_cursor, checked = False)
		self.slice_checkbox = checkbox(text= "Show slice" , bind=self.update_visibility_slice, checked = False)
		self.slice_checkbox.disabled = True
		self.cursor = sphere(pos=scene.center, color=color.yellow, radius=2, mode = "cursor", visible = False, locked = False)
		self.slice = None
		self.slice_plane = [None, None]

		scene.append_to_caption('\tPath ')
		self.image_file_winput = winput(text="", bind = self.load_image, width=200)
		scene.append_to_caption('\tOpacity ')
		self.slice_opacity_slider = slider(bind = self.slice_opacity, value = 1, length = 150, width = slider_width)


		scene.append_to_caption('\nExport  ')

		self.save_button = button(text = "Save", bind=self.save)
		self.save_menu = menu(choices = ['centerline', 'model', 'surface mesh', 'volume mesh'], selected = 'centerline', index=0, bind = self.do_nothing)
		self.save_directory = ""
		scene.append_to_caption('\tOutput directory ')
		self.save_winput = winput(text="", bind = self.update_save_directory, width=200)
		self.save_filename = "vascular_network"
		scene.append_to_caption('\tOutput filename ')
		self.save_filename_winput = winput(text="vascular_network", bind = self.update_save_filename, width=200)


		scene.append_to_caption('\n\n')

		# Check boxes
		self.checkboxes = {'full' : checkbox(text= "Data  ", bind=self.update_visibility_state, checked=True, mode = "full")}
		self.update_buttons = {'full' : button(text = "Apply", bind=self.update_graph, mode = 'full')}
		scene.append_to_caption('\t\t\t\t\t\t\t')
		self.checkboxes['topo'] = checkbox(text= "Topology  ", bind=self.update_visibility_state, checked = False, mode = "topo")
		self.update_buttons['topo'] = button(text = "Apply", bind=self.update_graph, mode = 'topo')
		self.crop_button = button(text = " Crop ", bind=self.crop_network, mode = 'topo', activated = False)
		scene.append_to_caption('\t\t\t\t\t')
		self.checkboxes['model'] = checkbox(text= "Model  ", bind=self.update_visibility_state, mode = "model", checked = False)
		self.update_buttons['model'] = button(text = "Apply", bind=self.update_graph, mode = 'model')
		self.extension_button = button(text="Extend", bind=self.manage_extensions)
		self.extension_state = False
		scene.append_to_caption('\t\t\t\t\t')
		self.checkboxes['mesh'] = checkbox(text= "Mesh  " , bind=self.update_visibility_state, mode="mesh", checked = False)

		self.surface_button = button(text="Surface", bind=self.mesh_surface)
		self.deform_mesh_button = button(text="Deform", bind=self.deform_mesh)

		self.check_mesh_button = button(text="Check", bind=self.check_mesh)
		self.check_state = False
		self.close_mesh_button = button(text="Close", bind=self.close_mesh)
		self.closing_state = False
		self.volume_button = button(text="Volume", bind=self.mesh_volume)
		

		scene.append_to_caption("\n\nOpacity\t\t\t\t\t\t\t\t\t\t")

		self.angle_checkbox_topo = checkbox(text="Show angles", bind=self.update_visibility_angle, checked=False, mode = "topo")
		scene.append_to_caption("\t\t\t\t\t\t\t\t")

		# Display bifurcations and control points
		self.angle_checkbox_model = checkbox(text="Show angles", bind=self.update_visibility_angle, checked=False, mode = "model")
		scene.append_to_caption("\t")
		self.furcation_checkbox = checkbox(text="Show furcations", bind=self.update_visibility_furcations, checked=False)

		scene.append_to_caption('\t\t\tDisplay ')

		self.mesh_representation_menu = menu(choices = ['default', 'wireframe', 'sections', 'solid'], selected = 'default', index=0, bind = self.update_mesh_representation)

		scene.append_to_caption('\n')

		# Transparency slides
		self.opacity_sliders  = {'full' : slider(bind = self.update_opacity_state, value = 1, length = slider_length, width = slider_width, right = slider_right_margin)}
		#self.opacity_sliders['topo'] = slider(bind = self.update_opacity_state, value = 1, length = slider_length, width = slider_width, right = slider_right_margin - 3)
		scene.append_to_caption('\t\t\t\t\t\t\t\t\t\t\t')
		self.opacity_value = {'full' : 1}

		self.control_pts_checkbox = checkbox(text="Show ctrl pts", bind=self.update_visibility_control_pts, checked=False)
		scene.append_to_caption("\t")
		self.control_radius_checkbox = checkbox(text="Show ctrl radius", bind=self.update_visibility_control_radius, checked=False)

		# Size sliders
		scene.append_to_caption('\nEdge radius\t\t\t\t\t\t\t\t\tEdge radius\t\t\t\t\t\t\t\t\tEdge radius\t\t\t\t\t\t\t\t\tEdge radius\n')
		
		self.edge_size_sliders = {'full' :  slider(bind = self.update_edge_size, value = 0.2, min=0, max = 0.5, length=slider_length, width = slider_width, right = slider_right_margin, mode = "full")}
		self.edge_size_sliders['topo'] = slider(bind = self.update_edge_size, value = 0.2, min=0, max = 0.5, length=slider_length, width = slider_width, right = slider_right_margin, mode  = "topo")
		self.edge_size_sliders['model'] = slider(bind = self.update_edge_size, value = 0.2, min=0, max = 0.5, length=slider_length, width = slider_width, right = slider_right_margin, mode  = "model")
		self.edge_size_sliders['mesh'] = slider(bind = self.update_edge_size, value = 0.05, min=0, max = 0.2, length=slider_length, width = slider_width, right = slider_right_margin, mode  = "mesh")

		self.edge_size = {'full' : 0.2, 'topo' : 0.2, 'model': 0.2, 'mesh' : 0.05}
		
		scene.append_to_caption('\nResample\t\t\t\t\t\t\t\t\tNode radius\t\t\t\t\t\t\t\t\tNode radius\n')
		self.node_size_sliders = {'full' : slider(bind = self.resample_nodes, value = 1, min=0, max = 1, length=slider_length, width = slider_width, left= 10, right = slider_right_margin -10, mode  = "full")}
		self.node_size_sliders['topo'] = slider(bind = self.update_node_size, value = 0.5, min=0, max = 1, length=slider_length, width = slider_width, left= 10, right = slider_right_margin -10, mode  = "topo")
		self.node_size_sliders['model'] = slider(bind = self.update_node_size, value = 0.5, min=0, max = 1, length=slider_length, width = slider_width, right = slider_right_margin, mode  = "model")


		self.node_size = {'topo' : 0.5, 'model' : 0.5}


		scene.append_to_caption('Nb nodes (nx8) ')
		self.parameters_winput = {'N' : winput(text=str(24), bind = self.update_mesh_parameters, width=50, parameter = 'N')}
		scene.append_to_caption('\tSection density [0,1] ')
		self.parameters_winput['d'] = winput(text=str(0.2), bind = self.update_mesh_parameters, width=50, parameter = 'd')
		scene.append_to_caption('\n')
		
		scene.append_to_caption('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t') 
		self.smooth_checkboxes = {'spatial' : checkbox(text= "Smooth spatial  ", bind = self.select_smooth_parameter, checked = False, parameter = 'spatial')}
		self.smooth_checkboxes['radius']  = checkbox(text= "Smooth radius", bind = self.select_smooth_parameter, checked = False, parameter = 'radius')

		scene.append_to_caption('\t\t\t Target mesh path ')
		self.parameters_winput['path'] = winput(text="", bind = self.update_mesh_parameters, width=200, parameter = 'path')
		self.target_mesh = None

		scene.append_to_caption('\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t') 
		
		self.lbds = 0
		scene.append_to_caption('Lbd spatial : ')
		self.lbds_text = wtext(text="")
		self.lbdr = 0
		scene.append_to_caption('\t\tLbd radius : ')
		self.lbdr_text = wtext(text="")
		scene.append_to_caption('\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t')  


		# Widget to add pathologies to the network
		self.pathology_checkbox =  checkbox(text= "Add pathology", bind = self.locate_pathology, checked = False)
		self.pathology_markers = [sphere(pos=scene.center, color=color.yellow, radius=0.3, mode = "marker", visible = False, locked = False), sphere(pos=scene.center, color=color.yellow, radius=0.3, mode = "marker", visible = False, locked = False)]
		self.pathology_edg = None
		scene.append_to_caption('  ') 
		self.pathology_directory_winput = winput(text="pathology_templates/default/", bind = self.load_pathology_template, width=200)

		self.template = None
		self.load_pathology_template() # Load the default pathology

		self.pathology = []
		self.pathology_output_dir = "pathology_templates/new_template/"


		# Meshing default parameter
		self.N = 24 # Number of nodes in one cross section (nx8)
		self.d = 0.2 # Density of cross sections

		self.mesh_display_limit = 40000

		# Display parameters
		self.display_spline_step = 10 # Step for displaying spline points
		self.temp_num_nds = 50 # Number of nodes in the template for pathology
		self.temp_num_crsec = 10 # Number of crsec in the template for pathology
		self.temp_rad = 10 # Radius of the template cross sections
		self.temp_center = scene.center # position of the center 

		# Link actions to functions
		scene.bind('click', self.select)
		scene.bind('mousemove', self.move)
		scene.bind('mouseup', self.drop)
		scene.bind('keydown', self.keyboard_control)

		# Storing the selected edges and nodes
		self.selected_node = None
		self.selected_edge = None

		# Draging and running state attributes
		self.drag = False
		self.running = False
		self.edge_drag = False
		self.edge_n_id = None

		# Storing of the elements to display 
		self.trash_elements = {'curves' : [], 'quads' : []}
		self.modified_elements = {'splines' : []}
		self.modified = {"off" : False, "full" : False, "topo" : False, "model" : False, "mesh" : False, "pathology" : False, "crop" : False}

		# Disable elements
		self.disable(True, checkboxes = False)
		# Generate full graph
		self.create_elements('full')
		self.disable(False, ["full"])

		self.create_template_pathology()
		

	###########################################
	################ VISIBILITY ###############
	###########################################

	def reset_scene(self):

		# Import tree
		file = self.centerline_file_winput.text 
		if file[-4:] == ".swc" or file[-4:] == ".vtp" or file[-4:] == ".txt":
			tree = ArterialTree("Unknown", "Unknown", file)

		elif file[-4:] == ".obj":
			f = open(file, 'rb') 
			tree = pickle.load(f) 
		else:
			tree = None
			self.output_message("The input centerline file does not exist or have a wrong extension. The extension supported are .swc, .vtp, .txt, .obj.", "error")

		if tree is not None:

			self.tree = tree

			# Erase objects
			self.hide("full")
			self.hide("topo")
			self.hide("model")
			self.hide("mesh")

			# Reset buttons
			self.tree = tree
			self.barycenter = self.barycenter_finder()
			scene.center = vector(self.barycenter[0], self.barycenter[1], self.barycenter[2])

			self.elements = {'full' : {}, 'topo' : {}, 'model' : {}, 'mesh' : {}} 
			self.edition_menu.selected = 'off'
			self.edition_mode = 'off'
			self.checkboxes['full'].checked = True
			self.checkboxes['topo'].checked = False
			self.checkboxes['model'].checked = False
			self.checkboxes['mesh'].checked = False

			self.node_size_sliders['topo'].value = 0.5
			self.node_size_sliders['model'].value = 0.5
			self.node_size = {'topo' : 0.5, 'model' : 0.5}

			self.disable(True, checkboxes = False)
			# Generate full graph
			self.create_elements('full')
			self.disable(False, ["full"])
			self.output_message("Input centerline successfully imported from " + file + ".")



	def do_nothing(self):
		pass

	def node_color(self, n, mode):
		""" Returns the node color, depending on the mode"""

		if mode == 'full' or mode=="data":
			res = color.red
		else:
			if mode =='topo':
				G = self.tree.get_topo_graph()
			else:
				G = self.tree.get_model_graph()

			typ = G.nodes[n]['type']
			if typ == 'end' and G.in_degree(n) == 0:
				typ = 'inlet'

			col = {'end' : color.blue, 'inlet' : color.orange, 'bif' : color.red, 'reg' : color.green, 'sep': color.purple, 'sink' : color.green}
			res = col[typ]

		return res



	def barycenter_finder(self, edg = []):

		""" Finds the scene barycenter by averaging the network nodes """
		if len(edg) == 0:
			coords = list(nx.get_node_attributes(self.tree.get_full_graph(), 'coords').values())
			barycenter = sum(coords) / len(coords)
			return barycenter[:3]
		else:
			topo = self.tree.get_topo_graph()
			sm = np.array([0,0,0,0])
			num = 0
			for e in edg:
				sm = sm + np.sum(topo.edges[e]['coords'], axis=0)
				num += len(topo.edges[e]['coords'])
			return sm[:3]/num
		
	def crop_network(self, b):

		""" Crop network to the selected edges TMP VERSION """

		if not b.activated: 
			# Lock everything
			if len(self.crop_selection) == 0:
				self.output_message("Please select one or more edges from the crop menu first.", "warning")
			else:

				self.lock("full")
				#self.lock("topo")

				# Unlock only the selected edges, if the representation checkbox is checked
				for e in self.crop_selection:
					topo_edg = [t for t in self.tree.get_topo_graph().edges()]
					if e not in topo_edg:
						print("This edge does not exist.")
					#self.elements['topo']['nodes'][e[0]].locked = False
					#self.elements['topo']['nodes'][e[1]].locked = False
					#self.elements['topo']['edges'][e].locked = False

					# Get full id 
					ids = self.tree.get_topo_graph().edges[e]['full_id']
					id0 = self.tree.get_topo_graph().nodes[e[0]]['full_id']
					id1 = self.tree.get_topo_graph().nodes[e[1]]['full_id']

					self.elements['full']['nodes'][id0].locked = False
					self.elements['full']['edges'][(id0, ids[0])].locked = False
					for i in range(len(ids)-1):
						self.elements['full']['nodes'][ids[i]].locked = False
						self.elements['full']['edges'][(ids[i], ids[i+1])].locked = False

					self.elements['full']['nodes'][ids[len(ids)-1]].locked = False
					self.elements['full']['edges'][(ids[len(ids)-1], id1)].locked = False
					self.elements['full']['nodes'][id1].locked = False

				# Hide and show all
				self.hide("full")
				#self.hide("topo")

				if self.checkboxes["full"].checked:
					self.show("full")

				#if self.checkboxes["topo"].checked:
					#self.show("topo")

				# Recompute barycenter
				barycenter = self.barycenter_finder(edg = self.crop_selection)
				scene.center = vector(barycenter[0], barycenter[1], barycenter[2])
				self.cursor.pos=scene.center
				b.activated = True
				b.text = "Uncrop"

		else:
			self.unlock("full")
			#self.unlock("topo")

			if self.checkboxes["full"].checked:
				self.show("full")

			#if self.checkboxes["topo"].checked:
			#	self.show("topo")

			scene.center = vector(self.barycenter[0], self.barycenter[1], self.barycenter[2])
			self.cursor.pos=scene.center
			b.activated = False
			#self.crop_selection = []
			b.text = " Crop "


	def get_closest_data_point(self, coords, n):

		# Get the id of the closest data point AMONG THE VISIBLE NODES after projection on the plane of given normal
		pos = []
		ids = []
		for sph in self.elements["full"]["nodes"].values():
			if not sph.locked:
				pos.append([sph.pos.x, sph.pos.y, sph.pos.z])
				ids.append(sph.id)


		pos = np.array(pos)


		# Project all coords to the same plane
		tmp = np.array([0,0,1])
		u1 = cross(n, tmp)
		u2 = cross(u1, n)

		# u1 and u2 are a basis of the plane
		proj = np.zeros((pos.shape[0], 2))

		for i in range(pos.shape[0]):
			proj[i, 0] = dot(pos[i, :], u1) 
			proj[i, 1] = dot(pos[i, :], u2) 

		proj_pt = np.array([dot(coords, u1), dot(coords, u2)])


		kdtree = KDTree(proj)
		d, idx = kdtree.query(proj_pt)
		idx = ids[idx]

		return self.elements["full"]["nodes"][idx]


	def update_visibility_state(self, b):
		
		""" Show / hide network representation when the corresponding checkbox is checked / unchecked. """
		mode = b.mode
		if self.modified[mode] and not b.checked:
			# Do nothing as the changes must be validated 
			b.checked = True
			self.output_message("Please apply the modifications by clicking 'Apply' before leaving this representation mode.","error")

		else:
			if mode == 'model':
				categories = ['edges', 'nodes']
			elif mode == 'topo':
				categories = ['edges', 'nodes']
			else:
				categories = []

			if b.checked:
				self.show(mode, categories)

				if mode == "model":
					self.update_visibility_furcations()
					self.update_visibility_control_pts()
					self.update_visibility_control_radius()
					if self.angle_checkbox_model.checked:
						self.update_visibility_angle(self.angle_checkbox_model)
				if mode == "topo":
					if self.angle_checkbox_topo.checked:
						self.update_visibility_angle(self.angle_checkbox_topo)

				self.disable(False, [mode], checkboxes = False)
			else:
				self.hide(mode)
				self.disable(True, [mode], checkboxes = False)


	def show_mesh_selection(self, show):

		""" Show / hide the edges selected for meshing. """

		for e in self.mesh_selection:
			if show:
				self.elements['model']['edges'][e].color = color.red
			else:
				self.elements['model']['edges'][e].color = color.black


	def show_crop_selection(self, show):

		""" Show / hide the edges selected for meshing. """

		for e in self.crop_selection:
			if show:
				self.elements['topo']['edges'][e].color = color.red
			else:
				self.elements['topo']['edges'][e].color = color.black


	def update_visibility_cursor(self):
	
		self.cursor.visible = self.cursor_checkbox.checked

	def update_visibility_slice(self):

		self.slice.visible = self.slice_checkbox.checked

	def load_image(self):

		filename = self.image_file_winput.text
		try:
			if len(filename) == 0: 
				self.output_message("No image directory found. Please write the path in the text box and hit enter.", "warning")
			else:
				# Load image
				self.image  = nib.load(filename)
				self.output_message("Image loaded from " + filename + ".")
				self.slice_button.disabled = False

		except FileNotFoundError:
			self.output_message("The output directory does not exist.", "error")


	def compute_slice(self):

		""" Return a MRA image patch oriented normally to the artery tangent.

		Keyword arguments:
		img -- image volume as np array
		pix_dim -- dimension of img (mm)
		pt -- origin coordinates (mm)
		tg -- unit tangent vector
		dim -- patch dimension (vx)
		dist -- patch dimension (mm)
		"""
		outfolder = "img/"

		try:
			os.makedirs(outfolder)
		except OSError as error:
			pass

		def fill_patch(img, c):
			if c[0] > 0 and c[0]<img.shape[0] and c[1] > 0 and c[1] < img.shape[1] and c[2] > 0 and c[2] < img.shape[2]:
				return img[c[0], c[1], c[2]]
			else:
				return 0


		self.output_message("Cutting slice in the medical image.")
		self.disable(True, checkboxes = True)

		img = np.array(self.image.dataobj)
		pix_dim = self.image.header['pixdim'][1:4]
		dim = 20
		dist = 20

		if self.selected_node is None:
			center = self.cursor.pos
		else:
			center = self.selected_node.pos


		vec_norm = scene.camera.axis
		self.slice_plane[0] = vec_norm
		self.slice_plane[1] = center

		pt = np.array([center.x, center.y, center.z])
		tg = np.array([vec_norm.x, vec_norm.y, vec_norm.z])
		tg = tg/norm(tg)
		
		nr = cross(np.array([0, 0.1, 0.9]), tg) # Normal vector
		bnr = cross(tg, nr) # Binormal vector

		nr = nr / norm(nr) # Normalize
		bnr = bnr / norm(bnr)

		# Coord conversion 
		patch_pix = np.array([float(dist) / float(dim)]*2)
		step = np.linspace(0, dist, dim + 1)[1:]

		patch = np.zeros((dim * 2 + 1, dim * 2 + 1))

		ct = (pt / pix_dim).astype(int)
		patch[dim, dim] = img[ct[0], ct[1], ct[2]]

		# Fill patch
		for j in range(dim):
			# Fill cross
			c1 = ((pt - nr * step[::-1][j]) / pix_dim).astype(int)
			c2 = ((pt + nr * step[j]) / pix_dim).astype(int)

			c3 = ((pt - bnr * step[::-1][j]) / pix_dim).astype(int)
			c4 = ((pt + bnr * step[j]) / pix_dim).astype(int)

			patch[dim, j] = fill_patch(img, c1)
			patch[dim, dim + 1 + j] = fill_patch(img, c2)
			patch[j, dim] = fill_patch(img, c3)
			patch[dim + 1 + j, dim] = fill_patch(img, c4)


		for j in range(dim):

			c1mm = (pt - nr * step[::-1][j])
			c2mm = (pt + nr * step[j])
				
			c1 = (c1mm / pix_dim).astype(int)
			c2 = (c2mm / pix_dim).astype(int)
				

			patch[dim + 1, j] = fill_patch(img, c1)
			patch[dim + 1, dim + 1 + j] = fill_patch(img, c2)

			for k in range(dim):

				c3 = ((c1mm - bnr * step[::-1][k]) / pix_dim).astype(int)
				c4 = ((c1mm + bnr * step[k]) / pix_dim).astype(int)

				c5 = ((c2mm - bnr * step[::-1][k]) / pix_dim).astype(int)
				c6 = ((c2mm + bnr * step[k]) / pix_dim).astype(int)

				patch[k, j] = fill_patch(img, c3)
				patch[dim + 1 + k, j] = fill_patch(img, c4)

				patch[k, dim + 1 + j] = fill_patch(img, c5)
				patch[dim + 1 + k, dim + 1 + j] = fill_patch(img, c6)

		# Write the image

		pos1 = pt - bnr * dist - nr * dist
		pos2 = pt + bnr * dist - nr * dist
		pos3 = pt + bnr * dist + nr * dist
		pos4 = pt - bnr * dist + nr * dist

		if self.slice is None:

			imt.imsave(outfolder + "image.jpg", patch, cmap='gray')

			v1 = vertex(pos=vector(pos1[0],pos1[1],pos1[2]), normal=vector(0,0,1), texpos=vector(0,1,0), shininess= 0, opacity = self.slice_opacity_slider.value)
			v2 = vertex(pos=vector(pos2[0],pos2[1],pos2[2]), normal=vector(0,0,1), texpos=vector(0,0,0), shininess= 0, opacity = self.slice_opacity_slider.value)
			v3 = vertex(pos=vector(pos3[0],pos3[1],pos3[2]), normal=vector(0,0,1), texpos=vector(1,0,0), shininess= 0, opacity = self.slice_opacity_slider.value)
			v4 = vertex(pos=vector(pos4[0],pos4[1],pos4[2]), normal=vector(0,0,1), texpos=vector(1,1,0), shininess= 0, opacity = self.slice_opacity_slider.value)
				
			Q = quad(vs=[v1, v2, v3, v4], texture=outfolder + "image.jpg", locked = False)
			self.slice = Q
			self.nb_im = 0
	
		else:
			self.nb_im = self.nb_im + 1
			imt.imsave(outfolder + "image" + str(self.nb_im) +".jpg", patch, cmap='gray')
		
			self.slice.vs[0].pos = vector(pos1[0],pos1[1],pos1[2])
			self.slice.vs[1].pos = vector(pos2[0],pos2[1],pos2[2])
			self.slice.vs[2].pos = vector(pos3[0],pos3[1],pos3[2])
			self.slice.vs[3].pos = vector(pos4[0],pos4[1],pos4[2])
			
			self.slice.texture = outfolder + "image" + str(self.nb_im) +".jpg"
			#self.slice = quad(vs=self.slice.vs, texture='image2.jpg')
		

		self.disable(False, checkboxes = True)
		self.slice_checkbox.disabled = False
		self.slice_checkbox.checked = True

	def slice_opacity(self):

		alpha = self.slice_opacity_slider.value
		self.slice.vs[0].opacity = alpha
		self.slice.vs[1].opacity = alpha
		self.slice.vs[2].opacity = alpha
		self.slice.vs[3].opacity = alpha


	def create_template_pathology(self):

		# Create image slice (as mesh in pyvista)
		# Use barycenter as center and camera axis as normal

		# Create the crsec outline with a curve + prec curve + baseline curve
		
		center = np.array([self.temp_center.x, self.temp_center.y, self.temp_center.z])

		outline_coords = np.zeros((self.temp_num_nds,3))
		angle = 2 * pi / self.temp_num_nds
		angle_list = angle * np.arange(self.temp_num_nds)

		nds = np.zeros((self.temp_num_nds, 3))
		for i in range(self.temp_num_nds):
			vec = rotate_vector(np.array([0,1,0]), np.array([0,0,1]), angle_list[i])
			outline_coords[i, :] = center + vec * (self.temp_rad)

		outline = curve(color = color.black, radius = 0.1, mode = "pathology", visible = False, locked = False, category = "outline")
		previous = [] 
		current = curve(color = color.red, radius = 0.1, mode = "pathology", visible = False, locked = False, category = "current")

		gray_color = np.linspace(0.1, 1, self.temp_num_crsec)[::-1]
		for i in range(self.temp_num_crsec):
			previous.append(curve(color = color.gray(gray_color[i]), radius = 0.1, mode = "pathology", visible = True, locked = False, category = "previous"))

		current_point = []

		for i in range(len(outline_coords)):
			pos = vector(outline_coords[i, 0], outline_coords[i, 1], outline_coords[i, 2])
			outline.append(pos)
			current.append(pos)

			current_point.append(sphere(pos=pos, color=color.red, radius=0.2, mode = "pathology", category = "current", id = i, visible = False, locked = False))

		outline.append(outline.point(0)['pos'])
		current.append(current.point(0)['pos'])

		self.pathology.append(np.vstack((outline_coords,outline_coords[0, :]))) # The stenosis start with a cicle shape to preserve the smoothness

		self.elements['pathology']['edges'] = [outline, previous, current]
		self.elements['pathology']['nodes'] = current_point
		self.elements['pathology']['text'] = [label(pos=scene.center, text="crsec number "  + str(len(self.pathology)) + " / " + str(self.temp_num_crsec), box = False, visible = False, locked = False)]

	


	def show_hide_template(self, show = True):

		if show:
			# Hide everything (without unchecking boxes)
			self.hide("full")
			self.hide("topo")
			self.hide("model")
			self.hide("mesh")
			if self.slice is not None:
				self.slice.visible = False 

			self.cursor.visible = False
			# Show the template
			self.show("pathology")
			self.output_message("Move the points to edit the current cross section. Press 'n' to move to the next cross section.")
			

		else:
			# Hide the slice
			self.hide("pathology")

			# Show the elements if the boxes are checked
			for mode in ["full", "topo", "model", "mesh"]:
				if self.checkboxes[mode].checked:
					
					self.show(mode)
					if mode == "model":
						self.update_visibility_furcations()
						self.update_visibility_control_pts()
						self.update_visibility_control_radius()
						if self.angle_checkbox_model.checked:
							self.update_visibility_angle(self.angle_checkbox_model)
					if mode == "topo":
						if self.angle_checkbox_topo.checked:
							self.update_visibility_angle(self.angle_checkbox_topo)

			# Show the slice and cursor if the boxes are checked

			if self.slice_checkbox.checked:
				self.update_visibility_slice()
			if self.cursor_checkbox.checked:
				self.update_visibility_cursor()

	def save_pathology_template(self):

		# If the directory doesn't exist, create it 
		try:
			os.makedirs(self.pathology_output_dir)
			self.output_message("Saving the pathology template in " + self.pathology_output_dir)
		except OSError as error:
			self.output_message("Overwriting the pathology template in " + self.pathology_output_dir, "warning")


		for i in range(len(self.pathology)):
			num = str(i)
			if len(num) == 1:
				num = "0" + num
			if len(num) == 2:
				num = "0" + num

			f = open(self.pathology_output_dir+ "crsec_" + num + ".txt", 'w') 
			# Write coordinates in file
			for j in range(len(self.pathology[i])):
				f.write(str(self.pathology[i][j, 0]) + "\t" + str(self.pathology[i][j, 1]) + "\t" + str(self.pathology[i][j, 2]) + "\n")

			f.close()

		# Write info file
		f = open(self.pathology_output_dir + "info.txt", 'w') 
		f.write("center_x\tcenter_y\tcenter_z\tradius\n")
		f.write(str(self.temp_center.x) + "\t" + str(self.temp_center.y) + "\t" + str(self.temp_center.z) +"\t" + str(self.temp_rad) + "\n")
		f.close()


	def next_template(self):

		self.unselect("node")

		def curve_to_coords(c):

			n = c.npoints
			coords = np.zeros((n, 3))
			for i in range(c.npoints):
				pos = c.point(i)["pos"]
				coords[i,:] = np.array([pos.x, pos.y, pos.z])
			return coords

		def coords_to_curve(coords, c):
			c.clear()
			for i in range(len(coords)):
				c.append(vec(coords[i, 0], coords[i, 1], coords[i, 2]))

		if len(self.pathology) == self.temp_num_crsec:

			self.show_hide_template(False)
			# Reset all the curves
			for i in range(self.temp_num_crsec):
				 self.elements["pathology"]["edges"][1][i].clear()
			outline_coords = curve_to_coords(self.elements["pathology"]["edges"][0])
			coords_to_curve(outline_coords, self.elements["pathology"]["edges"][-1])

			self.elements['pathology']['text'][0].text = "crsec number "  + str(1) + " / " + str(self.temp_num_crsec)
			self.pathology.append(np.vstack((outline_coords,outline_coords[0, :])))

			self.save_pathology_template()

			
			self.pathology = [self.pathology[0]]
			self.edition_mode = "off"
			self.edition_menu.selected = "off"
			self.unselect("node")
			self.disable(disabled = False, checkboxes = True)

		else:

			# Save the current outline coords
			current_coords = curve_to_coords(self.elements["pathology"]["edges"][-1])
	
			self.pathology.append(current_coords)

			# Modify the prec curve to keep track of previous outline
			coords_to_curve(current_coords, self.elements["pathology"]["edges"][1][len(self.pathology)-2])
			self.elements['pathology']['text'][0].text = "crsec number "  + str(len(self.pathology)) + " / " + str(self.temp_num_crsec)


	def locate_pathology(self, b):

		if b.checked :

			# Check if crsec graph exist
			if self.tree.get_model_graph() is None:
				self.output_message("Please compute the mesh before adding pathology.", "warning")
			else:

				if self.pathology_edg!= self.selected_edge.id:
					self.pathology_edg = self.selected_edge.id
					spl = self.tree.get_model_graph().edges[self.pathology_edg]["spline"]
					pos1 = spl.point(0.2)
					pos2 = spl.point(0.8)

					self.pathology_markers[0].pos = vec(pos1[0], pos1[1], pos1[2])
					self.pathology_markers[1].pos = vec(pos2[0], pos2[1], pos2[2])

				self.pathology_markers[0].visible = True
				self.pathology_markers[1].visible = True

				self.modified["model"] = True

		else:
			self.pathology_markers[0].visible = False 
			self.pathology_markers[1].visible = False 



	def add_pathology(self):

		pt0 = self.pathology_markers[0].pos
		pt0 = np.array([pt0.x, pt0.y, pt0.z])

		pt1 = self.pathology_markers[1].pos
		pt1 = np.array([pt1.x, pt1.y, pt1.z])

		spl = self.tree.get_model_graph().edges[self.pathology_edg]["spline"]

		t0 = spl.project_point_to_centerline(pt0)
		t1 = spl.project_point_to_centerline(pt1)
		print(t0,t1)
		self.tree.deform_surface_to_template(self.pathology_edg, t0, t1, self.template[0], self.template[1], self.template[2], rotate = 120)

		
	def update_visibility_angle(self, checkbox):


		if checkbox.checked: # Create angles labels
			angles = self.tree.angle(None, mode=checkbox.mode)

			if "angles" not in self.elements[checkbox.mode].keys():
				self.elements[checkbox.mode]["angles"] = []

				for a in angles:
					L = label(pos=vec(a[1][0], a[1][1], a[1][2]), text=str(a[2])+ "°", box = False, locked = False)
					self.elements[checkbox.mode]["angles"].append(L)

			else:
				for i in range(len(angles)):
					a = angles[i]
					if i < len(self.elements[checkbox.mode]["angles"]):
						self.elements[checkbox.mode]["angles"][i].visible = True
						self.elements[checkbox.mode]["angles"][i].pos = vec(a[1][0], a[1][1], a[1][2])
						self.elements[checkbox.mode]["angles"][i].text = str(a[2])+ "°"
					else:

						L = label(pos=vec(a[1][0], a[1][1], a[1][2]), text=str(a[2])+ "°", box = False, locked = False)
						self.elements[checkbox.mode]["angles"].append(L)

		else: # Hide angle labels
			for elt in self.elements[checkbox.mode]["angles"]:
				elt.visible = False



	def update_visibility_control_pts(self):

		""" Show/ hide model control points """

		if self.control_pts_checkbox.checked:
			self.control_radius_checkbox.disabled = False
			self.show('model', ['control_edges', 'control_nodes'])
		else:

			# Control point radius can be displayed only if the control points are already displayed
			self.control_radius_checkbox.disabled = True
			self.control_radius_checkbox.checked = False
			self.hide('model',['control_edges', 'control_nodes'])

			for k in self.elements["model"]["control_nodes"].keys():
				for i in range(len(self.elements["model"]["control_nodes"][k])):
					self.elements["model"]["control_nodes"][k][i].radius = 0.5



	def update_visibility_control_radius(self):

		""" Show/ hide model control points radius """
		if self.control_radius_checkbox.checked:

			G = self.tree.get_model_graph()
			for e in G.edges():
				crtlpts = G.edges[e]['spline'].get_control_points()
				for i in range(len(crtlpts)):
					self.elements["model"]["control_nodes"][e][i].radius = crtlpts[i, 3]

		else:
			for k in self.elements["model"]["control_nodes"].keys():
				for i in range(len(self.elements["model"]["control_nodes"][k])):
					self.elements["model"]["control_nodes"][k][i].radius = 0.5


	def update_visibility_furcations(self):

		""" Show / hide the furcation models """

		if self.furcation_checkbox.checked:
			self.show('model', ['furcation_edges', 'furcation_nodes'])
		else:
			self.hide('model', ['furcation_edges', 'furcation_nodes'])




	def update_mesh_representation(self, b):

		""" Switch the mesh representation mode (wireframe / sections/ solid)"""

		if self.checkboxes["mesh"].checked:
			representation = self.mesh_representation_menu.selected
			if representation == "wireframe":
				self.hide("mesh", ["surface"])
				self.show("mesh", ["connecting_edges", "section_edges"])

			elif representation == "sections":
				self.show("mesh", ["section_edges"])
				self.hide("mesh", ["surface", "connecting_edges"])

			elif representation == 'solid':
				self.show("mesh", ["surface"])
				self.hide("mesh",["section_edges","connecting_edges"]) 
			else: 
				self.show("mesh")


			self.output_message("Mesh representation mode switched to " +  representation + ".")



	def show(self, mode, categories = []):

		""" Show elements for a given mode and category 

		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
		categories -- list of object categrories (node, edges...) 
		"""

		def set_visible(elt, args):
			if not elt.locked:
				elt.visible = args[0]

		if len(self.elements[mode]) == 0:
			# Creation of the mode objects
			self.create_elements(mode)
		else:
			self.apply_function(mode, func = set_visible, args=[True], categories = categories)

	


	def hide(self, mode, categories=[]):

		""" Hide elements for a given mode and category 

		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
		categories -- list of object categrories (node, edges...) 
		"""

		def set_visible(elt, args):
			elt.visible = args[0]

		self.apply_function(mode, func = set_visible, args=[False], categories = categories)


	def lock(self, mode, categories=[]):

		""" Hide elements for a given mode and category 

		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
		categories -- list of object categrories (node, edges...) 
		"""

		def set_lock(elt, args):
			elt.locked = args[0]

		self.apply_function(mode, func = set_lock, args=[True], categories = categories)


	def unlock(self, mode, categories=[]):

		""" Hide elements for a given mode and category 

		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
		categories -- list of object categrories (node, edges...) 
		"""

		def set_lock(elt, args):
			elt.locked = args[0]

		self.apply_function(mode, func = set_lock, args=[False], categories = categories)


	def disable(self, disabled=True, mode = ["full", "topo", "model", "mesh"], checkboxes = True):


		""" Disable user interaction widgets for a given mode.

		Keyword arguments:
		mode -- list of newtwork modes (full, topo, model, mesh) to disable
		checkboxes -- if true, disable also the mode checkbox 
		"""

		if not disabled:
			for m in mode:
				self.checkboxes[m].disabled = disabled
				if not self.checkboxes[m].checked:
					mode.remove(m)
	
		for m in mode:
			
			if checkboxes:
				self.checkboxes[m].disabled = disabled
			if m in self.update_buttons.keys():
				self.update_buttons[m].disabled = disabled

			if m in self.opacity_sliders.keys():
				self.opacity_sliders[m].disabled = disabled
			if m in self.edge_size_sliders:
				self.edge_size_sliders[m].disabled = disabled
			if m in self.node_size_sliders:
				self.node_size_sliders[m].disabled = disabled

			if m == "topo":
				self.angle_checkbox_topo.disabled = disabled
				self.crop_button.disabled = disabled
				#self.angle_checkbox_topo.checked = False

			if m == "model":
				self.extension_button.disabled = disabled
				self.control_pts_checkbox.disabled = disabled
				#self.control_pts_checkbox.checked = False 
				if disabled:
					self.control_radius_checkbox.disabled = disabled
					#self.control_radius_checkbox.checked = False
				self.furcation_checkbox.disabled = disabled
				#self.furcation_checkbox.checked = False

				self.angle_checkbox_model.disabled = disabled
				#self.angle_checkbox_model.checked = False


				if disabled:
					for k in self.smooth_checkboxes.keys():
						self.smooth_checkboxes[k].disabled = disabled

					self.pathology_checkbox.disabled = disabled

			if m == "mesh":
				self.deform_mesh_button.disabled = disabled
				self.check_mesh_button.disabled = disabled
				self.mesh_representation_menu.disabled = disabled
				self.close_mesh_button.disabled = disabled
				self.volume_button.disabled = disabled
				self.surface_button.disabled = disabled



	###########################################
	################### SAVE ##################
	###########################################


	def update_save_directory(self):

		""" Update the path to the output folder """

		self.save_directory = self.save_winput.text
		self.output_message("The output directory set to " + self.save_directory + ".")
	

	def update_save_filename(self):

		""" Update the output filename """

		self.save_filename = self.save_filename_winput.text
		self.output_message("The output filename set to " + self.save_filename + ".")

	def load_pathology_template(self):

		""" Load the pathology file """
		try:
			path = self.pathology_directory_winput.text
			info_file = None 
			crsec_files = []
			for root, dirs, files in os.walk(path):
				for file in files:
					if file == "info.txt":
						info_file = file
					elif file[:5] == "crsec":
						crsec_files.append(file)
			crsec_files.sort()
			if info_file is None:
				self.output_message("Info file not found.", "error")
			elif len(crsec_files) == 0:
				self.output_message("Crsec files not found.", "error")
			else:
				data = []
				if self.template is not None:
					self.output_message("Pathology template loaded from " + path + ".")

				for i in range(len(crsec_files)):
					data.append(np.loadtxt(path + crsec_files[i], skiprows=0))

				infos = np.loadtxt(path + info_file, skiprows=1)
				self.template = [data, int(infos[-1]), infos[:-1]]

		except FileNotFoundError:
			self.output_message("No template found at" + path + ".", "error")


	def save(self):

		""" Save the network or the mesh in a given output directory """

		try:
			if self.save_directory is None: 
				self.output_message("No output directory found. Please write the path in the text box and hit enter.", "warning")
			else:
				if self.save_menu.selected == "model":

					if self.save_filename[-4:] != ".obj":
						file = self.save_directory + self.save_filename + ".obj"
					else:
						file = self.save_directory + self.save_filename

					f = open(file, 'wb') 
					pickle.dump(self.tree, f)
					self.output_message("Vessel model saved in " + file + ".")

				elif self.save_menu.selected == "centerline":
					file = self.save_directory + self.save_filename

					if self.save_filename[-4:] == ".swc":
						self.tree.write_swc(file)
					elif self.save_filename[-4:] == ".txt":
						self.tree.write_edg_nds(file)
					else:
						self.tree.write_swc(file)
					
					self.output_message("Vessel centerline saved in " + file + ".")

				elif self.save_menu.selected == "surface mesh":

					if self.save_filename[-4:] not in [".vtk", ".stl", ".ply"]:
						file = self.save_directory + self.save_filename + ".vtk"
					else:
						file = self.save_directory + self.save_filename

					mesh = self.tree.get_surface_mesh()
					if mesh is None:
						self.output_message("No mesh found. Please compute and/or update the mesh first.")
					else:
						self.tree.write_surface_mesh(file)
						self.output_message("Surface mesh saved in " + file + ".")

				else:
					if self.save_filename[-4:] not in [".vtk", ".vtu", ".msh"]:
						file = self.save_directory + self.save_filename + ".vtk"
					else:
						file = self.save_directory + self.save_filename

					mesh = self.tree.get_volume_mesh()
					if mesh is None:
						self.output_message("No volume mesh found. Please compute and/or update the mesh first.")
					else:
						self.tree.write_volume_mesh(file)
						self.output_message("Volume mesh saved in " + file + ".")


		except FileNotFoundError:
			self.output_message("The output directory does not exist.", "error")

		except PermissionError:
			self.output_message("Writting permission denied. Please check the permissions on output folder.", "error")



		
	###########################################
	########## CREATION / UPDATE ##############
	###########################################


	def apply_function(self, mode, func, args, categories=[]):

		""" Apply a modifier function to all elements of given mode and category.

		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
		categories -- list of object categrories (node, edges...) 
		"""


	
		if len(categories) == 0:
			categories = list(self.elements[mode].keys())

		for category in categories:
			if category in list(self.elements[mode].keys()):
				obj_cat = self.elements[mode][category]
	
				if type(obj_cat) == dict:
					for elt in obj_cat.values():
						if type(elt) == list:
							for e in elt:
								func(e, args)
						else:
							func(elt, args)

				else:
					if type(obj_cat) == list:
						for elt in obj_cat:
							if type(elt) == list:
								for e in elt:
									func(e, args)

							else:
								func(elt, args)

	def create_elements(self, mode):

		""" Create all 3D objects (nodes, edges) for a given mode.
		
		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
		"""

		if mode == 'full':

			# Import full graph
			G = self.tree.get_full_graph()

			# Show data points as spheres
			nodes = {}
			for n in G.nodes():
				pos = vector((G.nodes[n]['coords'][0]), (G.nodes[n]['coords'][1]), (G.nodes[n]['coords'][2]))
				radius = G.nodes[n]['coords'][3]
				ball = sphere(pos=pos, color=color.red, radius=radius, mode = 'full', category = 'nodes', id = n, locked = False)
				nodes[n] = ball
		
			# Show data edge as cylinders
			edges = {}
			for e in G.edges():
				pos = vector((G.nodes[e[0]]['coords'][0]), (G.nodes[e[0]]['coords'][1]), (G.nodes[e[0]]['coords'][2]))
				axis = G.nodes[e[1]]['coords'][:-1] - G.nodes[e[0]]['coords'][:-1]
				length = norm(axis)
				direction = axis / length
				c = cylinder(pos=pos, axis=vector(direction[0], direction[1], direction[2]), length=length, radius=self.edge_size_sliders["full"].value, color=color.black, mode = 'full', category = 'edges', id = e, locked = False)
				edges[e] = c

			self.elements['full']['nodes'] = nodes
			self.elements['full']['edges'] = edges


		elif mode == 'topo':

			G = self.tree.get_topo_graph()
			
			nodes = {}
			# Show nodes of the topo graph as spheres
			for n in G.nodes():
				pos = vector((G.nodes[n]['coords'][0]), (G.nodes[n]['coords'][1]), (G.nodes[n]['coords'][2]))
				ball = sphere(pos=pos, color=self.node_color(n, 'topo'), radius=self.node_size_sliders["topo"].value, mode = 'topo', category = 'nodes', id = n, locked = False)
				nodes[n] = ball

			edges = {}
			# Show edges of the topo graph as curves
			for e in G.edges():
				coords = np.vstack((G.nodes[e[0]]['coords'], G.edges[e]['coords'], G.nodes[e[1]]['coords']))

				c = curve(color = color.black, radius = self.edge_size_sliders["topo"].value, mode = 'topo', category = 'edges', id = e, locked = False)
				for i in range(len(coords)):
					c.append(vector(coords[i, 0], coords[i, 1], coords[i, 2]))

				edges[e] = c

			self.elements['topo']['nodes'] = nodes
			self.elements['topo']['edges'] = edges

		elif mode == 'model':

			# Check if graph is valid
			if not self.tree.check_full_graph():
				self.output_message("The model cannot be computed as the input centerline contains invalid elements.", "error")
			else:

				self.mesh_selection = [] # Reset the list of edges selected for meshing

				def create_crsec_coords(coord, normal):

					num = 20
					v = np.zeros((num, 3))

					angle = 2 * pi / num
					angle_list = angle * np.arange(num)

					ref = cross(normal, np.array([0,0,1]))
					ref = ref/norm(ref)
					for i in range(num):

						n = rotate_vector(ref, normal, angle_list[i])
						v[i, :] = coord[:-1] + coord[-1]*n

					return v
			

				# Node and edges
				G = self.tree.get_model_graph()

				if G is None: # Compute model network if not done
					self.output_message("Computing model...")

					self.disable(True, checkboxes = True)
					self.tree.model_network()
					G = self.tree.get_model_graph()
					
					self.output_message("Model complete!")
					self.disable(False, checkboxes = True)

				nodes = {}
				furcation_nodes = {}
				furcation_edges = {}
			
				for n in G.nodes():
					pos = vector((G.nodes[n]['coords'][0]), (G.nodes[n]['coords'][1]), (G.nodes[n]['coords'][2]))
					pt_type = G.nodes[n]['type']

					ball = sphere(pos=pos, color=self.node_color(n, 'model'), radius=self.node_size_sliders["model"].value, mode = 'model', category = 'nodes', id = n, locked = False)
					nodes[n] = ball

					if pt_type == "bif":
						n_list = []
						c_list = []
						bif = G.nodes[n]['bifurcation']

						AP = bif.get_AP()
						for pt in AP:
							n_list.append(sphere(pos = vector(pt[0], pt[1], pt[2]), radius=self.node_size_sliders["model"].value, color = color.red, visible = False, mode = 'model', category = 'furcation_nodes', locked = False))

						apexsec = bif.get_apexsec()
						for l in apexsec:
							for sec in l:
								coords = create_crsec_coords(sec[0], sec[1][:-1])
								c = curve(color = color.black, radius = 0.1, visible=False, mode = 'model', category = 'furcation_edges', locked = False)
								for pt in coords:
									c.append(vector(pt[0], pt[1], pt[2]))
								c.append(vector(coords[0][0], coords[0][1], coords[0][2]))
								c_list.append(c)
								n_list.append(sphere(pos = vector(sec[0][0], sec[0][1], sec[0][2]), radius=self.node_size_sliders["model"].value, color = color.black, visible = False, mode = 'model', category = 'furcation_nodes', locked = False))
										
						endsec = bif.get_endsec()
						for sec in endsec:
							coords = create_crsec_coords(sec[0], sec[1][:-1])
							c = curve(color = color.black, radius = 0.1, visible=False, mode = 'model', category = 'furcation_edges', locked = False)
							for pt in coords:
								c.append(vector(pt[0], pt[1], pt[2]))
							c.append(vector(coords[0][0], coords[0][1], coords[0][2]))
							c_list.append(c)
							n_list.append(sphere(pos = vector(sec[0][0], sec[0][1], sec[0][2]), radius=self.node_size_sliders["model"].value, color = color.black, visible = False, mode = 'model', category = 'furcation_nodes', locked = False))

						furcation_nodes[n] = n_list
						furcation_edges[n] = c_list


				edges = {}
				control_edges = {}
				control_nodes = {}
				for e in G.edges():
					spl = G.edges[e]['spline']

					coords = spl.get_points()
					coords = np.vstack((coords[0, :], coords[::self.display_spline_step, :], coords[-1, :]))
					pos = vector(coords[0][0], coords[0][1], coords[0][2])
					c = curve(pos=pos, color = color.black, radius = self.edge_size_sliders["model"].value, mode = 'model', category = 'edges', id = e, locked = False)

					for i in range(1,len(coords)):
						c.append(vector(coords[i][0], coords[i][1], coords[i][2]))

					edges[e] = c

					coords = spl.get_control_points()
					pos = vector(coords[0][0], coords[0][1], coords[0][2])
					c2 = curve(pos=pos, color = color.gray(0.5), radius = self.edge_size_sliders["model"].value, visible=False, mode = 'model', category = 'control_edges', id = e, locked = False)
					n_list = [sphere(pos = pos, color=color.gray(0.5), radius=self.node_size_sliders["model"].value, visible=False, mode = 'model', category = 'control_nodes', id = (e, 0), locked = False)]
					
					for i in range(1,len(coords)):
						c2.append(vector(coords[i][0], coords[i][1], coords[i][2]))
						n_list.append(sphere(pos = vector(coords[i][0], coords[i][1], coords[i][2]), color=color.gray(0.5), radius=self.node_size_sliders["model"].value, visible=False, mode = 'model', category = 'control_nodes', id = (e, i), locked = False))
					control_edges[e] = c2
					control_nodes[e] = n_list

					self.mesh_selection.append(e) # Add all the edges to the default meshing selection
					

				self.elements['model']['nodes'] = nodes
				self.elements['model']['edges'] = edges

				self.elements['model']['control_edges'] = control_edges
				self.elements['model']['control_nodes'] = control_nodes

				self.elements['model']['furcation_edges'] = furcation_edges
				self.elements['model']['furcation_nodes'] = furcation_nodes

				self.control_pts_checkbox.disabled = False
				self.furcation_checkbox.disabled = False

			
		elif mode == 'mesh':

			# Check if graph is valid
			if not self.tree.check_full_graph():
				self.output_message("The model cannot be computed as the input centerline contains invalid elements.", "error")
			else:

				if self.tree.get_model_graph() is None:
					self.create_elements("model")
					self.checkboxes["model"].checked = True

				if self.tree.get_crsec_graph() is None:

					self.output_message("Computing mesh...")

					self.disable(True, checkboxes = True)
					self.tree.compute_cross_sections(self.N, self.d)
					self.disable(False, checkboxes = True)

					self.output_message("Mesh complete!")

				mesh = self.tree.get_surface_mesh()
				if mesh is None: 
					# Convert mesh selection from model graph to crsec graph
					self.disable(True, checkboxes = True)
					mesh = self.tree.mesh_surface(edg = self.converted_selection())
					self.disable_close_check()
					self.disable(False, checkboxes = True)

				vertices = mesh.points
				faces = mesh.faces.reshape((-1, 5))
				faces.astype('int32')
				print('number of faces: ',  len(faces))

				# If too much faces, don't display it
				if len(faces) > self.mesh_display_limit:
					self.output_message("The entire mesh cannot be displayed has it has too many faces. Please use the mesh selection menu to select a subset of the network, or modify the meshing parameters.", 'warning')
				else:

					v_obj = []
					for i in range(len(vertices)):
						v = vertices[i]
						v_obj.append(vertex(pos=vector(v[0], v[1], v[2]), color=color.gray(0.75), id = i))

					
					quads = []
					for i in range(len(faces)):
						f = faces[i]
						q = quad(v0=v_obj[f[1]], v1=v_obj[f[2]], v2=v_obj[f[3]], v3=v_obj[f[4]], mode = 'mesh', category = 'surface', id = i, locked = False)
						quads.append(q)
						self.trash_elements['quads'].append(q)

					curves = {}
					lines = {}
					G = self.tree.get_crsec_graph()
					node_list = []

					for e in self.converted_selection():
						if e[0] not in node_list:
							node_list.append(e[0])
						if e[1] not in node_list:
							node_list.append(e[1])

					for n in node_list:
						crsec = G.nodes[n]['crsec']
						if G.nodes[n]['type'] == "bif":
							N = (self.tree._N // 2) - 1
							curve_list = []
							s = 2
							for i in range(len(crsec) // N):
								c = curve(pos = vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]), radius=self.edge_size_sliders["mesh"].value, color = color.black, mode = 'mesh', category = 'section_edges', locked = False)
								for k in range(N):
									c.append(vector(crsec[s+k, 0], crsec[s+k, 1], crsec[s+k, 2]))
								s += k + 1
								c.append(vector(crsec[1, 0], crsec[1, 1], crsec[1, 2]))
								self.trash_elements['curves'].append(c)
								curve_list.append(c)
							curves[n] = curve_list


						else:
							c = curve(radius=self.edge_size_sliders["mesh"].value, color = color.black, mode = 'mesh', category = 'section_edges', locked = False)
							for i in range(len(crsec)):
								c.append(vector(crsec[i, 0], crsec[i, 1], crsec[i, 2]))
							c.append(vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
							self.trash_elements['curves'].append(c)
							curves[n] = [c]

					for e in self.converted_selection():
						crsec_list = G.edges[e]['crsec']
						# Longitudinal curves
						lines_list = []
						for i in range(crsec_list.shape[1]):
							c = curve(radius=self.edge_size_sliders["mesh"].value, color = color.black, mode = 'mesh', category = 'connecting_edges', locked = False)
							# Starting point
							start_sec = G.nodes[e[0]]['crsec']
							c.append(vector(start_sec[i, 0], start_sec[i, 1], start_sec[i, 2]))
							for j in range(crsec_list.shape[0]):
								c.append(vector(crsec_list[j, i, 0], crsec_list[j, i, 1], crsec_list[j, i, 2]))
							# Ending point
							end_sec = G.nodes[e[1]]['crsec']
							l = G.edges[e]['connect'][i]
							c.append(vector(end_sec[l][0], end_sec[l][1], end_sec[l][2]))
							self.trash_elements['curves'].append(c)

							lines_list.append(c)
						lines[e] = lines_list

						# Cross section curves
						curve_list = []
						for i in range(len(crsec_list)):
							crsec = crsec_list[i]
							c = curve(radius=self.edge_size_sliders["mesh"].value, color = color.black, mode = 'mesh', category = 'section_edges', locked = False)
							for j in range(len(crsec)):
								c.append(vector(crsec[j, 0], crsec[j, 1], crsec[j, 2]))
							c.append(vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
							self.trash_elements['curves'].append(c)
							curve_list.append(c)
						curves[e] = curve_list


					self.elements['mesh']['surface'] = quads
					self.elements['mesh']['section_edges'] = curves
					self.elements['mesh']['connecting_edges'] = lines

					self.update_mesh_representation(self.mesh_representation_menu)

		else:
			print("Unknown mode.")

		if mode in self.edge_size_sliders.keys():
			self.edge_size_sliders[mode].disabled = False
		if mode in self.node_size_sliders.keys():
			self.node_size_sliders[mode].disabled = False
		if mode in self.opacity_sliders.keys():
			self.opacity_sliders[mode].disabled = False


	def update_graph(self, b):

		""" Update the modified elements in the tree object by modifying the graphs"""

		self.disable(True, checkboxes = True)
		mode = b.mode # The mode of the button is the representation it controls
		self.modified[mode] = False

		if mode == "full":
			self.refresh_display("full")
			self.tree.set_full_graph(self.tree.get_full_graph())
			self.refresh_display("topo")
			self.refresh_display("model")
			self.refresh_display("mesh")

			if self.node_size_sliders["full"].value != 1:
				self.node_size_sliders["full"].value = 1

			self.output_message("Full graph updated.")

		elif mode == "topo":

			self.refresh_display("topo")
			self.tree.topo_to_full()
			self.refresh_display("full")
			self.refresh_display("model")
			self.refresh_display("mesh")

			self.crop_selection = []

			self.output_message("Topo graph updated.")

		elif mode == "model":

			self.refresh_display("model")

			if self.pathology_checkbox.checked: # Add pathology 
				self.add_pathology()
				self.pathology_markers[0].visible = False 
				self.pathology_markers[1].visible = False 

			if self.tree.get_crsec_graph() is not None: 
			# Recompute crsec for modified model edges and update mesh (cannot be done in real time)
				for e in self.modified_elements['splines']:
					self.tree.vessel_cross_sections(e)
				self.tree.mesh_surface()

			self.refresh_display("mesh")

			self.mesh_selection = []

			self.output_message("Model graph updated.")


		self.unselect("node", mode)
		self.unselect("edge", mode)
		self.disable(False, checkboxes = True)


	def converted_selection(self):

		converted_selection = []
		for e in self.mesh_selection:
			if self.tree.get_model_graph().nodes[e[0]]['type'] == 'bif':
				converted_selection.append((e[1], e[0]))
			else:
				converted_selection.append(e)
		return converted_selection




	def refresh_display(self, mode):

		""" Modify the edges and nodes position according to the tree object modified graph, for a given mode.

		Keyword arguments:
		mode -- network mode (full, topo, model, mesh)
		"""


		if self.elements[mode]:# If there is elements in the mode dictionnary

			if self.checkboxes[mode].checked:
				visible = True
			else:
				visible = False

			if mode == "full":

				nds = [n for n in self.tree.get_full_graph().nodes()]
				nds_elt = list(self.elements[mode]["nodes"].keys())[:]
				for n in nds_elt:
					if n not in nds:
						self.elements[mode]["nodes"][n].visible = False
						self.elements[mode]["nodes"].pop(n)


				for n in nds:
					coords = self.tree.get_full_graph().nodes[n]['coords']
					if n in nds_elt: # Modify node
						self.elements[mode]["nodes"][n].pos = vector(coords[0], coords[1], coords[2])
						self.elements[mode]["nodes"][n].radius = coords[3] 
						self.elements[mode]["nodes"][n].visible = visible
						if self.elements[mode]["nodes"][n].locked:
							self.elements[mode]["nodes"][n].visible = False

					else: 
						self.elements[mode]["nodes"][n] = sphere(pos=vector(coords[0], coords[1], coords[2]), color=color.red, radius=coords[3], mode = 'full', category = 'nodes', id = n, visible = visible, locked = False) # Create node

				edj = [e for e in self.tree.get_full_graph().edges()]
				edj_elt = list(self.elements[mode]["edges"].keys())[:]
				for e in edj_elt:
					if e not in edj:
						self.elements[mode]["edges"][e].visible = False
						self.elements[mode]["edges"].pop(e)

				for e in edj:
					coords_0 = self.tree.get_full_graph().nodes[e[0]]['coords']
					coords_1 = self.tree.get_full_graph().nodes[e[1]]['coords']

					pos = vector(coords_0[0], coords_0[1], coords_0[2])
					axis = coords_1[:-1] - coords_0[:-1]
					length = norm(axis)
					axis = axis / norm(axis)
					axis = vector(axis[0], axis[1], axis[2]) 

					if e in edj_elt:
						self.elements[mode]["edges"][e].pos = pos
						self.elements[mode]["edges"][e].axis = axis
						self.elements[mode]["edges"][e].length = length
						self.elements[mode]["edges"][e].visible = visible
						if self.elements[mode]["edges"][e].locked:
							self.elements[mode]["edges"][e].visible = False

					else:
						self.elements[mode]["edges"][e] = cylinder(pos=pos, axis=axis, length=length, radius=self.edge_size_sliders["full"].value, color=color.black, mode = 'full', category = 'edges', id = e, visible = visible, locked = False)

			elif mode == "topo":

				if not self.tree.check_full_graph():
					self.output_message("The input centerline contains invalid elements.", "warning")
			
				# Empty the crop selection
				self.crop_selection = []
				for elt in self.elements['topo']['edges'].values():
					elt.color = color.black


				G = self.tree.get_topo_graph()
				nds = [n for n in G.nodes()]
				nds_elt = list(self.elements[mode]["nodes"].keys())[:]
				for n in nds_elt:
					if n not in nds:
						self.elements[mode]["nodes"][n].visible = False
						self.elements[mode]["nodes"].pop(n)

				for n in nds:
					coords = G.nodes[n]['coords']
					if n not in nds_elt:
						pos = vector(coords[0], coords[1], coords[2])
						pt_type = G.nodes[n]['type']
						self.elements[mode]["nodes"][n] = sphere(pos=pos, color=self.node_color(n, 'topo'), radius=self.node_size_sliders["topo"].value, mode = 'topo', category = 'nodes', id = n, visible = visible, locked = False) # Create new point
					else:
						self.elements[mode]["nodes"][n].pos = vector(coords[0], coords[1], coords[2])
						self.elements[mode]["nodes"][n].color  = self.node_color(n, 'topo')

						self.elements[mode]["nodes"][n].visible = visible


				edj = [e for e in G.edges()]
				edj_elt = list(self.elements[mode]["edges"].keys())[:]
				for e in edj_elt:
					if e not in edj:
						self.elements[mode]["edges"][e].visible = False
						self.elements[mode]["edges"].pop(e)

				for e in edj:

					coords = np.vstack((G.nodes[e[0]]['coords'], G.edges[e]['coords'], G.nodes[e[1]]['coords']))

					if e in edj_elt:
						c = self.elements[mode]["edges"][e]
						c.clear()
						for i in range(len(coords)):
							c.append(vector(coords[i][0], coords[i][1], coords[i][2]))

					else:
						c = curve(color = color.black, radius = self.edge_size_sliders["topo"].value, mode = 'topo', category = 'edges', id = e, locked = False)

						for i in range(len(coords)):
							c.append(vector(coords[i][0], coords[i][1], coords[i][2]))

						self.elements[mode]["edges"][e] = c
					self.elements[mode]["edges"][e].visible = visible


			elif mode == "model":

				
				G = self.tree.get_model_graph()
				if G is not None:
			
					# Remove nodes if required
					nds = [n for n in G.nodes()]
					nds_elt = list(self.elements[mode]["nodes"].keys())[:]
					for n in nds_elt:
						if n not in nds:
							self.elements[mode]["nodes"][n].visible = False
							self.elements[mode]["nodes"].pop(n)

							if n in self.elements[mode]["furcation_nodes"].keys(): # Remove the bifurcation nodes and edges
								for i in range(len(self.elements[mode]["furcation_nodes"][n])):
									self.elements[mode]["furcation_nodes"][n][i].visible = False

								for i in range(len(self.elements[mode]["furcation_nodes"][n])):
									self.elements[mode]["furcation_nodes"][n][i].visible = False

					# Modify or add node
					for n in nds:
						coords = G.nodes[n]['coords']

						if n not in nds_elt:
							pos = vector(coords[0], coords[1], coords[2])
							self.elements[mode]["nodes"][n] = sphere(pos=pos, color=self.node_color(n, 'model'), radius=self.node_size_sliders["model"].value, mode = 'model', category = 'nodes', id = n, visible = visible, locked = False) # Create new point
						else:
							self.elements[mode]["nodes"][n].pos = vector(coords[0], coords[1], coords[2]) # Change coordinates
							self.elements[mode]["nodes"][n].color = self.node_color(n, 'model')

						self.elements[mode]["nodes"][n].visible = visible

					# Remove edge if required
					edj = [e for e in G.edges()]
					self.mesh_selection = edj[:] #Update the mesh selection

					edj_elt = list(self.elements[mode]["edges"].keys())[:]
					for e in edj_elt:
						if e not in edj:
							self.elements[mode]["edges"][e].visible = False
							self.elements[mode]["edges"].pop(e)

							self.elements[mode]["control_edges"][e].visible = False
							self.elements[mode]["control_edges"].pop(e)

							for i in range(len(self.elements[mode]["control_nodes"][e])):
								self.elements[mode]["control_nodes"][e][i].visible = False

							self.elements[mode]["control_nodes"].pop(e)

					# Modify or add edge
					for e in G.edges(): 
		
						spl = G.edges[e]['spline']
						if e in edj_elt: # Modify existing edge
							# Update the control points positions
							coords = spl.get_control_points()
							c = self.elements[mode]["control_edges"][e]

							if len(coords) == c.npoints: 

								for i in range(len(coords)):
									c.modify(i, vector(coords[i][0], coords[i][1], coords[i][2]))
									self.elements[mode]["control_nodes"][e][i].pos = vector(coords[i][0], coords[i][1], coords[i][2])
							else:
								c.clear()
								for i in len(self.elements[mode]["control_nodes"][e]):
									self.elements[mode]["control_nodes"][e][i].visible = self.control_pts_checkbox.checked

								self.elements[mode]["control_nodes"][e] = []

								for i in range(len(coords)):
									c.append(pos=vector(coords[i][0], coords[i][1], coords[i][2]), radius=self.edge_size_sliders["model"].value)
									self.elements[mode]["control_nodes"][e].append(sphere(pos = vector(coords[i][0], coords[i][1], coords[i][2]), color=color.gray(0.5), radius=self.node_size_sliders["model"].value, visible=False, mode = 'model', category = 'control_nodes', id = (e, i), locked = False))


							# Update the spline curve point positions
							coords = spl.get_points()
							coords = np.vstack((coords[0, :], coords[::self.display_spline_step, :], coords[-1, :]))
							c = self.elements[mode]["edges"][e]

							c.clear()
							for i in range(len(coords)):
								c.append(pos=vector(coords[i][0], coords[i][1], coords[i][2]), radius=self.edge_size_sliders["model"].value)
							self.elements[mode]["control_edges"][e].visible = self.control_pts_checkbox.checked

						else: # Create new edge

							# Create control nodes
							coords = spl.get_control_points()
							pos = vector(coords[0][0], coords[0][1], coords[0][2])
							c2 = curve(pos=pos, color = color.black, radius = self.edge_size_sliders["model"].value, visible=False, mode = 'model', category = 'control_edges', id = e, locked = False)
							n_list = [sphere(pos = pos, color=color.gray(0.5), radius=self.node_size_sliders["model"].value, visible=False, mode = 'model', category = 'control_nodes', id = (e, 0), locked = False)]
					
							for i in range(1,len(coords)):
								c2.append(vector(coords[i][0], coords[i][1], coords[i][2]))
								n_list.append(sphere(pos = vector(coords[i][0], coords[i][1], coords[i][2]), color=color.gray(0.5), radius=self.node_size_sliders["model"].value, visible=False, mode = 'model', category = 'control_nodes', id = (e, i), locked = False))

							self.elements[mode]["control_edges"][e] = c2
							self.elements[mode]["control_nodes"][e] = n_list
					
							# Create spline curve
							coords = spl.get_points()
							coords = np.vstack((coords[0, :], coords[::self.display_spline_step, :], coords[-1, :]))
							pos = vector(coords[0][0], coords[0][1], coords[0][2])
							c = curve(pos=pos, color = color.black, radius = self.edge_size_sliders["model"].value, mode = 'model', category = 'edges', id = e, locked = False)

							for i in range(1,len(coords)):
								c.append(vector(coords[i][0], coords[i][1], coords[i][2]))

							self.elements[mode]["edges"][e] = c

					if self.edition_mode == "mesh":
						self.show_mesh_selection(True)

				else: # Hide current model

					self.hide("model")
					self.elements["model"] = {}
					self.checkboxes['model'].checked = False

			else:

				mesh = self.tree.get_surface_mesh()

				if self.tree.get_crsec_graph() is None: # Hide current mesh

					self.hide("mesh")
					self.elements["mesh"] = {}
					self.checkboxes['mesh'].checked = False

				else:
					if mesh is None:
						self.disable(True, checkboxes = True)
						mesh = self.tree.mesh_surface(edg = self.converted_selection())
						self.disable(False, checkboxes = True)
						self.closing_state = False # The surface will be opened after the recomputation
						self.close_mesh_button.text = "Close"
						
					self.hide("mesh")

					vertices = mesh.points
					faces = mesh.faces.reshape((-1, 5))
					faces.astype('int32')

					qc = 0
					cc = 0

					v_obj = []
					for i in range(len(vertices)):
						v = vertices[i]
						v_obj.append(vertex(pos=vector(v[0], v[1], v[2]), color=color.gray(0.75), id = i, locked = False))
								
					quads = []
					for i in range(len(faces)):
						f = faces[i]
						if qc > len(self.trash_elements['quads']) - 1:
							new_v = [vertex(pos=vector(0, 0, 0), color=color.gray(0.75), id = 0), vertex(pos=vector(0, 0, 0), color=color.gray(0.75), id = 0), vertex(pos=vector(0, 0, 0), color=color.gray(0.75), id = 0), vertex(pos=vector(0, 0, 0), color=color.gray(0.75), id = 0)]
							self.trash_elements['quads'].append(quad(v0 = new_v[0], v1 = new_v[1], v2 = new_v[2], v3 = new_v[3], mode = 'mesh', category = 'surface', id = 0, locked = False))
						self.trash_elements['quads'][qc].v0 = v_obj[f[1]]
						self.trash_elements['quads'][qc].v1 = v_obj[f[2]]
						self.trash_elements['quads'][qc].v2 = v_obj[f[3]]
						self.trash_elements['quads'][qc].v3 = v_obj[f[4]]
						self.trash_elements['quads'][qc].id = i


						quads.append(self.trash_elements['quads'][qc])
						qc += 1
					

					curves = {}
					lines = {}
					G = self.tree.get_crsec_graph()
					node_list = []

					for e in self.converted_selection():
						if e[0] not in node_list:
							node_list.append(e[0])
						if e[1] not in node_list:
							node_list.append(e[1])

					for n in node_list:
						crsec = G.nodes[n]['crsec']
						if G.nodes[n]['type'] == "bif":
							N = (self.tree._N // 2) - 1
							curve_list = []
							s = 2
							for i in range(len(crsec) // N):
								if cc > len(self.trash_elements['curves']) - 1:
									self.trash_elements['curves'].append(curve(radius=self.edge_size_sliders["mesh"].value, color = color.black, mode = 'mesh', category = 'section_edges', locked = False))
								c = self.trash_elements['curves'][cc]
								c.clear()
								c.category = 'section_edges'
								c.append(vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
								for k in range(N):
									c.append(vector(crsec[s+k, 0], crsec[s+k, 1], crsec[s+k, 2]))
								s += k + 1
								c.append(vector(crsec[1, 0], crsec[1, 1], crsec[1, 2]))
								cc+=1
								curve_list.append(c)
							curves[n] = curve_list


						else:
							if cc > len(self.trash_elements['curves']) - 1:
								self.trash_elements['curves'].append(curve(radius=self.edge_size_sliders["mesh"].value, color = color.black, mode = 'mesh', category = 'section_edges', locked = False))
							c = self.trash_elements['curves'][cc]
							c.clear()
							c.category = 'section_edges'
							for i in range(len(crsec)):
								c.append(vector(crsec[i, 0], crsec[i, 1], crsec[i, 2]))
							c.append(vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
							cc+=1
							curves[n] = [c]

					for e in self.converted_selection():
						crsec_list = G.edges[e]['crsec']
						# Longitudinal curves
						lines_list = []
						for i in range(crsec_list.shape[1]):
							if cc > len(self.trash_elements['curves']) - 1:
									self.trash_elements['curves'].append(curve(radius=self.edge_size_sliders["mesh"].value, color = color.black, mode = 'mesh', category = 'section_edges', locked = False))

							c = self.trash_elements['curves'][cc]
							c.clear()
							c.category = 'connecting_edges'
								
							# Starting point
							start_sec = G.nodes[e[0]]['crsec']
							c.append(vector(start_sec[i, 0], start_sec[i, 1], start_sec[i, 2]))
							for j in range(crsec_list.shape[0]):
								c.append(vector(crsec_list[j, i, 0], crsec_list[j, i, 1], crsec_list[j, i, 2]))
							# Ending point
							end_sec = G.nodes[e[1]]['crsec']
							l = G.edges[e]['connect'][i]
							c.append(vector(end_sec[l][0], end_sec[l][1], end_sec[l][2]))
							cc+=1
							lines_list.append(c)
						lines[e] = lines_list

						# Cross section curves
						curve_list = []
						for i in range(len(crsec_list)):
							crsec = crsec_list[i]
							if cc > len(self.trash_elements['curves']) - 1:
									self.trash_elements['curves'].append(curve(radius=self.edge_size_sliders["mesh"].value, color = color.black, mode = 'mesh', category = 'section_edges', locked = False))
							c = self.trash_elements['curves'][cc]
							c.clear()
							c.category = 'section_edges'
							for j in range(len(crsec)):
								c.append(vector(crsec[j, 0], crsec[j, 1], crsec[j, 2]))
							c.append(vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
							cc+=1
								
							curve_list.append(c)
						curves[e] = curve_list


					self.elements['mesh']['surface'] = quads
					self.elements['mesh']['section_edges'] = curves
					self.elements['mesh']['connecting_edges'] = lines

					if visible:
						self.show("mesh")
					else:
						self.hide("mesh")

					self.update_mesh_representation(self.mesh_representation_menu)




	###########################################
	########### USER INTERACTION ##############
	###########################################

	def output_message(self, str, mode = 'msg'):

		""" Print a message under the 3D view.

		Keyword arguments: 
		str -- text to print
		mode -- message mode (msg, warning, error)
		"""

		if mode == "error":
			self.text_output.text = "<b>\n" + "Error : " + str  + "</b>\n-----------"
			
		elif mode == "warning":
			self.text_output.text = "<b>\n" + "Warning : " + str  + "</b>\n-----------"
		else:
			self.text_output.text = "\n" + str  + "\n-----------"



	def keyboard_control(self, evt):

		""" Handle user keyboard input """
		if not self.running: # Wait for the execution of previous commands 

			# Actions on data mode edges
			if self.edition_mode == "full" and self.selected_edge is not None and self.selected_edge.mode == "full":
				
				if evt.key == "delete": # Delete an edge
					self.selected_edge.visible = False 
					self.tree.delete_data_edge(self.selected_edge.id, False)
					self.refresh_display("full")
					self.unselect('edge', 'full')
					self.modified["full"] = True

			if self.edition_mode == "full":
				if evt.key == "a": # Adding a regular node
					pos = scene.mouse.project(normal = scene.mouse.ray)

					if not self.slice_checkbox.checked:
						# The projection plane is determined by the closest points
						close_sph = self.get_closest_data_point(np.array([pos.x, pos.y, pos.z]), np.array([scene.mouse.ray.x, scene.mouse.ray.y, scene.mouse.ray.z]))
						pos = scene.mouse.project(normal = scene.camera.axis, point = vec(close_sph.pos.x,close_sph.pos.y, close_sph.pos.z))
						self.tree.add_data_point(np.array([pos.x, pos.y, pos.z, close_sph.radius]), idx = close_sph.id, branch = False, apply = False)
					else:
						# The projection plane is determined by the normal of the image slice
						close_sph = self.get_closest_data_point(np.array([pos.x, pos.y, pos.z]), np.array([scene.mouse.ray.x, scene.mouse.ray.y, scene.mouse.ray.z]))
						pos = scene.mouse.project(normal = self.slice_plane[0], point = self.slice_plane[1])
						self.tree.add_data_point(np.array([pos.x, pos.y, pos.z, close_sph.radius]), idx = close_sph.id, branch = False, apply = False)
					
					self.refresh_display("full")
					self.modified["full"] = True

				if evt.key == "b": # Adding a branch node
					pos = scene.mouse.project(normal = scene.mouse.ray)

					if not self.slice_checkbox.checked:
						# The projection plane is determined by the closest points
						close_sph = self.get_closest_data_point(np.array([pos.x, pos.y, pos.z]), np.array([scene.mouse.ray.x, scene.mouse.ray.y, scene.mouse.ray.z]))
						pos = scene.mouse.project(normal = scene.camera.axis, point = vec(close_sph.pos.x,close_sph.pos.y, close_sph.pos.z))
						self.tree.add_data_point(np.array([pos.x, pos.y, pos.z, close_sph.radius]), idx = close_sph.id, branch = True, apply = False)
					else:
						# The projection plane is determined by the normal of the image slice
						close_sph = self.get_closest_data_point(np.array([pos.x, pos.y, pos.z]), np.array([scene.mouse.ray.x, scene.mouse.ray.y, scene.mouse.ray.z]))
						pos = scene.mouse.project(normal = self.slice_plane[0], point = self.slice_plane[1])
						self.tree.add_data_point(np.array([pos.x, pos.y, pos.z, close_sph.radius]), idx = close_sph.id, branch = True, apply = False)

					self.refresh_display("full")
					self.modified["full"] = True

			# Actions on data mode nodes
			if self.edition_mode == "full" and self.selected_node is not None and self.selected_node.mode != "cursor":

				ids = self.selected_node.id
				
				if evt.key == "d":
					self.selected_node.radius = self.selected_node.radius - 0.01
					self.tree.modify_data_point_radius(self.selected_node.id, self.selected_node.radius, False)
					self.modified["full"] = True

				if evt.key == "u":
					self.selected_node.radius = self.selected_node.radius + 0.01
					self.tree.modify_data_point_radius(self.selected_node.id, self.selected_node.radius, False)
					self.modified["full"] = True

				if evt.key == "delete":
					self.selected_node.visible = False 
					self.tree.delete_data_point(self.selected_node.id, False)
					self.refresh_display("full")
					self.unselect('node', 'topo')
					self.modified["full"] = True

				if evt.key == "e":

					self.edge_drag = True
					self.edge_n_id = self.selected_node.id
					self.output_message("Adding edge : Please select the target node by clicking.")
					self.modified["full"] = True


			# Actions on topo mode edges
			if self.edition_mode == "topo" and self.selected_edge is not None:


				if evt.key == "u" or evt.key == "d":
					if evt.key == "u":
						eps = 0.01
					else:
						eps = -0.01

					self.tree.modify_branch_radius(self.selected_edge.id, eps, apply = False)
					self.refresh_display("full")
					self.modified["topo"] = True

				if (evt.key == "r" or evt.key == "R"): # Rotate branch

					# Get rotation angle and normal 
					normal = scene.camera.axis
					normal = np.array([normal.x, normal.y, normal.z])
					normal = normal/norm(normal)
					
					if evt.key == "r":
						alpha = 0.1
					else:
						alpha = -0.1

					self.tree.rotate_branch(normal, self.selected_edge.id, alpha)
					self.refresh_display("topo")
					self.selected_edge.color = color.green
					self.modified["topo"] = True
				
				if evt.key == "delete": # Delete branch
					self.output_message("Removing branch...")

					self.disable(True, checkboxes = True)
					edg = self.selected_edge.id
					self.tree.remove_branch(edg)
					self.refresh_display("topo")
					self.modified["topo"] = True
					self.disable(False, checkboxes = True)

			# Actions on topo mode nodes
			if self.edition_mode == "topo" and self.selected_node is not None and self.selected_node.mode != "cursor":

				if evt.key == "i": # Make inlet
					if self.tree.get_topo_graph().in_degree(self.selected_node.id) == 1 and self.tree.get_topo_graph().out_degree(self.selected_node.id)  == 0:
						self.output_message("Adding new inlet.", mode = "msg")
						self.tree.make_inlet(self.selected_node.id)
						self.refresh_display("topo")
						self.selected_node = None
						self.modified["topo"] = True
					else:
						self.output_message("This node is not an outlet, it cannot be changed to inlet.", mode = "warning")

				if evt.key == "delete": # Delete branch

					self.output_message("Removing branches.")
					n = self.selected_node.id
					self.tree.remove_branch((n, n), from_node = True)
					self.refresh_display("topo")
					self.modified["topo"] = True

			# Actions on model mode nodes
			if self.edition_mode == "model" and self.selected_node is not None and self.selected_node.mode != "cursor":

				ids = self.selected_node.id
				
				if evt.key == "d":
					self.selected_node.radius = self.selected_node.radius - 0.01
					self.tree.modify_control_point_radius(self.selected_node.id[0], self.selected_node.id[1], self.selected_node.radius)
					if self.selected_node.id[0] not in self.modified_elements['splines']: # To update mesh locally when modifications are applied
						self.modified_elements['splines'].append(self.selected_node.id[0])
					self.modified["model"] = True

				if evt.key == "u":
					self.selected_node.radius = self.selected_node.radius + 0.01
					self.tree.modify_control_point_radius(self.selected_node.id[0], self.selected_node.id[1], self.selected_node.radius)
					if self.selected_node.id[0] not in self.modified_elements['splines']: # To update mesh locally when modifications are applied
						self.modified_elements['splines'].append(self.selected_node.id[0])
					self.modified["model"] = True


			if self.edition_mode == "model" and self.selected_edge is not None:
				if self.smooth_checkboxes['radius'].checked and (evt.key == "d" or evt.key == "u"):

					if evt.key == "d":
						self.lbdr -= 0.5
						if self.lbdr < 0:
							self.lbdr = 0
						self.lbdr_text.text = str(round(self.lbdr, 2))

					if evt.key == "u":
						self.lbdr += 0.5
						self.lbdr_text.text = str(round(self.lbdr, 2))

					self.running = True
					self.tree.smooth_spline(self.selected_edge.id, self.lbdr, radius = True)
					self.update_spline(self.selected_edge)

					if self.selected_edge.id not in self.modified_elements['splines']: # To update mesh locally when modifications are applied
						self.modified_elements['splines'].append(self.selected_edge.id)
					self.running = False
					self.modified["model"] = True

			
				if self.smooth_checkboxes['spatial'].checked and (evt.key == "d" or evt.key == "u"):

					if evt.key == "d":
						self.lbds -= 0.5
						if self.lbds < 0:
							self.lbds = 0
						self.lbds_text.text = str(round(self.lbds, 2))

					if evt.key == "u":
						self.lbds += 0.5
						self.lbds_text.text = str(round(self.lbds, 2))

					self.running = True
					self.tree.smooth_spline(self.selected_edge.id, self.lbds, radius = False)
					self.update_spline(self.selected_edge)
					if self.selected_edge.id not in self.modified_elements['splines']: # To update mesh locally when modifications are applied
						self.modified_elements['splines'].append(self.selected_edge.id)
					self.running = False
					self.modified["model"] = True
				

			if self.edition_mode == "mesh":

				if evt.key == "a":
					# Select all edges for meshing
					self.mesh_selection = []
					for e in self.tree.get_model_graph().edges():
						self.mesh_selection.append(e)
						self.elements['model']['edges'][e].color = color.red

				elif evt.key == "n":
					# Deselect all edges for meshing
					self.mesh_selection = []
					for e in self.tree.get_model_graph().edges():
						self.elements['model']['edges'][e].color = color.black

			if self.edition_mode == "crop":

				if evt.key == "a":
					# Select all edges for meshing
					self.crop_selection = []
					for e in self.tree.get_topo_graph().edges():
						self.crop_selection.append(e)
						self.elements['topo']['edges'][e].color = color.red

				elif evt.key == "n":
					# Deselect all edges for meshing
					self.crop_selection = []
					for e in self.tree.get_topo_graph().edges():
						self.elements['topo']['edges'][e].color = color.black

			if self.edition_mode == "pathology" and evt.key == "n":
				self.next_template()


	###########################################
	############# NODE EDITION ################
	###########################################

	def update_edition_mode(self):

		""" Update the edition mode chosen by the user in the edition menu """
		selected = self.edition_menu.selected
		if selected == "data":
			selected = "full"

		if self.modified[self.edition_mode]:
			# Do nothing as the changes must be validated 
			self.output_message("Please apply the modifications by clicking 'Apply' before leaving this edition mode.","error")
			if self.edition_mode == "full":
				self.edition_menu.selected = "data"
			else:
				self.edition_menu.selected = self.edition_mode

		else:

			self.unselect("node", self.edition_mode)
			self.unselect("edge", self.edition_mode)

			if self.edition_mode == "mesh":
				self.show_mesh_selection(False)

			if self.edition_mode == "crop":
				self.show_crop_selection(False)

			if self.edition_mode == "pathology" and selected != "pathology":
				self.show_hide_template(False)
				self.disable(disabled=False, checkboxes = True)

			self.edition_mode = selected
			if self.edition_mode == "mesh":
				if self.tree.get_model_graph() is None:
					self.edition_menu.selected = "off"
					self.edition_mode = "off"
					self.output_message("No model found. The meshing selection mode is not available. Please compute the model first.", "warning")

				else:
					self.show_mesh_selection(True)
					self.output_message("Edition mode switched to " + self.edition_mode + ". The edges to mesh (in red) can be selected from the model representation. Press 'a' to select all and 'n' to deselect all. Click on an edge to select/unselect it." )
			
			if self.edition_mode == "crop":
				self.show_crop_selection(True)
				

			if self.edition_mode == "pathology":
				self.show_hide_template(True)
				self.disable(disabled=True, checkboxes = True)
			

			#self.output_message("Edition mode switched to " + self.edition_mode + ".")


	def resample_nodes(self, b):

		""" Resample nodes """
		p = b.value
		self.tree.low_sample(p, apply = False)
		self.refresh_display("full")


	def move(self, evt):

		""" Move the selected node with mouse cursor """
		if self.selected_node is not None and self.drag:
			if self.selected_node.mode == "cursor":
				if self.running == False:
					self.running = True
					self.selected_node.pos = scene.mouse.project(normal = scene.mouse.ray, point = self.selected_node.pos)#scene.mouse.pos #evt.pos
					self.running = False

			elif self.selected_node.mode == "marker":
				if self.running == False:
					self.running = True
					pos = scene.mouse.project(normal = scene.mouse.ray, point = self.selected_node.pos)#scene.mouse.pos #evt.pos
					pos = np.array([pos.x, pos.y, pos.z])
					spl = self.tree.get_model_graph().edges[self.pathology_edg]["spline"]
					t = spl.project_point_to_centerline(pos)
					new_pos = spl.point(t)
					self.selected_node.pos = vec(new_pos[0], new_pos[1], new_pos[2])
					self.running = False

			else:

				if self.edition_mode == "pathology":
					if self.running == False:
						self.running = True
						new_pos = scene.mouse.project(normal = vec(0,0,-1), point = scene.center)
						self.selected_node.pos = new_pos
						# Move the curve point
						self.elements["pathology"]["edges"][-1].modify(self.selected_node.id, new_pos)
						if self.selected_node.id == 0:
							self.elements["pathology"]["edges"][-1].modify(-1, new_pos)


						self.move_edges(self.edition_mode, self.selected_node.id, self.selected_node.pos)
							
						self.running = False

				elif self.edition_mode == "full" or self.edition_mode == "model":

					if self.running == False:
						self.running = True
						self.selected_node.pos = scene.mouse.project(normal = scene.mouse.ray, point = self.selected_node.pos)#scene.mouse.pos #evt.pos

						if self.selected_node.mode != "cursor":
							self.move_edges(self.edition_mode, self.selected_node.id, self.selected_node.pos)
							
						self.running = False
				else:
					pass




	def move_edges(self, edition_mode, ids, new_pos):
		""" Move the edges as the nodes are modified """
		if edition_mode == "full":

			# Move the associated edge
			G = self.tree.get_full_graph()
			in_edges = list(G.in_edges(ids))
			out_edges = list(G.out_edges(ids))

			for e in in_edges:
				# Modify cylinder
				axis = new_pos - self.elements["full"]["edges"][e].pos
			
				self.elements["full"]["edges"][e].axis = axis
				self.elements["full"]["edges"][e].length = mag(axis)

			for e in out_edges:
				# Modify cylinder
				e1 = self.elements["full"]["edges"][e].pos + self.elements["full"]["edges"][e].axis / mag(self.elements["full"]["edges"][e].axis)*self.elements["full"]["edges"][e].length
				axis = e1 - new_pos
				self.elements["full"]["edges"][e].pos = new_pos
				self.elements["full"]["edges"][e].axis = axis
				self.elements["full"]["edges"][e].length = mag(axis)

		elif edition_mode == "model":

			c = self.elements['model']['control_edges'][ids[0]]
			c.modify(ids[1], pos = new_pos) # Move control polygon

			# Move spline
			spl = copy.deepcopy(self.tree.get_model_graph().edges[ids[0]]['spline'])
			new_P = spl.get_control_points().copy()
			for i in range(c.npoints):
				new_P[i,0] = c.point(i)['pos'].x
				new_P[i,1] = c.point(i)['pos'].y
				new_P[i,2] = c.point(i)['pos'].z
			spl.set_control_points(new_P.tolist())
			coords = spl.get_points()
			coords = np.vstack((coords[0, :], coords[::self.display_spline_step, :], coords[-1, :]))
			cspl = self.elements['model']['edges'][ids[0]]
			cspl.clear()

			for i in range(len(coords)):
				cspl.append(pos=vector(coords[i][0], coords[i][1], coords[i][2]), radius=0.2)


	def update_spline(self, curve):

		""" Change the spline smoothing parameter and update render. """

		ids = curve.id
		spl = self.tree.get_model_graph().edges[ids]['spline']
		coords = spl.get_control_points()

		c = self.elements["model"]["control_edges"][ids]

		# Update control points and control polygon
		for i in range(len(coords)):
			c.modify(i, vector(coords[i][0], coords[i][1], coords[i][2]))
			self.elements["model"]["control_nodes"][ids][i].pos = vector(coords[i][0], coords[i][1], coords[i][2])
			self.elements["model"]["control_nodes"][ids][i].radius = coords[i][3]

		# Update spline
		coords = spl.get_points()
			
		curve.clear()
		for i in range(len(coords)):
			curve.append(pos=vector(coords[i][0], coords[i][1], coords[i][2]), color=color.green, radius=0.2)


		
	def drop(self, evt):

		""" Drop the node when mouse is released """

		# Save new position in changes
		if self.selected_node is not None:

			if self.selected_node.mode == "cursor":
				self.unselect("node")
				self.drag = False

			if self.selected_node.mode == "marker":
				self.unselect("node")
				self.drag = False

			elif self.selected_node.mode == "full":
				self.drag = False
				pos = self.selected_node.pos
				self.tree.modify_data_point_coords(self.selected_node.id, np.array([pos.x, pos.y, pos.z]), False)

				self.unselect("node")
				self.modified["full"] = True

			elif self.selected_node.mode == "pathology":
				self.drag = False

			elif self.selected_node.mode == "model":

				self.drag = False
				pos = self.selected_node.pos
				self.tree.modify_control_point_coords(self.selected_node.id[0], self.selected_node.id[1], np.array([pos.x, pos.y, pos.z]))
				if self.selected_node.id[0] not in self.modified_elements['splines']: # To update mesh locally when modifications are applied
					self.modified_elements['splines'].append(self.selected_node.id[0])
				self.unselect("node")
				self.modified["model"] = True

			else:
				pass
				


	def select(self):

		""" Select a node or edge by mouse clicking """

		obj = scene.mouse.pick
		if obj is not None:
		
			if self.edge_drag:
				if type(obj) == sphere and obj.mode == "full":

					self.output_message("Adding new edge : (" + str(self.edge_n_id) + "," + str(obj.id) + ")")
					self.tree.add_data_edge(self.edge_n_id, obj.id, False)
					self.edge_drag = False
					self.refresh_display("full")
					self.unselect('node', 'full')
				else:
					self.edge_drag = False

			else:

				if obj.mode == "cursor":
					
					self.selected_node = obj
					self.drag = True
					self.output_message("Origin cursor selected. Move it to the center of the image cut.")

				elif obj.mode == "marker":
					
					self.selected_node = obj
					self.drag = True
					self.output_message("Marker selected. Move it to the desired position.")

				else:

					if self.edition_mode == "pathology":
						if type(obj) == sphere:

							self.unselect("node")
							self.unselect("edge")

							self.selected_node = obj
							self.selected_node.color = color.yellow
							self.drag = True

					elif self.edition_mode =="full":
						if obj.mode == "full":
							if type(obj) == sphere:
				
								self.unselect("node")
								self.unselect("edge")

								self.selected_node = obj
								self.selected_node.color = color.yellow
								self.drag = True
								self.output_message("Node " + str(obj.id) + " selected. Move it using the mouse. Press 'u' or 'd' to increase or lower the radius, 'suppr.' to delete the node, and 'e' to start a new edge.")

							if type(obj) == cylinder:
				
								self.unselect("edge")
								self.unselect("node")

								self.selected_edge = obj
								self.selected_edge.color = color.green
								self.output_message("Edge " + str(obj.id) + " selected. Press 'suppr.' to delete this edge. Press 'u' or 'd' to increase or lower the radius of the branch.")

					elif self.edition_mode =="topo":
						if obj.mode == "topo":
							if type(obj) == sphere:
				
								self.unselect("node")
								self.unselect("edge")

								self.selected_node = obj
								self.selected_node.color = color.yellow
								self.output_message("Node " + str(obj.id) + " selected. This node can be changed to inlet by pressing 'i'.")

							if type(obj) == curve:

								self.unselect("edge")
								self.unselect("node")

								self.selected_edge = obj
								self.selected_edge.color = color.green

								self.output_message("Edge " + str(obj.id) + " selected. Press suppr. to cut the corresponding branch. Press r (resp. R) to rotate the branch clockwise (resp. counterclockwise).")

					elif self.edition_mode =="model":
						if obj.mode == "model":
							if type(obj) == sphere and obj.category == "control_nodes":

								nb_ctrl = len(self.elements['model']['control_nodes'][obj.id[0]]) 
								if self.tree.get_model_graph().nodes[obj.id[0][0]]['type']!= "bif" and  self.tree.get_model_graph().nodes[obj.id[0][1]]['type']!= "bif" and obj.id[1] not in [0, 1, nb_ctrl-1, nb_ctrl-2]:
							
									self.unselect("node")
									self.unselect("edge")

									self.selected_node = obj
									self.selected_node.color = color.yellow
									self.drag = True
									self.output_message("Node "  + str(obj.id) + " selected. Move it using the mouse. Press 'u' or 'd' to increase or lower the radius. Press 'suppr.' to delete it.")


							if type(obj) == curve :
						
								self.unselect("edge")
								self.unselect("node")

								self.selected_edge = obj
								self.selected_edge.color = color.green

								self.smooth_checkboxes['spatial'].disabled = False
								self.smooth_checkboxes['radius'].disabled = False
								self.pathology_checkbox.disabled = False

								self.output_message("Spline " + str(obj.id) +  " selected. Check a smoothing box and use the slider smooth or unsmooth.")

					elif self.edition_mode == "mesh":
						if type(obj) == curve:
							if obj.id not in self.mesh_selection:
								self.mesh_selection.append(obj.id)
								self.elements['model']['edges'][obj.id].color = color.red
							else:
								self.mesh_selection.remove(obj.id)
								self.elements['model']['edges'][obj.id].color = color.black

					elif self.edition_mode == "crop":
						if type(obj) == curve:
							if obj.id not in self.crop_selection:
								self.crop_selection.append(obj.id)
								self.elements['topo']['edges'][obj.id].color = color.red
							else:
								self.crop_selection.remove(obj.id)
								self.elements['topo']['edges'][obj.id].color = color.black
					else:
						pass  

					

	def unselect(self, elt = "node", mode = None):

		""" Unselect the selected edge or elements if it is associated to a given mode.

		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
		"""

		if elt == "node":
			if self.selected_node is not None:
				if mode is None or self.selected_node.mode == mode:
					if self.selected_node.mode == "full" or self.selected_node.mode == "topo":
						self.selected_node.color = self.node_color(self.selected_node.id, self.selected_node.mode)
					if self.selected_node.mode == "model":
						self.selected_node.color = color.gray(0.5)
					if self.selected_node.mode == "pathology":
						self.selected_node.color = color.red

					self.selected_node = None

		else:
			if self.selected_edge is not None:
				if mode is None or self.selected_edge.mode == mode:
					if type(self.selected_edge) == curve:

						self.selected_edge.color = color.black
						self.selected_edge = None

						self.smooth_checkboxes['spatial'].disabled = True
						self.smooth_checkboxes['radius'].disabled = True
						self.pathology_checkbox.disabled = True
						self.smooth_checkboxes['spatial'].checked = False
						self.smooth_checkboxes['radius'].checked = False
						self.pathology_checkbox.checked = False
						self.lbds_text.text = ""
						self.lbdr_text.text = ""

					else:
						if self.selected_edge.mode == "full":
							self.selected_edge.color = color.black

						elif self.selected_edge.id[0] in self.elements["topo"]["edges"].keys():
							for elt in self.elements["topo"]["edges"][self.selected_edge.id[0]]:
								elt.color = color.black

						self.selected_edge = None




	###########################################
	############# EDGE EDITION ################
	###########################################


	def select_smooth_parameter(self, b):

		""" Update the user choice for the smoothing parameter (radius or spatial) to work with """


		if b.checked:

			ids = self.selected_edge.id
			lbd = self.tree.get_model_graph().edges[ids]['spline'].get_lbd()

			if b.parameter == "spatial":
				if len(lbd) == 0:
					self.output_message("There are no models associated with this spline.", mode = "warning")
					self.smooth_checkboxes["spatial"].checked = False
				else:
					self.output_message("Spatial smoothing enabled. Press 'u' to smooth the model and 'd' to unsmooth it.")
					self.smooth_checkboxes["radius"].checked = False
					self.lbdr_text.text = ""
					self.lbds = lbd[0]
					self.lbds_text.text = str(round(self.lbds, 2))

			else:

				if len(lbd) < 2:
					self.output_message("There are no models associated with this spline.", mode = "warning")
					self.smooth_checkboxes["radius"].checked = False
				else:
					self.output_message("Radius smoothing enabled. Press 'u' to smooth the model and 'd' to unsmooth it.")
					self.smooth_checkboxes["spatial"].checked = False
					self.lbds_text.text = ""
					self.lbdr = lbd[1]
					self.lbdr_text.text = str(round(self.lbdr, 2))
		else:
			if b.parameter == "spatial":
				self.lbds_text.text = ""
			else:
				self.lbdr_text.text = ""




	###########################################
	############# MESH EDITION ################
	###########################################


	def update_mesh_parameters(self, b):

		""" Update user choice of meshing parameters """

		if b.parameter == "N":
			self.N = int(self.parameters_winput['N'].text)
			self.output_message("Number of cross section nodes set to " + str(self.N) + ".")
			
		elif b.parameter == "d":
			self.d = float(self.parameters_winput['d'].text)
			self.output_message("Density of cross section set to " + str(self.d) + ".")
			
		else:
			try:
				self.target_mesh = pv.read(self.parameters_winput['path'].text)
				self.output_message("Target mesh is now : " + self.parameters_winput['path'].text + ".")
				self.parameters_winput['path'].text = ""
		
			except FileNotFoundError:	
				self.output_message("The target mesh file does not exist.", "error")



	def deform_mesh(self):

		""" Deform mesh to a given target mesh """

		if self.target_mesh is not None:
			self.output_message("Mesh deformation...")
			self.disable(True, checkboxes = True)

			self.tree.deform_surface_to_mesh(self.target_mesh)

			self.refresh_display("mesh")
			
			self.disable(False, checkboxes = True)
			self.output_message("Mesh deformation complete!")


	def check_mesh(self):
		""" Checks the mesh quality and display the mesh segments not compatible with simulation in red """

		self.disable(True, checkboxes = True)

		if self.check_state == False:

			self.output_message("Checking mesh...")
			field, failed_edges, failed_bifs = self.tree.check_mesh(edg = self.converted_selection())
			
			for i in range(len(field)):
				if field[i] == 1:

					self.elements["mesh"]["surface"][i].v0.color = color.red
					self.elements["mesh"]["surface"][i].v1.color = color.red
					self.elements["mesh"]["surface"][i].v2.color = color.red
					self.elements["mesh"]["surface"][i].v3.color = color.red


			self.check_state = True
			self.check_mesh_button.text = "Uncheck"
			self.output_message(str(len(failed_edges)) +" vessels and " + str(len(failed_bifs)) + " furcations failed the test.")

		else:
			for i in range(len(self.elements["mesh"]["surface"])):
				
				self.elements["mesh"]["surface"][i].v0.color = color.gray(0.75)
				self.elements["mesh"]["surface"][i].v1.color = color.gray(0.75)
				self.elements["mesh"]["surface"][i].v2.color = color.gray(0.75)
				self.elements["mesh"]["surface"][i].v3.color = color.gray(0.75)

			self.check_state = False
			self.check_mesh_button.text = "Check"

		self.disable(False, checkboxes = True)


	def close_mesh(self):
		""" Checks the mesh quality and display the mesh segments not compatible with simulation in red """

		self.disable(True, checkboxes = True)

		if self.closing_state == False:

			self.output_message("Closing surface inlets and outlets...")
			self.tree.close_surface(edg = self.converted_selection())
			self.refresh_display("mesh")
			self.closing_state = True
			self.close_mesh_button.text = "Unclose"
			self.output_message("Surface closed.")

		else:
			self.tree.open_surface(edg = self.converted_selection())
			self.refresh_display("mesh")
			self.closing_state = False
			self.close_mesh_button.text = "Close"
			self.output_message("Surface opened.")

		self.disable(False, checkboxes = True)


	def disable_close_check(self):

		if self.closing_state:
			self.closing_state = False
			self.close_mesh_button.text = "Close"

		if self.check_state:
			self.check_state = False
			self.check_mesh_button.text = "Check"


	def manage_extensions(self):

		""" Add and remove inlet and outlet extensions """
		self.disable(True, checkboxes = True)

		if self.extension_state == False:

			self.extension_state = True
			self.output_message("Adding inlet and outlet extensions...")
			self.tree.add_extensions(edg = self.converted_selection())
			self.refresh_display("model")
			#self.refresh_display("mesh")
			self.extension_button.text = "Unextend"
			self.output_message("Extensions added.")
			

		else:
			self.extension_state = False
			self.output_message("Removing inlet and outlet extensions...")
			self.tree.remove_extensions(edg = self.converted_selection())
			self.refresh_display("model")
			#self.refresh_display("mesh")
			self.extension_button.text = "Extend"
			self.output_message("Extensions removed.")
			
		self.disable(False, checkboxes = True)


	def mesh_volume(self):

		self.disable(True, checkboxes = True)
		self.output_message("Meshing the volume...")
		self.tree.mesh_volume(edg = self.converted_selection())
		self.output_message("Volume meshed.")
		self.disable(False, checkboxes = True)

	def mesh_surface(self):
		self.output_message("Meshing surface.")

		self.disable(True, checkboxes = True)

		self.tree.compute_cross_sections(self.N, self.d)
		self.tree.mesh_surface(edg = self.converted_selection())
		self.closing_state = False # The surface will be opened after the recomputation
		self.close_mesh_button.text = "Close"

		self.refresh_display("mesh")
		self.disable_close_check()

		self.disable(False, checkboxes = True)



	###########################################
	############# CUSTOMIZATION ###############
	###########################################
		

	def update_edge_size(self):

		""" Change edge size using slider """

		def set_edge_size(elt, args):
			if type(elt) == curve:
				if args[0] > 0:
					elt.visible = True
					for n in range(elt.npoints):
						elt.modify(n, radius=args[0])
				else:
					elt.visible = False

			else:
				elt.radius = args[0]

		for mode in self.edge_size_sliders.keys():
			if self.edge_size_sliders[mode].value != self.edge_size[mode]:
	
				if mode == "mesh":
					categories = ['section_edges', 'connecting_edges']
				elif mode == "model":
					categories = ['edges', 'control_edges']

				else:
					categories = ['edges']

				self.apply_function(mode, func=set_edge_size, args=[self.edge_size_sliders[mode].value], categories = categories)
				self.edge_size[mode] = self.edge_size_sliders[mode].value



	def update_node_size(self, b):

		""" Change node size using slider """
		mode = b.mode

		def set_node_size(elt, args):
			elt.radius = args[0]

		
		if self.node_size_sliders[mode].value != self.node_size[mode]:

			if mode == "model":
				categories = ['nodes', 'control_nodes']
			else:
				categories = ['nodes']

			self.apply_function(mode, func=set_node_size, args=[self.node_size_sliders[mode].value], categories = categories)
			self.node_size[mode] = self.node_size_sliders[mode].value



	def update_opacity_state(self):

		""" Change opacity using slider """

		def set_opacity(elt, args):
			elt.opacity = args[0]

		for mode in self.opacity_sliders.keys():
			if self.opacity_sliders[mode].value != self.opacity_value[mode]:
				self.apply_function(mode, func=set_opacity, args=[self.opacity_sliders[mode].value])
				self.opacity_value[mode] = self.opacity_sliders[mode].value



	












