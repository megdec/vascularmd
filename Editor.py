import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imt
from mpl_toolkits.mplot3d import Axes3D  # 3D display
import pickle
import pyvista as pv
from scipy.spatial import ckdtree
from vpython import *
import os
from numpy.linalg import norm
import copy
import nibabel as nib

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
		slider_right_margin = 90

		self.scene = scene

		self.elements = {'full' : {}, 'topo' : {}, 'model' : {}, 'mesh' : {}} 

		scene.append_to_caption('\n\nEdit  ')
		self.edition_menu = menu(choices = ['off', 'data', 'topo', 'model', 'mesh'], selected = 'off', index=0, bind = self.update_edition_mode)
		self.edition_mode = 'off'

		scene.append_to_caption('\nImport centerline  ')

		self.centerline_file_winput = winput(text="", bind = self.reset_scene, width=200)

		scene.append_to_caption('\nImport image  ')
		self.slice_button = button(text= "Cut slice  " , bind=self.compute_slice, disabled = True)
		self.cursor_checkbox = checkbox(text= "Show origin  " , bind=self.update_visibility_cursor, checked = False)
		self.slice_checkbox = checkbox(text= "Show slice" , bind=self.update_visibility_slice, checked = False)
		self.slice_checkbox.disabled = True
		self.cursor = sphere(pos=scene.center, color=color.yellow, radius=2, mode = "cursor", visible = False)
		self.slice = None

		scene.append_to_caption('\tPath ')
		self.image_file_winput = winput(text="", bind = self.load_image, width=200)

		scene.append_to_caption('\nExport  ')

		self.save_button = button(text = "Save", bind=self.save)
		self.save_menu = menu(choices = ['centerline', 'model', 'mesh'], selected = 'centerline', index=0, bind = self.do_nothing)
		self.save_directory = ""
		scene.append_to_caption('\tOutput directory ')
		self.save_winput = winput(text="", bind = self.update_save_directory, width=200)
		self.save_filename = "vascular_network"
		scene.append_to_caption('\tOutput filename ')
		self.save_filename_winput = winput(text="vascular_network", bind = self.update_save_filename, width=200)


		scene.append_to_caption('\n\n')

		# Check boxes
		self.checkboxes = {'full' : checkbox(text= "Data  ", bind=self.update_visibility_state, checked=True, mode = "full")}
		self.update_buttons = {'full' : button(text = "Update", bind=self.update_graph, mode = 'full')}
		self.reset_buttons = {'full' : button(text = "Reset", bind=self.reset_graph, mode = 'full')}
		scene.append_to_caption('\t\t\t\t\t')
		self.checkboxes['topo'] = checkbox(text= "Topology  ", bind=self.update_visibility_state, checked = False, mode = "topo")
		self.update_buttons['topo'] = button(text = "Update", bind=self.update_graph, mode = 'topo')
		self.reset_buttons['topo'] = button(text = "Reset", bind=self.reset_graph, mode = 'topo')
		scene.append_to_caption('\t\t\t\t')
		self.checkboxes['model'] = checkbox(text= "Model  ", bind=self.update_visibility_state, mode = "model", checked = False)
		self.update_buttons['model'] = button(text = "Update", bind=self.update_graph, mode = 'model')
		self.reset_buttons['model'] = button(text = "Reset", bind=self.reset_graph, mode = 'model')
		scene.append_to_caption('\t\t\t\t\t')
		self.checkboxes['mesh'] = checkbox(text= "Mesh  " , bind=self.update_visibility_state, mode="mesh", checked = False)
		self.update_buttons['mesh'] = button(text="Update", bind=self.update_graph, mode = 'mesh')
		self.reset_buttons['mesh'] = button(text="Reset", bind=self.reset_graph, mode = 'mesh')

		self.deform_mesh_button = button(text="Deform", bind=self.deform_mesh)
		self.check_mesh_button = button(text=" Check ", bind=self.check_mesh)
		self.check_state = False

		scene.append_to_caption("\n\nOpacity\t\t\t\t\t\t\t\t\t")

		self.angle_checkbox_topo = checkbox(text="Show angles", bind=self.update_visibility_angle, checked=False, mode = "topo")
		scene.append_to_caption("\t\t\t\t\t\t\t")

		# Display bifurcations and control points
		self.angle_checkbox_model = checkbox(text="Show angles", bind=self.update_visibility_angle, checked=False, mode = "topo")
		scene.append_to_caption("\t")
		self.furcation_checkbox = checkbox(text="Show furcations", bind=self.update_visibility_furcations, checked=False)

		scene.append_to_caption('\t\t\t\tDisplay ')

		self.mesh_representation_menu = menu(choices = ['default', 'wireframe', 'sections', 'solid'], selected = 'default', index=0, bind = self.update_mesh_representation)

		scene.append_to_caption('\n')

		# Transparency slides
		self.opacity_sliders  = {'full' : slider(bind = self.update_opacity_state, value = 1, length = slider_length, width = slider_width, right = slider_right_margin)}
		#self.opacity_sliders['topo'] = slider(bind = self.update_opacity_state, value = 1, length = slider_length, width = slider_width, right = slider_right_margin - 3)
		scene.append_to_caption('\t\t\t\t\t\t\t\t\t\t')
		self.opacity_value = {'full' : 1}

		self.control_pts_checkbox = checkbox(text="Show ctrl pts", bind=self.update_visibility_control_pts, checked=False)
		scene.append_to_caption("\t")
		self.control_radius_checkbox = checkbox(text="Show ctrl radius", bind=self.update_visibility_control_radius, checked=False)

		# Size sliders
		scene.append_to_caption('\nEdge radius\t\t\t\t\t\t\t\tEdge radius\t\t\t\t\t\t\t\tEdge radius\t\t\t\t\t\t\t\tEdge radius\n')
		
		self.edge_size_sliders = {'full' :  slider(bind = self.update_edge_size, value = 0.2, min=0, max = 0.5, length=slider_length, width = slider_width, right = slider_right_margin, mode = "full")}
		self.edge_size_sliders['topo'] = slider(bind = self.update_edge_size, value = 0.2, min=0, max = 0.5, length=slider_length, width = slider_width, right = slider_right_margin, mode  = "topo")
		self.edge_size_sliders['model'] = slider(bind = self.update_edge_size, value = 0.2, min=0, max = 0.5, length=slider_length, width = slider_width, right = slider_right_margin, mode  = "model")
		self.edge_size_sliders['mesh'] = slider(bind = self.update_edge_size, value = 0.05, min=0, max = 0.2, length=slider_length, width = slider_width, right = slider_right_margin, mode  = "mesh")

		self.edge_size = {'full' : 0.2, 'topo' : 0.2, 'model': 0.2, 'mesh' : 0.05}
		
		scene.append_to_caption('\nResample\t\t\t\t\t\t\t\tNode radius\t\t\t\t\t\t\t\tNode radius\n')
		self.node_size_sliders = {'full' : slider(bind = self.resample_nodes, value = 1, min=0, max = 1, length=slider_length, width = slider_width, left= 10, right = slider_right_margin -10, mode  = "full")}
		self.node_size_sliders['topo'] = slider(bind = self.update_node_size, value = 0.5, min=0, max = 1, length=slider_length, width = slider_width, left= 10, right = slider_right_margin -10, mode  = "topo")
		self.node_size_sliders['model'] = slider(bind = self.update_node_size, value = 0.5, min=0, max = 1, length=slider_length, width = slider_width, right = slider_right_margin, mode  = "model")


		self.node_size = {'topo' : 0.5, 'model' : 0.5}


		scene.append_to_caption('Nb nodes (nx8) ')
		self.parameters_winput = {'N' : winput(text=str(24), bind = self.update_mesh_parameters, width=50, parameter = 'N')}
		scene.append_to_caption('\tSection density [0,1] ')
		self.parameters_winput['d'] = winput(text=str(0.2), bind = self.update_mesh_parameters, width=50, parameter = 'd')
		scene.append_to_caption('\n')
		
		scene.append_to_caption('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t') 
		self.smooth_checkboxes = {'spatial' : checkbox(text= "Smooth spatial  ", bind = self.select_smooth_parameter, checked = False, parameter = 'spatial')}
		self.smooth_checkboxes['radius']  = checkbox(text= "Smooth radius", bind = self.select_smooth_parameter, checked = False, parameter = 'radius')

		scene.append_to_caption('\t\t\t Target mesh path ')
		self.parameters_winput['path'] = winput(text="", bind = self.update_mesh_parameters, width=200, parameter = 'path')
		self.target_mesh = None

		scene.append_to_caption('\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t') 
		
		self.lbds = 0
		scene.append_to_caption('Lbd spatial : ')
		self.lbds_text = wtext(text="")
		self.lbdr = 0
		scene.append_to_caption('\t\tLbd radius : ')
		self.lbdr_text = wtext(text="")
		scene.append_to_caption('\t\t\t\t') 

		#scene.append_to_caption('Max. projection distance  ')
		#self.parameters_winput['search_dist'] = winput(text=str(40), bind = self.update_mesh_parameters, width=50, parameter = 'search_dist')
		
		self.N = 24
		self.d = 0.2
		self.display_spline_step = 10
		self.search_dist = 40
		scene.bind('click', self.select)
		scene.bind('mousedown', self.select)
		scene.bind('mousemove', self.move)
		scene.bind('mouseup', self.drop)
		scene.bind('keydown', self.keyboard_control)

		self.selected_node = None
		self.selected_edge = None
		self.drag = False
		self.running = False
		self.modified_elements = {'full' : {'move' : [], 'add' : [], 'delete' : []}, 'topo' : {'delete' : [], 'move' : [], 'merge' : []}, 'model' : {'move' : [], 'add' : [], 'delete' : [], 'lambda' : []}, 'mesh' : {'move' : [], 'parameter' : [], 'deform' : [], 'crsec' : []}} # [categories, id (id dict, if not index), new pos, new_radius]

		# Disable elements
		self.disable(True, checkboxes = False)
		# Generate full graph
		self.create_elements('full')
		self.disable(False, ["full"])



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


	def barycenter_finder(self):

		""" Finds the scene barycenter by averaging the network nodes """

		coords = list(nx.get_node_attributes(self.tree.get_full_graph(), 'coords').values())
		barycenter = sum(coords) / len(coords)
		return barycenter[:3]
		

	def update_visibility_state(self, b):
		
		""" Show / hide network representation when the corresponding checkbox is checked / unchecked. """

		mode = b.mode

		if mode == 'model':
			categories = ['edges', 'nodes']
		elif mode == 'topo':
			categories = ['edges', 'nodes']
		else:
			categories = []

		if b.checked:
			self.show(mode, categories)
			self.disable(False, [mode], checkboxes = False)
		else:
			self.hide(mode)
			self.disable(True, [mode], checkboxes = False)

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

		def fill_patch(img, c):
			if c[0] > 0 and c[0]<img.shape[0] and c[1] > 0 and c[1] < img.shape[1] and c[2] > 0 and c[2] < img.shape[2]:
				return img[c[0], c[1], c[2]]
			else:
				return 0


		self.output_message("Cutting slice in the medical image.")
		self.disable(True)

		img = np.array(self.image.dataobj)
		pix_dim = self.image.header['pixdim'][1:4]
		dim = 20
		dist = 20

		vec_norm = scene.camera.axis
		center = self.cursor.pos

		pt = np.array([center.x, center.y, center.z])
		tg = np.array([vec_norm.x, vec_norm.x, vec_norm.z])
		tg = tg/norm(tg)
		nr = cross(np.array([0, 0, 1]), tg) # Normal vector
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
			imt.imsave('image.jpg', patch, cmap='gray')

			v1 = vertex(pos=vector(pos1[0],pos1[1],pos1[2]), normal=vector(0,0,1), texpos=vector(0,1,0), shininess= 0)
			v2 = vertex(pos=vector(pos2[0],pos2[1],pos2[2]), normal=vector(0,0,1), texpos=vector(0,0,0), shininess= 0)
			v3 = vertex(pos=vector(pos3[0],pos3[1],pos3[2]), normal=vector(0,0,1), texpos=vector(1,0,0), shininess= 0)
			v4 = vertex(pos=vector(pos4[0],pos4[1],pos4[2]), normal=vector(0,0,1), texpos=vector(1,1,0), shininess= 0)
				
			Q = quad(vs=[v1, v2, v3, v4], texture='image.jpg')
			self.slice = Q
			self.nb_im = 0
	
		else:
			self.nb_im = self.nb_im + 1
			imt.imsave("image" + str(self.nb_im) +".jpg", patch, cmap='gray')
		
			self.slice.vs[0].pos = vector(pos1[0],pos1[1],pos1[2])
			self.slice.vs[1].pos = vector(pos2[0],pos2[1],pos2[2])
			self.slice.vs[2].pos = vector(pos3[0],pos3[1],pos3[2])
			self.slice.vs[3].pos = vector(pos4[0],pos4[1],pos4[2])
			
			self.slice.texture = "image" + str(self.nb_im) +".jpg"
			#self.slice = quad(vs=self.slice.vs, texture='image2.jpg')
		

		self.disable(False)
		self.slice_checkbox.disabled = False
		self.slice_checkbox.checked = True



	def update_visibility_angle(self, checkbox):


		if checkbox.checked: # Create angles labels
			angles = self.tree.angle(1.0, mode=checkbox.mode)

			if "angles" not in self.elements["model"].keys():
				self.elements[checkbox.mode]["angles"] = []

				for a in angles:
					L = label(pos=vec(a[1][0], a[1][1], a[1][2]), text=str(a[2])+ "°", box = False)
					self.elements[checkbox.mode]["angles"].append(L)

			else:
				for i in range(len(angles)):
					a = angles[i]
					if i < len(self.elements["model"]["angles"]):
						self.elements[checkbox.mode]["angles"][i].visible = True
						self.elements[checkbox.mode]["angles"][i].pos = vec(a[1][0], a[1][1], a[1][2])
						self.elements[checkbox.mode]["angles"][i].text = str(a[2])
					else:

						L = label(pos=vec(a[1][0], a[1][1], a[1][2]), text=str(a[2])+ "°", box = False)
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
			self.update_buttons[m].disabled = disabled

			if m in self.reset_buttons.keys():
				self.reset_buttons[m].disabled = disabled

			if m in self.opacity_sliders.keys():
				self.opacity_sliders[m].disabled = disabled
			if m in self.edge_size_sliders:
				self.edge_size_sliders[m].disabled = disabled
			if m in self.node_size_sliders:
				self.node_size_sliders[m].disabled = disabled

			if m == "topo":
				self.angle_checkbox_topo.disabled = disabled
				self.angle_checkbox_topo.checked = False

			if m == "model":
				self.control_pts_checkbox.disabled = disabled
				self.control_pts_checkbox.checked = False 
				if disabled:
					self.control_radius_checkbox.disabled = disabled
					self.control_radius_checkbox.checked = False
				self.furcation_checkbox.disabled = disabled
				self.furcation_checkbox.checked = False

				self.angle_checkbox_model.disabled = disabled
				self.angle_checkbox_model.checked = False


				if disabled:
					for k in self.smooth_checkboxes.keys():
						self.smooth_checkboxes[k].disabled = disabled


			if m == "mesh":
				self.deform_mesh_button.disabled = disabled
				self.check_mesh_button.disabled = disabled
				self.mesh_representation_menu.disabled = disabled





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


	def save(self):

		""" Save the network or the mesh in a given output directory """

		try:
			if self.save_directory is None: 
				self.output_message("No output directory found. Please write the path in the text box and hit enter.", "warning")
			else:
				if self.save_menu.selected == "model":
					file = open(self.save_directory + self.save_filename + ".obj", 'wb') 
					pickle.dump(self.tree, file)
					self.output_message("Vessel model saved in " + self.save_directory + self.save_filename + ".obj" + ".")

				elif self.save_menu.selected == "centerline":
					file = self.save_directory + self.save_filename + ".swc"
					self.tree.write_swc(file)
					self.output_message("Vessel centerline saved in " + self.save_directory + self.save_filename + ".swc" + ".")

				else:
					mesh = self.tree.get_surface_mesh()
					if mesh is None:
						self.output_message("No mesh found. Please compute and/or update the mesh first.")
					else:
						mesh.save(self.save_directory + self.save_filename + ".vtk")
						self.output_message("Surface mesh saved in " + self.save_directory + self.save_filename + ".vtk" + ".")


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
					for elt in obj_cat:
						func(elt, args)



	def create_elements(self, mode):

		""" Create all 3D objects (nodes, edges) for a given mode.
		
		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
		"""

		if mode == 'full':

			# Import full graph
			G = self.tree.get_full_graph()

			# plot data points by iterating over the nodes
			nodes = {}
			for n in G.nodes():
				pos = vector((G.nodes[n]['coords'][0]), (G.nodes[n]['coords'][1]), (G.nodes[n]['coords'][2]))
				radius = G.nodes[n]['coords'][3]
				ball = sphere(pos=pos, color=color.red, radius=radius, mode = 'full', category = 'nodes', id = n)
				nodes[n] = ball
		
			# plot edges
			edges = {}
			for e in G.edges():
				pos = vector((G.nodes[e[0]]['coords'][0]), (G.nodes[e[0]]['coords'][1]), (G.nodes[e[0]]['coords'][2]))
				axis = G.nodes[e[1]]['coords'][:-1] - G.nodes[e[0]]['coords'][:-1]
				length = norm(axis)
				direction = axis / length
				c = cylinder(pos=pos, axis=vector(direction[0], direction[1], direction[2]), length=length, radius=0.2, color=color.black, mode = 'full', category = 'edges', id = e)
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
				ball = sphere(pos=pos, color=col[pt_type], radius=0.5, mode = 'topo', category = 'nodes', id = n)
				nodes[n] = ball

			edges = {}
			for e in G.edges():
				coords = np.vstack((G.nodes[e[0]]['coords'], G.edges[e]['coords'], G.nodes[e[1]]['coords']))

				c = curve(color = color.black, radius = 0.2, mode = 'topo', category = 'edges', id = e)
				for i in range(len(coords)):
					c.append(vector(coords[i, 0], coords[i, 1], coords[i, 2]))

				edges[e] = c

	

			self.elements['topo']['nodes'] = nodes
			self.elements['topo']['edges'] = edges

		elif mode == 'model':

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

				self.disable(True)
				self.tree.model_network()
				G = self.tree.get_model_graph()
				self.disable(False)

				self.output_message("Model complete!")

			nodes = {}
			furcation_nodes = {}
			furcation_edges = {}
			col = {'end': color.blue, 'bif' : color.red, 'reg' : color.green, 'sep': color.purple}

			for n in G.nodes():
				pos = vector((G.nodes[n]['coords'][0]), (G.nodes[n]['coords'][1]), (G.nodes[n]['coords'][2]))
				pt_type = G.nodes[n]['type']

				if pt_type != "sep":
					ball = sphere(pos=pos, color=col[pt_type], radius=0.5, mode = 'model', category = 'nodes', id = n)
					nodes[n] = ball

				if pt_type == "bif":
					n_list = []
					c_list = []
					bif = G.nodes[n]['bifurcation']

					AP = bif.get_AP()
					for pt in AP:
						n_list.append(sphere(pos = vector(pt[0], pt[1], pt[2]), radius=0.3, color = color.red, visible = False, mode = 'model', category = 'furcation_nodes'))

					apexsec = bif.get_apexsec()
					for l in apexsec:
						for sec in l:
							coords = create_crsec_coords(sec[0], sec[1][:-1])
							c = curve(color = color.black, radius = 0.1, visible=False, mode = 'model', category = 'furcation_edges')
							for pt in coords:
								c.append(vector(pt[0], pt[1], pt[2]))
							c.append(vector(coords[0][0], coords[0][1], coords[0][2]))
							c_list.append(c)
							n_list.append(sphere(pos = vector(sec[0][0], sec[0][1], sec[0][2]), radius=0.3, color = color.black, visible = False, mode = 'model', category = 'furcation_nodes'))
									
					endsec = bif.get_endsec()
					for sec in endsec:
						coords = create_crsec_coords(sec[0], sec[1][:-1])
						c = curve(color = color.black, radius = 0.1, visible=False, mode = 'model', category = 'furcation_edges')
						for pt in coords:
							c.append(vector(pt[0], pt[1], pt[2]))
						c.append(vector(coords[0][0], coords[0][1], coords[0][2]))
						c_list.append(c)
						n_list.append(sphere(pos = vector(sec[0][0], sec[0][1], sec[0][2]), radius=0.3, color = color.black, visible = False, mode = 'model', category = 'furcation_nodes'))

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
				c = curve(pos=pos, color = color.black, radius = 0.2, mode = 'model', category = 'edges', id = e)

				for i in range(1,len(coords)):
					c.append(vector(coords[i][0], coords[i][1], coords[i][2]))

				edges[e] = c

				coords = spl.get_control_points()
				pos = vector(coords[0][0], coords[0][1], coords[0][2])
				c2 = curve(pos=pos, color = color.black, radius = 0.2, visible=False, mode = 'model', category = 'control_edges', id = e)
				n_list = [sphere(pos = pos, color=color.gray(0.5), radius=0.5, visible=False, mode = 'model', category = 'control_nodes', id = (e, 0))]
				
				for i in range(1,len(coords)):
					c2.append(vector(coords[i][0], coords[i][1], coords[i][2]))
					n_list.append(sphere(pos = vector(coords[i][0], coords[i][1], coords[i][2]), color=color.gray(0.5), radius=0.5, visible=False, mode = 'model', category = 'control_nodes', id = (e, i)))
				control_edges[e] = c2
				control_nodes[e] = n_list
				

			self.elements['model']['nodes'] = nodes
			self.elements['model']['edges'] = edges

			self.elements['model']['control_edges'] = control_edges
			self.elements['model']['control_nodes'] = control_nodes

			self.elements['model']['furcation_edges'] = furcation_edges
			self.elements['model']['furcation_nodes'] = furcation_nodes

			self.control_pts_checkbox.disabled = False
			self.furcation_checkbox.disabled = False

			
		elif mode == 'mesh':

			if self.tree.get_model_graph() is None:
				self.create_elements("model")
				self.checkboxes["model"].checked = True

			if self.tree.get_crsec_graph() is None:

				self.output_message("Computing mesh...")

				self.disable(True)
				self.tree.compute_cross_sections(self.N, self.d)
				self.disable(False)

				self.output_message("Mesh complete!")

			mesh = self.tree.mesh_surface()

			vertices = mesh.points
			faces = mesh.faces.reshape((-1, 5))
			faces.astype('int32')

			v_obj = []
			for i in range(len(vertices)):
				v = vertices[i]
				v_obj.append(vertex(pos=vector(v[0], v[1], v[2]), color=color.gray(0.75), id = i))

			
			quads = []
			for i in range(len(faces)):
				f = faces[i]
				q = quad(v0=v_obj[f[1]], v1=v_obj[f[2]], v2=v_obj[f[3]], v3=v_obj[f[4]], mode = 'mesh', category = 'surface', id = i)
				quads.append(q)

			curves = {}
			lines = {}
			G = self.tree.get_crsec_graph()
			for n in G.nodes():
				crsec = G.nodes[n]['crsec']
				if G.nodes[n]['type'] == "bif":
					N = (self.tree._N // 2) - 1
					curve_list = []
					s = 2
					for i in range(len(crsec) // N):
						c = curve(pos = vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]), radius=0.05, color = color.black, mode = 'mesh', category = 'section_edges')
						for k in range(N):
							c.append(vector(crsec[s+k, 0], crsec[s+k, 1], crsec[s+k, 2]))
						s += k + 1
						c.append(vector(crsec[1, 0], crsec[1, 1], crsec[1, 2]))
						curve_list.append(c)
					curves[n] = curve_list


				else:
					c = curve(radius=0.05, color = color.black, mode = 'mesh', category = 'section_edges')
					for i in range(len(crsec)):
						c.append(vector(crsec[i, 0], crsec[i, 1], crsec[i, 2]))
					c.append(vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
					curves[n] = [c]

			for e in G.edges():
				crsec_list = G.edges[e]['crsec']
				# Longitudinal curves
				lines_list = []
				for i in range(crsec_list.shape[1]):
					c = curve(radius=0.05, color = color.black, mode = 'mesh', category = 'connecting_edges')
					# Starting point
					start_sec = G.nodes[e[0]]['crsec']
					c.append(vector(start_sec[i, 0], start_sec[i, 1], start_sec[i, 2]))
					for j in range(crsec_list.shape[0]):
						c.append(vector(crsec_list[j, i, 0], crsec_list[j, i, 1], crsec_list[j, i, 2]))
					# Ending point
					end_sec = G.nodes[e[1]]['crsec']
					l = G.edges[e]['connect'][i]
					c.append(vector(end_sec[l][0], end_sec[l][1], end_sec[l][2]))

					lines_list.append(c)
				lines[e] = lines_list

				# Cross section curves
				curve_list = []
				for i in range(len(crsec_list)):
					crsec = crsec_list[i]
					c = curve(radius=0.05, color = color.black, mode = 'mesh', category = 'section_edges')
					for j in range(len(crsec)):
						c.append(vector(crsec[j, 0], crsec[j, 1], crsec[j, 2]))
					c.append(vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
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

		mode = b.mode # The mode of the button is the representation it controls
		if mode == "full":
			for elt in self.modified_elements[mode]['move']:
				category = elt[0]
				ids = elt[1]
				new_pos = elt[2]
				radius = elt[3]

				if category == "nodes":
					self.tree.get_full_graph().nodes[ids]['coords'][:3] = np.array([new_pos.x, new_pos.y, new_pos.z])
					if radius is not None:
						self.tree.get_full_graph().nodes[ids]['coords'][3] = radius

			if len(self.modified_elements[mode]['move']) > 0:
				self.modified_elements[mode]['move'] = []
				self.tree.set_full_graph(self.tree.get_full_graph())
				self.refresh_display("full")
			
			for n in self.modified_elements[mode]['delete']:
				prec = list(self.tree.get_full_graph().predecessors(n))[0]
				succ = list(self.tree.get_full_graph().successors(n))[0]
				self.tree.get_full_graph().remove_node(n)
				self.tree.get_full_graph().add_edge(prec, succ, coords = np.array([]).reshape(0,4))

				self.elements["full"]["nodes"].pop(n)
				self.elements["full"]["edges"][(n, succ)].visible = False
				self.elements["full"]["edges"].pop((n, succ))
				self.elements["full"]["edges"][(prec, succ)] = self.elements["full"]["edges"][(prec, n)]
				self.elements["full"]["edges"][(prec, succ)].id = (prec, succ) 
				self.elements["full"]["edges"].pop((prec, n))


			if len(self.modified_elements[mode]['delete']) > 0:
				self.modified_elements[mode]['delete'] = []
				self.tree.set_full_graph(self.tree.get_full_graph())
				self.refresh_display("full")
				self.node_size_sliders['full'].value = 1


			# Empty the model and mesh objects and make it invisible
			self.hide("topo")
			self.elements["topo"] = {}
			self.checkboxes['topo'].checked = False

			self.hide("model")
			self.elements["model"] = {}
			self.checkboxes['model'].checked = False

			self.hide("mesh")
			self.elements["mesh"] = {}
			self.checkboxes['mesh'].checked = False
			self.output_message("Full graph updated.")


		elif mode == "topo":

			for elt in self.modified_elements["topo"]["move"]:

				G = self.tree.get_topo_graph()
				if elt[0] == "edges":
					G.edges[elt[1]]['coords'] = elt[2]

				else:
					G.nodes[elt[1]]['coords'] = elt[2]

			if len(self.modified_elements["topo"]["move"]) > 0:

				self.tree.set_topo_graph(G)

				self.modified_elements["topo"]["move"] = []
				self.refresh_display("topo")
				self.refresh_display("full")

				self.hide("model")
				self.elements["model"] = {}
				self.checkboxes['model'].checked = False

				self.hide("mesh")
				self.elements["mesh"] = {}
				self.checkboxes['mesh'].checked = False

			for elt in self.modified_elements["topo"]["merge"]:
				self.tree.merge_branch(elt)

			if len(self.modified_elements["topo"]["merge"]) > 0:

				self.modified_elements["topo"]["merge"] = []
				self.refresh_display("topo")
				self.refresh_display("full")

				self.hide("model")
				self.elements["model"] = {}
				self.checkboxes['model'].checked = False

				self.hide("mesh")
				self.elements["mesh"] = {}
				self.checkboxes['mesh'].checked = False


			for e in self.modified_elements["topo"]["delete"]:
				self.tree.remove_branch(e)

			if len(self.modified_elements["topo"]["delete"]) > 0:

				self.modified_elements["topo"]["delete"] = []
				self.refresh_display("topo")
				self.refresh_display("full")

				if len(self.elements["model"]) > 0:
					self.refresh_display("model")

				if len(self.elements["mesh"]) > 0:
					self.refresh_display("mesh")
				

			self.output_message("Topo graph updated.")

		elif mode == "model":

			G = self.tree.get_model_graph()

			for elt in self.modified_elements[mode]['move']:
				category = elt[0]
				ids = elt[1]
				new_pos = elt[2]
				radius = elt[3]

				if category == "control_nodes":

					P = G.edges[ids[0]]['spline'].get_control_points()
					P[ids[1], :3] = [new_pos.x, new_pos.y, new_pos.z]
					if radius is not None:
						P[ids[1], 3] = radius
					G.edges[ids[0]]['spline'].set_control_points(P.tolist())

				if category == "nodes":
					pass

			for elt in self.modified_elements[mode]['lambda']:
				G.edges[elt[0]]['spline'] = elt[1]


			self.tree.set_model_graph(G, down_replace = False)

			for k in self.modified_elements[mode]:
				self.modified_elements[mode][k] = []

			self.refresh_display("model")
			self.hide("mesh")
			self.elements["mesh"] = {}
			self.checkboxes['mesh'].checked = False
			self.output_message("Model graph updated.")
			

		else:

			# Parameter update
			if len(self.modified_elements[mode]['parameter']) > 0:
				self.disable(True)
				self.tree.compute_cross_sections(self.N, self.d)
				self.tree.mesh_surface()
				self.disable(False)
				
				self.modified_elements[mode]['parameter'] = []


			else:
				# Deform update
				for G in self.modified_elements[mode]['deform']:
					self.tree.set_crsec_graph(G)

				self.modified_elements[mode]['deform'] = []
				self.output_message("Mesh updated.")

			self.refresh_display("mesh")

		self.unselect("node", mode)
		self.unselect("edge", mode)





	def refresh_display(self, mode):

		""" Modify the edges and nodes position according to the tree object modified graph, for a given mode.

		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
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
					else: 
						self.elements[mode]["nodes"][n] = sphere(pos=vector(coords[0], coords[1], coords[2]), color=color.red, radius=coords[3], mode = 'full', category = 'nodes', id = n, visible = visible) # Create node

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

					else:
						self.elements[mode]["edges"][e] = cylinder(pos=pos, axis=axis, length=length, radius=0.2, color=color.black, mode = 'full', category = 'edges', id = e, visible = visible)



			elif mode == "topo":
				col = {'end': color.blue, 'bif' : color.red, 'reg' : color.green}

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
						self.elements[mode]["nodes"][n] = sphere(pos=pos, color=col[pt_type], radius=0.5, mode = 'topo', category = 'nodes', id = n, visible = visible) # Create new point
					else:
						self.elements[mode]["nodes"][n].pos = vector(coords[0], coords[1], coords[2])
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
						for i in range(len(coords)):
							c.modify(i, vector(coords[i][0], coords[i][1], coords[i][2]))

					else:

						c = curve(color = color.black, radius = 0.2, mode = 'topo', category = 'edges', id = e)

						for i in range(len(coords)):
							c.append(vector(coords[i][0], coords[i][1], coords[i][2]))

						self.elements[mode]["edges"][e] = c
					self.elements[mode]["edges"][e].visible = visible


			elif mode == "model":

				
				G = self.tree.get_model_graph()
				col = {'end': color.blue, 'bif' : color.red, 'reg' : color.green, 'sep': color.purple}

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
					pt_type = G.nodes[n]['type']
					if n not in nds_elt:
						pos = vector(coords[0], coords[1], coords[2])
						pt_type = G.nodes[n]['type']
						self.elements[mode]["nodes"][n] = sphere(pos=pos, color=col[pt_type], radius=0.5, mode = 'model', category = 'nodes', id = n, visible = visible) # Create new point
					else:
						self.elements[mode]["nodes"][n].pos = vector(coords[0], coords[1], coords[2]) # Change coordinates
					self.elements[mode]["nodes"][n].visible = visible


				# Remove edge is if required
				edj = [e for e in G.edges()]
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
								c.append(pos=vector(coords[i][0], coords[i][1], coords[i][2]), color=color.black, radius=0.2)
								self.elements[mode]["control_nodes"][e].append(sphere(pos = vector(coords[i][0], coords[i][1], coords[i][2]), color=color.gray(0.5), radius=0.5, visible=False, mode = 'model', category = 'control_nodes', id = (e, i)))


						# Update the spline curve point positions
						coords = spl.get_points()
						coords = np.vstack((coords[0, :], coords[::self.display_spline_step, :], coords[-1, :]))
						c = self.elements[mode]["edges"][e]

						c.clear()
						for i in range(len(coords)):
							c.append(pos=vector(coords[i][0], coords[i][1], coords[i][2]), color=color.black, radius=0.2)
						self.elements[mode]["control_edges"][e].visible = self.control_pts_checkbox.checked

					else: # Create new edge

						# Create control nodes
						coords = spl.get_control_points()
						pos = vector(coords[0][0], coords[0][1], coords[0][2])
						c2 = curve(pos=pos, color = color.black, radius = 0.2, visible=False, mode = 'model', category = 'control_edges', id = e)
						n_list = [sphere(pos = pos, color=color.gray(0.5), radius=0.5, visible=False, mode = 'model', category = 'control_nodes', id = (e, 0))]
				
						for i in range(1,len(coords)):
							c2.append(vector(coords[i][0], coords[i][1], coords[i][2]))
							n_list.append(sphere(pos = vector(coords[i][0], coords[i][1], coords[i][2]), color=color.gray(0.5), radius=0.5, visible=False, mode = 'model', category = 'control_nodes', id = (e, i)))

						self.elements[mode]["control_edges"][e] = c2
						self.elements[mode]["control_nodes"][e] = n_list
				
						# Create spline curve
						coords = spl.get_points()
						coords = np.vstack((coords[0, :], coords[::self.display_spline_step, :], coords[-1, :]))
						pos = vector(coords[0][0], coords[0][1], coords[0][2])
						c = curve(pos=pos, color = color.black, radius = 0.2, mode = 'model', category = 'edges', id = e)

						for i in range(1,len(coords)):
							c.append(vector(coords[i][0], coords[i][1], coords[i][2]))

						self.elements[mode]["edges"][e] = c

			else:

				mesh = self.tree.get_surface_mesh()
				if mesh is None:
					mesh = self.tree.mesh_surface()


				if mesh.n_faces != len(self.elements["mesh"]["surface"]):
					self.hide("mesh")
					self.elements["mesh"] = {}

					self.create_elements("mesh")
				else:

					mesh = self.tree.get_surface_mesh()
					if mesh is None:
						self.output_message("Computing mesh...")
						self.disable_checkboxes(True)
						mesh = self.tree.mesh_surface()
						self.disable_checkboxes(False)

					vertices = mesh.points
					faces = mesh.faces.reshape((-1, 5))
					faces.astype('int32')

					v_obj = []
					for v in vertices:
						v_obj.append(vertex(pos=vector(v[0], v[1], v[2]), color=color.gray(0.75)))

					i = 0
					for f in faces:
						self.elements["mesh"]["surface"][i].v0 = v_obj[f[1]]
						self.elements["mesh"]["surface"][i].v1 = v_obj[f[2]]
						self.elements["mesh"]["surface"][i].v2 = v_obj[f[3]]
						self.elements["mesh"]["surface"][i].v3 = v_obj[f[4]]
						i+=1

					G = self.tree.get_crsec_graph()
					for n in G.nodes():
						crsec = G.nodes[n]['crsec']

						if G.nodes[n]['type'] == "bif":

							N = (self.tree._N // 2) - 1
							s = 2
							for i in range(len(crsec) // N):
								self.elements["mesh"]["section_edges"][n][i].modify(0, vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
								c = 1
								for k in range(N):
									self.elements["mesh"]["section_edges"][n][i].modify(c, vector(crsec[s+k, 0], crsec[s+k, 1], crsec[s+k, 2]))
									c +=1
								s += k + 1
								self.elements["mesh"]["section_edges"][n][i].modify(c,vector(crsec[1, 0], crsec[1, 1], crsec[1, 2]))

						else:
							
							for i in range(len(crsec)):
								self.elements["mesh"]["section_edges"][n][0].modify(i, vector(crsec[i, 0], crsec[i, 1], crsec[i, 2]))

							self.elements["mesh"]["section_edges"][n][0].modify(len(crsec), vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))
					
					for e in G.edges():

						crsec_list = G.edges[e]['crsec']
						
						# Longitudinal curves
						for i in range(crsec_list.shape[1]):
							
							# Starting point
							start_sec = G.nodes[e[0]]['crsec']
							self.elements["mesh"]["connecting_edges"][e][i].modify(0, vector(start_sec[i, 0], start_sec[i, 1], start_sec[i, 2]))
							c = 1
							for j in range(crsec_list.shape[0]):
								self.elements["mesh"]["connecting_edges"][e][i].modify(c, vector(crsec_list[j, i, 0], crsec_list[j, i, 1], crsec_list[j, i, 2]))
								c+=1
								
							# Ending point
							end_sec = G.nodes[e[1]]['crsec']
							l = G.edges[e]['connect'][i]
							self.elements["mesh"]["connecting_edges"][e][i].modify(c,vector(end_sec[l][0], end_sec[l][1], end_sec[l][2]))


						# Cross section curves
						for i in range(len(crsec_list)):
							crsec = crsec_list[i]
							
							for j in range(len(crsec)):
								self.elements["mesh"]["section_edges"][e][i].modify(j, vector(crsec[j, 0], crsec[j, 1], crsec[j, 2]))
							self.elements["mesh"]["section_edges"][e][i].modify(len(crsec),vector(crsec[0, 0], crsec[0, 1], crsec[0, 2]))




	def reset_graph(self, b):

		""" Empty the modified element and refresh the display to show the unmodified tree object """

		mode = b.mode
		refresh = True


		if refresh:
			self.refresh_display(mode)

		if mode == "mesh":
			self.output_message("Reset mesh.")
		else:
			self.output_message("Reset " + mode + " graph.")

		self.node_size_sliders['full'].value = 1

	
		self.unselect("node", mode)
		self.unselect("edge", mode)



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

		if self.edition_mode == "topo" and self.selected_edge is not None and self.selected_edge.mode != "cursor":

			if evt.key == "i":

				# Make the in node invisible
				e = self.selected_edge.id
				pred = list(self.tree.get_topo_graph().predecessors(e[0]))[0]
				if self.tree.get_topo_graph().nodes[pred]["type"] == "end":
					self.output_message("Edge can not be merged.")
				else:
					self.elements["topo"]["nodes"][e[0]].visible = False

				# Add the merging to modified elements
				self.modified_elements["topo"]["merge"] = [e]

			if (evt.key == "r" or evt.key == "R") and not self.running:

				self.running = True
				# Get rotation angle and normal 
				normal = scene.camera.axis
				normal = np.array([normal.x, normal.y, normal.z])
				normal = normal/norm(normal)
			
				
				rot_center = self.selected_edge.point(0)['pos'] 
				rot_center = np.array([rot_center.x, rot_center.y, rot_center.z]) # Coordinates of the first point of edge

				if evt.key == "r":
					alpha = 0.1
				else:
					alpha = -0.1

				edg = self.selected_edge.id
				G = self.tree.get_topo_graph()
				edg_list = [edg] + list(nx.dfs_edges(G, source=edg[1]))
				nds_list = list(nx.dfs_preorder_nodes(G, source=edg[1]))

				for e in edg_list:
					new_coords = np.vstack((np.zeros((1,4)), G.edges[e]['coords'], np.zeros((1,4))))
					for i in range(self.elements["topo"]["edges"][e].npoints):
						if e != edg or (e == edg and i > 0):
							coord = self.elements["topo"]["edges"][e].point(i)['pos'] 
							coord = np.array([coord.x, coord.y, coord.z])
							l = norm(coord - rot_center)
							v = (coord - rot_center) / l
							
							new_v = rotate_vector(v, normal, alpha)
							new_pos = rot_center + l* new_v

							self.elements["topo"]["edges"][e].modify(i, vector(new_pos[0], new_pos[1], new_pos[2]))

							new_coords[i, :3] = new_pos
							

					self.modified_elements["topo"]["move"].append(["edges", e, new_coords[1:-1, :]])

				for n in nds_list:
					new_coords = G.nodes[n]['coords'].copy()

					coord = self.elements["topo"]["nodes"][n].pos 
					coord = np.array([coord.x, coord.y, coord.z])
					l = norm(coord - rot_center)
					v = (coord - rot_center) / l
						
					new_v = rotate_vector(v, normal, alpha)
					new_pos = rot_center + l* new_v

					self.elements["topo"]["nodes"][n].pos = vector(new_pos[0], new_pos[1], new_pos[2])
					new_coords[:3] = np.array([new_pos[0], new_pos[1], new_pos[2]])
					self.modified_elements["topo"]["move"].append(["nodes", n, new_coords])

				self.running = False

			
			
			if evt.key == "delete":

				edg = self.selected_edge.id
				self.modified_elements["topo"]["delete"].append(edg)
				G = self.tree.get_topo_graph()

				# Hide the edges
				for e in [edg] + list(nx.dfs_edges(G, source=edg[1])):
					self.elements["topo"]["edges"][e].visible = False
					#for elt in self.elements["topo"]["edges"][e]:
					#	elt.visible = False

				# Hide the nodes 
				for n in list(nx.dfs_preorder_nodes(G, source=edg[1])):
					self.elements["topo"]["nodes"][n].visible = False
				

		if self.edition_mode == "model" and self.selected_edge is not None and not self.running:
			if self.smooth_checkboxes['radius'].checked:
			
				self.running = True

				if evt.key == "d":
					self.lbdr -= 0.5
					if self.lbdr < 0:
						self.lbdr = 0
					self.lbdr_text.text = str(round(self.lbdr, 2))

					self.smooth_spline()

				if evt.key == "u":
					self.lbdr += 0.5
					self.lbdr_text.text = str(round(self.lbdr, 2))
					self.smooth_spline()

				self.running = False

			if self.smooth_checkboxes['spatial'].checked:

				self.running = True
				
				if evt.key == "d":
					self.lbds -= 0.5
					if self.lbds < 0:
						self.lbds = 0
					self.lbds_text.text = str(round(self.lbds, 2))
					self.smooth_spline()

				if evt.key == "u":
					self.lbds += 0.5
					self.lbds_text.text = str(round(self.lbds, 2))
					self.smooth_spline()

				self.running = False


		if (self.edition_mode == "data" or self.edition_mode == "model") and self.selected_node is not None and self.selected_node.mode != "cursor":

			ids = self.selected_node.id
		
			if evt.key == "d":
				self.selected_node.radius = self.selected_node.radius - 0.01

			if evt.key == "u":
				self.selected_node.radius = self.selected_node.radius + 0.01

			# Save new position in changes
			already_moved = False

			for elt in self.modified_elements[self.edition_mode]['move']:
				if elt[0] == self.selected_node.category and elt[1] == self.selected_node.id:
					already_moved = True
					elt[3] = self.selected_node.radius
		
			if not already_moved:
				self.modified_elements[self.edition_mode]['move'].append([self.selected_node.category, self.selected_node.id, self.selected_node.pos, self.selected_node.radius])


	###########################################
	############# NODE EDITION ################
	###########################################

	def update_edition_mode(self):

		""" Update the edition mode chosen by the user in the edition menu """

		self.unselect("node", self.edition_mode)
		self.unselect("edge", self.edition_mode)

		self.edition_mode = self.edition_menu.selected
		if self.edition_mode == "data":
			self.edition_mode = "full"

		self.output_message("Edition mode switched to " + self.edition_mode + ".")


	def resample_nodes(self, b):

		""" Resample nodes """

		self.show("full", ["nodes"])
		self.modified_elements["full"]["delete"] = []

		p = b.value

		show_id = []
		for n in self.tree.get_topo_graph().nodes():
			show_id.append(self.tree.get_topo_graph().nodes[n]['full_id'])

		for e in self.tree.get_topo_graph().edges():
			pts_id = self.tree.get_topo_graph().edges[e]['full_id']

			if len(pts_id)!=0:

				if p!=0:
					# Resampling
					step = int(len(pts_id)/(p*len(pts_id)))
					if step > 0 and len(pts_id[:-1:step]) > 0:
						show_id = show_id +  pts_id[:-1:step]
						
					else:
						show_id = show_id + [pts_id[int(len(pts_id)/2)]]
				else:
					show_id = show_id + [pts_id[int(len(pts_id)/2)]]
					

		for k in self.elements["full"]["nodes"].keys():
			if k not in show_id:
				self.elements["full"]["nodes"][k].visible = False
				self.modified_elements["full"]["delete"].append(k)



	def move(self, evt):

		""" Move the selected node with mouse cursor """
		if self.selected_node is not None and self.drag:
			if self.edition_mode == "full" or self.edition_mode == "model" or self.selected_node.mode == "cursor":
				self.selected_node.pos = scene.mouse.project(normal = scene.mouse.ray, point = self.selected_node.pos)#scene.mouse.pos #evt.pos



		
	def drop(self, evt):

		""" Drop the node when mouse is released """

		# Save new position in changes
		if self.selected_node is not None:

			if self.selected_node.mode == "cursor":
				self.unselect("node")
				self.drag = False

			else:
				already_moved = False
				for elt in self.modified_elements[self.edition_mode]['move']:
					if elt[0] == self.selected_node.category and elt[1] == self.selected_node.id:
						already_moved = True
						elt[2] = self.selected_node.pos
				if not already_moved:
					self.modified_elements[self.edition_mode]['move'].append([self.selected_node.category, self.selected_node.id, self.selected_node.pos, None])

				self.unselect("node")
				self.drag = False


	def select(self):

		""" Select a node or edge by mouse clicking """

		obj = scene.mouse.pick

		if type(obj) == sphere and obj.mode == self.edition_mode:
	
			self.unselect("node")

			self.selected_node = obj
			self.selected_node.color = color.yellow
			self.drag = True
			self.output_message("Node "  + str(obj.id) + " selected. Move it using the mouse. Press 'u' or 'd' to increase or lower the radius.")

		elif type(obj) == curve and obj.mode == "model" and self.edition_mode == "model":
			
			self.unselect("edge")

			self.selected_edge = obj
			self.selected_edge.color = color.green

			self.smooth_checkboxes['spatial'].disabled = False
			self.smooth_checkboxes['radius'].disabled = False

			self.output_message("Spline " + str(obj.id) +  " selected. Check a smoothing box and use the slider smooth or unsmooth.")

		elif type(obj) == curve and self.edition_mode == "topo":

			self.unselect("edge")
			self.selected_edge = obj
			self.selected_edge.color = color.green

			self.output_message("Edge " + str(obj.id) + " selected. Press suppr. to cut the corresponding branch. Press r (resp. R) to rotate the branch clockwise (resp. counterclockwise).")

		elif type(obj) == sphere and obj.mode == "cursor":
			
			self.selected_node = obj
			self.drag = True
			self.output_message("Origin cursor selected. Move it to the center of the image cut.")




	def unselect(self, elt = "node", mode = None):

		""" Unselect the selected edge or elements if it is associated to a given mode.

		Keyword arguments:
		mode -- newtwork mode (full, topo, model, mesh)
		"""

		if elt == "node":
			if self.selected_node is not None:
				if mode is None or self.selected_node.mode == mode:
					if self.selected_node.mode == "full":
						self.selected_node.color = color.red
					if self.selected_node.mode == "model":
						self.selected_node.color = color.gray(0.5)

					self.selected_node = None

		else:
			if self.selected_edge is not None:
				if mode is None or self.selected_edge.mode == mode:
					if type(self.selected_edge) == curve:

						self.selected_edge.color = color.black
						self.selected_edge = None

						self.smooth_checkboxes['spatial'].disabled = True
						self.smooth_checkboxes['radius'].disabled = True
						self.smooth_checkboxes['spatial'].checked = False
						self.smooth_checkboxes['radius'].checked = False
						self.lbds_text.text = ""
						self.lbdr_text.text = ""

					else:
						if self.selected_edge.id[0] in self.elements["topo"]["edges"].keys():
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


	def smooth_spline(self):

		""" Change the spline smoothing parameter and update render. """

		G = self.tree.get_model_graph()
		ids = self.selected_edge.id

		spl = copy.deepcopy(G.edges[ids]["spline"])

		if len(G.edges[ids]["spline"].get_lbd())>0:
			if self.smooth_checkboxes["spatial"].checked:
				spl._set_lambda_model([self.lbds, None])
			else:
				spl._set_lambda_model([None, self.lbdr])

			# Modify spline display
			coords = spl.get_control_points()
			c = self.elements["model"]["control_edges"][ids]

			for i in range(len(coords)):
				c.modify(i, vector(coords[i][0], coords[i][1], coords[i][2]))
				self.elements["model"]["control_nodes"][ids][i].pos = vector(coords[i][0], coords[i][1], coords[i][2])
				if self.control_radius_checkbox.checked:
					self.elements["model"]["control_nodes"][ids][i].radius = coords[i][3]

			coords = spl.get_points()
			coords = np.vstack((coords[0, :], coords[::self.display_spline_step, :], coords[-1, :]))
			
			self.selected_edge.clear()
			for i in range(len(coords)):
				self.selected_edge.append(pos=vector(coords[i][0], coords[i][1], coords[i][2]), radius = 0.2) 

			# Add lambda modification
			self.modified_elements["model"]["lambda"].append([ids, spl])

		else:
			self.output_message("There are no models associated with this spline.", mode = "warning")


	###########################################
	############# MESH EDITION ################
	###########################################


	def update_mesh_parameters(self, b):

		""" Update user choice of meshing parameters """

		if b.parameter == "N":
			self.N = int(self.parameters_winput['N'].text)
			self.output_message("Number of cross section nodes set to " + str(self.N) + ".")
			if len(self.elements['mesh']['surface']) > 0: # If mesh is already computed
				self.modified_elements['mesh']['parameter'].append(True)

		elif b.parameter == "d":
			self.d = float(self.parameters_winput['d'].text)
			self.output_message("Density of cross section set to " + str(self.d) + ".")
			if len(self.elements['mesh']['surface']) > 0: # If mesh is already computed
				self.modified_elements['mesh']['parameter'].append(True)

		elif b.parameter == "search_dist":
			self.search_dist  = float(self.parameters_winput['search_dist'].text)
			self.output_message("Maximum projection distance set to " + str(self.search_dist) + ".")

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
			self.disable(True)

			G = self.tree.get_crsec_graph().copy()
			self.tree.deform_surface_to_mesh(self.target_mesh, search_dist = self.search_dist)

			self.modified_elements['mesh']['deform'].append(self.tree.get_crsec_graph().copy())

			self.refresh_display("mesh")
			self.tree.set_crsec_graph(G)

			self.disable(False)
			self.output_message("Mesh deformation complete!")


	def check_mesh(self):
		""" Checks the mesh quality and display the mesh segments not compatible with simulation in red """
		self.disable(True)

		if self.check_state == False:

			self.output_message("Checking mesh...")
			field, failed_edges, failed_bifs = self.tree.check_mesh()
			
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

		self.disable(False)





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



	












