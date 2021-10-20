import numpy as np
import vtk
import os
import pyvista as pv # Meshing


from ArterialTree import ArterialTree


class Simulation:

	""" Class to parametrize, run and analyse cases using openFoam solver via python """


	def __init__(self, arterial_tree, output_dir):

		"""
		Keyword argument:
		arterial_tree -- Arterial Tree object containing the mesh information of the case (see ArterialTree class) 
		output_dir -- path to the case directory
		"""

		if arterial_tree.get_volume_mesh() is None:
			raise ValueError('No volume mesh found.')

		self.arterial_tree = arterial_tree
		self.output_dir = output_dir
		

		crsec_graph = self.arterial_tree.get_crsec_graph()

		self.inlet_nodes = []
		self.outlet_nodes = []

		for n in crsec_graph.nodes():
			if crsec_graph.nodes[n]['type'] == "end":
				if crsec_graph.in_degree(n) == 0: # Inlet case
					self.inlet_nodes.append(n)
				else:
					self.outlet_nodes.append(n)

		self.boundary_patches = self.__boundary_patches()
		self.mesh = self.arterial_tree.get_volume_mesh()



	#####################################
	############### MESH  ###############
	#####################################


	def __boundary_patches(self):

		""" Writes the boundary patches as a vtk MultiBlock data. The walls, inlet and oulet surfaces are stored as blocks, with appropriate names
		"""

		# Arterial walls
		if self.arterial_tree.get_surface_mesh() is None:
			wall = self.arterial_tree.mesh_surface()
		else: 
			wall = self.arterial_tree.get_surface_mesh()

		boundary_blocks = pv.MultiBlock()
		boundary_blocks["wall"] = wall

		# Inlets and outlets
		inlet_num = 0
		outlet_num = 0

		crsec_graph = self.arterial_tree.get_crsec_graph()

		for i in range(len(self.inlet_nodes)):
			n = self.inlet_nodes[i]
			# Get crsec
			crsec = crsec_graph.nodes[n]['crsec']
			center = (crsec[0] + crsec[crsec.shape[0]//2])/2.0
			vertices, faces = self.arterial_tree.ogrid_pattern(center, crsec)
			pattern = pv.PolyData(vertices, faces)

			name = "inlet_" + str(i)
			boundary_blocks[name] = pattern

		for i in range(len(self.outlet_nodes)):
			n = self.outlet_nodes[i]
			# Get crsec
			crsec = crsec_graph.nodes[n]['crsec']
			center = (crsec[0] + crsec[crsec.shape[0]//2])/2.0
			vertices, faces = self.arterial_tree.ogrid_pattern(center, crsec)
			pattern = pv.PolyData(vertices, faces)

			name = "outlet_" + str(i)
			boundary_blocks[name] = pattern

		return boundary_blocks


	#####################################
	############# OPENFOAM  #############
	#####################################

	def file_description(self, ver, fmt, cls, obj):
		""" Returns the OpenFoam file description 

		Keyword arguments:
		ver -- version
		fmt -- format
		cls -- class
		obj -- objet 
		"""

		foamfile = """
FoamFile
{
	version     %.1f;
	format      %s;
	class       %s;
	object      %s;
}
""" % (ver, fmt, cls, obj)

		return foamfile


	def file_header(self):
		""" Returns the OpenFoam file header """

		file_header = """/*--------------------------------*- C++ -*----------------------------------*\
=========                 |
\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
 \\    /   O peration     | Website:  https://openfoam.org
  \\  /    A nd           | Version:  8
   \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
"""
		return file_header


	def top_separator(self):
		""" Returns the OpenFoam top separator """

		top_separator = """
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""
		return top_separator


	def bottom_separator(self):
		""" Returns the OpenFoam top separator """

		bottom_separator = """
// ************************************************************************* //
"""

		return bottom_separator


	def write_mesh_files(self):

		""" Writes the mesh openFoam files 
		"""

		output_dir = self.output_dir + "/constant/polyMesh/"
		volume = self.mesh
		surfaces =  self.boundary_patches

		file_header = self.file_header()
		top_separator = self.top_separator()
		bottom_separator = self.bottom_separator()


		def get_midpoint(cell):

			num_p = cell.GetNumberOfPoints()
			pts = cell.GetPoints()
			midpoint = np.array([0.0,0.0,0.0])

			for i in range(num_p):
				midpoint += pts.GetPoint(i)
			midpoint /= num_p

			return midpoint

		def write_face(face_points):
			return "%d(%s)\n" % (len(face_points), " ".join([str(p) for p in face_points]))


		boundary_names = [name for name in surfaces.keys()]

		boundary_mid_points = vtk.vtkPoints()
		boundary_id_array = vtk.vtkIntArray()

		iter = surfaces.NewIterator()
		iter.UnRegister(None)
		iter.InitTraversal()
		boundary_id = 0
		while not iter.IsDoneWithTraversal():
			polyData = iter.GetCurrentDataObject()

			if not polyData.IsA("vtkPolyData"):
				print("ERROR: unexpected input (not vtkPolyData)")
				exit(1)

			nc = polyData.GetNumberOfCells()

			for i in range(nc):
				cell = polyData.GetCell(i)
				midpoint = get_midpoint(cell)
				boundary_mid_points.InsertNextPoint(midpoint)
				boundary_id_array.InsertNextValue(boundary_id)
			boundary_id += 1
			iter.GoToNextItem()
		num_boundaries = boundary_id

		loc = vtk.vtkKdTreePointLocator()
		boundary_dataset = vtk.vtkPolyData()
		boundary_dataset.SetPoints(boundary_mid_points)
		loc.SetDataSet(boundary_dataset)
		loc.BuildLocator()

		# map from boundaries to a list of tuples containing point ids and the cell id
		boundary_faces = []
		for i in range(num_boundaries):
			boundary_faces.append([])

		internal_faces = []

		volume.BuildLinks()
		nc = volume.GetNumberOfCells()

		for cell_id in range(nc):

			cell = volume.GetCell(cell_id)
			nf = cell.GetNumberOfFaces()
			cell_internal_faces = {}
			for face_id in range(nf):
				face = cell.GetFace(face_id)
				neighbour_cell_ids = vtk.vtkIdList()
				face_point_ids = face.GetPointIds()
				volume.GetCellNeighbors(cell_id, face.GetPointIds(), neighbour_cell_ids)
				nn = neighbour_cell_ids.GetNumberOfIds()
				if nn == 0:
					# boundary
					face_midpoint = get_midpoint(face)
					boundary_id = boundary_id_array.GetValue(loc.FindClosestPoint(face_midpoint))
					boundary_faces[boundary_id].append((
						[face.GetPointId(p) for p in range(face.GetNumberOfPoints())],
						cell_id))
				elif nn == 1:
					# internal
					neighbour_cell_id = neighbour_cell_ids.GetId(0)
					if cell_id < neighbour_cell_id:
						cell_internal_faces[neighbour_cell_id] = (
							[face.GetPointId(p) for p in range(face.GetNumberOfPoints())],
							cell_id,
							neighbour_cell_id)
				else:
					print("ERROR: face associated with more than 2 cells")
					exit(1)

			ids = list(cell_internal_faces.keys())
			ids.sort()
			for f in ids:
				internal_faces.append(cell_internal_faces[f])

		for i in range(num_boundaries):
			print(boundary_names[i] + ":", len(boundary_faces[i]),  "faces")

		# write files
		points_file = open(os.path.join(output_dir, "points"), "w")
		points_file.write(file_header)
		points_file.write(self.file_description(2.0, "ascii", "vectorField", "points"))
		points_file.write(top_separator)
		num_p = volume.GetNumberOfPoints()
		pts = volume.GetPoints()
		points_file.write("%d\n(\n" % num_p)

		for i in range(num_p):
			points_file.write("(%f %f %f)\n" % pts.GetPoint(i))

		points_file.write(")\n")
		points_file.write(bottom_separator)
		points_file.close()


		faces_file = open(os.path.join(output_dir, "faces"), "w")
		faces_file.write(file_header)
		faces_file.write(self.file_description(2.0, "ascii", "faceList", "faces"))
		faces_file.write(top_separator)
		total_faces = len(internal_faces)
		for i in range(num_boundaries):
			total_faces += len(boundary_faces[i])
		faces_file.write("%d\n(\n" % total_faces)
		for i in range(len(internal_faces)):
			faces_file.write(write_face(internal_faces[i][0]))
		for b in boundary_faces: 
			for j in range(len(b)):
				faces_file.write(write_face(b[j][0]))
		faces_file.write(")\n")
		faces_file.write(bottom_separator)
		faces_file.close()

		neighbour_file = open(os.path.join(output_dir, "neighbour"), "w")
		neighbour_file.write(file_header)
		neighbour_file.write(self.file_description(2.0, "ascii", "labelList", "neighbour"))
		neighbour_file.write(top_separator)
		neighbour_file.write("%d\n(\n" % len(internal_faces))
		for i in range(len(internal_faces)):
			neighbour_file.write("%d\n" % internal_faces[i][2])
		neighbour_file.write(")\n")
		neighbour_file.write(bottom_separator)
		neighbour_file.close()

		owner_file = open(os.path.join(output_dir, "owner"), "w")
		owner_file.write(file_header)
		owner_file.write(self.file_description(2.0, "ascii", "labelList", "owner"))
		owner_file.write(top_separator)
		owner_file.write("%d\n(\n" % total_faces)
		for i in range(len(internal_faces)):
			owner_file.write("%d\n" % internal_faces[i][1])
		for b in boundary_faces: 
			for j in range(len(b)):
				owner_file.write("%d\n" % b[j][1])
		owner_file.write(")\n")
		owner_file.write(bottom_separator)
		owner_file.close()

		boundary_file = open(os.path.join(output_dir, "boundary"), "w")
		boundary_file.write(file_header)
		boundary_file.write(self.file_description(2.0, "ascii", "polyBoundaryMesh", "boundary"))
		boundary_file.write(top_separator)
		start_face = len(internal_faces)
		boundary_file.write("%d\n(\n" % num_boundaries)

		for i in range(num_boundaries):
			boundary_file.write(boundary_names[i] + 
		"""
		{
			type patch;
			nFaces %d;
			startFace %d;
		}
		""" % (len(boundary_faces[i]), start_face))
			start_face += len(boundary_faces[i])

		boundary_file.write(")\n")
		boundary_file.write(bottom_separator)
		boundary_file.close()
		

	def write_pressure_boundary_condition_file(self):

		""" Writes the boundary condition files for pressure """

		boundary_names = [name for name in self.boundary_patches.keys()]

		file = open(self.output_dir + "/0/" + "p", "w")
		file.write(self.file_header())
		file.write(self.file_description(2.0, "ascii", "volScalarField", "p"))
		file.write(self.top_separator())

		file.write("dimensions		[0 2 -2 0 0 0 0];\n\n")
		file.write("internalField		uniform 0;\n\n")
		file.write("boundaryField\n{\n\n")

		for i in range(len(boundary_names)):
			if boundary_names[i] == "wall" or boundary_names[i][:5] == "inlet":
				tp =  "type		zeroGradient;"
				value = None
			else:
				tp = "type		fixedValue;"
				value = "value		uniform 0;"

			file.write("\n\t" + boundary_names[i] + "\n\t{\n")
			file.write("\t\t" + tp + "\n")

			if value is not None:
				file.write("\t\t" + value + "\n")
			file.write("\t}\n")
		file.write("}\n")

		file.write(self.bottom_separator())
		file.close()


	def write_velocity_boundary_condition_file(self, vel_mag):

		""" Writes the boundary condition files for velocity 

		Keyword argument:
		vel_mag -- list of inlet velocity magnitude in m/s
		"""

		boundary_names = [name for name in self.boundary_patches.keys()]

		file = open(self.output_dir + "/0/" + "U", "w")
		file.write(self.file_header())
		file.write(self.file_description(2.0, "ascii", "volVectorField", "U"))
		file.write(self.top_separator())

		file.write("dimensions		[0 1 -1 0 0 0 0];\n\n")
		file.write("internalField		uniform (0 0 0);\n\n")
		file.write("boundaryField\n{\n\n")

		U_inlet = []
		spl_graph = self.arterial_tree.get_spline_graph()

		for i in range(len(self.inlet_nodes)):
			succ = list(spl_graph.successors(self.inlet_nodes[i]))[0]
			U_inlet.append(spl_graph.edges[(self.inlet_nodes[i], succ)]['spline'].tangent(0.0) * vel_mag[i])

		for i in range(len(boundary_names)):
			if boundary_names[i] == "wall":
				tp =  "type		noSlip;"
				value = None
			elif boundary_names[i][:5] == "inlet":
				num = int(boundary_names[i][-1])
				tp =  "type		fixedValue;"
				value = "value		uniform (" + str(U_inlet[num][0]) + " " + str(U_inlet[num][1]) + " " + str(U_inlet[num][2])+");"
			else:
				tp = "type		zeroGradient;"
				value = None

			file.write("\n\t" + boundary_names[i] + "\n\t{\n")
			file.write("\t\t" + tp + "\n")

			if value is not None:
				file.write("\t\t" + value + "\n")
			file.write("\t}\n")
		file.write("}\n")

		file.write(self.bottom_separator())
		file.close()



	#####################################
	########## VISUALIZATION  ###########
	#####################################

	def import_field(self, filename, fieldname, outfieldname):
		""" Imports a scalar or vector field from the simulation result as a vtk multiblock file and add it to the volume mesh

		Keyword arguments:
		filename -- name of the vtk multiblock file
		fieldname -- name of the cell array with the field of interest"""

		res = pv.read(filename)
		mesh = res.get(0).get(0)
		field = mesh.cell_arrays([fieldname])
		self.mesh[outfieldname] = field

	

	def display_crsec(self, step, fieldname):
		
		""" Displays cross sections along the vessels 

		Keyword arguments:
		step -- step size for slicing
		fieldname -- field to display"""

		#slc = self.mesh.slice(normal=, origin=)

		p = pv.Plotter()
		p.add_mesh(slc) 
		p.show() 






