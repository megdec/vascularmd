import numpy
import vtk
import os




def write_files(volume, surfaces, output_dir):

	""" Writes the mesh files necessary to run OpenFoam cases 

	Keyword arguments:
	volume -- VTK Unstructured volume mesh
	surfaces -- VTK MultiBlock of surface of boundaries
	output_dir --  output directory for files
	"""


	def write_FoamFile(ver, fmt, cls, obj):

		return """
	FoamFile
	{
		version     %.1f;
		format      %s;
		class       %s;
		object      %s;
	}
	""" % (ver, fmt, cls, obj)


	def get_midpoint(cell):

		np = cell.GetNumberOfPoints()
		pts = cell.GetPoints()
		midpoint = numpy.array([0.0,0.0,0.0])

		for i in range(np):
			midpoint += pts.GetPoint(i)
		midpoint /= np

		return midpoint

	def write_face(face_points):
		return "%d(%s)\n" % (len(face_points), " ".join([str(p) for p in face_points]))



	file_header = """
	/*--------------------------------*- C++ -*----------------------------------*\\
	| =========                 |                                                 |
	| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
	|  \\\\    /   O peration     | Version:  2.1.1                                 |
	|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
	|    \\\\/     M anipulation  |                                                 |
	 \*---------------------------------------------------------------------------*/
	"""

	top_separator = """
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
	"""

	bottom_separator = """
	// ************************************************************************* //
	"""
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
		print(boundary_names[i] + ":", len(boundary_faces[i]))

	# write files
	points_file = open(os.path.join(output_dir, "points"), "w")
	points_file.write(file_header)
	points_file.write(write_FoamFile(2.0, "ascii", "vectorField", "points"))
	points_file.write(top_separator)
	np = volume.GetNumberOfPoints()
	pts = volume.GetPoints()
	points_file.write("%d\n(\n" % np)

	for i in range(np):
		points_file.write("(%f %f %f)\n" % pts.GetPoint(i))

	points_file.write(")\n")
	points_file.write(bottom_separator)
	points_file.close()



	faces_file = open(os.path.join(output_dir, "faces"), "w")
	faces_file.write(file_header)
	faces_file.write(write_FoamFile(2.0, "ascii", "faceList", "faces"))
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
	neighbour_file.write(write_FoamFile(2.0, "ascii", "labelList", "neighbour"))
	neighbour_file.write(top_separator)
	neighbour_file.write("%d\n(\n" % len(internal_faces))
	for i in range(len(internal_faces)):
		neighbour_file.write("%d\n" % internal_faces[i][2])
	neighbour_file.write(")\n")
	neighbour_file.write(bottom_separator)
	neighbour_file.close()

	owner_file = open(os.path.join(output_dir, "owner"), "w")
	owner_file.write(file_header)
	owner_file.write(write_FoamFile(2.0, "ascii", "labelList", "owner"))
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
	boundary_file.write(write_FoamFile(2.0, "ascii", "polyBoundaryMesh", "boundary"))
	boundary_file.write(top_separator)
	start_face = len(internal_faces)
	boundary_file.write("%d\n(\n" % num_boundaries)
	for i in range(num_boundaries):
		boundary_file.write("""boundary_%d
	{
		type patch;
		nFaces %d;
		startFace %d;
	}
	""" % (i, len(boundary_faces[i]), start_face))
		start_face += len(boundary_faces[i])
	boundary_file.write(")\n")
	boundary_file.write(bottom_separator)
	boundary_file.close()