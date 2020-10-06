import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # 3D display
import pickle 
import pyvista as pv
from geomdl import BSpline, operations

from Bifurcation import Bifurcation
from ArterialTree import ArterialTree
from Spline import Spline
from utils import quality, distance

from numpy.linalg import norm 


def test_bifurcation_class():


	S0 =[[ 32.08761717, 167.06666271, 137.34338173,   1.44698439], [ 0.65163598, -0.50749161,  0.56339026, -0.02035281]]
	S1 = [[ 32.54145209, 166.84075994, 141.89954624,   0.73235938], [-0.7741084 ,  0.39475545,  0.49079378, -0.06360652]]
	S2 = [[ 37.10561944, 165.62299463, 140.86549835,   1.08367909], [ 0.95163039, -0.03598218,  0.30352055, -0.03130735]]


	bif = Bifurcation(S0, S1, S2, 0.5)

	mesh = bif.surface_mesh()
	mesh.plot(show_edges=True)
	mesh.save("Results/bifurcation.vtk")


def test_bifurcation_smoothing():

	S0 =[[ 32.08761717, 167.06666271, 137.34338173,   1.44698439], [ 0.65163598, -0.50749161,  0.56339026, -0.02035281]]
	S1 = [[ 32.54145209, 166.84075994, 141.89954624,   0.73235938], [-0.7741084 ,  0.39475545,  0.49079378, -0.06360652]]
	S2 = [[ 37.10561944, 165.62299463, 140.86549835,   1.08367909], [ 0.95163039, -0.03598218,  0.30352055, -0.03130735]]

	bif_ref = Bifurcation(S0, S1, S2, 0)
	mesh_ref = bif_ref.surface_mesh()
	mesh_ref.plot(show_edges=True)

	mean_distance = [distance(mesh_ref, mesh_ref)[0]]
	mean_quality = [quality(mesh_ref)[0]]
	n_iter = [0]

	for i in range(10):
		bif_ref.smooth(100)
		mesh = bif_ref.surface_mesh()
		mesh.plot(show_edges=True)
		mean_distance.append(distance(mesh, mesh_ref, True)[0])
		mean_quality.append(quality(mesh, True)[0])
		n_iter.append(n_iter[-1] + 100)
 
	fig, ax = plt.subplots() 
	ax.set_ylabel('distance', fontsize=60) 
	ax.set_xlabel('n_iter', fontsize=60) 
	ax.plot(n_iter, mean_distance, color = 'red')
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.show() 

	fig, ax = plt.subplots() 
	ax.set_ylabel('cell quality', fontsize=60) 
	ax.set_xlabel('n_iter', fontsize=60) 
	ax.plot(n_iter, mean_quality, color = 'red')
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.show() 


def test_tree_class():

	tree = ArterialTree("TestPatient", "BraVa", "Results/refence_mesh_simplified_centerline.swc")

	#tree.deteriorate_centerline(0.05, [0.0, 0.0, 0.0, 0.0])
	#tree.write_swc("Results/refence_mesh_simplified_centerline.swc")

	tree.show()

	tree.spline_approximation()
	tree.show(False)



def test_ogrid_pattern():
 
	crsec = np.array([[34.476392923549895, 161.46798074886593, 120.75770866695724], [34.14434732316482, 161.07809281313507, 120.85266059483581], [33.721564070035285, 160.77760980972167, 120.9000635198961], [33.23685514417969, 160.58700915873672, 120.89668701113229], [32.72325265787573, 160.51927997956236, 120.84276117203669], [32.21575777179679, 160.57903790383153, 120.741960959427], [31.748955423754175, 160.76221052776, 120.60115574121237], [31.354657422339653, 161.05631493973408, 120.42994116032247], [31.059734524741355, 161.44130841010823, 120.2399852074995], [30.88428523904748, 161.8909542701083, 120.04423306701126], [30.840266144134407, 162.37460989745014, 119.85602492373634], [30.930677068575473, 162.85931496040916, 119.68818685161025], [31.149356657281132, 163.31203760995768, 119.55215673788287], [31.481402257666204, 163.70192554568854, 119.4572048100043], [31.904185510795745, 164.00240854910194, 119.40980188494402], [32.388894436651334, 164.1930092000869, 119.41317839370782], [32.902496922955294, 164.26073837926126, 119.46710423280342], [33.40999180903424, 164.20098045499208, 119.5679044454131], [33.87679415707685, 164.0178078310636, 119.70870966362774], [34.27109215849138, 163.72370341908953, 119.87992424451764], [34.56601505608967, 163.33870994871538, 120.06988019734061], [34.74146434178355, 162.88906408871534, 120.26563233782885], [34.785483436696616, 162.40540846137347, 120.45384048110377], [34.69507251225556, 161.92070339841445, 120.62167855322986]])
	center = np.array([ 32.81287479, 162.39000918, 120.1549327])

	#crsec = crsec[::2]
	crsec2 = np.array([[34.32744144745753, 167.33359178216872, 139.75840924324035], [34.02026945083537, 167.34870016821867, 139.78879039934256], [33.717862093845795, 167.3207500583885, 139.81445169543412], [33.41116926252608, 167.25102779158624, 139.83805920914952], [33.09448875488814, 167.12439902551372, 139.85835730156177], [32.78605798115766, 166.91164385302693, 139.86737007275417], [32.537606322492735, 166.59780269739528, 139.86109149865445], [32.411822721680636, 166.2061153348049, 139.83727652409027], [32.435828653114044, 165.79536598763164, 139.7994385261573], [32.59051958808648, 165.422213099016, 139.75307819960594], [32.83814973648061, 165.11874782079002, 139.70256968617386], [33.142389907519934, 164.8945993393273, 139.64864671546368], [33.484522694152, 164.74686960921488, 139.5986381770904], [33.70360182717187, 164.66118767066132, 139.2665953281323], [33.96316055938328, 164.6932594996356, 138.93025095926774], [34.23899573891874, 164.85862855770316, 138.6305189005574], [34.49986491538833, 165.14530394793616, 138.40738592683581], [34.71365729401687, 165.5226760533064, 138.2931738619465], [34.85524594230583, 165.94304533567075, 138.30055429363728], [34.91259833660647, 166.35478921418178, 138.4201371010893], [34.89697326703166, 166.7166072735139, 138.62530602382026], [34.81623529422784, 167.003712522333, 138.8890770141987], [34.68513863756305, 167.203361721164, 139.18298166989322], [34.51708658360998, 167.31167009634225, 139.48026132516557]])
	center2 = (crsec2[0] + crsec2[12])/2
	
	tree = ArterialTree("TestPatient", "BraVa")
	tree.ogrid_pattern(center2, crsec2, [0.5, 0.4, 0.1], 5, 25)


def test_meshing():

	#tree = ArterialTree("TestPatient", "BraVa", "Data/braVa_p2_full.swc")
	tree = ArterialTree("TestPatient", "BraVa", "Data/refence_mesh_simplified_centerline.swc")

	#tree.deteriorate_centerline(1, [0.0, 0.0, 0.0, 0.0])
	tree.show(True, False, False)
	tree.spline_approximation(0.05)
	tree.show(True, True, False)


	#file = open('Results/tree_crsec_ref_mesh.obj', 'rb') 	 
	#tree = pickle.load(file)
	#tree.show()
	
	tree.subgraph([4,5])
	tree.show(True, False, False)


	tree.compute_cross_sections(24, 0.3, bifurcation_model=False)

	file = open('Results/tube_tree.obj', 'wb') 
	pickle.dump(tree, file)

	mesh = tree.mesh_surface()

	print("plot mesh")
	mesh.plot(show_edges=True)
	mesh.save("Results/surface_mesh.vtk")
	mesh.save("Results/surface_mesh.stl")

	mesh = tree.mesh_volume([0.2, 0.3, 0.5], 5, 10)
	mesh.plot(show_edges=True)
	mesh.save("Results/volume_mesh.vtk")



def test_fitting():

	D = np.array([[93.91046111154787, 181.88254391779975, 214.13217284407793, 0.9308344650856465], [93.06141936094309, 178.7939280727319, 213.35499871161392, 1.285582084835041], [87.12149585611041, 173.88847286748307, 213.0933392632536, 1.245326754665641], [88.72064373772372, 167.85629776284827, 213.23791337736043, 1.2156057018939763], [95.63034638734848, 166.47106157620692, 217.3498134331477, 0.5916894764860996], [87.59871877556981, 172.384069768833, 219.1565118077874, 0.16595909410171705], [96.06576337092793, 175.64169053013188, 221.47013007324222, 0.44741711425241704], [96.49090672544531, 174.13046182636438, 223.0878945823948, 0.28742055566016933]])
	#D = np.array([[ 95.17, 184.3606, 213.4994, 0.93], [ 93.93, 181.8806, 214.1194, 0.93], [ 91.76, 178.1606, 213.4994, 1.24], [ 89.28, 175.0606, 212.8794, 1.24], [ 87.42, 172.5806, 213.4994, 1.24], [ 88.01283199, 172.01515466, 213.12311999, 1.26105746]])
	deriv = [D[1] - D[0], D[-2] - D[-1]]
	clip = [D[0], D[-1]]

	n = len(D)
	lbd = [0.0, 0.01, 0.02, 0.05, 0.1]
	#lbd = [0.0]
	spl = Spline()
	spl.distance_constraint_approximation(D, 0.0, clip=clip, deriv=deriv)
	spl.show(data=D)

	"""
	for l in lbd:

		fig = plt.figure(figsize=(10,7))
		ax = Axes3D(fig)
		ax.set_facecolor('white')

		spl.pspline_approximation(D, n, l, clip = clip, deriv=deriv, derivative = False)

		points = spl.get_points()
		dist = spl.distance(D)
		print(np.mean(dist))
		for pt in D : 
			t = spl.project_point_to_centerline(pt[:-1])
			proj = spl.point(t)
			ax.scatter(proj[0], proj[1], proj[2],  c='red', s = 40)
			ax.plot([proj[0], pt[0]], [proj[1], pt[1]], [proj[2], pt[2]])

		ax.plot(points[:,0], points[:,1], points[:,2],  c='black')
		ax.scatter(D[:,0], D[:,1], D[:,2],  c='blue', s = 40)

		# Set the initial view
		ax.view_init(90, -90) # 0 is the initial angle

		# Hide the axes
		ax.set_axis_off()
		plt.show()
	"""



#test_tree_class()
#test_ogrid_pattern()
test_meshing()
#test_bifurcation_smoothing()
#test_bifurcation_class()
#test_fitting()