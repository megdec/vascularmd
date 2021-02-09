import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # 3D display
import pickle 
import pyvista as pv
from geomdl import BSpline, operations

from Bifurcation import Bifurcation
from Trifurcation import Trifurcation
from ArterialTree import ArterialTree
from Spline import Spline
from Model import Model
from utils import quality, distance, lin_interp

from numpy.linalg import norm 
import time


def test_bifurcation_class():


	S0 =[[ 32.08761717, 167.06666271, 137.34338173,   1.44698439], [ 0.65163598, -0.50749161,  0.56339026, -0.02035281]]
	S1 = [[ 32.54145209, 166.84075994, 141.89954624,   0.73235938], [-0.7741084 ,  0.39475545,  0.49079378, -0.06360652]]
	S2 = [[ 37.10561944, 165.62299463, 140.86549835,   1.08367909], [ 0.95163039, -0.03598218,  0.30352055, -0.03130735]]


	bif = Bifurcation(np.array([S0, S1, S2]), 0.5)
	bif.cross_sections(24, 0.2)


	mesh = bif.mesh_surface()
	bif.show(nodes = True)

	#bif.local_smooth(0)
	#quality(mesh, display=True, metric='scaled_jacobian')
	mesh.plot(show_edges=True)
	#mesh.save("Results/bifurcation.vtk")


def test_trifurcation_class():

	S0 =[[ 32.08761717, 167.06666271, 137.34338173,   1.44698439], [ 0.65163598, -0.50749161,  0.56339026, -0.02035281]]
	S1 = [[ 32.54145209, 166.84075994, 141.89954624,   0.73235938], [-0.7741084 ,  0.39475545,  0.49079378, -0.06360652]]
	S2 = [[ 37.10561944, 165.62299463, 140.86549835,   1.08367909], [ 0.95163039, -0.03598218,  0.30352055, -0.03130735]]
	S3 = (np.array(S1) + np.array(S2)) / 2
	S3[0] = S3[0] + 4* S3[1]
	S3 = S3.tolist()


	trif = Trifurcation(np.array([S0, S3, S2, S1]), 0.5)
	mesh = trif.mesh_surface()
	mesh.plot(show_edges=True)


def test_trifurcation_nonplanar():

	S0 =[[ 32.08761717, 167.06666271, 137.34338173,   1.44698439], [ 0.65163598, -0.50749161,  0.56339026, -0.02035281]]
	S1 = [[ 32.54145209, 166.84075994, 141.89954624,   0.73235938], [-0.7741084 ,  0.39475545,  0.49079378, -0.06360652]]
	S2 = [[ 37.10561944, 165.62299463, 140.86549835,   1.08367909], [ 0.95163039, -0.03598218,  0.30352055, -0.03130735]]
	S3 = (np.array(S1) + np.array(S2)) / 2
	S3[0] = S3[0] + 4* S3[1]
	S3[0, :-1] = S3[0, :-1] + 3* np.array([0.2,0.8,0])
	S3 = S3.tolist()


	trif = Trifurcation(np.array([S0, S3, S2, S1]), 0.5)
	mesh = trif.mesh_surface()
	mesh.plot(show_edges=True)



def test_bifurcation_smoothing():

	S0 =[[ 32.08761717, 167.06666271, 137.34338173,   1.44698439], [ 0.65163598, -0.50749161,  0.56339026, -0.02035281]]
	S1 = [[ 32.54145209, 166.84075994, 141.89954624,   0.73235938], [-0.7741084 ,  0.39475545,  0.49079378, -0.06360652]]
	S2 = [[ 37.10561944, 165.62299463, 140.86549835,   1.08367909], [ 0.95163039, -0.03598218,  0.30352055, -0.03130735]]

	bif_ref = Bifurcation(S0, S1, S2, 0)
	mesh_ref = bif_ref.mesh_surface()
	mesh_ref.plot(show_edges=True)

	mean_distance = [distance(mesh_ref, mesh_ref)[0]]
	mean_quality = [quality(mesh_ref)[0]]
	n_iter = [0]

	for i in range(10):
		bif_ref.smooth(100)
		mesh = bif_ref.mesh_surface()
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
	tree.ogrid_pattern(24, center2, crsec2, [0.5, 0.4, 0.1], 5, 25)

def test_bif_ogrid_pattern():

	crsec24 = np.array([[28.472897149872587, 16.647843302505702, 25.447880980319386], [31.502682236294486, 18.33460378012935, 24.869209603934415], [28.56528143, 16.58960056, 25.01218854], [28.71130444, 16.57125149, 24.60448866], [28.91409403, 16.59404931, 24.22232322], [29.17878826, 16.66244053, 23.87075631], [29.5113068 , 16.78504919, 23.56896745], [29.907562686141308, 16.968895904030084, 23.353170324741292], [30.34150796, 17.21101339, 23.27230498], [30.75996517, 17.48903152, 23.36410605], [31.10246859, 17.76459016, 23.62221757], [31.33387429, 18.0031732 , 23.99636518], [31.4599088 , 18.19182849, 24.42393355], [28.34220219, 17.07771078, 25.43174422], [28.29583079, 17.51642797, 25.40261713], [28.34245508, 17.95534558, 25.35994537], [28.47570745, 18.38538174, 25.30539786], [28.72289127, 18.7738559 , 25.23767493], [29.117923855202317, 19.007499266473335, 25.161104275470123], [29.54503796, 19.14013629, 25.0881257 ], [29.98425684, 19.14023168, 25.02422383], [30.39964512, 19.03592831, 24.97232508], [30.78234036, 18.85951748, 24.9310783 ], [31.142985  , 18.62685482, 24.89763884], [28.6127879 , 16.4021492 , 25.84277083], [28.88617401, 16.21141298, 26.23739254], [29.28738932, 16.1065126 , 26.58848509], [29.7878501 , 16.11184639, 26.84898294], [30.33726855, 16.2365906 , 26.98005583], [30.879496054470728, 16.464770173411626, 26.972512755818382], [31.3602181 , 16.77777477, 26.8215766 ], [31.71637317, 17.1480424 , 26.52945955], [31.90482738, 17.52906104, 26.13371446], [31.91309777, 17.873516  , 25.68931998], [31.76222318, 18.14602388, 25.2536228 ]])
	crsec32 = np.array([[28.472897149872587, 16.647843302505702, 25.447880980319386], [31.502682236294486, 18.33460378012935, 24.869209603934415], [28.53695265, 16.6002297 , 25.11822942], [28.63140865, 16.57541017, 24.80514352], [28.75699139, 16.57333665, 24.50676019], [28.91409403, 16.59404931, 24.22232322],[29.10633337, 16.64055209, 23.95491674],[29.33635974, 16.71629485, 23.71156018], [29.60489654, 16.82504979, 23.50495863], [29.907562686141308, 16.968895904030084, 23.353170324741292], [30.23199204, 17.14589528, 23.27741207], [30.55673645, 17.34773985, 23.29562693], [30.85487119, 17.55975331, 23.41415192],[31.10246859, 17.76459016, 23.62221757], [31.28676357, 17.94814397, 23.89558113], [31.40789093, 18.10335627, 24.20702618], [31.47606079, 18.23074765, 24.53492039], [28.36677038, 16.96912593, 25.43704896], [28.30802957, 17.29629106, 25.41884261], [28.2998665 , 17.6265537 , 25.39302533], [28.34245508, 17.95534558, 25.35994537], [28.43745032, 18.27738982, 25.31979344], [28.58445881, 18.58771236, 25.27303349], [28.80939071, 18.85120686, 25.21876688], [29.117923855202317, 19.007499266473335, 25.161104275470123], [29.43527451, 19.11983373, 25.10575331], [29.76590132, 19.15606586, 25.05469373], [30.09116786, 19.12283944, 25.01009338], [30.39964512, 19.03592831, 24.97232508], [30.68959182, 18.90962781, 24.94047322], [30.96393075, 18.74950435, 24.91365748], [31.23185437, 18.55983195, 24.89019113], [28.56562446, 16.45972526, 25.74254673], [28.7328811 , 16.29814806, 26.04245924], [28.97513612, 16.17592945, 26.33116808], [29.28738932, 16.1065126 , 26.58848509], [29.65584306, 16.09918042, 26.79494139], [30.06019109, 16.15943365, 26.93251798], [30.4751948 , 16.28545363, 26.99009698], [30.879496054470728, 16.464770173411626, 26.972512755818382], [31.24910851, 16.69278544, 26.87293781], [31.55666011, 16.95865236, 26.69087783], [31.78026465, 17.24422842, 26.43827591], [31.90482738, 17.52906104, 26.13371446], [31.92707236, 17.79303815, 25.80164819], [31.8554447 , 18.0203202 , 25.46722867],[31.70329565, 18.20010923, 25.15166022]])

	center = (crsec24[0] + crsec24[1]) /2
	tree = ArterialTree("TestPatient", "BraVa")
	vertices = tree.bif_ogrid_pattern_vertices(center, crsec24, [0.5, 0.4, 0.1], 5, 25)
	faces = tree.bif_ogrid_pattern_faces(24, 5, 25)

	face_ord = tree.reorder_faces([[0, 1, 0], [1, 0, 1]], faces, 24, 5, 25)

	for i in range(1, face_ord.shape[0] - 1, 20):
		mesh = pv.PolyData(vertices, face_ord[:i])
		mesh.plot(show_edges = True)
	

def test_deformation():

	ref_surface = pv.read("Data/reference_mesh_aneurisk.vtp")

	tree = ArterialTree("TestPatient", "BraVa", "Data/refence_mesh_simplified_centerline.swc")
	tree.spline_approximation()
	tree.compute_cross_sections(24, 0.2, bifurcation_model=False)
	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)

	tree.deform(ref_surface)
	mesh = tree.mesh_surface()
	mesh.plot(show_edges=True)
	mesh.save("Results/Aneurisk/Deformation/deformed_surface.vtk")

	mesh = tree.mesh_volume([0.2, 0.3, 0.5], 5, 10)
	mesh.plot(show_edges=True)
	mesh.save("Results/Aneurisk/Deformation/deformed_volume.vtk")


def test_meshing():

	#tree = ArterialTree("TestPatient", "BraVa", "Data/braVa_p3_full.swc")
	tree = ArterialTree("TestPatient", "BraVa", "Data/refence_mesh_simplified_centerline.swc")

	#tree.subgraph([1,2,3,4,5,6])
	#tree.subgraph([1,2])
	
	#tree.write_vtk("full", "Results/test.vtk")
	tree.deteriorate_centerline(1, [0.0, 0.0, 0.0, 0.0])
	tree.show(True, False, False)
	#tree.write_vtk("full", "Results/test_deteriorate.vtk")
	tree.spline_approximation()
	tree.show(True, True, False)
	#tree.write_vtk("spline", "Results/test_fitting.vtk")


	#file = open('Results/tree_crsec_ref_mesh.obj', 'rb') 	 
	#tree = pickle.load(file)
	#tree.show()
	
	
	#tree.show(True, False, False)

	t1 = time.time()
	tree.compute_cross_sections(24, 0.2)
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )
	#file = open('Results/tube_tree.obj', 'wb') 
	#pickle.dump(tree, file)
	t1 = time.time()
	mesh = tree.mesh_surface()
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	print("plot mesh")
	mesh.plot(show_edges=True)
	mesh.save("Results/Aneurisk/mesh_surface.vtk")
	#mesh.save("Results/mesh_surface.stl")

	mesh = tree.mesh_volume([0.2, 0.3, 0.5], 1, 1)
	mesh = mesh.compute_cell_quality()
	mesh['CellQuality'] = np.absolute(mesh['CellQuality'])
	mesh.plot(show_edges=True, scalars= 'CellQuality')
	mesh.save("Results/Aneurisk/volume_mesh.vtk")



def test_fitting():

	tree = ArterialTree("TestPatient", "BraVa", "Data/refence_mesh_simplified_centerline.swc")

	crsec = tree.get_topo_graph()
	D = crsec.edges[(4,5)]["coords"][::2]

	#D = np.array([[93.91046111154787, 181.88254391779975, 214.13217284407793, 0.9308344650856465], [93.0 6141936094309, 178.7939280727319, 213.35499871161392, 1.285582084835041], [87.12149585611041, 173.88847286748307, 213.0933392632536, 1.245326754665641], [88.72064373772372, 167.85629776284827, 213.23791337736043, 1.2156057018939763], [95.63034638734848, 166.47106157620692, 217.3498134331477, 0.5916894764860996], [87.59871877556981, 172.384069768833, 219.1565118077874, 0.16595909410171705], [96.06576337092793, 175.64169053013188, 221.47013007324222, 0.44741711425241704], [96.49090672544531, 174.13046182636438, 223.0878945823948, 0.28742055566016933]])
	#D = np.array([[ 95.17, 184.3606, 213.4994, 0.93], [ 93.93, 181.8806, 214.1194, 0.93], [ 91.76, 178.1606, 213.4994, 1.24], [ 89.28, 175.0606, 212.8794, 1.24], [ 87.42, 172.5806, 213.4994, 1.24], [ 88.01283199, 172.01515466, 213.12311999, 1.26105746]])

	values = np.vstack((D[0], (D[1] - D[0])*15, (D[-1] - D[-2])*15, D[-1]))

	n = 6
	
	spl = Spline()
	spl.approximation(D, [True, False, True, True], values, True, False, lbd = 0, criterion="None")
	spl.show(False, False, data=D)


def test_fitting_angle(): 

	p0 = np.array([0,0])
	p1 = np.array([5,15])
	p2 = np.array([10,0])
	num = 5

	D = np.array(lin_interp(p0, p1, num) + lin_interp(p1, p2, num))
	values = np.vstack((D[0], np.zeros((2,2)), D[-1]))
	values = np.vstack((D[0], (D[1] - D[0]), (D[-1] - D[-2]), D[-1]))

	spl = Spline()
	spl.approximation(D, [1,0,0,1], values, False, False, lbd = 100, criterion = "None")
	spl.show(data=D)


def test_quality_model2D():

	N = 50 # number of data points
	t = np.linspace(0, 4*np.pi, N)
	f = 1.15247
	data = 3.0*np.sin(f*t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise


	D = np.zeros((N, 2))
	D[:,0] = t
	D[:,1] = data
	plt.scatter(D[:,0], D[:,1],  c='blue', s = 40)
	plt.show()

	n = 20
	spl = Spline()
	spl.automatic_approximation(D, clip = [D[0], D[-1]])

	points = spl.get_points()

	plt.plot(points[:,0], points[:,1],  c='black')
	plt.scatter(D[:,0], D[:,1],  c='blue', s = 40)
	plt.plot(D[:,0], D[:,1],  c='blue')
	plt.show()


def test_quality_model3D():

	# Import reference network
	tree = ArterialTree("TestPatient", "BraVa", "Data/refence_mesh_simplified_centerline.swc")

	# Extract one vessel
	e = [4,5]

	# Get data
	pts = tree.get_topo_graph().edges[e]['coords'][:, :-1]
	spl_ref = Spline()
	spl_ref.automatic_approximation(pts)
	print(len(pts))
	pts = spl_ref.get_points()
	print(len(pts))

	# Add noise or resample
	noise = [0.02, 0.02, 0.02]
	p = 0.2

	if p != 0:
		# Resampling
		step = int(pts.shape[0]/(p*pts.shape[0]))
		if step > 0:
			pts =  pts[:-1:step]
		else:
			pts = pts[int(pts.shape[0]/2)]
	else: 
		pts = pts[int(pts.shape[0]/2)]

	rand = np.hstack((np.random.normal(0, noise[0], (pts.shape[0], 1)), np.random.normal(0, noise[1], (pts.shape[0], 1)), np.random.normal(0, noise[2], (pts.shape[0], 1)))) #np.random.normal(0, noise[3], (pts.shape[0], 1))))
	pts += pts * rand

	spl_aicc = Spline()
	spl_aicc.automatic_approximation(pts, criteria="AICC")


	spl_cv = Spline()
	spl_cv.automatic_approximation(pts, criteria="CV")


	# Compare the splines 
	fig = plt.figure(figsize=(10,7))
	ax = Axes3D(fig)
	ax.set_facecolor('white')

	points = spl_ref.get_points()
	ax.plot(points[:,0], points[:,1], points[:,2],  c='black')

	points = spl_aicc.get_points()
	ax.plot(points[:,0], points[:,1], points[:,2],  c='red')

	points = spl_cv.get_points()
	ax.plot(points[:,0], points[:,1], points[:,2],  c='green')

	ax.scatter(pts[:,0], pts[:,1], pts[:,2],  c='blue', s = 40)

	# Set the initial view
	ax.view_init(90, -90) # 0 is the initial angle

	# Hide the axes
	ax.set_axis_off()
	plt.show()



def test_Model():

	D = np.array([[93.91046111154787, 181.88254391779975, 214.13217284407793, 0.9308344650856465], [93.06141936094309, 178.7939280727319, 213.35499871161392, 1.285582084835041], [87.12149585611041, 173.88847286748307, 213.0933392632536, 1.245326754665641], [88.72064373772372, 167.85629776284827, 213.23791337736043, 1.2156057018939763], [95.63034638734848, 166.47106157620692, 217.3498134331477, 0.5916894764860996], [87.59871877556981, 172.384069768833, 219.1565118077874, 0.16595909410171705], [96.06576337092793, 175.64169053013188, 221.47013007324222, 0.44741711425241704], [96.49090672544531, 174.13046182636438, 223.0878945823948, 0.28742055566016933]])
	#D = np.array([[ 95.17, 184.3606, 213.4994, 0.93], [ 93.93, 181.8806, 214.1194, 0.93], [ 91.76, 178.1606, 213.4994, 1.24], [ 89.28, 175.0606, 212.8794, 1.24], [ 87.42, 172.5806, 213.4994, 1.24], [ 88.01283199, 172.01515466, 213.12311999, 1.26105746]])
	values = np.vstack((D[0], D[1] - D[0], D[-2] - D[-1], D[-1]))

	n = len(D)
	lbd = 0
	#lbd = [0.0]
	model = Model(D, n, 3, [True, True, True, True], values, False, lbd)
	model.spl.show(data=D)
	print(model.quality("SSE"))
	model.set_lambda(1)
	print(model.quality("SSE"))
	model.spl.show(data=D)
	print(model.get_magnitude())

def test_rotations():

		file = open("tmp.obj", 'rb') 
		tree = pickle.load(file)

		t1 = time.time()
		tree.compute_cross_sections(32, 0.2)
		t2 = time.time()
		print("The cross section computation process took ", t2 - t1, "seconds." )

def test_brava():

	for i in [2]:

		filename = "P" + str(i) + ".swc"
		print(filename)
		tree = ArterialTree("TestPatient", "BraVa", "/home/decroocq/Documents/Thesis/Data/BraVa/Centerlines/Registered/" + filename)
		tree.write_vtk("topo", "Results/BraVa/registered/topo/P" + str(i) + ".vtk")
	
		t1 = time.time()
		tree.spline_approximation()
		t2 = time.time()
		print("The approximation process took ", t2 - t1, "seconds." )
		tree.write_vtk("spline", "Results/BraVa/registered/splines/P" + str(i) + ".vtk")
	
		
		t1 = time.time()
		tree.compute_cross_sections(32, 0.2)
		t2 = time.time()
		print("The cross section computation process took ", t2 - t1, "seconds." )

		
		file = open("Results/BraVa/registered/crsec/P" + str(i) + ".obj", 'wb') 
		pickle.dump(tree, file)


		t1 = time.time()
		mesh = tree.mesh_surface()
		mesh.save("Results/BraVa/registered/surface/P" + str(i) + ".vtk")
		t2 = time.time()
		print("The surface meshing process took ", t2 - t1, "seconds." )
		"""
		t1 = time.time()
		mesh = tree.mesh_volume([0.2, 0.3, 0.5], 5, 10)
		mesh.save("Results/BraVa/volume/P" + str(i) + ".vtk")
		t2 = time.time()
		print("The volume meshing process took ", t2 - t1, "seconds." )
		"""
		
def meshing_brava():
	i = 2

	file = open("Results/BraVa/registered/crsec/P" + str(i) + ".obj", 'rb') 
	tree = pickle.load(file)
	tree.show(False, True, False)
	
	tree.cut_branch((49,50), preserve_shape = True)
	tree.write_vtk("spline", "Results/BraVa/registered/splines/remove_branch_preserveP" + str(i) + ".vtk")
	
	tree.compute_cross_sections(24, 0.4)
	
	t1 = time.time()
	mesh = tree.mesh_surface()

	mesh.save("Results/BraVa/registered/surface/remove_branchP" + str(i) +".vtk")
	t2 = time.time()
	print("The surface meshing process took ", t2 - t1, "seconds." )

	
	"""
	t1 = time.time()
	mesh = tree.mesh_volume([0.2, 0.3, 0.5], 5, 10)
	mesh.save("Results/BraVa/volume/P" + str(i) + ".vtk")
	t2 = time.time()
	print("The surface meshing process took ", t2 - t1, "seconds." )
	"""
	
def test_bifurcation_resampling():

	# Create a bifurcation
	S0 =[[ 32.08761717, 167.06666271, 137.34338173,   1.44698439], [ 0.65163598, -0.50749161,  0.56339026, -0.02035281]]
	S1 = [[ 32.54145209, 166.84075994, 141.89954624,   0.73235938], [-0.7741084 ,  0.39475545,  0.49079378, -0.06360652]]
	S2 = [[ 37.10561944, 165.62299463, 140.86549835,   1.08367909], [ 0.95163039, -0.03598218,  0.30352055, -0.03130735]]

	bif = Bifurcation(S0, S1, S2, 0.5)

	# Initial surface mesh
	init_mesh = bif.mesh_surface()
	init_mesh.plot(show_edges=True)
	init_mesh.save('Results/Bifurcation/bifurcation.ply')

	# Smooth 
	bif.smooth(5)
	smooth_mesh = bif.mesh_surface()
	smooth_mesh.plot(show_edges=True)

	# Backprojection
	bif.deform(init_mesh)
	resampled_mesh = bif.mesh_surface()
	resampled_mesh.plot(show_edges=True)


def test_bifurcation_local_smooth():

	# Create a bifurcation
	S0 =[[ 32.08761717, 167.06666271, 137.34338173,   1.44698439], [ 0.65163598, -0.50749161,  0.56339026, -0.02035281]]
	S1 = [[ 32.54145209, 166.84075994, 141.89954624,   0.73235938], [-0.7741084 ,  0.39475545,  0.49079378, -0.06360652]]
	S2 = [[ 37.10561944, 165.62299463, 140.86549835,   1.08367909], [ 0.95163039, -0.03598218,  0.30352055, -0.03130735]]

	bif = Bifurcation(S0, S1, S2, 0.5)

	# Initial surface mesh
	init_mesh = bif.mesh_surface()
	init_mesh.plot(show_edges=True)

	init_mesh['curvature'] = np.abs(init_mesh.curvature())

	# Smooth 		
	p = pv.Plotter()
	p.add_mesh(init_mesh, scalars = 'curvature')
	p.show()

	bif.local_smooth(0.8)
	smooth_mesh = bif.mesh_surface()
	smooth_mesh.plot(show_edges=True)



def test_nb_control_points():

	tree = ArterialTree("TestPatient", "BraVa", "Data/refence_mesh_simplified_centerline.swc")
	tree.deteriorate_centerline(0.5, [0.01, 0.01, 0.01, 0.0])

	crsec = tree.get_topo_graph()
	D = crsec.edges[(1,2)]["coords"]
	values = np.vstack((D[0], (D[1] - D[0])*30, (D[-1] - D[-2])*30, D[-1]))
	print(len(D))

	spl_prec = Spline()
	spl_prec.approximation(D, [False, False, False, False], values, derivatives = False, n = 4)
	distance = []
	for n in range(5, 80):
		
		spl = Spline()
		spl.approximation(D, [False, False, False, False], values, derivatives = False, n = n)
		if n in [20, 50, 70]:
			spl.show(data=D)
		
		p1 = spl_prec.get_points()
		p2 = spl.get_points()
		dist = 0
		for i in range(len(p1)):
			dist += norm(p1[i] - p2[i])
		distance.append(dist)

		spl_prec = spl

	plt.plot(range(5,80), distance)
	plt.show()
		

def test_compare_image():

	i = 5

	filename = "P" + str(i) + ".swc"
	print(filename)

	#tree = ArterialTree("TestPatient", "BraVa", "/home/decroocq/Documents/Thesis/Data/BraVa/Centerlines/Registered/" + filename)

	#tree.spline_approximation()
	#tree.write_vtk("spline", "Results/BraVa/registered/splines/P" + str(i) + ".vtk")

	#tree.compute_cross_sections(16, 0.8, bifurcation_model=False)

	#file = open("Results/BraVa/registered/crsec/P" + str(i) + ".obj", 'wb') 
	#pickle.dump(tree, file)

	file = open("Results/BraVa/registered/crsec/P" + str(i) + ".obj", 'rb') 
	tree = pickle.load(file)
	mesh = tree.mesh_surface()
	mesh.save("Results/BraVa/registered/surface/P" + str(i) + ".vtk")

	tree.compare_image("/home/decroocq/Documents/Thesis/Data/BraVa/MRA/renamed/MRA" + str(i) + ".nii.gz", data_type = "spline")


def register_swc_nii():

	import nibabel as nib

	for i in range(1, 59):
		try :
			data = nib.load("/home/decroocq/Documents/Thesis/Data/BraVa/MRA/renamed/MRA" + str(i) +".nii.gz")
		except:
			print("File " + "MRA" + str(i) +".nii.gz"+" is missing")
		else:

			pix_dim = np.array(data.header['pixdim'][1:4]) # Dimensions
			vox_dim = np.array(data.header['dim'][1:4])

			dim = pix_dim * vox_dim
			print(dim)

			file_in = np.loadtxt("/home/decroocq/Documents/Thesis/Data/BraVa/Centerlines/Renamed/" + "P" + str(i) + ".swc", skiprows=0)
			file_out = open("/home/decroocq/Documents/Thesis/Data/BraVa/Centerlines/Registered/" + "P" + str(i) + ".swc", 'w') 

			for j in range(0, file_in.shape[0]):

				# Brava database conversion to nii (x, ysize- z, zsize - y)
				txt = str(int(file_in[j, 0])) + '\t' + str(int(file_in[j, 1])) + '\t' + str(file_in[j, 2])
				txt = txt + '\t' + str(dim[1] - file_in[j, 4]) + '\t' + str(dim[2] + file_in[j, 3]) + '\t' + str(file_in[j, 5]) 
				txt = txt + '\t' + str(int(file_in[j, 6])) + '\n'

				file_out.write(txt) 
				

		file_out.close()


def register_centerlines():

	for i in range(1,59):

		filename = "P" + str(i) + ".swc"
		print(filename)

		try:
			tree = ArterialTree("TestPatient", "BraVa", "/home/decroocq/Documents/Thesis/Data/BraVa/Centerlines/Registered/" + filename)
		except : 
			print("No MRA for " + "P" + str(i))
		else:
			tree.write_vtk("full", "Results/BraVa/registered/centerlines/P" + str(i) + ".vtk")
	

def test_cut_branch():

		
	tree = ArterialTree("TestPatient", "BraVa", "Data/refence_mesh_simplified_centerline.swc")

	tree.deteriorate_centerline(1, [0.0, 0.0, 0.0, 0.0])
	tree.show(True, False, False)

	tree.spline_approximation()
	
	#tree.cut_branch((2,4), preserve_shape = True)
	tree.show(False, True, False)


	t1 = time.time()
	tree.compute_cross_sections(24, 0.2)
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	t1 = time.time()
	mesh = tree.mesh_surface()
	t2 = time.time()
	print("The process took ", t2 - t1, "seconds." )

	print("plot mesh")
	mesh.plot(show_edges=True)
	mesh.save("Results/Aneurisk/mesh_surface_full.vtk")





#test_tree_class()
#test_ogrid_pattern()
#test_bif_ogrid_pattern()
#test_meshing()
#test_bifurcation_smoothing()
#test_bifurcation_class()
#test_fitting()
#test_quality_model2D()
#test_quality_model3D()
#test_Model()
#test_fitting_angle()
#test_brava()
#meshing_brava()
#test_deformation()
#test_bifurcation_resampling()
#test_bifurcation_local_smooth()
#test_nb_control_points()
#test_bifurcation_class()
#test_trifurcation_class()
#test_trifurcation_nonplanar()
#test_compare_image()
#register_swc_nii()
#register_centerlines()
#test_rotations()
test_cut_branch()