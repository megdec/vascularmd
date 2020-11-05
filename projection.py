import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # 3D display
import pickle 
import pyvista as pv
from geomdl import BSpline, operations
import vtk
from Bifurcation import Bifurcation
from ArterialTree import ArterialTree
from Spline import Spline
from utils import quality, distance
import Pymesh


from numpy.linalg import norm


# Import mesh

mesh = pv.read("/home/maury/Documents/Erwan/Meghane/Complet/ref_mesh_full.stl")
mesh.save("/home/maury/Documents/Erwan/Meghane/Complet/ref_mesh_full.vtk")


filename = '/home/maury/Documents/Erwan/Meghane/Complet/network_tree.obj'
file = open(filename, 'rb')
tree = pickle.load(file)
crsec_graph = tree.get_crsec_graph()

p = pv.Plotter()
p.add_mesh(mesh)  #scalars = 'implicit_distance'



def projection_arterialTree_object(filename, mesh):
    # Import python object
    file = open(filename, 'rb')      
    tree = pickle.load(file)
    crsec_graph = tree.get_crsec_graph()
    for e in crsec_graph.edges():

        
        # Get the start cross section only if it is a terminating edge
        if crsec_graph.in_degree(e[0]) == 0:

            crsec = crsec_graph.nodes[e[0]]['crsec']
            center = (crsec[0] + crsec[int(crsec.shape[0]/2)])/2.0 # Compute the center of the section
            
            nouvelle_crsec = np.zeros([crsec.shape[0], 3])
            for i in range(crsec.shape[0]):
                nouvelle_crsec[i] = intersection(mesh, crsec, center, i)
            crsec_graph.nodes[e[0]]['crsec'] = nouvelle_crsec

                # Get the connection cross sections

        crsec_array = crsec_graph.edges[e]['crsec']  # In edges, the sections are stored as an array of cross section arrays
        
        nouvelle_crsec = np.zeros([crsec_array.shape[0], crsec_array.shape[1], 3])
        for i in range(crsec_array.shape[0]):
                crsec = crsec_array[i] # Get indivisual cross section
                center = (crsec[0] + crsec[int(crsec.shape[0]/2)])/2.0
            
                for j in range(crsec.shape[0]):
                    nouvelle_crsec[i, j, :] = intersection(mesh, crsec, center, j)

        crsec_graph.edges[e]['crsec'] = nouvelle_crsec

                # Get the end cross section

        # Handle bifurcation case here (later)
        if crsec_graph.nodes[e[1]]['type'] == "bif": # If bifurcation
            crsec = np.array(crsec_graph.nodes[e[1]]['crsec'])
            center = crsec_graph.nodes[e[1]]['coords']
            

        else:
            crsec = crsec_graph.nodes[e[1]]['crsec']
            center = (crsec[0] + crsec[int(crsec.shape[0]/2)])/2.0
        nouvelle_crsec = np.zeros([crsec.shape[0], 3])

        for i in range(crsec.shape[0]):
            nouvelle_crsec[i, :] = intersection(mesh, crsec, center, i)
            

        if crsec_graph.nodes[e[1]]['type'] == "bif": # If bifurcation
            crsec_graph.nodes[e[1]]['crsec'] = nouvelle_crsec.tolist()
            
        else :
            crsec_graph.nodes[e[1]]['crsec'] = nouvelle_crsec
                        
    mesh_deform = tree.mesh_surface()
    mesh_deform.save("/home/maury/Documents/Erwan/Meghane/Complet/mesh_deform.vtk") 
    mesh_deform.plot(show_edges = True)

def intersection(mesh, crsec, center,i):
    coord = crsec[i] # Coordinates of the node
    normal = coord - center
    normal = normal / norm(normal) # Normal=direction of the projection 
    symetric = 2*coord - center #Coordinates of the symetric of center
    points, ind = mesh.ray_trace(center, symetric)
    if len(points) > 0 :
        inter = points[0]
    else :
        inter = coord
    return inter



filename = '/home/maury/Documents/Erwan/Meghane/Complet/network_tree.obj'

projection_arterialTree_object(filename, mesh)
p = pv.Plotter()
p.add_mesh(mesh)  #scalars = 'implicit_distance'

p.show()

            
