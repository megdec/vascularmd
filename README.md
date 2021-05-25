# Structured meshing of arterial trees

This code provide several tools to work with arterial network: 

- A data structure based on spatial graph that can be used to visualize and edit vascular trees
- A parametric modeling framework
- A meshing method to obtain high quality hexahedral surface and volume mesh
- A set of functions to interact with numerical simulation tool OpenFoam for computational fluid dynamics

## Installation

The code runs with python3.

1) Install the following python3 packages are required : **numpy, pyvista, matplotlib, math, goemdl, networkx, scipy.spatial, pickle **. They can be installed using pip3 or anaconda.

2) Clone the git repository

## Example use

### Nfurcations

<img src="Documentation/bifurcation_model.svg" alt="Bifurcation model parameters" width="400"/>

*Bifurcation model parameters*

### Arterial trees
 
## Code overview

It is divided in four main classes:

<img src="Documentation/class_diagram.svg" alt="Class diagram" width="1300"/>


**ArterialTree**: Vascular network object. Gathers functions to store, visualize, generate parametric model, apply post_treatments and mesh vascular trees from centerlines.

<img src="Documentation/graphs.svg" alt="Spatial graph attributes of the ArterialTree class" width="800"/>

*Spatial graph attributes of the ArterialTree class*


**Nfurcation**: Nfurcation object. Gathers functions to store, visualize, apply post_treatments and mesh bifurcations, trifurcations, nfurcations from a set of parameters. Used in ArterialTree.

**Spline**: Spline object with tools to evualuate the coordinates and derivatives and to approximate 3D data points. Used in ArterialTree and Nfurcation.

**Model**: Computation of the control points of the spline model. Used in Spline class.

**Simulation**: Interaction with openFoam simulation software via vtk for numerical simulation and result visualization.



