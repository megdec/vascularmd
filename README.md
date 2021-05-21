# Structured meshing of arterial trees

This code provide several tools to work with arterial network: 

- A data structure based on spatial graph that can be used to visualize and edit vascular trees
- A parametric modeling framework
- A meshing method to obtain high quality hexahedral surface and volume mesh
- A set of functions to interact with numerical simulation tool OpenFoam for computational fluid dynamics

It is divided in four main classes:

**ArterialTree**: Vascular network object. Gathers functions to store, visualize, generate parametric model, apply post_treatments and mesh vascular trees from centerlines.

**Nfurcation**: Nfurcation object. Gathers functions to store, visualize, apply post_treatments and mesh bifurcations, trifurcations, nfurcations from a set of parameters. Used in ArterialTree.

**Spline**: Spline object with tools to evualuate the coordinates and derivatives and to approximate 3D data points. Used in ArterialTree and Nfurcation.

**Model**: Computation of the control points of the spline model. Used in Spline class.

**Simulation**: Interaction with openFoam simulation software via vtk for numerical simulation and result visualization.

<img src="Documentation/class_diagram.svg" alt="Class diagram" width="1300"/>

*Required python3 packages: numpy, pyvista, matplotlib, math, goemdl, networkx, scipy.spatial*


<img src="Documentation/graphs.svg" alt="Spatial graph attributes of the ArterialTree class" width="800"/>

**Figure. Spatial graph attributes of the ArterialTree class**

