import pip 

try : 
    import numpy 
except : 
    pip.main(['install' , 'numpy']) 
try : 
    import pyvista 
except : 
    pip.main(['install' , 'pyvista']) 
try : 
    import scipy 
except : 
    pip.main(['install' , 'scipy']) 
try : 
    import networkx 
except : 
    pip.main(['install' , 'networkx']) 
try : 
    import nibabel 
except : 
    pip.main(['install' , 'nibabel']) 
try : 
    import geomdl 
except : 
    pip.main(['install' , 'geomdl']) 


print("********* TEST EXIT FUNCTION ****************")

try : 
    import vpython 
except : 
    pip.main(['install' , 'vpython']) 
