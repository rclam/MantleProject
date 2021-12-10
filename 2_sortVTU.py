import os
import numpy as np

from vtk import vtkXMLUnstructuredGridReader
from vtk.util.numpy_support import vtk_to_numpy

numb = 50


# empty list of time steps
a_T = []
print(type(a_T))


#command: docker run -it -v "${HOME}/projects/aspect/cookbooks/mantle_outputs/output_mantle_config_0050/solution:/home/pv-user/data" kitware/paraview:pv-v5.7.1-osmesa-py3
#command: /opt/paraview/bin/pvpyton ./2_sortVTU.py
directory = '/home/pv-user/data'

#command: docker run -it -v "${HOME}/projects/aspect/cookbooks/mantle_outputs:/home/pv-user/data" kitware/paraview:pv-v5.7.1-osmesa-py3
#directory = "/home/pv-user/data/output_mantle_config_0050/solution"


for f in os.listdir(directory):
    if f.endswith(".vtu"):
        filename = f
        #print(filename)
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        data = reader.GetOutput()
        T_hist = vtk_to_numpy(data.GetPointData().GetVectors('T'))

        a_T.append((T_hist))

#np.save('a_T_0050.npy',a_T, allow_pickle=True)
np.save('a_T_00{0}.npy'.format(numb),a_T, allow_pickle=True)

#print(a_T[0])


