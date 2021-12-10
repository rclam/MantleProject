#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read in all npy files and save in one npy

@author: rclam
"""

import numpy as np
#from numpy import load

import os



#data_dir = "/home/pv-user/data"  #docker
data_dir = "/Users/rclam/Documents/F21- Thomas-ML/ML scripts/MantleProject_npy"

a_T = []

for f in os.listdir(data_dir):
    if f.startswith("a_T_"):
        filename = f
        # print(filename)
        a_T.append((filename))


n1, n2 = 21, 6912  # n1=timesteps, n2=points of data

combined_data = np.zeros((len(a_T), n1, n2))
for i, fn in enumerate(a_T):
    combined_data[i, :, :] = np.load(fn)

np.save('2_allData.npy',combined_data)

