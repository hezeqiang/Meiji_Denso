# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os
from math import sqrt

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

q0 = np.array([0, 39.6, 102.8, 0, -52.38, 0]) *2*np.pi / 360 # initial configuration

dt = 0.01*1               # DDP time step

N_STARTUP = int(200/1)    # horizon size to seek for the first point with maximum x-axis stiffness
N = int(30/1)             # horizon size

dt_sim = 1e-3

use_viewer = True
show_floor = False

PRINT_T = 1                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]

