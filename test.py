import ddp_manipulator_stiffness_trajectory_joint_space
import matplotlib.pyplot as plt
import utils.plot_utils as plut
import time
from utils.robot_loaders import load_ee_frame
from utils.robot_wrapper import RobotWrapper
import ddp_manipulator_conf as conf
import numpy as np
np.set_printoptions(precision=3, suppress=True)

''' Test DDP with a manipulator
'''
print("".center(conf.LINE_WIDTH,'#'))
print(" DDP - Manipulator ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

N = conf.N               # horizon size
N_STARTUP = conf.N_STARTUP # startup horizon size to seek for the first point with maximum x-axis stiffness

dt = conf.dt             # control time step
mu = 1e-4                # initial regularization factor for not invertible system Quu: the influence of  input u on the system dynamics
phi = 0 # angle of the trajectory in radians, used to calculate the stiffness matrix

ddp_params = {}
ddp_params['alpha_factor'] = 1 # line search factor of the step size, default 1 means full step, 0 means no step of u
ddp_params['min_alpha_to_increase_mu'] = 0.1 # use small step size for convergence around the local minimum
ddp_params['max_line_search_iter'] = 6

ddp_params['mu_factor'] = 3. # the scaling factor of the mu: regularization term, for not invertible system Quu
ddp_params['mu_max'] = 1e0
ddp_params['mu_min'] = 1e-8

ddp_params['min_cost_impr'] = 1e-1 # minimum cost improvement to increase mu. Large mu -> slow convergence
ddp_params['exp_improvement_threshold'] = 1e-3 # threshold to stop iteration
ddp_params['max_iter'] = 50 #  full loop itaration number forward + backward pass
DEBUG = False


r = load_ee_frame() # example_robot_data.load("ur5", True) #loadUR()#

robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
nq, nv = robot.nq, robot.nv
