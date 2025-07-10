import matplotlib.pyplot as plt
import utils.plot_utils as plut
import time
import pinocchio as pin
from utils.robot_loaders import load_ee_frame
from utils.robot_wrapper import RobotWrapper
import ddp_manipulator_conf as conf
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf
from utils.meshcat_arrow import draw_arrow

np.set_printoptions(precision=6, suppress=True)

''' draw a circle trajectory for the end-effector of a robot '''

Kv = np.array([121000, 136000, 59100, 15300, 15100, 4110]) # stiffness vector of 6 of the robot joint
Kv_inv = 1/Kv
stiffness_matrix_inv = np.diag(1/Kv)

r = load_ee_frame() # example_robot_data.load("ur5", True) #loadUR()#

robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
nq, nv = robot.nq, robot.nv
ee_frame_id = robot.model.getFrameId("tool_frame") # get the end-effector frame id
dt = 0.1 # time step for the IK solver

robot.initVisualization(robot.q0)
pose = robot.oMf_calculation_ee(robot.q0, ee_frame_id) # calculate the end-effector pose in the world frame
pose_matrix = pose.homogeneous
draw_arrow(robot, "stiffness", 1,1,1,1, pose_matrix) # draw the end-effector frame in the world frame


print("Displaying start")
time.sleep(1.)

# data structure to store the trajectory
q_list = np.zeros((robot.nq, 3600)) # 0-360 degrees, 1 configuration for each 0.1 degree

# circle trajectory parameters
radius = 0.1 # radius of the circle in meters
angle= np.linspace(0, 2 * np.pi, 3600) # angles from 0 to 2*pi radians (360 degrees)
x_list = np.zeros((3, 3600)) # (x,y) of end effector in the plane. 0-360 degrees, 1 state for each 0.1 degree
x_list[0,:] = 0.6 + radius * np.cos(angle) # x coordinates
x_list[1,:] = radius * np.sin(angle) # y coordinates
x_list[2,:] = 0 # rotation around z axis, constant at 0
# print("x_list:", x_list[:,1800])

plant_height= 0.34 # height of the end effector in meters
# z coordinate of end effector, constant at 0.34 m

# The target pose of the end-effector in the world frame
target_pose = np.zeros((4, 4))

target_pose[:3, 3] = x_list[:, -1]
target_pose[3, 3] = 1.0

for i in range(3600):
    # print("alpha, beta, theta of x:", x)
    alpha= x_list[0, i]
    beta = x_list[1, i]
    theta_z = x_list[2, i]

    # Calculate the end-effector pose in the world frame
    oMf = pin.SE3.Identity()  # create an identity SE3 object
    oMf.translation[0] = alpha
    oMf.translation[1] = beta
    oMf.translation[2] = plant_height  # set the height of the end-effector
    oMf.rotation = pin.rpy.rpyToMatrix(0, 0, theta_z) @ oMf.rotation

    # if i == 0:
    #     print("Initial end-effector pose in world frame:")
    #     print(oMf)
    q_list[:, i] = robot.IK_step(ee_frame_id, oMf, dt=dt)

# print(q_list)
# oMf_zero = robot.oMf_calculation_ee(q_list[:, 1800], ee_frame_id) # calculate the end-effector pose in the world frame
# print("End-effector pose in world frame at the end of the trajectory:")
# print(oMf_zero)

print("Starting continuous circular motion. Press Ctrl+C to stop...")
try:
    while True:
        for i in range(3600):
            if i % 10 == 0:
                robot.display(q_list[:, i])  # display the robot configuration
                pose = robot.oMf_calculation_ee(q_list[:, i], ee_frame_id) # calculate the end-effector pose in the world frame
                pose_matrix = pose.homogeneous
                _, stiffness_wrench_align_world = robot.spatial_deformation_calculation(q_list[:, i], stiffness_matrix_inv, ee_frame_id)

                if i == 0 or i == 450:
                    print(stiffness_wrench_align_world*10e3)  # print the stiffness in m/kN


                draw_arrow(robot, "stiffness", x_length = float(stiffness_wrench_align_world[0,0])*10e4, y_length = float(stiffness_wrench_align_world[1,1])*10e4, z_length = 1, XY_cross = float(stiffness_wrench_align_world[1,0])*10e4, pose_matrix = pose_matrix) # draw the end-effector frame in the world frame
                time.sleep(dt/5)  # wait for dt seconds


except KeyboardInterrupt:
    print("\nMotion stopped by user.")
    print("Final position displayed.")