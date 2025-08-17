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
from spatial_deformation_cal import spatial_deformation_calculation
import pandas as pd


np.set_printoptions(precision=6, suppress=True)

''' draw a circle trajectory for the end-effector of a robot '''

Kv = np.array([48000, 121000, 16100, 15300, 15100, 4110]) # stiffness vector of 6 of the robot joint
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
x_list[0,:] = 0.4 + radius * np.cos(angle) # x coordinates
x_list[1,:] = 0.1+ radius * np.sin(angle) # y coordinates
x_list[2,:] = 0 # rotation around z axis, constant at 0
# print("x_list:", x_list[:,1800])

plant_height= 0.34 # height of the end effector in meters
# z coordinate of end effector, constant at 0.34 m

# The target pose of the end-effector in the world frame
target_pose = np.zeros((4, 4))

target_pose[:3, 3] = x_list[:, -1]
target_pose[3, 3] = 1.0

lmbda_stiff =50 # stiffness regularization parameter
cost_list = np.zeros(3600) # store the cost for each step
sigma_alpha_list = np.zeros(3600) # store the stiffness in x direction
sigma_beta_list = np.zeros(3600) # store the stiffness in y direction
sigma_alphabeta_list = np.zeros(3600) # store the stiffness in xy direction

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
    phi= i / 10 * np.pi / 180  # convert to radians
    pose = robot.oMf_calculation_ee(q_list[:, i], ee_frame_id) # calculate the end-effector pose in the world frame
    pose_matrix = pose.homogeneous
    _, stiffness_wrench_align_world = robot.spatial_deformation_calculation(q_list[:, i], stiffness_matrix_inv, ee_frame_id)

    # if i == 0 or i == 450:
    #     print(stiffness_wrench_align_world*10e3)  # print the stiffness in m/kN
    sigma_alpha_list[i] = stiffness_wrench_align_world[0, 0]*1000000 # convert to um/N
    sigma_beta_list[i] = stiffness_wrench_align_world[1, 1]*1000000 # convert to um/N
    sigma_alphabeta_list[i] = stiffness_wrench_align_world[0, 1]*1000000 # convert to um/N
    cost = 0.5 * lmbda_stiff * (np.cos(phi)**2*sigma_alpha_list[i] + np.sin(phi)**2*sigma_beta_list[i] + 2*np.cos(phi)*np.sin(phi)*sigma_alphabeta_list[i])**2
    cost_list[i] = cost  # store the cost for each step


print("cost_list:", cost_list)

# Set font to Times New Roman with fallback options
plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 20

# Create a single plot showing the circle trajectory with radius affected by cost
plt.figure(figsize=(8, 6))

# Normalize cost squared to reasonable radius modulation
cost_squared = np.sqrt(cost_list)
cost_normalized = cost_squared / np.max(cost_squared) * 0.02  # scale to max 2cm modulation
modulated_radius = radius + cost_normalized

# Calculate modulated circle coordinates
x_modulated = modulated_radius * np.cos(angle)
y_modulated = modulated_radius * np.sin(angle)

plt.plot(x_list[1, :]-0.1, x_list[0, :]-0.4, 'b--', alpha=0.5, label='Original Circle')
plt.plot(y_modulated, x_modulated, 'r-', linewidth=2, label='Cost-Modulated Circle')
plt.xlabel('X Position (m)', fontsize=18)
plt.ylabel('Y Position (m)', fontsize=18)
plt.title('Circle with Radius Modulated by Cost²', fontsize=20)
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.savefig('/home/he/Meiji_Denso/circle_trajectory_analysis_1.png', dpi=300, bbox_inches='tight')
plt.show()

# Export cost list and stiffness data to Excel
data_dict = {
    'Angle_deg': np.arange(0, 360, 0.1),
    'X_position': x_list[0, :],
    'Y_position': x_list[1, :],
    'X_modulated': x_modulated,
    'Y_modulated': y_modulated,
    'Modulated_radius': modulated_radius,
    'Cost': cost_list,
    'Cost_squared': cost_squared,
    'Sigma_alpha_um_per_N': sigma_alpha_list,
    'Sigma_beta_um_per_N': sigma_beta_list,
    'Sigma_alphabeta_um_per_N': sigma_alphabeta_list
}

df = pd.DataFrame(data_dict)
csv_filename = '/home/he/Meiji_Denso/trajectory_cost_analysis_1.csv'
df.to_csv(csv_filename, index=False)
print(f"Data exported to {csv_filename}")

# print(q_list)
# oMf_zero = robot.oMf_calculation_ee(q_list[:, 1800], ee_frame_id) # calculate the end-effector pose in the world frame
# print("End-effector pose in world frame at the end of the trajectory:")
# print(oMf_zero)

print("Starting continuous circular motion. Press Ctrl+C to stop...")
try:
    while True:
        for i in range(3600):
            if i % 10 == 0:
                pose = robot.oMf_calculation_ee(q_list[:, i], ee_frame_id) # calculate the end-effector pose in the world frame
                pose_matrix = pose.homogeneous
                robot.display(q_list[:, i])  # display the robot configuration
                draw_arrow(robot, "stiffness", x_length = float(sigma_alpha_list[i])*10e-2, y_length = float(sigma_beta_list[i])*10e-2, z_length = 1, XY_cross = float(sigma_alphabeta_list[i])*10e-2, pose_matrix = pose_matrix) # draw the end-effector frame in the world frame
                time.sleep(dt/5)  # wait for dt seconds


except KeyboardInterrupt:
    print("\nMotion stopped by user.")
    print("Final position displayed.")