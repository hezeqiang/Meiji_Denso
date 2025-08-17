from pathlib import Path
from sys import argv

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from stiffnesscal import calculate_stiffness_matrix_local, calculate_stiffness_matrix_align_world


def spatial_deformation_calculation(model, data, q, stiffness_matrix_inv, ee_frame_id, spatial_wrench_align_world=np.zeros(6), stiffness_matrix_frame=0):
    """
    This function calculates the spatial deformation of a robot model.
    """
    # Compute forward kinematics + frame placements
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # end effector frame in world frame
    oMf = data.oMf[ee_frame_id] 
    # Ad = oMf.toActionMatrix()
    # print(f"oMf = {oMf}")
    R_of = oMf.rotation # 3×3 matrix

    if stiffness_matrix_frame == 0: # stiffness matrix in LOCAL_ALIGN_WORLD frame
       
        stiffness_wrench_align_world = calculate_stiffness_matrix_align_world(model, data, q, stiffness_matrix_inv, ee_frame_id) 
        # print(f"stiffness_wrench_align_world= {stiffness_wrench_align_world}")
        body_align_twist_dt = stiffness_wrench_align_world @ spatial_wrench_align_world
        print(f"body_align_twist_dt = {body_align_twist_dt}")
        # print(stiffness_wrench_align_world[:,3:6])
        return body_align_twist_dt, stiffness_wrench_align_world[0:3,0:3]
    
    else: # stiffness matrix in LOCAL frame
        spatial_wrench_world_align_pin = pin.Force(
            spatial_wrench_align_world[3:6], # torque in LOCAL_WORLD_ALIGNED axes
            spatial_wrench_align_world[0:3]) # force in LOCAL_WORLD_ALIGNED axes
        # notice that order is [force torque] in the data

        body_wrench = pin.Force( 
            R_of.T @ spatial_wrench_world_align_pin.angular,  # new torque
            R_of.T @ spatial_wrench_world_align_pin.linear )  # new force
        # print(f"body_wrench  = {body_wrench }")
        # notice that order is [force torque] in the data

        stiffness_wrench_local = calculate_stiffness_matrix_local(model, data, q, stiffness_matrix_inv, ee_frame_id)
    
        # for row in Stiffness_wrench:
        #     print("[" + "  ".join(f"{val:8.8f}" for val in row) + "]")

        # The twist of the end effector frame over the time
        body_twist_dt = stiffness_wrench_local @ body_wrench # [force torque] in body frame
        body_twist_dt_pin = pin.Motion(body_twist_dt[0:3],body_twist_dt[3:6]) # [v w]

        body_align_twist_dt =  pin.Motion(
                        R_of @ body_twist_dt_pin.linear, R_of @ body_twist_dt_pin.angular)

        print(f"body_align_twist_dt = {body_align_twist_dt}")

        return body_align_twist_dt, stiffness_wrench_local[0:3,0:3]

    # stiffness calculation in x y z direction and around x y z
    # stiffness_x = body_twist_dt[0]/body_wrench[0]
    # stiffness_y = body_twist_dt[1]/body_wrench[1]
    # stiffness_z = body_twist_dt[2]/body_wrench[2]


if __name__ == "__main__":
    # 1) Parent directory of the package:
    root = os.getcwd()     # e.g. "/home/he/Meiji_Denso"
    mesh_dir = root
    print(mesh_dir)
    # 2) Path to your URDF:
    urdf_model_path = os.path.join(root, "r068_description", "r068.urdf")

    # # Load model
    # model = pin.buildModelFromUrdf(urdf_path, package_dirs=[model_dir])
    # data = model.createData()

    # # Build the full robot (model + geometry), here as a fixed‐base
    # robot = RobotWrapper.BuildFromURDF(
    #     urdf,
    #     root,                # where to look up package://… meshes
    # )
    # model = robot.model
    # data  = robot.data
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_model_path, mesh_dir
    )
    # model, collision_model, visual_model = pin.buildModelsFromUrdf(
    #     urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
    # )
    data = model.createData()

    Kv = np.array([48000, 121000, 16100, 15300, 15100, 4110]) # stiffness vector of 6 of the robot joint
    Kv_inv = 1/Kv
    # Initialize the stiffness matrix
    stiffness_matrix_inv = np.diag(Kv_inv) # Diagonal matrix with Kv as diagonal elements

    # Use the default (zero) configuration:
    q0 = pin.neutral(model)
    # print(f"q0 = {q0}")
    v0 = pin.utils.zero(model.nv)
    a0 = pin.utils.zero(model.nv)

    # Compute forward kinematics + frame placements
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    ee_frame_id = model.getFrameId("joint_6")
    
    # (printing code is identical to above)
    # print(f"Robot has {model.njoints} joints (incl. universe)")
    # print("\n⎯⎯ Joint Tree ⎯⎯")
    # for i, name in enumerate(model.names):
    #     parent = model.parents[i]
    #     pname  = model.names[parent] if parent!=0 else "universe"
    #     print(f"[{i:2d}] {name:20s}  ← parent: {pname}")

    # print("\n⎯⎯ Link Inertia ⎯⎯")
    # for i in range(1, model.njoints):
    #     I = model.inertias[i]
    #     print(f"\n• Link “{model.names[i]}”")
    #     print(f"   Mass:         {I.mass:.6g}")
    #     print(f"   CoM (lever):  {I.lever.T}")
    #     print(f"   Inertia (3×3):\n{I.inertia.T}")

    # compute and print each joint frame’s world placement --
    # oMi the placement of joint i’s joint frame in the world (“origin”) frame
    # oMf: the placement of frame f in the world frame
    # print("\n⎯⎯ Joint Frame Placements ⎯⎯")
    # for i, name in enumerate(model.names):
    #     M = data.oMi[i]
    #     # M.translation is a (3,) vector; M.rotation is a 3×3 matrix
    #     print(f"[{i:2d}] {name:20s}")
    #     print(f"   Position:      {M.translation.T}")
    #     print(f"   Orientation (R):\n{M.rotation}\n")

    # q = np.array([0, 0, 0, 0, 0, 0]) *2*np.pi / 360
    # pin.forwardKinematics(model, data, q)
    # pin.updateFramePlacements(model, data)
    # print("initial oMf",data.oMf[ee_frame_id])

    q = np.array([0, 39.6, 102.8, 0, -52.38, 0]) *2*np.pi / 360
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    print("initial oMf",data.oMf[ee_frame_id])

    # [force torque]
    spatial_wrench_align_world = np.array([10, 0, 0, 0, 0, 0]) # force and torque in world frame

    space_linear_movement, Stiffness_matrix = spatial_deformation_calculation(model, 
                                    data, 
                                    q, 
                                    stiffness_matrix_inv, 
                                    ee_frame_id, 
                                    spatial_wrench_align_world=spatial_wrench_align_world,
                                    stiffness_matrix_frame=0
    )
    # print(f"space_linear_movement_rotation = {space_linear_movement}")
    print (f"Stiffness_matrix = {Stiffness_matrix}")


    # VISUALIZER = MeshcatVisualizer

    # # Added code for Meshcat visualizer realization
    # if VISUALIZER == MeshcatVisualizer:
    #     try:
    #         viz = MeshcatVisualizer(model, collision_model, visual_model)
    #         viz.initViewer(open=True)
    #     except ImportError as err:
    #         print(
    #             "Error while initializing the viewer. "
    #             "It seems you should install Python meshcat"
    #         )
    #         print(err)
    #         sys.exit(0)

    # # Added code for Meshcat visualizer realization
    # if VISUALIZER == GepettoVisualizer:
    #     try:
    #         viz = GepettoVisualizer(model, collision_model, visual_model)
    #         viz.initViewer()
    #     except ImportError as err:
    #         print(
    #             "Error while initializing the viewer. "
    #             "It seems you should install Python meshcat"
    #         )
    #         print(err)
    #         sys.exit(0)

    # viz.loadViewerModel()
    # viz.display(q)
    # viz.displayVisuals(True)

    input("Press ENTER to continue...")

    # Replace the previous q0 figure with a 6*1 grid of subplots for each q0 element
    # fig_q0 = plt.figure()
    # axes = fig_q0.subplots(6, 1)
    # for i, ax in enumerate(axes.flat):
    #     ax.bar([0], [q0[i]])
    #     ax.set_title(f'q0[{i}] = {q0[i]:.2f}')
    #     ax.set_xticks([])
    #     ax.set_ylabel('Value')

    # # Added figure for the end effector position
    # fig_ee = plt.figure()
    # axes_ee = fig_ee.subplots(3, 1)
    # ee_pos = data.oMi[model.njoints - 1].translation  # end effector position
    # for i, ax_ee in enumerate(axes_ee.flat):
    #     ax_ee.bar([0], [ee_pos[0]])
    #     ax_ee.set_xlabel('X')
    #     ax_ee.set_ylabel('Y')
    #     ax_ee.set_title('End Effector Position')

    # plt.show()
