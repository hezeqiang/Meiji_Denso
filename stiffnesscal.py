
from pathlib import Path
from sys import argv

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
import matplotlib.pyplot as plt
import numpy as np

import os
import sys


def calculate_stiffness_matrix_local(model, data, q, stiffness_matrix_inv, ee_frame_id):
    """
    Compute the end-effector stiffness in the LOCAL frame:
      K_task = J_local @ Kq⁻¹ @ J_localᵀ
    where J_local is the 6×nv Jacobian expressed in the end-effector joint frame.
    """
    # 1) update kinematics
    pin.computeJointJacobians(model, data, q)
    pin.framesForwardKinematics(model, data, q)
    # 2) get LOCAL jacobian (6×nv)
    Je_local = pin.getFrameJacobian(
        model, data, ee_frame_id, pin.LOCAL
    )
    # print(f"Je_local = {Je_local}")
    # 3) task-space stiffness
    return Je_local @ stiffness_matrix_inv @ Je_local.T


def calculate_stiffness_matrix_align_world(model, data, q, stiffness_matrix_inv, ee_frame_id):

    # Calculate the Jacobian of the end effector frame 
    pin.computeJointJacobians(model, data, q)
    pin.framesForwardKinematics(model, data, q)
    Je = pin.getFrameJacobian(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED) # for [v w]
    # one-shot, returns a 6×nv matrix expressed in LWA axes

    return Je @ stiffness_matrix_inv @ Je.T

if __name__ == "__main__":
    pass