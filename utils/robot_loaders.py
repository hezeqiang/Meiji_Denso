#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 07:09:47 2020

@author: student
"""
import sys
import os
from os.path import dirname, exists, join

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from example_robot_data.robots_loader import getModelPath, readParamsFromSrdf

def load(URDF_FILENAME='', path=''):
    try:
        path = os.getcwd()     # e.g. "/home/he/Meiji_Denso"
        urdf = path + "/r068_description/r068.urdf"        
        mesh_dir = path

        print("Loading URDF from:", urdf)
        robot = RobotWrapper.BuildFromURDF(urdf, [mesh_dir, ])
        print("loaded robot:",robot)
        return robot

    except Exception as e:
        print("Error loading robot model:", e)
        pass


def load_ee_frame(URDF_FILENAME='', path=''):
    try:
        path = os.getcwd()     # e.g. "/home/he/Meiji_Denso"
        urdf = path + "/r068_description/r068.urdf"        
        mesh_dir = path

        print("Loading URDF from:", urdf)
        robot = RobotWrapper.BuildFromURDF(urdf, [mesh_dir, ])
        model = robot.model
        # ------------------------------------------------------------------
        # Identify the joint that owns the link you care about
        # ------------------------------------------------------------------
        joint_6_id = model.getJointId("joint_6")

        # ------------------------------------------------------------------
        # Build the SE3 placement of the new frame w.r.t. that link
        # (pure translation, identity rotation)
        # ------------------------------------------------------------------
        offset_xyz   = np.array([0.1, 0.0, 0.05]) # expressed in the coordinate frame of the link that follows, not world frame
        M_offset     = pin.SE3.Identity()
        M_offset.rotation = np.array([[0, 0, -1],    # identity rotation
                                      [0, 1, 0],
                                      [1, 0, 0]])
        M_offset.translation[:] = offset_xyz  # R = I, p = xyz

        # ------------------------------------------------------------------
        # 4.  Add the frame to the Model as a *FIXED* frame
        #     parent = jid   (the joint that carries the link)
        #     previousFrame = jid   (standard for link-attached frames)
        # ------------------------------------------------------------------
        frame_name = "tool_frame"
        new_fid = model.addFrame(
                    pin.Frame(frame_name,
                            joint_6_id,               # parent joint
                            joint_6_id,               # previous frame (same joint frame)
                            M_offset,
                            pin.FrameType.JOINT))

        print(f"Added frame '{frame_name}' with id {new_fid}")

        robot.data = model.createData()

        print("loaded robot:",robot)
        return robot

    except Exception as e:
        print("Error loading robot model:", e)
        pass

if __name__ == "__main__":
    robot = load_ee_frame()
    model = robot.model
    data = robot.data
    q = pin.neutral(model)                       # any configuration you want
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)       # fills data.oMf

    print("Robot loaded successfully")


    for fr in model.frames:
        print(fr.name)

    ee_joint_id = model.getFrameId("joint_6")
    print(f"ee_joint_id = {ee_joint_id}")
    print("Joint 6 pose in world frame:\n",
        robot.data.oMf[ee_joint_id])  # SE3 of the joint frame
    
    tool_frame_id = model.getFrameId("tool_frame")
    print(f"tool_frame_id = {tool_frame_id}")
    print("Joint tool pose in world frame:\n",
        robot.data.oMf[tool_frame_id])  # SE3 of the joint frame


