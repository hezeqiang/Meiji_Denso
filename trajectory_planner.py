import numpy as np
from ddp import DDPSolver
from ddp_linear import DDPSolverLinearDyn
import pinocchio as pin
import os

from utils.robot_loaders import load_ee_frame
from utils.robot_wrapper import RobotWrapper

r = load_ee_frame()

robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

model = robot.model
data = robot.data

q = pin.neutral(model)                       # any configuration you want
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)       # fills data.oMf

# Print the robot tool frame model and data
tool_frame_id = model.getFrameId("tool_frame")
print(f"tool_joint_id = {tool_frame_id}")
print("Joint tool pose in world frame:\n",
      robot.data.oMf[tool_frame_id])  # SE3 of the joint frame






