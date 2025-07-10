import time

from pinocchio.robot_wrapper import RobotWrapper as PinocchioRobotWrapper
from pinocchio.deprecation import deprecated
import pinocchio as pin
import pinocchio.utils as utils
from pinocchio.explog import exp
import numpy as np
from pinocchio.visualize import GepettoVisualizer
from pinocchio.visualize import MeshcatVisualizer
import sys
import subprocess
import os
import math
import gepetto.corbaserver
from stiffnesscal import calculate_stiffness_matrix_local, calculate_stiffness_matrix_align_world

class RobotWrapper(PinocchioRobotWrapper):
    
    @staticmethod
    def BuildFromURDF(filename, package_dirs=None, root_joint=None, verbose=False, meshLoader=None):
        robot = RobotWrapper()
        robot.initFromURDF(filename, package_dirs, root_joint, verbose, meshLoader)
        return robot
    
    @property
    def na(self):
        if(self.model.joints[0].nq==7): # floating base robot
            return self.model.nv-6
        return self.model.nv

    def mass(self, q, update=True):
        if(update):
            return pin.crba(self.model, self.data, q)
        return self.data.M

    def nle(self, q, v, update=True):
        if(update):
            return pin.nonLinearEffects(self.model, self.data, q, v)
        return self.data.nle
        
    def com(self, q=None, v=None, a=None, update=True):
        if(update==False or q is None):
            return PinocchioRobotWrapper.com(self, q);
        if a is None:
            if v is None:
                return PinocchioRobotWrapper.com(self, q)
            return PinocchioRobotWrapper.com(self, q, v)
        return PinocchioRobotWrapper.com(self, q, v,a)
        
    def Jcom(self, q, update=True):
        if(update):
            return pin.jacobianCenterOfMass(self.model, self.data, q) # three by nv
        return self.data.Jcom
        
    def momentumJacobian(self, q, v, update=True):
        if(update):
            pin.ccrba(self.model, self.data, q, v); # computes the centroidal momentum matrix of the robot by q and v
        return self.data.Ag;


    def computeAllTerms(self, q, v):
        ''' pin.computeAllTerms is equivalent to calling:
        pinocchio::forwardKinematics
        Computes the positions and orientations of all joints by propagating the kinematic chain from the base using the given joint configuration (and optionally velocities and accelerations).

        pinocchio::crba
        Implements the Composite Rigid Body Algorithm to compute the joint-space mass (inertia) matrix of the robot, which is used in dynamics calculations.
        
        pinocchio::ccrba
        Computes the centroidal dynamics quantities of the robot. In particular, it’s used to efficiently compute the momentum (both linear and angular) of the robot and provides the centroidal momentum matrix, which is valuable for tasks related to balance and dynamic motions.

        pinocchio::nonLinearEffects
        Calculates the combined effects of gravity, Coriolis, and centrifugal forces acting on the robot, given the current configuration and velocity.

        pinocchio::computeJointJacobians
        Computes the Jacobian matrices for all joints, mapping joint velocities to the spatial velocities of the corresponding bodies.

        pinocchio::centerOfMass
        Determines the overall center of mass of the robot model by aggregating the contributions of all bodies based on their masses and positions.

        pinocchio::jacobianCenterOfMass
        Computes the Jacobian of the robot’s center of mass with respect to its joint coordinates, useful for tasks like balance and motion planning.

        pinocchio::kineticEnergy
        Computes the total kinetic energy of the robot given its mass distribution and joint velocity configuration.

        pinocchio::potentialEnergy
        Calculates the potential energy of the robot, typically due to gravity, based on the configuration and the positions of its centers of mass.
            This is too much for our needs, so we call only the functions
            we need, including those for the frame kinematics
        '''
#        pin.computeAllTerms(self.model, self.data, q, v);
        pin.forwardKinematics(self.model, self.data, q, v, np.zeros(self.model.nv))
        pin.computeJointJacobians(self.model, self.data)
        pin.updateFramePlacements(self.model, self.data)
        pin.crba(self.model, self.data, q)
        pin.nonLinearEffects(self.model, self.data, q, v)
        
        
    def forwardKinematics(self, q, v=None, a=None):
        if v is not None:
            if a is not None:
                pin.forwardKinematics(self.model, self.data, q, v, a)
            else:
                pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q)
               
    def frameJacobian(self, q, index, update=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        ''' Call computeFrameJacobian if update is true. If not, user should call computeFrameJacobian first.
            Then call getFrameJacobian and return the Jacobian matrix.
            ref_frame can be: ReferenceFrame.LOCAL, ReferenceFrame.WORLD, ReferenceFrame.LOCAL_WORLD_ALIGNED

            ReferenceFrame.LOCAL: origin expressed in the local frame, representing the linear and angular velocities of the frame in the local frame.
            ReferenceFrame.WORLD: point coinciding with the origin of the world frame and the velocities are projected in the basis of the world frame, representing the spatial velocity and angular velocities of the frame in the world frame.
            ReferenceFrame.LOCAL_WORLD_ALIGNED: origin coinciding with the local frame, but the XYZ axes are aligned with the world frame. In this way, the velocities are projected in the basis of the world frame, representing the linear and angular velocities of the frame in the world frame.

        '''
        if(update): 
            pin.computeFrameJacobian(self.model, self.data, q, index)
        return pin.getFrameJacobian(self.model, self.data, index, ref_frame)
        
    def frameVelocity(self, q, v, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v) 
            pin.updateFramePlacements(self.model, self.data)
        v_local = pin.getFrameVelocity(self.model, self.data, index) # default expressed in the local frame
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return v_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            v_world = H.act(v_local)
            return v_world # spatial velocity expressed in the world frame
        
        Hr = pin.SE3(H.rotation, np.zeros(3))
        v = Hr.act(v_local) # linear velocity expressed in the world frame
        return v
            

    def frameAcceleration(self, q, v, a, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v, a)
            pin.updateFramePlacements(self.model, self.data)
        a_local = pin.getFrameAcceleration(self.model, self.data, index) #  default expressed in the local frame
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return a_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            a_world = H.act(a_local) # spatial acceleration expressed in the world frame
            return a_world
        
        Hr = pin.SE3(H.rotation, np.zeros(3))
        a = Hr.act(a_local) # linear acceleration expressed in the world frame
        return a
        
        
    def deactivateCollisionPairs(self, collision_pair_indexes):
        for i in collision_pair_indexes:
            self.collision_data.deactivateCollisionPair(i);
            
    def addAllCollisionPairs(self):
        self.collision_model.addAllCollisionPairs();
        self.collision_data = pin.GeometryData(self.collision_model);
        
    def isInCollision(self, q, stop_at_first_collision=True):
        return pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, np.asmatrix(q).reshape((self.model.nq,1)), stop_at_first_collision);

    def findFirstCollisionPair(self, consider_only_active_collision_pairs=True):
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    return (i, self.collision_model.collisionPairs[i]);
        return None;
        
    def findAllCollisionPairs(self, consider_only_active_collision_pairs=True):
        res = [];
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    res += [(i, self.collision_model.collisionPairs[i])];
        return res;

    # initialize the visualization
    def initVisualization(self, q0):
        try:
            self.initViewer(loadModel=True, open=True)
            print(self.viz)
        except ImportError as err:
            print(
                "Error while initializing the viewer. "
                "It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)

        # self.viz.setCameraPosition(conf.CAMERA_TRANSFORM)

        self.displayCollisions(False)
        self.displayVisuals(True)
        self.display(q0)

    def oMf_calculation_ee(self, q, ee_frame_id=None) -> pin.SE3:
        """
        Calculate the end-effector pose in the world frame based on the state x, q.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[ee_frame_id]
        return oMf

    def IK_step(self, ee_frame_id: int, target_pose: pin.SE3, dt: float = 0.01,
                q_init: np.ndarray = np.array([0, 39.6, 102.8, 0, -52.38, 0]) *2*np.pi / 360,
                tol: float = 1e-3,
                max_iter: int = 50,
                damping: float = 1e-6) -> np.ndarray:
        """
        Performs one IK routine to compute the desired joint configuration (q_des)
        that achieves the target end-effector pose.

        Args:
            ee_frame_id (int): The frame id of the end-effector.
            target_pose (pin.SE3): The desired end-effector pose.
            q_init (np.ndarray): The initial guess for the joint configuration.
            tol (float): Convergence tolerance for the error norm.
            max_iter (int): Maximum number of iterations.
            damping (float): Damping factor used in the pseudo-inverse computation.

        Returns:
            np.ndarray: The computed joint configuration (q_des) that (approximately) achieves the target pose.
        """
        
        q = q_init.copy()

        # Tuning parameters for error weighting:
        # You can increase w_rot if the rotational error is too large relative to the translation error.
        w_rot = 2.0      # Weight for the orientation error (first 3 components)
        w_trans = 1.0    # Weight for the translation error (last 3 components)
        # We use self.datapin and self.modelpin for the computations.

        for i in range(max_iter):
            # Update forward kinematics for current configuration
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Get current end-effector pose
            current_pose = self.data.oMf[ee_frame_id]

            pose_des = current_pose.actInv(target_pose) # current_pose.inverse() * target_pose

            # # Compute error transformation: T_error = current_pose⁻¹ * target_pose
            # error_transform = current_pose.inverse() * target_pose
            # Get the 6D error (twist) using the logarithm map
            error_twist = pin.log6(pose_des).vector   # in joint frame
            # print(f"Iteration {i}: error_twist = {error_twist}")

            # Apply separate weights to rotation and translation errors.
            error_twist[:3] *= w_rot
            error_twist[3:] *= w_trans  
            
            error_norm = np.linalg.norm(error_twist)
            # Check convergence
            if error_norm < tol:
                # print(f"IK converged in {i} iterations with error norm: {error_norm:.6f}")
                break
            
            # Compute the Jacobian of the end-effector frame.
            J = pin.computeFrameJacobian(self.model, self.data, q , ee_frame_id, pin.ReferenceFrame.LOCAL)

            # Compute the joint velocity update using a damped least-squares pseudo-inverse of the Jacobian.
            J_pinv = np.linalg.pinv(J, rcond=damping)
            dq = J_pinv.dot(error_twist)
            # print(f"Iteration {i}: dq = {dq}")

            # Update configuration taking the manifold structure into account.
            q = pin.integrate(self.model, q, dq * dt)
            self.q = q  # Update instance variable
            # Update the joint configuration
            # print(f"Iteration {i}: Updated joint configuration q = {self.q}")
        
        # print(f"IK completed with final error norm: {error_norm:.6f} after {i+1} iterations.")

        return self.q

    def spatial_deformation_calculation(self, q, stiffness_matrix_inv, ee_frame_id, spatial_wrench_align_world=np.zeros(6), stiffness_matrix_frame=0):
        """ This function calculates the spatial deformation of a robot model."""
        # model: pinocchio model of the robot
        # Compute forward kinematics + frame placements
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # end effector frame in world frame
        oMf = self.data.oMf[ee_frame_id] 
        # Ad = oMf.toActionMatrix()
        # print(f"oMf = {oMf}")
        R_of = oMf.rotation # 3×3 matrix

        if stiffness_matrix_frame == 0: # stiffness matrix in LOCAL_ALIGN_WORLD frame

            stiffness_wrench_align_world = calculate_stiffness_matrix_align_world(self.model, self.data, q, stiffness_matrix_inv, ee_frame_id) 
            # print(f"stiffness_wrench_align_world= {stiffness_wrench_align_world}")
            body_align_twist_dt = stiffness_wrench_align_world @ spatial_wrench_align_world
            # print(f"body_align_twist_dt = {body_align_twist_dt}")
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

            stiffness_wrench_local = calculate_stiffness_matrix_local(self.robot.model, self.robot.data, q, stiffness_matrix_inv, ee_frame_id)
        
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


__all__ = ['RobotWrapper']
