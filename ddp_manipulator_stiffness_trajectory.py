# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:52:58 2016

@author: adelpret
"""

import numpy as np
from ddp import DDPSolver
import pinocchio as pin
import os
from stiffnesscal import calculate_stiffness_matrix_local, calculate_stiffness_matrix_align_world
from utils.robot_wrapper import RobotWrapper
from utils.robot_loaders import load_ee_frame
from utils.plot_traj import plot_frame, FrameAnimator


os.environ["PINOCCHIO_VIEWER"] = "meshcat"

class DDPSolverManipulatorStiffness_SE2(DDPSolver):
    ''' 
    The nonlinear system dynamics are defined
    This task is to find the optimal trajectory of a manipulator in SE(2) space that exhibit the maximum stiffness in the tangent direction.
    The trajectory (circles) is divieded into N=360 segments by angle phi 
    Kinematic model:
    x = [alpha, beta, theta]
    x_(k+1)= x_k + h * [v*cos(theta), v*sin(theta), omega], h = dt
    The cost function for a specific phi:
    J = l_f(x_N, phi) + sum_{k=0}^{N-1}l_k(x_k, u_k, phi)
    l_f(x_N) =0.5 * lmbda_stiff (cos(phi)^2*sigma_alpha + sin(phi)^2*sigma_beta + 2*cos(phi)*sin(phi)*sigma_theta)^2
    l_k= 0.5 * u_k' * lmbda  u_k + 0.5 * (cos(phi)^2*sigma_alpha + sin(phi)^2*sigma_beta + 2*cos(phi)*sin(phi)*sigma_theta)^2
    or
    l_k= 0.5 * u_k' * lmbda  u_k
    
    '''
    
    def __init__(self, name, robot:RobotWrapper, ddp_params, lmbda_stiff, lmbda, dt, DEBUG=False, simu=None):
        DDPSolver.__init__(self, name, ddp_params, DEBUG)
        self.robot = robot
        self.ee_frame_id = robot.model.getFrameId("tool_frame") # end effector
        self.alpha_factor = ddp_params['alpha_factor'] # scaling index for alpha

        self.lmbda = lmbda # control regularization
        self.lmbda_stiff = lmbda_stiff # final cost regularization

        self.nx = 3  # state size: [alpha, beta, theta]
        self.nu = 2 # control size: [v, omega]
        self.q0= np.array([0, 39.6, 102.8, 0, -52.38, 0]) *2*np.pi / 360 # initial configuration of the robot in radians
        self.q_guess_IK = self.q0 # initial guess for the inverse kinematics, used in the cost function
        self.phi = 0
        self.best_alpha_stiffness = 100 # in um/N
        self.best_beta_stiffness = 100 # in um/N
        self.best_alphabeta_stiffness = 100 # in um/N
        self.final_cost = 0.0 # final cost of the trajectory, for comparison with the previous iteration in line search
        self.max_alphabeta = 0.3 # maximum alpha and beta in meters, used to limit the trajectory

        pin.forwardKinematics(self.robot.model, self.robot.data, self.q0)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        self.initial_oMf = self.robot.data.oMf[self.ee_frame_id] # initial end effector frame in world frame
        Kv = np.array([48000, 121000, 16100, 15300, 15100, 4110]) # stiffness vector of 6 of the robot joint
        self.stiffness_matrix_inv = np.diag(1/Kv) # Diagonal matrix with Kv_inv as diagonal elements

        self.plant_height = self.initial_oMf.translation[2] # height of the plant, used to calculate the stiffness matrix
        self.q = np.zeros(self.robot.nq) # robot configuration

        self.dt = dt
        self.simu = simu # simulate object    
        
        self.Fx = np.eye(self.nx) # 3 times 3
        self.Fu = np.zeros((self.nx, self.nu)) # 3 times 2
        self.dx = np.zeros(self.nx) # 3 

        
    ''' System dynamics '''
    def f(self, x, u):

        ''' System dynamics: x_(k+1)= x_k + h * [v*cos(theta), v*sin(theta), omega], h = dt '''
        self.dx[0] = u[0] * np.cos(x[2])
        self.dx[1] = u[0] * np.sin(x[2])
        self.dx[2] = u[1]
        return x + self.dt * self.dx
           
    def f_x(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        self.Fx[0, 2]= -self.dt * u[0] * np.sin(x[2])  # d(alpha)/d(theta)
        self.Fx[1, 2]= self.dt * u[0] * np.cos(x[2]) # d(beta)/d(theta)
        return self.Fx
        
    def f_u(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. u '''
        self.Fu[0, 0] = self.dt * np.cos(x[2])  # d(alpha)/d(v)
        self.Fu[1, 0] = self.dt * np.sin(x[2])  # d(beta)/d(v)
        self.Fu[2, 1] = self.dt              # d(theta)/d(omega)
        return self.Fu
    
    # cost functions related to the task

    def oMf_calculation(self, x):
        """
        Calculate the end-effector pose in the world frame based on the state x.
        The state x is [alpha, beta, theta].
        """
        # print("alpha, beta, theta of x:", x)
        alpha, beta, theta = x
        # Calculate the end-effector pose in the world frame
        oMf = self.initial_oMf.copy()
        oMf.translation[0] += alpha
        oMf.translation[1] += beta
        oMf.translation[2] = self.plant_height  # set the height of the end-effector
        oMf.rotation = pin.rpy.rpyToMatrix(0, 0, theta) @ oMf.rotation
        # create a rotation matrix around the world z-axis
        return oMf

    def IK_step(self, target_x: np.ndarray,
                q_init: np.ndarray = np.array([0, 39.6, 102.8, 0, -52.38, 0]) *2*np.pi / 360,
                tol: float = 1e-4,
                max_iter: int = 10,
                damping: float = 1e-10) -> np.ndarray:
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

        target_pose = self.oMf_calculation(target_x)  # Convert target_x to SE3 pose
        # Tuning parameters for error weighting:
        # You can increase w_rot if the rotational error is too large relative to the translation error.
        w_rot = 2.0      # Weight for the orientation error (first 3 components)
        w_trans = 1.0    # Weight for the translation error (last 3 components)
        # We use self.datapin and self.modelpin for the computations.

        for i in range(max_iter):
            # Update forward kinematics for current configuration
            pin.forwardKinematics(self.robot.model, self.robot.data, q)
            pin.updateFramePlacements(self.robot.model, self.robot.data)

            # Get current end-effector pose
            current_pose = self.robot.data.oMf[self.ee_frame_id]

            pose_des = current_pose.actInv(target_pose) # current_pose.inverse() * target_pose

            # # Compute error transformation: T_error = current_pose⁻¹ * target_pose
            # error_transform = current_pose.inverse() * target_pose
            # Get the 6D error (twist) using the logarithm map
            error_twist = pin.log6(pose_des).vector   # in joint frame
            # print("error_twist:", error_twist)

            # Apply separate weights to rotation and translation errors.
            error_twist[:3] *= w_rot
            error_twist[3:] *= w_trans  
            
            error_norm = np.linalg.norm(error_twist)
            # Check convergence
            # if error_norm < tol:
            #     if self.DEBUG:
            #         print(f"IK converged in {i} iterations with error norm: {error_norm:.6f}")
            #     break
            
            # Compute the Jacobian of the end-effector frame.
            J = pin.computeFrameJacobian(self.robot.model, self.robot.data, q , self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            
            # Compute the joint velocity update using a damped least-squares pseudo-inverse of the Jacobian.
            J_pinv = np.linalg.pinv(J, rcond=damping)
            dq = J_pinv.dot(error_twist)

            # Update configuration taking the manifold structure into account.
            q = pin.integrate(self.robot.model, q, dq * self.dt)
            self.q = q # Update instance variable

        # else:
        #     if self.DEBUG:
        #         print(f"IK did not converge within {max_iter} iterations. Final error norm: {error_norm:.6f}")

        return self.q

    def spatial_deformation_calculation(self, q, stiffness_matrix_inv, ee_frame_id, spatial_wrench_align_world=np.zeros(6), stiffness_matrix_frame=0):
        """ This function calculates the spatial deformation of a robot model."""
        # model: pinocchio model of the robot
        # Compute forward kinematics + frame placements
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        # end effector frame in world frame
        oMf = self.robot.data.oMf[ee_frame_id] 
        # Ad = oMf.toActionMatrix()
        # print(f"oMf = {oMf}")
        R_of = oMf.rotation # 3×3 matrix

        if stiffness_matrix_frame == 0: # stiffness matrix in LOCAL_ALIGN_WORLD frame
        
            stiffness_wrench_align_world = calculate_stiffness_matrix_align_world(self.robot.model, self.robot.data, q, stiffness_matrix_inv, ee_frame_id) 
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

    def cost(self, X, U, record_result=False):
        ''' total cost (running+final) for state trajectory X and control trajectory U '''
        N = U.shape[0]
        cost = self.cost_final(X[-1,:],record_result=record_result)
        # print(f"cost_final: {cost:.5g}")
        for i in range(N):
            cost += self.cost_running(i, X[i,:], U[i,:])
        return cost
        
    def cost_running(self, i, x, u):
        ''' Running cost at time step i for state x and control u '''
        cost = 0.5 * u.T @ self.lmbda @ u 
        return cost

    def cost_final(self, x, record_result=False):
        ''' Final cost for state x '''
        ''' Running cost at time step i for state x and control u '''
        # calculate the desired joint configuration for the end-effector pose
        q_des = self.IK_step(x , self.q_guess_IK)
        # self.q_guess_IK = q_des # update the initial guess for the next IK step
        deviation, stiffness_matrix = self.spatial_deformation_calculation(q_des, self.stiffness_matrix_inv, self.ee_frame_id)
        # stiffness_matrix in R 3x3 =[sigma_alpha, sigma_alphabeta, sigma_xz; sigma_alphabeta, sigma_beta, sigma_yz; sigma_zx, sigma_zy, sigma_zz]
        sigma_alpha = stiffness_matrix[0, 0]*1000000 # convert to um/N
        sigma_beta = stiffness_matrix[1, 1]*1000000 # convert to um/N
        sigma_alphabeta  = stiffness_matrix[0, 1]*1000000 # convert to um/N

        cost = 0.5 * self.lmbda_stiff * (np.cos(self.phi)**2*sigma_alpha + np.sin(self.phi)**2*sigma_beta + 2*np.cos(self.phi)*np.sin(self.phi)*sigma_alphabeta)**2

        if (abs(x[0]) > self.max_alphabeta):
            cost += 0.5 * 1000 * (abs(x[0])-self.max_alphabeta)**2
            print("alpha is outside of the range")
        if (abs(x[1]) > self.max_alphabeta):
            cost += 0.5 * 1000 * (abs(x[1])-self.max_alphabeta)**2
            print("beta is outside of the range")

        if cost < self.min_final_cost:
            self.final_cost = cost
            self.best_alpha_stiffness = sigma_alpha
            self.best_beta_stiffness = sigma_beta
            self.best_alphabeta_stiffness = sigma_alphabeta
        return cost

    # cost functions differentiation 
        
    def cost_running_x(self, i, x, u):
        ''' Gradient of the running cost w.r.t. x '''
        ''' l_k= 0.5 * u_k' * lmbda  u_k '''
        return np.zeros(self.nx)

    def cost_running_u(self, i, x, u):
        ''' Gradient of the running cost w.r.t. u '''
        ''' l_k= 0.5 * u_k' * lmbda  u_k '''
        return self.lmbda @ u
    
    def cost_running_xx(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x '''
        ''' l_k= 0.5 * u_k' * lmbda  u_k '''
        return np.zeros((self.nx, self.nx))

    def cost_running_uu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. u '''
        ''' l_k= 0.5 * u_k' * lmbda  u_k '''
        return self.lmbda @ np.eye(self.nu)
        
    def cost_running_xu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x and then w.r.t. u '''
        ''' l_k= 0.5 * u_k' * lmbda  u_k '''
        return np.zeros((self.nx, self.nu))
    
    # final cost functions differentiation based on symmetrical finite-difference
    # central second-order finite differences
    def cost_final_x(self, x, h=1e-4):
        ''' Gradient of the final cost w.r.t. x '''
        ''' l_f(x_N) =0.5 * lmbda_stiff (cos(phi)^2*sigma_alpha + sin(phi)^2*sigma_beta + 2*cos(phi)*sin(phi)*sigma_theta)^2 '''
        """∂L/∂x  (shape: (n_x,))"""
        g = np.zeros(self.nx)
        for i in range(self.nx):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g[i] = (self.cost_final(xp) - self.cost_final(xm)) / (2.0 * h)
        return g

    def cost_final_xx(self, x,  h=1e-4):
        ''' Hessian of the final cost w.r.t. x '''
        ''' l_f(x_N) =0.5 * lmbda_stiff (cos(phi)^2*sigma_alpha + sin(phi)^2*sigma_beta + 2*cos(phi)*sin(phi)*sigma_theta)^2 '''
        """∂²L/∂x²  (shape: (n_x, n_x))"""
        H = np.zeros((self.nx, self.nx))
        for j in range(self.nx):
            xp = x.copy(); xp[j] += h
            xm = x.copy(); xm[j] -= h
            H[:, j] = (self.cost_final_x(xp, h) -
                       self.cost_final_x(xm, h)) / (2.0 * h)
        return 0.5 * (H + H.T)

    def backward_pass(self, X_bar, U_bar, mu):
        n = X_bar.shape[1] # number of state variables
        m = U_bar.shape[1] # number of control inputs
        N = U_bar.shape[0] # step number
        rx = list(range(0,n))
        ru = list(range(0,m))
        
        # the task is defined by a quadratic cost: 
        # sum_{i=0}^N 0.5 x' l_{xx,i} x + l_{x,i} x +  0.5 u' l_{uu,i} u + l_{u,i} u + x' l_{xu,i} u
        
        # define the dimension of the matrix to store the derivatives of the cost function
        V_xx = np.zeros((N+1, n, n))
        V_x  = np.zeros((N+1, n))
        
        # dynamics derivatives
        A = np.zeros((N, n, n))
        B = np.zeros((N, n, m))
        
        # initialize value function

        self.l_x[-1,:]  = self.cost_final_x(X_bar[-1,:])
        self.l_xx[-1,:,:] = self.cost_final_xx(X_bar[-1,:])
        # print("cost of the last iteration", self.cost(X_bar, U_bar, record_result=True))
        # print("Best stiffenss alpha, beta, alphabeta", self.best_alpha_stiffness, self.best_beta_stiffness, self.best_alphabeta_stiffness)
        # here V is the P in my notes, the derivative of value function with respect to x and xx        
        V_xx[N,:,:] = self.l_xx[N,:,:] # initial value function at the last time step
        V_x[N,:]    = self.l_x[N,:] # initial value function at the last time step
        # if (self.DEBUG  ):
        #     print("V_x V_xx at the last time step",  (V_x[N,:]),  (V_xx[N,:,:]))
        #     print("X_bar, U_bar at the last time step",  (X_bar[-1,:]),  (U_bar[-1,:]))

        for i in range(N-1, -1, -1):
            if(self.DEBUG):
                print("\n *** Time step %d ***" % i)
                
            # compute dynamics Jacobians
            # _bar means that the value is computed as the current iteration nominal trajectory, which is used for linearization
            A[i,:,:] = self.f_x(X_bar[i,:], U_bar[i,:])
            B[i,:,:] = self.f_u(X_bar[i,:], U_bar[i,:])
                
            # compute the gradient of the cost function at X=X_bar
            self.l_x[i,:]    = self.cost_running_x(i, X_bar[i,:], U_bar[i,:])
            self.l_xx[i,:,:] = self.cost_running_xx(i, X_bar[i,:], U_bar[i,:])
            self.l_u[i,:]    = self.cost_running_u(i, X_bar[i,:], U_bar[i,:])
            self.l_uu[i,:,:] = self.cost_running_uu(i, X_bar[i,:], U_bar[i,:])
            self.l_xu[i,:,:] = self.cost_running_xu(i, X_bar[i,:], U_bar[i,:])
            # if(self.DEBUG  ):
            #     print("l_x, l_xx, l_u, l_uu, l_xu",  (self.l_x[i,rx]),  (self.l_xx[i,rx,:]),  (self.l_u[i,ru]),  (self.l_uu[i,ru,:]),  (self.l_xu[i,rx,0]))

            # compute regularized cost-to-go
            self.Q_x[i,:]     = self.l_x[i,:] + A[i,:,:].T @ V_x[i+1,:]
            self.Q_u[i,:]     = self.l_u[i,:] + B[i,:,:].T @ V_x[i+1,:]
            self.Q_xx[i,:,:]  = self.l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
            self.Q_uu[i,:,:]  = self.l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            self.Q_xu[i,:,:]  = self.l_xu[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            
            # if(self.DEBUG  ):
                # print("Q_x, Q_u, Q_xx, Q_uu, Q_xu",  (self.Q_x[i,rx]),  (self.Q_u[i,ru]), 
                #     (self.Q_xx[i,rx,:]),  (self.Q_uu[i,ru,:]),  (self.Q_xu[i,rx,0]))
                
            # regularize Q_uu by adding a small value to the diagonal
            # Qbar_uu = Q_uu + mu*I
            Qbar_uu       = self.Q_uu[i,:,:] + mu*np.identity(m)
            # pseudo-inverse of a matrix
            Qbar_uu_pinv  = np.linalg.pinv(Qbar_uu) 
            self.kk[i,:]       = - Qbar_uu_pinv @ self.Q_u[i,:] # dk in my note
            self.KK[i,:,:]     = Qbar_uu_pinv @ self.Q_xu[i,:,:].T # Kk in my note
            # if(self.DEBUG):
            #     print("Qbar_uu, Qbar_uu_pinv", (Qbar_uu),  (Qbar_uu_pinv))
            #     print("dk, Kk",  self.kk[i,:],  self.KK[i,:,:] )
                
            # update Value function pk and Pk
            V_x[i,:]    = self.Q_x[i,:]  - self.Q_xu[i,:,:] @ Qbar_uu_pinv @ self.Q_u[i,:]
            V_xx[i,:]   = self.Q_xx[i,:] - self.Q_xu[i,:,:] @ Qbar_uu_pinv @ self.Q_xu[i,:,:].T
                    
        return (self.kk, self.KK)
   

    ''' Simulate system forward with computed control law '''
    # notice here U_bar <- U_bar + alpha*self.kk
    def simulate_system(self, x0, U_bar, KK, X_bar, record_result=False):
        n = x0.shape[0] # number of state variables
        m = U_bar.shape[1] # number of control inputs
        N = U_bar.shape[0] # step number
        X = np.zeros((N+1, n))
        U = np.zeros((N, m))
        X[0, :] = x0
        for i in range(N):
            # if X_bar is None:
            #     U[i] = U_bar[i]
            #     X[i+1,:] = self.f(X[i,:], U[i,:]) # discrete-time system dynamics 
            # else:
                # feedback law: u = u_bar + alpha * K_i ( x - x_b
                U[i,:] = U_bar[i,:] - np.dot(KK[i,:,:], (X[i,:]-X_bar[i,:])) # control law
                X[i+1,:] = self.f(X[i,:], U[i,:]) # discrete-time system dynamics 
            
            # if np.any(np.abs(X[i+1,:2]) > self.max_alphabeta):
            #     U[i,:] = U_bar[i,:] - np.dot(KK[i,:,:], (X[i,:]-X_bar[i,:])) # control law
            #     U[i,0]    = 0
            #     X[i+1,:] = self.f(X[i,:], U[i,:]) # discrete-time system dynamics 
                           
        return (X,U)


    def solve(self, x0, U_bar, mu):                    
        # each control law is composed by a feedforward kk and a feedback KK
        self.N = N = U_bar.shape[0]
        m = U_bar.shape[1]
        n = x0.shape[0]
        self.kk  = np.zeros((N,m))
        self.KK  = np.zeros((N,m,n))
        self.min_cost_q =  np.zeros((N,6)) # minimum cost q for the inverse kinematics, used to record the best trajectory
        X_bar = np.zeros((N,n))   # initial nominal state trajectory should be None, waiting to be rollout by the initial guess U_bar

        # derivatives of the cost function
        self.l_x = np.zeros((N+1, n))
        self.l_xx = np.zeros((N+1, n, n))
        self.l_u = np.zeros((N, m))
        self.l_uu = np.zeros((N, m, m))
        self.l_xu = np.zeros((N, n, m))
        
        # the cost-to-go is defined by a quadratic function: 0.5 x' Q_{xx,i} x + Q_{x,i} x + ...
        self.Q_xx = np.zeros((N, n, n))
        self.Q_x  = np.zeros((N, n))
        self.Q_uu = np.zeros((N, m, m))
        self.Q_u  = np.zeros((N, m))
        self.Q_xu = np.zeros((N, n, m))
        
        converged = False
        
        (self.min_cost_X, self.min_cost_U) = self.simulate_system(x0,
                                                  U_bar, 
                                                  np.zeros((N,m,n)), 
                                                  X_bar,
                                                  record_result=True)

        for j in range(self.max_iter):
            print("\n*** Iter %d" % j)
            
            # compute nominal state trajectory X_bar
            (X_bar, U_bar) = self.simulate_system(x0,
                                                  self.min_cost_U , 
                                                  np.zeros((N,m,n)), 
                                                  self.min_cost_X,
                                                  record_result=True
                                                  )
            # mod 2 * pi in the theta
            X_bar[:,2] = np.mod(X_bar[:,2], 2*np.pi) # keep theta in [0, 2*pi]

            self.backward_pass(X_bar, U_bar, mu)
            # print("kk:", self.kk)
            # print("KK:", self.KK)
            
            # forward pass - line search
            alpha = 1 # feedforward gain of the kk (dk in my note)
            line_search_succeeded = False
            # compute costs for nominal trajectory and expected improvement model
            cst = self.cost(X_bar, U_bar, record_result=True)
            # if cst < self.min_cost:
            #     self.min_cost = cst
            #     self.min_cost_U = U_bar.copy()
            #     self.min_cost_X = X_bar.copy()
            #     for i in range(N):
            #         self.min_cost_q[i,:] = self.IK_step(self.min_cost_X[i,:] , self.q_guess_IK)
            #     print("New minimum cost found: %.3f" % cst)
            # print("Current minimum cost X_N", self.min_cost_X[-1,:])

            self.update_expected_cost_improvement()


            # trajectory optimization with line search in each iteration
            # find the optimal of 2nd-order function (gradient descent), not directly find the optimal value by zero gradient point. 
            # X_bar and U_bar are not updated in the line search, only the alpha is changed
            for jj in range(self.max_line_search_iter):
                (X,U) = self.simulate_system(x0, U_bar + alpha*self.kk, self.KK, X_bar)
                # print("Line search iteration %d" % (jj), U)

                new_cost = self.cost(X, U)


                exp_impr = alpha*self.d1 + 0.5*(alpha**2)*self.d2 
                relative_impr = (new_cost-cst)/cst # (1-2)/1 =-1
                # print("X:", X[-1,:],"U(0):", U[0,:])
                # print("Real cost", new_cost)

                if(relative_impr < 0):
                    print(" Cost improved from %.3f to %.3f. Rel. impr. %.1f%%" % ( cst, new_cost,  1e2*relative_impr))
                    line_search_succeeded = True

                if(line_search_succeeded):
                    # update control input
                    U_bar += alpha*self.kk # forward input update
                    cst = new_cost
                    break
                else:
                    # line search failed due to overshoot, reduce alpha to reduce the step size
                    alpha = self.alpha_factor*alpha 
        
            if new_cost < self.min_cost:
                self.min_cost = new_cost
                self.min_cost_U = U.copy()
                self.min_cost_X = X.copy()
                for i in range(N):
                    self.min_cost_q[i,:] = self.IK_step(self.min_cost_X[i,:] , self.q_guess_IK)
                print("New minimum cost found: %.3f" % new_cost)
            print("Current minimum cost X_N", self.min_cost_X[-1,:])

            # after the all steps, update the mu if necessary before start the next iteration
            if(not line_search_succeeded):
                mu = mu*self.mu_factor
                print("No cost improvement, increasing mu to", mu)
                if(mu>self.mu_max):
                    print("Max regularization reached. Algorithm failed to converge.")
                    converged = True
            else:
                print("Line search succeeded with alpha", alpha)
                if(alpha>=self.min_alpha_to_increase_mu):
                    mu = mu/self.mu_factor
                    if(mu<self.mu_min):
                        mu = self.mu_min
                    else:
                        print("Decreasing mu to ", mu)
                else:
                    mu = mu*self.mu_factor
                    print("Alpha is small => increasing mu to", mu)
                    if(mu>self.mu_max):
                        print("Max regularization reached. Algorithm failed to converge.")
                        converged = True
                self.callback(X_bar, U_bar)
                    
            exp_impr = self.d1 + 0.5*self.d2
            if(abs(exp_impr) < self.exp_improvement_threshold):
                print("Algorithm converged. Expected improvement", exp_impr)
                converged = True
                
            if(converged):
                break
                    
        # compute nominal state trajectory X_bar
        (X_bar, U_bar) = self.simulate_system(x0, U_bar, self.KK, X_bar)
    
        return (X_bar, U_bar, self.KK)
        

    def print_statistics(self, X, U):
        # simulate system forward with computed control law
        print("\n**************************************** RESULTS ****************************************")
        
        # compute cost of each task
        print("Min Effort", U[0,:])
        print("Min X_N   ", X[-1,:])
        print("Min Cost  ", self.cost(X, U))

        # print("best alpha stiffness", self.best_alpha_stiffness)
        # print("best beta stiffness", self.best_beta_stiffness)
        # print("best alphabeta stiffness", self.best_alphabeta_stiffness)
        # print("Final q", self.min_cost_q[-1,:])
        # pin.forwardKinematics(self.robot.model, self.robot.data, self.min_cost_q[-1,:])
        # pin.updateFramePlacements(self.robot.model, self.robot.data)

        # # end effector frame in world frame
        # oMf = self.robot.data.oMf[self.ee_frame_id] 
        # # print("Final real end effector pose in world frame:", oMf)
        # print("Final real X in world frame:", oMf.translation[0]-0.6, oMf.translation[1], pin.rpy.matrixToRpy(oMf.rotation)[2])

        print("X_bar trajectory:", X)

        for x in X:
            # calculate the desired joint configuration for the end-effector pose
            q_des = self.IK_step(x , self.q_guess_IK)
            # self.q_guess_IK = q_des # update the initial guess for the next IK step
            deviation, stiffness_matrix = self.spatial_deformation_calculation(q_des, self.stiffness_matrix_inv, self.ee_frame_id)
            # stiffness_matrix in R 3x3 =[sigma_alpha, sigma_alphabeta, sigma_xz; sigma_alphabeta, sigma_beta, sigma_yz; sigma_zx, sigma_zy, sigma_zz]
            sigma_alpha = stiffness_matrix[0, 0]*1000000 # convert to um/N
            sigma_beta = stiffness_matrix[1, 1]*1000000 # convert to um/N
            sigma_alphabeta  = stiffness_matrix[0, 1]*1000000 # convert to um/N
            cost = 0.5 * self.lmbda_stiff * (np.cos(self.phi)**2*sigma_alpha + np.sin(self.phi)**2*sigma_beta + 2*np.cos(self.phi)*np.sin(self.phi)*sigma_alphabeta)**2
            print("stiffness values:", sigma_alpha, sigma_beta, sigma_alphabeta,"cost:", cost)

    def callback(self, X, U):
        pass
        # rendering the robot in the viewer
        # for i in range(0, N):
        #     time_start = time.time()
        #     self.simu.display(X[i,:self.robot.nq])
        #     time_spent = time.time() - time_start
        #     if(time_spent < self.dt):
        #         time.sleep(self.dt-time_spent)
                
    def start_simu(self, X, U, KK, dt_sim):
        pass
        # t = 0.0
        # simu = self.simu
        # simu.init(X[0,:self.robot.nq])
        # ratio = int(self.dt/dt_sim)
        # N_sim = N * ratio
        # self.N_sim = N_sim

        # print("Start simulation")
        # time.sleep(1)
        # for i in range(0, N_sim):
        #     time_start = time.time()
    
        #     # compute the index corresponding to the DDP time step
        #     j = int(np.floor(i/ratio))
        #     # compute joint torques
        #     x = np.hstack([simu.q, simu.v])
        #     tau = U[j,:] + KK[j,:,:] @ (X[j,:] - x)        
        #     # send joint torques to simulator
        #     simu.simulate(tau, dt_sim)
            
        #     t += dt_sim
        #     time_spent = time.time() - time_start
        #     if(time_spent < dt_sim):
        #         time.sleep(dt_sim-time_spent)
        # print("Simulation finished")

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import utils.plot_utils as plut
    import time
    from utils.robot_loaders import load_ee_frame
    from utils.robot_wrapper import RobotWrapper
    import ddp_manipulator_conf as conf
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
    ddp_params['max_line_search_iter'] = 5
    
    ddp_params['mu_factor'] = 3. # the scaling factor of the mu: regularization term, for not invertible system Quu
    ddp_params['mu_max'] = 1e0
    ddp_params['mu_min'] = 1e-8

    ddp_params['min_cost_impr'] = 1e-2 # minimum cost improvement to increase mu. Large mu -> slow convergence
    ddp_params['exp_improvement_threshold'] = 1e-3 # threshold to stop iteration
    ddp_params['max_iter'] = 10 #  full loop itaration number forward + backward pass
    DEBUG = 0


    r = load_ee_frame() # example_robot_data.load("ur5", True) #loadUR()#

    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    nq, nv = robot.nq, robot.nv
    
    n = 3                   # state size in SE2
    m = 2                   # control size in SE2

    # create to store the trajectory of the robot configuration
    q_list = np.zeros((robot.nq, 360)) # 0-360 degrees, 1 configuration for each degree
    x_list = np.zeros((n, 360)) # 0-360 degrees, 1 state for each degree

    ''' COST FUNCTION  '''
    lmbda = np.array([[3, 0], [0, 3]])         # control regularization move and rotation
    lmbda_stiff = 300    # final stiffness cost regularization

    robot.initVisualization(robot.q0) # initialize the robot configuration
    print("Displaying start")
    time.sleep(1.)

    # startup to seek for the first point with maximum x-axis stiffness
    solver = DDPSolverManipulatorStiffness_SE2("Denso", robot, ddp_params, lmbda_stiff, lmbda, dt, DEBUG=DEBUG)
    solver.phi = 0/360*2*np.pi

    x0 = np.zeros(n) #initial state when q0 is used
    U_bar = np.zeros((N_STARTUP, m))        # initial guess for control inputs
    U_bar[:,0] = 0 # initial guess for control inputs, v
    (X,U,KK) = solver.solve(x0, U_bar, mu)
    # x0 is the initial state in SE2, [alpha, beta, theta]
    # U_bar is the initial guess for control inputs in SE2, [v, omega]

    # plot the trajectory of the robot configuration in the xy plane
    # given x y and orientation, how to plot the frame in the 2D plane


    # plot_frame(X[N_STARTUP-1, 0], X[N_STARTUP-1, 1], X[N_STARTUP-1, 2], ax=plt.gca())

    # create the anime of the trajectory of the robot configuration
    print(X[-1,:])
    animator = FrameAnimator(
        [X[i,:] for i in range(0, len(X), 5)],
        length=0.5,
        interval=100,
        xlim=(-1, 1),
        ylim=(-1, 1)
    )
    ani = animator.animate()
    plt.show()


    solver.print_statistics(X,U)


    # KK (K_k in my note) is the feedback gain matrix, used to compute the control inputs
    # kk (d_k in my note) is the feedforward gain matrix, used to compute the control inputs


    # seek for the optimal trajectory from the startup point with maximum stiffness in the tangent direction
    # for i in range(0,360): 
    #     solver.phi = i * 2 * np.pi / 360
    #     U_bar = np.zeros((N, m))        # initial guess for control inputs
    #     (X,U,KK) = solver.solve(x0, U_bar, mu)


    # time.sleep(10.)
    # for i in range(360):
    #     robot.display(q_list[:, i]) # display the initial configuration
    #     time.sleep(0.01) # wait for a short time to display the configuration

    input("Press ENTER to continue...")
    

    # print("Show reference motion")
    # for i in range(0, N+1):
    #     time_start = time.time()
    #     simu.display(X[i,:nq])
    #     time_spent = time.time() - time_start
    #     if(time_spent < dt):
    #         time.sleep(dt-time_spent)
    # print("Reference motion finished")
    # time.sleep(1)
    
    # print("Show real simulation")
    # for i in range(10):
    #     solver.start_simu(X, U, KK, conf.dt_sim)
