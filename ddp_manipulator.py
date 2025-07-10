# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:52:58 2016

@author: adelpret
"""

import numpy as np
from ddp import DDPSolver
from ddp_linear import DDPSolverLinearDyn
import pinocchio as pin
import os
os.environ["PINOCCHIO_VIEWER"] = "gepetto"


class DDPSolverManipulator(DDPSolverLinearDyn):
    ''' The nonlinear system dynamics are defined
        The task is defined by a quadratic cost: sum_{i=0}^N 0.5 x' H_{xx,i} x + h_{x,i} x + h_{s,i}
        plus a control regularization: sum_{i=0}^{N-1} lmbda ||u_i||.
    '''
    
    def __init__(self, name, robot, ddp_params, H_xx, h_x, h_s, lmbda, dt, DEBUG=False, simu=None):
        DDPSolver.__init__(self, name, ddp_params, DEBUG)
        self.robot = robot
        #  x^T H_xx x + h_x x + h_s
        # we do not need to compute the l_xx, l_x, l_u, it is constant define by task
        # use in parent class DDPSolverLinearDyn for linear cost
        self.H_xx = H_xx # cost matrix, 
        self.h_x = h_x # cost vector
        self.h_s = h_s # cost scalar
        self.lmbda = lmbda # control regularization

        self.nx = h_x.shape[1]
        self.nu = robot.na
        self.dt = dt
        self.simu = simu
        
        nq, nv = self.robot.nq, self.robot.nv
        self.Fx = np.zeros((self.nx, self.nx))
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fu = np.zeros((self.nx, self.nu))
        self.dx = np.zeros(2*nv)
        
    ''' System dynamics '''
    def f(self, x, u):
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
        ddq = pin.aba(model, data, q, v, u)
        self.dx[:nv] = v
        self.dx[nv:] = ddq
        return x + self.dt * self.dx
           
    def f_x_fin_diff(self, x, u, delta=1e-8):
        # the method shows the basic way to cal derivative and is not used in the code
        f0 = self.f(x, u)
        Fx = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            xp = np.copy(x)
            xp[i] += delta
            fp = self.f(xp, u)
            Fx[:,i] = (fp-f0)/delta
        return Fx
        
    def f_u_fin_diff(self, x, u, delta=1e-8):
        # the method shows the basic way to cal derivative and is not used in the code
        ''' Partial derivatives of system dynamics w.r.t. u '''
        f0 = self.f(x, u)
        Fu = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            up = np.copy(u)
            up[i] += delta
            fp = self.f(x, up)
            Fu[:,i] = (fp-f0)/delta
                
        return Fu
        
    def f_x(self, x, u):
        # derivative calculation using pinocchio
        ''' Partial derivatives of system dynamics w.r.t. x '''
        nq = self.robot.nq
        nv = self.robot.nv
        # in this case, nq = nv
        
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
                
        # first compute Jacobians for continuous time dynamics
        # x = [q, v], dot(x) = [ dot (q), M^-1 * (tau - C(q,v) - G(q)) ]
        # then the linearized system is:
        # [dot(q), dot(v)] = [ 0_nq*nq  I_nq*nv , ddq_dq  dq_dv ] * [q, v] + [ 0_nq*nu, M^-1 ] * u


        pin.computeABADerivatives(model, data, q, v, u)
        self.Fx[:nv, :nv] = 0.0
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fx[nv:, :nv] = data.ddq_dq # Partial derivative of the joint acceleration vector with respect to the joint configuration.
        self.Fx[nv:, nv:] = data.ddq_dv # Partial derivative of the joint acceleration vector with respect to the joint 
        self.Fu[nv:, :] = data.Minv # Partial derivative of the joint acceleration vector with respect to the joint torque, being nothing more than the inverse of the inertia matrix
        
        # Convert them to discrete time
        self.Fx = np.identity(2*nv) + dt * self.Fx  # 1st order Euler
        self.Fu *= dt # 
        
        return self.Fx
    
    def f_u(self, x, u):
        # derivative calculation using pinocchio
        ''' Partial derivatives of system dynamics w.r.t. u '''
        return self.Fu

  
    def callback(self, X, U):
        # rendering the robot in the viewer
        for i in range(0, N):
            time_start = time.time()
            self.simu.display(X[i,:self.robot.nq])
            time_spent = time.time() - time_start
            if(time_spent < self.dt):
                time.sleep(self.dt-time_spent)
                
    def start_simu(self, X, U, KK, dt_sim):
        t = 0.0
        simu = self.simu
        simu.init(X[0,:self.robot.nq])
        ratio = int(self.dt/dt_sim)
        N_sim = N * ratio
        print("Start simulation")
        time.sleep(1)
        for i in range(0, N_sim):
            time_start = time.time()
    
            # compute the index corresponding to the DDP time step
            j = int(np.floor(i/ratio))
            # compute joint torques
            x = np.hstack([simu.q, simu.v])
            tau = U[j,:] + KK[j,:,:] @ (X[j,:] - x)        
            # send joint torques to simulator
            simu.simulate(tau, dt_sim)
            
            t += dt_sim
            time_spent = time.time() - time_start
            if(time_spent < dt_sim):
                time.sleep(dt_sim-time_spent)
        print("Simulation finished")


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import utils.plot_utils as plut
    import time
    import example_robot_data
    from utils.robot_loaders import loadUR
    from utils.robot_wrapper import RobotWrapper
    from utils.robot_simulator import RobotSimulator
    import ddp_manipulator_conf as conf
    np.set_printoptions(precision=3, suppress=True);
    
    ''' Test DDP with a manipulator
    '''
    print("".center(conf.LINE_WIDTH,'#'))
    print(" DDP - Manipulator ".center(conf.LINE_WIDTH, '#'))
    print("".center(conf.LINE_WIDTH,'#'), '\n')

    N = conf.N               # horizon size
    dt = conf.dt             # control time step
    mu = 1e-4                # initial regularization factor for not invertible system Quu: the influence of  input u on the system dynamics

    ddp_params = {}
    ddp_params['alpha_factor'] = 0.5 # line search factor of the step size, default 1 means full step, 0 means no step of u
    ddp_params['min_alpha_to_increase_mu'] = 0.1 # use small step size for convergence around the local minimum
    ddp_params['max_line_search_iter'] = 10
    
    ddp_params['mu_factor'] = 10. # the scaling factor of the mu: regularization term, for not invertible system Quu
    ddp_params['mu_max'] = 1e0
    ddp_params['mu_min'] = 1e-10

    ddp_params['min_cost_impr'] = 1e-1 # minimum cost improvement to increase mu. Large mu -> slow convergence
    ddp_params['exp_improvement_threshold'] = 1e-3 # threshold to stop iteration
    ddp_params['max_iter'] = 50 #  full loop itaration number forward + backward pass
    DEBUG = False


    r = loadUR() # example_robot_data.load("ur5", True) #loadUR()#

    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    nq, nv = robot.nq, robot.nv
    
    n = nq+nv                       # state size
    m = robot.na                    # control size
    U_bar = np.zeros((N, m))        # initial guess for control inputs
    x0 = np.concatenate((conf.q0, np.zeros(robot.nv)))  # initial state
    x_tasks = np.concatenate((conf.qT, np.zeros(robot.nv)))  # goal state
    
    # gravity torque
    tau_g = robot.nle(conf.q0, np.zeros(robot.nv))
    for i in range(N):
        U_bar[i, :] = tau_g
    
    ''' COST FUNCTION  '''
    lmbda = 1e-3           # control regularization
    H_xx = np.zeros((N+1, n, n))
    h_x  = np.zeros((N+1, n))
    h_s  = np.zeros(N+1)

    W = np.diagflat(np.concatenate([np.ones(nq), 1e-2*np.ones(nv)]))
    for i in range(0, N+1):
        H_xx[i,:,:]  = W
        h_x[i,:]     = -W @ x_tasks
        h_s[i]       = 0.5*x_tasks.T @ W @ x_tasks

    # W = np.diagflat(np.concatenate([np.zeros(nq), 1e-3*np.ones(nv)]))
    # for i in range(0,N):
    #     H_xx[i,:,:] = W
    #     h_x[i,:] = -W @ x_tasks
    #     h_s[i] = 0.5 * x_tasks.T @ W @ x_tasks

    print("Displaying desired goal configuration")
    simu = RobotSimulator(conf, robot)
    simu.display(conf.qT)
    time.sleep(1.)
    
    solver = DDPSolverManipulator("ur5", robot, ddp_params, H_xx, h_x, h_s, lmbda, dt, DEBUG, simu)
    
    (X,U,KK) = solver.solve(x0, U_bar, mu)
    solver.print_statistics(x0, U, KK, X)
    print("Computed Final state", X[-1,:])
    print("Desired  Final state", x_tasks)
    
    print("Show reference motion")
    for i in range(0, N+1):
        time_start = time.time()
        simu.display(X[i,:nq])
        time_spent = time.time() - time_start
        if(time_spent < dt):
            time.sleep(dt-time_spent)
    print("Reference motion finished")
    time.sleep(1)
    
    print("Show real simulation")
    for i in range(10):
        solver.start_simu(X, U, KK, conf.dt_sim)
