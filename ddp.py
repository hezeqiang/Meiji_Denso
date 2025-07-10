# -*- coding: utf-8 -*-
"""
Generic DDP solver class.
System dynamics and cost functions must be specified in child classes.

@author: adelpret
"""

import numpy as np

def a2s(a, format_string ='{0:.4f} '):
    ''' array to string '''
    if(len(a.shape)==0):
        return format_string.format(a);

    if(len(a.shape)==1):
        res = '[';
        for i in range(a.shape[0]):
            res += format_string.format(a[i]);
        return res+']';
        
    res = '[[';
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            res += format_string.format(a[i,j]);
        res = res[:-1]+'] [';
    return res[:-2]+']'; 
    #[format_string.format(v,i) for i,v in enumerate(a)]


class DDPSolver:
    
    def __init__(self, name, params, DEBUG=False):
        self.name = name
        # line search factor of the step size, default 1 means full step, 0 means no step of u
        # use small step size for convergence around the local minimum
        self.alpha_factor = params['alpha_factor']
        # if the min alpha still not converged, increase mu to decrease the step size   
        self.min_alpha_to_increase_mu = params['min_alpha_to_increase_mu']
        
        # initial regularization factor for not invertible system Quu: the influence of  input u on the system dynamics
        self.mu_factor = params['mu_factor']
        self.mu_max = params['mu_max']
        self.mu_min = params['mu_min']
        self.min_cost=10000
        self.min_final_cost=10000
   
        # minimum cost improvement to increase mu. Large mu -> slow convergence
        self.min_cost_impr = params['min_cost_impr']

        # maximum number of iterations for line search
        self.max_line_search_iter = params['max_line_search_iter']
        # expected cost improvement threshold to stop iteration
        self.exp_improvement_threshold = params['exp_improvement_threshold']
        # maximum number of iterations for DDP
        self.max_iter = params['max_iter']
        self.DEBUG = DEBUG

        self.d1 = 0.0 # index for improvement of the cost function
        self.d2 = 0.0 # index for improvement of the cost function
        self.final_cost = 0.0 # final cost of the trajectory, for comparison with the previous iteration in line search
        
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
            if X_bar is None:
                U[i] = U_bar[i]
            else:
                # feedback law: u = u_bar + alpha * K_i ( x - x_b
                U[i,:] = U_bar[i,:] - np.dot(KK[i,:,:], (X[i,:]-X_bar[i,:])) # control law

            X[i+1,:] = self.f(X[i,:], U[i,:]) # discrete-time system dynamics
        return (X,U)
        
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
        print("cost of the last iteration", self.cost(X_bar, U_bar, record_result=True))
        self.l_x[-1,:]  = self.cost_final_x(X_bar[-1,:])
        self.l_xx[-1,:,:] = self.cost_final_xx(X_bar[-1,:])
        # here V is the P in my notes, the derivative of value function with respect to x and xx        
        V_xx[N,:,:] = self.l_xx[N,:,:] # initial value function at the last time step
        V_x[N,:]    = self.l_x[N,:] # initial value function at the last time step
        if (self.DEBUG or 0):
            print("V_x V_xx at the last time step", a2s(V_x[N,:]), a2s(V_xx[N,:,:]))
            print("X_bar, U_bar at the last time step", a2s(X_bar[-1,:]), a2s(U_bar[-1,:]))

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
            if(self.DEBUG):
                if (i==1):
                    print("l_x, l_xx, l_u, l_uu, l_xu", a2s(self.l_x[i,rx]), a2s(self.l_xx[i,rx,:]), a2s(self.l_u[i,ru]), a2s(self.l_uu[i,ru,:]), a2s(self.l_xu[i,rx,0]))

            # compute regularized cost-to-go
            self.Q_x[i,:]     = self.l_x[i,:] + A[i,:,:].T @ V_x[i+1,:]
            self.Q_u[i,:]     = self.l_u[i,:] + B[i,:,:].T @ V_x[i+1,:]
            self.Q_xx[i,:,:]  = self.l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
            self.Q_uu[i,:,:]  = self.l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            self.Q_xu[i,:,:]  = self.l_xu[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            
            if(self.DEBUG):
                print("Q_x, Q_u, Q_xx, Q_uu, Q_xu", a2s(self.Q_x[i,rx]), a2s(self.Q_u[i,ru]), 
                        a2s(self.Q_xx[i,rx,:]), a2s(self.Q_uu[i,ru,:]), a2s(self.Q_xu[i,rx,0]))
                
            # regularize Q_uu by adding a small value to the diagonal
            # Qbar_uu = Q_uu + mu*I
            Qbar_uu       = self.Q_uu[i,:,:] + mu*np.identity(m)
            # pseudo-inverse of a matrix
            Qbar_uu_pinv  = np.linalg.pinv(Qbar_uu) 
            self.kk[i,:]       = - Qbar_uu_pinv @ self.Q_u[i,:] # dk in my note
            self.KK[i,:,:]     = Qbar_uu_pinv @ self.Q_xu[i,:,:].T # Kk in my note
            if(self.DEBUG):
                print("Qbar_uu, Qbar_uu_pinv",a2s(Qbar_uu), a2s(Qbar_uu_pinv))
                print("dk, Kk", a2s(self.kk[i,ru]), a2s(self.KK[i,ru,rx]))
                
            # update Value function pk and Pk
            V_x[i,:]    = self.Q_x[i,:]  - self.Q_xu[i,:,:] @ Qbar_uu_pinv @ self.Q_u[i,:]
            V_xx[i,:]   = self.Q_xx[i,:] - self.Q_xu[i,:,:] @ Qbar_uu_pinv @ self.Q_xu[i,:,:].T
                    
        return (self.kk, self.KK)
      
    def update_expected_cost_improvement(self):

        for i in range(self.N):
            self.d1 += self.kk[i,:].T @ self.Q_u[i,:]
            self.d2 += 0.5 * self.kk[i,:].T @ self.Q_uu[i,:,:] @ self.kk[i,:]
            
                        
    ''' Differential Dynamic Programming
        The pseudoinverses in the algorithm are regularized by the damping factor mu.
    '''
    def solve(self, x0, U_bar, mu):                    
        # each control law is composed by a feedforward kk and a feedback KK
        self.N = N = U_bar.shape[0]
        m = U_bar.shape[1]
        n = x0.shape[0]
        self.kk  = np.zeros((N,m))
        self.KK  = np.zeros((N,m,n))
                
        X_bar = None   # initial nominal state trajectory should be None, waiting to be rollout by the initial guess U_bar
        
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
        for j in range(self.max_iter):
            print("\n*** Iter %d" % j)
            
            # compute nominal state trajectory X_bar
            (X_bar, U_bar) = self.simulate_system(x0,
                                                  U_bar, 
                                                  self.KK, 
                                                  X_bar,
                                                  record_result=True
                                                  )
            
            self.backward_pass(X_bar, U_bar, mu)
            
            # forward pass - line search
            alpha = 0.1 # feedforward gain of the kk (dk in my note)
            line_search_succeeded = False
            # compute costs for nominal trajectory and expected improvement model
            cst = self.cost(X_bar, U_bar, record_result=True)
            if cst < self.min_cost:
                self.min_cost = cst
                self.min_cost_U = U_bar.copy()
                self.min_cost_X = X_bar.copy()
                print("New minimum cost found: %.3f" % cst)
                print("Current X_N", X_bar[-1,:])

            self.update_expected_cost_improvement()
            exp_impr = alpha*self.d1 + 0.5*(alpha**2)*self.d2 
            # exp_impr = self.d1 + 0.5*self.d2 
            print("Expected improvement (linearized system)", exp_impr)

            # trajectory optimization with line search in each iteration
            # find the optimal of 2nd-order function (gradient descent), not directly find the optimal value by zero gradient point. 
            # X_bar and U_bar are not updated in the line search, only the alpha is changed
            for jj in range(self.max_line_search_iter):
                (X,U) = self.simulate_system(x0, U_bar + alpha*self.kk, self.KK, X_bar)
                new_cost = self.cost(X, U)

                relative_impr = (new_cost-cst)/exp_impr
                print("Real improvement (considering nonlinearity)", new_cost-cst)
                
                # if(relative_impr > self.min_cost_impr):
                #     print("Cost improved from %.3f to %.3f. Exp. impr %.3f. Rel. impr. %.1f%%" % (cst, new_cost, exp_impr, 1e2*relative_impr))
                #     line_search_succeeded = True

                if(new_cost < self.final_cost):
                    print("Cost improved from %.3f to %.3f. Exp. impr %.3f. Rel. impr. %.1f%%" % (cst, new_cost, exp_impr, 1e2*relative_impr))
                    line_search_succeeded = True

                if(line_search_succeeded):
                    # update control input
                    U_bar += alpha*self.kk # forward input update
                    cst = new_cost
                    break
                else:
                    # line search failed due to overshoot, reduce alpha to reduce the step size
                    alpha = self.alpha_factor*alpha 
            
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
        
        
    def callback(self, X, U):
        ''' callback function called at every iteration '''
        pass
    
    def print_statistics(self):
        # simulate system forward with computed control law
        print("\n**************************************** RESULTS ****************************************")
        
        # compute cost of each task

        print("Min Cost  ", self.min_cost)
        print("Min Effort", np.linalg.norm(self.min_cost_U))
        print("Min Effort", self.min_cost_U[0,:])
        print("Min X_N   ", self.min_cost_X[-1,:])

    ''' Discrete-time system dynamics '''
    def f(x, u):
        return None
           
    ''' Partial derivatives of discrete-time system dynamics w.r.t. x '''
    def f_x(x, u):
        return None
    
    ''' Partial derivatives of discrete-time system dynamics w.r.t. u '''       
    def f_u(x, u):
        return None

    def cost(self, X, U, record_result=False):
        ''' total cost (running+final) for state trajectory X and control trajectory U '''
        return None
        
    def cost_running(self, i, x, u):
        ''' Running cost at time step i for state x and control u '''
        return None
        
    def cost_final(self, x):
        ''' Final cost for state x '''
        return None
        
    def cost_running_x(self, i, x, u):
        ''' Gradient of the running cost w.r.t. x '''
        return None
        
    def cost_final_x(self, x):
        ''' Gradient of the final cost w.r.t. x '''
        return None
        
    def cost_final_xx(self, x):
        ''' Gradient of the final cost w.r.t. xx '''
        return None
    
    def cost_running_u(self, i, x, u):
        ''' Gradient of the running cost w.r.t. u '''
        return None
        
    def cost_running_xx(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x '''
        return None
        
    def cost_running_uu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. u '''
        return None
        
    def cost_running_xu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x and then w.r.t. u '''
        return None
