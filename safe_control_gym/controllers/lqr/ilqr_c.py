import numpy as np
import cvxpy as cp
import scipy.linalg
from termcolor import colored

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import (compute_lqr_gain, discretize_linear_system,
                                                        get_cost_weight_matrix)
from safe_control_gym.controllers.lqr.lqr import LQR
from safe_control_gym.envs.benchmark_env import Task


class iLQR_C(LQR):
    def __init__(self, env_func, 
                 q_lqr: list = None, 
                 r_lqr: list = None, 
                 discrete_dynamics: bool = True,
                 CM_search: bool = False, 
                 **kwargs):
        super().__init__(env_func, q_lqr, r_lqr, discrete_dynamics, **kwargs)

        self.env = env_func()
        # Controller params.
        self.model = self.get_prior(self.env)
        print('self.model:', self.model)


        # self.discrete_dynamics = discrete_dynamics # not used since we are using continuous dynamics
        self.Q = get_cost_weight_matrix(q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(r_lqr, self.model.nu)
        self.env.set_cost_function_param(self.Q, self.R)

        self.dt = self.model.dt
        self.total_time = self.env.EPISODE_LEN_SEC
        self.N = int(self.total_time/self.dt) # number of timesteps
        self.T = 1 # MPC like horizon, only for get_references()
        
        self.alpha_range = [0.1, 5.0]
        self.alpha_step = 0.1
        if CM_search:
            self.line_search_for_CM()

    def get_cl_jacobian(self, x_lin, u_lin):
        '''Get the Jacobian of the closed-loop dynamics.'''
        # Linearize continuous-time dynamics
        df = self.model.df_func(x_lin, u_lin)
        A, B = df[0].toarray(), df[1].toarray()
        P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
        K = np.dot(np.linalg.inv(self.R), np.dot(B.T, P))
        J_cl = A - B @ K
        return J_cl

    def line_search_for_CM(self, d_bar=1.0):
        alpha = self.alpha_range[0]
        # total search steps
        Na = int((self.alpha_range[1] - self.alpha_range[0]) /self.alpha_step)+1
        print("========================================================")
        print("============= LINE SEARCH OF OPTIMAL ALPHA =============")
        print("========================================================")
        result_prev = np.Inf
        M_prev = None
        chi_prev = None
        for i in range(Na):
            result, M, chi, min_bound = self.compute_CM(alpha=alpha,
                                            d_bar=d_bar)
            print("Optimal value: Jcv =","{:.2f}".format(result),
                  "( alpha =","{:.3f}".format(alpha),
                    ", min_bound =","{:.3f}".format(min_bound),
                  ")")
            if result_prev <= result:
                alpha -= self.alpha_step
                self.result = result_prev
                self.M = M_prev
                self.chi = chi_prev
                break
            alpha += self.alpha_step
            # save the previous result
            result_prev = result
            M_prev = M
            chi_prev = chi
        self.alpha_opt = alpha
        print("Optimal contraction rate: alpha =","{:.3f}".format(alpha))
        print("Minimum bound: min_bound =","{:.3f}".format(min_bound))
        print("========================================================")
        print("=========== LINE SEARCH OF OPTIMAL ALPHA END ===========")
        print("========================================================\n\n")
        self.min_bound = d_bar / self.alpha_opt * np.sqrt(self.chi)


    def compute_CM(self, alpha, d_bar):
        J_ref = [self.get_cl_jacobian(self.env.X_GOAL[i], self.model.U_EQ)
                for i in range(self.N)]
        nx = self.model.nx
        chi = cp.Variable(nonneg=True)
        W_tilde = cp.Variable((nx, nx), symmetric=True)
        objective = cp.Minimize(chi * d_bar / alpha)
        constraints = [chi*np.identity(nx) - W_tilde >> 0,
                    W_tilde - np.identity(nx) >> 0,]
        for i in range(self.N):
            constraints += [ - W_tilde @ J_ref[i].T - J_ref[i] @ W_tilde - 2 * alpha * W_tilde
                            >> 1e-6*np.identity(nx)]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK, warm_start=True)
        # print('result:', result)
        min_bound = d_bar / alpha * np.sqrt(chi.value)
        M = np.linalg.inv(W_tilde.value)
        return result, M, chi.value, min_bound


    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        # step = self.extract_step(info)
        # self.goal_state = self.env.X_GOAL[self.traj_step]
        self.goal_state = self.get_references()

        if self.env.TASK == Task.STABILIZATION:
            # return -self.gain @ (obs - self.env.X_GOAL) + self.model.U_EQ
            raise NotImplementedError('iLQR not implemented for stabilization task.')
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.traj_step += 1
            # get the liearzation points
            # x_0 = self.env.X_GOAL[step]
            # x_0 = self.env.X_GOAL[self.traj_step]
            x_0 = self.goal_state[:, 0]
            # print('x_0:', x_0)
            u_0 = self.model.U_EQ
            # Linearize continuous-time dynamics
            df = self.model.df_func(x_0, u_0)
            A, B = df[0].toarray(), df[1].toarray()
            # print('A:\n', A)
            # print('B:\n', B)
            P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
            gain = np.dot(np.linalg.inv(self.R), np.dot(B.T, P))
            # control = -gain @ (obs - x_0) + u_0
            # action = -gain @ (obs - self.env.X_GOAL[step]) + self.model.U_EQ
            action = -gain @ (obs - x_0) + u_0

        # print('self.traj_step:', self.traj_step)
        return action
        pass
            # return -self.gain @ (obs - self.env.X_GOAL[step]) + self.model.U_EQ
    
    def get_references(self):
        '''Constructs reference states along mpc horizon.(nx, T+1).'''
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(self.traj_step, self.traj.shape[-1])
            end = min(self.traj_step + self.T + 1, self.traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            # end = start + 1
            # remain = max(0, 1 - (end - start))
            goal_states = np.concatenate([
                self.traj[:, start:end],
                np.tile(self.traj[:, -1:], (1, remain))
            ], -1)
            # goal_states = self.traj[:, start:end]
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).

    def reset(self):
        '''Prepares for evaluation.'''
        self.env.reset()
        if self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0

    def close(self):
        '''Cleans up resources.'''
        self.env.close()
