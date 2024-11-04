'''Control Contraction Metric(CCM).'''
import os
import time
from termcolor import colored

import numpy as np
import casadi as cs
import cvxpy as cp

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import compute_lqr_gain, get_cost_weight_matrix
from safe_control_gym.controllers.mpc.mpc_utils import reset_constraints
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.utils.utils import timing


class CCM(BaseController):
    '''Control Contraction Metric.'''

    def __init__(
            self,
            env_func,
            # Model args.
            discrete_dynamics: bool = False,
            active_state_dims: list = None,
            active_input_dims: list = None,
            **kwargs):
        '''Creates task and controller.

        Args:
            env_func (Callable): Function to instantiate task/environment.
            discrete_dynamics (bool): If to use discrete or continuous dynamics.
        '''

        super().__init__(env_func, **kwargs)

        self.env = env_func()
        # Controller params.
        self.model = self.get_prior(self.env)
        self.discrete_dynamics = discrete_dynamics

        # Active search dimensions considered for the CCM.
        self.active_state_dims = active_state_dims if active_state_dims is not None \
                                                   else np.arange(self.model.nx)
        self.active_input_dims = active_input_dims if active_input_dims is not None \
                                                   else np.arange(self.model.nu)

        self.constraints, self.state_constraints_sym, self.input_constraints_sym = \
            reset_constraints(self.env.constraints.constraints)

        # self.cm_path = 


    def reset(self):
        '''Prepares for evaluation.'''
        self.env.reset()

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def load_reference(self, ref=None):
        '''Load reference trajectory.'''
        # ref_path = /home/mingxuan/Repositories/scg_mx/examples/lqr/temp-data/ilqr_data_quadrotor_traj_tracking.pkl

    @timing
    def learn(self, env=None, **kwargs):
        ''' Compute the Contraction Metric for the system.
        
        NOTE: for simplicity, only constant contraction metric is considered.
        '''

        nx = self.model.nx
        nu = self.model.nu
        # differential dynamics
        x_sym = cs.MX.sym('x', nx)
        u_sym = cs.MX.sym('u', nu)
        df = self.model.df_func(x_sym, u_sym)
        A, B = df[0], df[1]
        A_func = cs.Function('A', [x_sym, u_sym], [A])
        B_func = cs.Function('B', [x_sym, u_sym], [B])
        self.B_func = B_func
        
        # search space
        state_search_space = np.hstack([self.env.constraints.state_constraints[0].upper_bounds[self.active_state_dims],
                                        self.env.constraints.state_constraints[0].lower_bounds[self.active_state_dims]]).reshape(-1, 2)
        input_search_space = np.hstack([self.env.constraints.input_constraints[0].upper_bounds[self.active_input_dims],
                                        self.env.constraints.input_constraints[0].lower_bounds[self.active_input_dims]]).reshape(-1, 2)
        num_grid_ax = 50
        search_space = np.hstack([state_search_space, input_search_space]).reshape(-1, 2)
        mesh = np.meshgrid(*[np.linspace(search_space[i, 0], 
                                         search_space[i, 1], num_grid_ax) for i in range(search_space.shape[0])])
        grid = np.array([m.flatten() for m in mesh]).T
        state_grid = grid[:, [0]]
        input_grid = grid[:, [1]]
        num_grid = grid.shape[0]
        state_grid = np.vstack([np.zeros((4, num_grid)), state_grid[:, [0]].T, np.zeros((1, num_grid))])
        input_grid = np.vstack([input_grid[:, [0]].T, np.zeros((1, num_grid))])

        # print('state_grid:', state_grid)
        # print('input_grid:', input_grid)

        alpha = 1.0
        # d_bar = 1.0
        # chi = cp.Variable(nonneg=True)
        W = cp.Variable((nx, nx), symmetric=True)
        rho = cp.Variable((1))
        epsilon = 1e-3
        # objective = cp.Minimize(chi * d_bar / alpha)

        constraints = []
        for i in range(num_grid):
            x = state_grid[:, [i]]
            u = input_grid[:, [i]]
            A_val = A_func(x, u).full()
            B_val = B_func(x, u).full()
            # print('A_val:', A_val)
            # print('B_val:', B_val)
             
            constraints += [A_val @ W + W @ A_val.T - rho * B_val @ B_val.T + 2 * alpha * W \
                            << -epsilon * np.identity(nx),]
            constraints += [W >> epsilon * np.identity(nx),]
        print(colored('CCM optimization started...', 'green'))
        time_begin = time.time()
        prob = cp.Problem(cp.Minimize(0), constraints)
        result = prob.solve(solver=cp.MOSEK, warm_start=True)
        time_end = time.time()
        print(colored(f'CCM optimization with {num_grid} grid points took {time_end - time_begin} seconds.', 'green'))
        print(colored('CCM optimization status: {}'.format(prob.status), 'green'))
        if 'optimal' not in prob.status:
            print(colored('CCM optimization failed. Exiting...', 'red'))
            exit()
        M = np.linalg.inv(W.value)
        rho = rho.value
        self.M = M
        self.rho = rho
        cm_file_name = 'cm_data.npy'
        np.save(os.path.join(self.output_dir, cm_file_name), {'M': M, 'rho': rho})
        print('CM saved to:', os.path.join(self.output_dir, cm_file_name))

        # cm_res = np.load(cm_load_path, allow_pickle=True)
        # M = cm_res.item()['M']
        # rho = cm_res.item()['rho']
        self.rho = rho
        self.M = M

        print('M:', M)
        print('rho:', rho)

        diff_gain = -0.5 * rho * B.T @ M
        self.diff_gain_func = cs.Function('gain', [x_sym, u_sym], [diff_gain]) # input-independet

    @timing
    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        step = self.extract_step(info)

        if self.env.TASK == Task.STABILIZATION:
            # get the straight-line from the current state to the goal state
            line = np.linspace(obs, self.env.X_GOAL, 100)
            # intergrate the gain along the line
            action = np.zeros(self.model.nu)
            dummy_action = np.zeros(self.model.nu)
            for i in range(1, 100):
                action += self.diff_gain_func(line[i], dummy_action).full() @ (line[i - 1] - line[i] ) 
            # action = - 0.5 * self.rho * self.B_func(self.env.X_GOAL, dummy_action).full().T @ self.M @ (obs - self.env.X_GOAL)
            action += self.model.U_EQ
            return action
        elif self.env.TASK == Task.TRAJ_TRACKING:
            raise NotImplementedError('Trajectory tracking not implemented for CCM controller.')
