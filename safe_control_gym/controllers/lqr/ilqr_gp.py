


import time
import os
from copy import deepcopy
from functools import partial

import casadi as cs
import gpytorch
import numpy as np
import scipy
import torch
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from skopt.sampler import Lhs
import munch

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.gp_utils import (GaussianProcessCollection, ZeroMeanIndependentGPModel,
                                                       covSEard, kmeans_centriods)
from safe_control_gym.controllers.mpc.gp_mpc import GPMPC
from safe_control_gym.controllers.lqr.ilqr_c import iLQR_C
from safe_control_gym.envs.benchmark_env import Task

class iLQR_GP(iLQR_C):
    '''Implements a GP-MPC controller with Acados optimization.'''
    def __init__(
            self,
            env_func,
            seed: int = 1337,
            horizon: int = 5,
            q_lqr: list = [1],
            r_lqr: list = [1],
            CM_search: bool = False,
            train_iterations: int = None,
            test_data_ratio: float = 0.2,
            overwrite_saved_data: bool = True,
            optimization_iterations: list = None,
            learning_rate: list = None,
            normalize_training_data: bool = False,
            use_gpu: bool = False,
            gp_model_path: str = None,
            prob: float = 0.955,
            input_mask: list = None,
            target_mask: list = None,
            prior_info: dict = None,
            prior_param_coeff: float = 1.0,
            terminate_run_on_done: bool = True,
            output_dir: str = 'results/temp',
            kernel: str = 'RBF',
            parallel: bool = False,
            **kwargs
    ):
        
        if prior_info is None or prior_info == {}:
            raise ValueError('ilqr GP requires prior_prop to be defined. You may use the real mass properties and then use prior_param_coeff to modify them accordingly.')
        prior_info['prior_prop'].update((prop, val * prior_param_coeff) for prop, val in prior_info['prior_prop'].items())
        self.prior_env_func = partial(env_func, inertial_prop=prior_info['prior_prop'])

        # Initialize the method using linear MPC.
        self.prior_ctrl = iLQR_C(
            self.prior_env_func,
            q_lqr=q_lqr,
            r_lqr=r_lqr,
            prior_info=prior_info,
            # runner args
            # shared/base args
            output_dir=output_dir,
        )
        self.prior_ctrl.reset()
        # super().__init__() # TODO: check the inheritance of the class
        super().__init__(
            env_func = env_func,
            seed= seed,
            horizon = horizon,
            q_lqr= q_lqr,
            r_lqr = r_lqr,
            CM_search= CM_search,
            train_iterations = train_iterations,
            test_data_ratio = test_data_ratio,
            overwrite_saved_data = overwrite_saved_data,
            optimization_iterations = optimization_iterations,
            learning_rate = learning_rate,
            normalize_training_data = normalize_training_data,
            use_gpu = use_gpu, 
            gp_model_path = gp_model_path,
            prob = prob,
            input_mask = input_mask,
            target_mask = target_mask,
            prior_info = prior_info,
            # inertial_prop: list = [1.0],
            prior_param_coeff = prior_param_coeff,
            terminate_run_on_done = terminate_run_on_done,
            output_dir = output_dir,
            **kwargs)
        # self.prior_ctrl.reset()
        # # Setup environments.
        self.env_func = env_func
        self.env = env_func(randomized_init=False, seed=seed)
        self.env_training = env_func(randomized_init=True, seed=seed)
        # No training data accumulated yet so keep the dynamics function as linear prior.
        self.train_data = None
        self.data_inputs = None
        self.data_targets = None
        self.prior_dynamics_func = self.prior_ctrl.dynamics_func # continuous-time prior
        self.X_EQ = self.prior_ctrl.model.X_EQ
        self.U_EQ = self.prior_ctrl.model.U_EQ
        # GP and training parameters.
        self.gaussian_process = None
        self.train_iterations = train_iterations
        self.test_data_ratio = test_data_ratio
        self.overwrite_saved_data = overwrite_saved_data
        self.optimization_iterations = optimization_iterations
        self.learning_rate = learning_rate
        self.gp_model_path = gp_model_path
        self.normalize_training_data = normalize_training_data
        self.prob = prob
        if input_mask is None:
            self.input_mask = np.arange(self.model.nx + self.model.nu).tolist()
        else:
            self.input_mask = input_mask
        if target_mask is None:
            self.target_mask = np.arange(self.model.nx).tolist()
        else:
            self.target_mask = target_mask
        Bd = np.eye(self.model.nx)
        self.Bd = Bd[:, self.target_mask]
        self.parallel = parallel
        self.kernel = kernel

        self.setup_prior_dynamics()

    def setup_prior_dynamics(self):
        self.prior_ctrl.set_dynamics_func()

    def setup_dynamics(self):
        pass

    #     self.model.x_sym
    #     self.model.u_sym


    #     A_lin = self.discrete_dfdx
    #     B_lin = self.discrete_dfdu

    #     if self.gaussian_process is None:
    #         f_disc = self.prior_dynamics_func(x0=acados_model.x- self.prior_ctrl.X_EQ[:, None], 
    #                                           p=acados_model.u- self.prior_ctrl.U_EQ[:, None])['xf'] \
    #             + self.prior_ctrl.X_EQ[:, None]
    #     else:
    #         z = cs.vertcat(acados_model.x, acados_model.u) # GP prediction point
    #         z = z[self.input_mask]
    #         if self.sparse_gp:
    #             raise NotImplementedError('Sparse GP not implemented for acados.')
    #         else:
    #             f_disc = self.prior_dynamics_func(x0=acados_model.x- self.prior_ctrl.X_EQ[:, None], 
    #                                               p=acados_model.u- self.prior_ctrl.U_EQ[:, None])['xf'] \
    #             + self.prior_ctrl.X_EQ[:, None]
    #             + self.Bd @ self.gaussian_process.casadi_predict(z=z)['mean']

    def select_action(self, obs, info=None):
        # print('current obs:', obs)
        time_before = time.time()
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            action = self.select_action_with_gp(obs)
        time_after = time.time()
        self.last_obs = obs
        self.last_action = action
        return action
    
    def select_action_with_gp(self, obs):
        nx, nu = self.model.nx, self.model.nu

        # set reference for the control horizon
        self.goal_state = self.get_references()
        if self.mode == 'tracking' or self.env.TASK == Task.TRAJ_TRACKING:
            self.traj_step += 1
        # linearization point
        x_0 = self.goal_state[:, 0]
        u_0 = self.model.U_EQ
        print('x_0:', x_0)
        print('u_0:', u_0)
        # linearize the prior dynamics
        df = self.prior_ctrl.model.df_func(x_0, u_0)
        A_prior, B_prior = df[0].toarray(), df[1].toarray()
        z = np.concatenate([x_0, u_0])
        A_gp = self.gaussian_process.casadi_linearized_predict(z=z)['A']
        B_gp = self.gaussian_process.casadi_linearized_predict(z=z)['B']
        print('A_gp:', A_gp)
        print('B_gp:', B_gp)
        assert A_gp.shape == (self.model.nx, self.model.nx)
        assert B_gp.shape == (self.model.nx, self.model.nu)
        A = A_prior + A_gp
        B = B_prior + B_gp
        print('A:', A)
        print('B:', B)
        P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
        gain = np.dot(np.linalg.inv(self.R), np.dot(B.T, P))
        A_cl = A - B @ gain
        eigenv = np.linalg.eigvals(A_cl)
        print('eigenv:', eigenv)
        action = -gain @ (obs - x_0) + u_0
        # pass
        return action
 
    def reset(self):
        '''Reset the controller before running.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            raise NotImplementedError('iLQR not implemented for stabilization task.')
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0
        # Dynamics model.
        print('=========== Resetting prior controller ===========')
        self.prior_ctrl.reset()

    def learn(self, env=None):
        '''Performs multiple epochs learning.
        '''

        train_runs = {0: {}}
        test_runs = {0: {}}

        if self.same_train_initial_state:
            train_envs = []
            for epoch in range(self.num_epochs):
                train_envs.append(self.env_func(randomized_init=True, seed=self.seed))
                train_envs[epoch].action_space.seed(self.seed)
        else:
            train_env = self.env_func(randomized_init=True, seed=self.seed)
            train_env.action_space.seed(self.seed)
            train_envs = [train_env] * self.num_epochs
        # init_test_states = get_random_init_states(env_func, num_test_episodes_per_epoch)
        test_envs = []
        if self.same_test_initial_state:
            for epoch in range(self.num_epochs):
                test_envs.append(self.env_func(randomized_init=True, seed=self.seed * 111))
                test_envs[epoch].action_space.seed(self.seed * 111)
        else:
            test_env = self.env_func(randomized_init=True, seed=self.seed * 111)
            test_env.action_space.seed(self.seed * 111)
            test_envs = [test_env] * self.num_epochs

        for episode in range(self.num_train_episodes_per_epoch):
            run_results = self.prior_ctrl.run(env=train_envs[0],
                                              terminate_run_on_done=self.terminate_train_on_done)
            train_runs[0].update({episode: munch.munchify(run_results)})
            self.reset()
        for test_ep in range(self.num_test_episodes_per_epoch):
            run_results = self.run(env=test_envs[0],
                                   terminate_run_on_done=self.terminate_test_on_done)
            test_runs[0].update({test_ep: munch.munchify(run_results)})
        self.reset()

        for epoch in range(1, self.num_epochs):
            # only take data from the last episode from the last epoch
            if self.rand_data_selection:
                x_seq, actions, x_dot_seq = self.gather_training_samples(train_runs, epoch - 1, self.num_samples, train_envs[epoch - 1].np_random)
            else:
                x_seq, actions, x_dot_seq = self.gather_training_samples(train_runs, epoch - 1, self.num_samples)
            train_inputs, train_outputs = self.preprocess_training_data(x_seq, actions, x_dot_seq)
            training_results = self.train_gp(input_data=train_inputs, target_data=train_outputs)

            # Test new policy.
            test_runs[epoch] = {}
            for test_ep in range(self.num_test_episodes_per_epoch):
                self.x_prev = test_runs[epoch - 1][episode]['obs'][:self.T + 1, :].T
                self.u_prev = test_runs[epoch - 1][episode]['action'][:self.T, :].T
                self.reset()
                run_results = self.run(env=test_envs[epoch],
                                       terminate_run_on_done=self.terminate_test_on_done)
                test_runs[epoch].update({test_ep: munch.munchify(run_results)})
            # gather training data
            train_runs[epoch] = {}
            for episode in range(self.num_train_episodes_per_epoch):
                self.reset()
                self.x_prev = train_runs[epoch - 1][episode]['obs'][:self.T + 1, :].T
                self.u_prev = train_runs[epoch - 1][episode]['action'][:self.T, :].T
                run_results = self.run(env=train_envs[epoch],
                                       terminate_run_on_done=self.terminate_train_on_done)
                train_runs[epoch].update({episode: munch.munchify(run_results)})

            lengthscale, outputscale, noise, kern = self.gaussian_process.get_hyperparameters(as_numpy=True)
        
        # save training data 
        np.savez(os.path.join(self.output_dir, 'data'),
                data_inputs=training_results['train_inputs'],
                data_targets=training_results['train_targets'])
        
        # close environments
        for env in train_envs:
            env.close()
        for env in test_envs:
            env.close()

        self.train_runs = train_runs
        self.test_runs = test_runs

        return train_runs, test_runs
    
    def gather_training_samples(self, all_runs, epoch_i, num_samples, rand_generator=None):
        n_episodes = len(all_runs[epoch_i].keys())
        num_samples_per_episode = int(num_samples / n_episodes)
        x_seq_int = []
        x_next_seq_int = []
        actions_int = []
        for episode_i in range(n_episodes):
            run_results_int = all_runs[epoch_i][episode_i]
            n = run_results_int['action'].shape[0]
            if num_samples_per_episode < n:
                if rand_generator is not None:
                    rand_inds_int = rand_generator.choice(n - 1, num_samples_per_episode, replace=False)
                else:
                    rand_inds_int = np.arange(num_samples_per_episode)
            else:
                rand_inds_int = np.arange(n - 1)
            next_inds_int = rand_inds_int + 1
            x_seq_int.append(run_results_int.obs[rand_inds_int, :])
            actions_int.append(run_results_int.action[rand_inds_int, :])
            x_next_seq_int.append(run_results_int.obs[next_inds_int, :])
        x_seq_int = np.vstack(x_seq_int)
        actions_int = np.vstack(actions_int)
        x_next_seq_int = np.vstack(x_next_seq_int)

        # return the x_seq, actions, and x_dot_seq
        x_dot_seq = (x_next_seq_int - x_seq_int) / self.dt
        return x_seq_int, actions_int, x_dot_seq
        # return x_seq_int, actions_int, x_next_seq_int

    def preprocess_training_data(self,
                                 x_seq,
                                 u_seq,
                                 x_dot_seq
                                 ):
        '''Converts trajectory data for GP trianing.

        Args:
            x_seq (list): state sequence of np.array (nx,).
            u_seq (list): action sequence of np.array (nu,).
            x_dot_seq (list): state derivative sequence of np.array (nx,).

        Returns:
            np.array: inputs for GP training, (N, nx+nu).
            np.array: targets for GP training, (N, nx).
        '''
        # Get the predicted dynamics. Nonlinear continuous-time dynamics.
        x_pred_seq = self.prior_dynamics_func(x=x_seq.T,
                                              u=u_seq.T)['f'].toarray()
        targets = (x_dot_seq.T - (x_pred_seq)).transpose()  # (N, nx).
        # check whether target is close to zero
        # print('target:', targets)
        # exit()
        inputs = np.hstack([x_seq, u_seq])  # (N, nx+nu).
        return inputs, targets
    
    def train_gp(self,
                 input_data=None,
                 target_data=None,
                 gp_model=None,
                 overwrite_saved_data: bool = None,
                 ):
        '''Performs GP training.

        Args:
            input_data, target_data (optiona, np.array): data to use for training
            gp_model (str): if not None, this is the path to pretrained models to use instead of training new ones.
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.

        Returns:
            training_results (dict): Dictionary of the training results.
        '''
        if gp_model is None:
            gp_model = self.gp_model_path
        if overwrite_saved_data is None:
            overwrite_saved_data = self.overwrite_saved_data
        self.reset()
        if input_data is None and target_data is None:
            # If no input data is provided, we will generate self.train_iterations
            # + (1+self.test_ratio)* self.train_iterations number of training points. This will ensure the specified
            # number of train iterations are run, and the correct train-test data spilt is achieved.
            train_inputs = []
            train_targets = []
            train_info = []

            ############
            # Use Latin Hypercube Sampling to generate states withing environment bounds.
            lhs_sampler = Lhs(lhs_type='classic', criterion='maximin')
            # limits = [(self.env.INIT_STATE_RAND_INFO[key].low, self.env.INIT_STATE_RAND_INFO[key].high) for key in
            #          self.env.INIT_STATE_RAND_INFO]
            limits = [(self.env.INIT_STATE_RAND_INFO['init_' + key]['low'], self.env.INIT_STATE_RAND_INFO['init_' + key]['high']) for key in self.env.STATE_LABELS]
            # TODO: parameterize this if we actually want it.
            num_eq_samples = 0
            validation_iterations = int(self.train_iterations * (self.test_data_ratio / (1 - self.test_data_ratio)))
            samples = lhs_sampler.generate(limits,
                                           self.train_iterations + validation_iterations - num_eq_samples,
                                           random_state=self.seed)
            if self.env.TASK == Task.STABILIZATION and num_eq_samples > 0:
                # TODO: choose if we want eq samples or not.
                delta_plus = np.array([0.1, 0.1, 0.1, 0.1, 0.03, 0.3])
                delta_neg = np.array([0.1, 0.1, 0.1, 0.1, 0.03, 0.3])
                eq_limits = [(self.prior_ctrl.env.X_GOAL[eq] - delta_neg[eq], self.prior_ctrl.env.X_GOAL[eq] + delta_plus[eq]) for eq in range(self.model.nx)]
                eq_samples = lhs_sampler.generate(eq_limits, num_eq_samples, random_state=self.seed)
                # samples = samples.append(eq_samples)
                init_state_samples = np.array(samples + eq_samples)
            else:
                init_state_samples = np.array(samples)
            input_limits = np.vstack((self.constraints.input_constraints[0].lower_bounds,
                                      self.constraints.input_constraints[0].upper_bounds)).T
            input_samples = lhs_sampler.generate(input_limits,
                                                 self.train_iterations + validation_iterations,
                                                 random_state=self.seed)
            input_samples = np.array(input_samples)  # not being used currently
            seeds = self.env.np_random.integers(0, 99999, size=self.train_iterations + validation_iterations)
            for i in range(self.train_iterations + validation_iterations):
                # For random initial state training.
                # init_state = init_state_samples[i,:]
                init_state = dict(zip(self.env.INIT_STATE_RAND_INFO.keys(), init_state_samples[i, :]))
                # Collect data with prior controller.
                run_env = self.env_func(init_state=init_state, randomized_init=False, seed=int(seeds[i]))
                episode_results = self.prior_ctrl.run(env=run_env, max_steps=1)
                run_env.close()
                x_obs = episode_results['obs'][-3:, :]
                u_seq = episode_results['action'][-1:, :]
                run_env.close()
                x_seq = x_obs[:-1, :]
                x_next_seq = x_obs[1:, :]
                train_inputs_i, train_targets_i = self.preprocess_training_data(x_seq, u_seq, x_next_seq)
                train_inputs.append(train_inputs_i)
                train_targets.append(train_targets_i)
            train_inputs = np.vstack(train_inputs)
            train_targets = np.vstack(train_targets)
            self.data_inputs = train_inputs
            self.data_targets = train_targets
        elif input_data is not None and target_data is not None:
            train_inputs = input_data
            train_targets = target_data
            if (self.data_inputs is None and self.data_targets is None) or overwrite_saved_data:
                self.data_inputs = train_inputs
                self.data_targets = train_targets
            else:
                self.data_inputs = np.vstack((self.data_inputs, train_inputs))
                self.data_targets = np.vstack((self.data_targets, train_targets))
        else:
            raise ValueError('[ERROR]: gp_mpc.learn(): Need to provide both targets and inputs.')

        total_input_data = self.data_inputs.shape[0]
        # If validation set is desired.
        if self.test_data_ratio > 0 and self.test_data_ratio is not None:
            train_idx, test_idx = train_test_split(
                list(range(total_input_data)),
                test_size=self.test_data_ratio,
                random_state=self.seed
            )

        else:
            # Otherwise, just copy the training data into the test data.
            train_idx = list(range(total_input_data))
            test_idx = list(range(total_input_data))

        train_inputs = self.data_inputs[train_idx, :]
        train_targets = self.data_targets[train_idx, :]
        self.train_data = {'train_inputs': train_inputs, 'train_targets': train_targets}
        test_inputs = self.data_inputs[test_idx, :]
        test_targets = self.data_targets[test_idx, :]
        self.test_data = {'test_inputs': test_inputs, 'test_targets': test_targets}

        train_inputs_tensor = torch.Tensor(train_inputs).double()
        train_targets_tensor = torch.Tensor(train_targets).double()
        test_inputs_tensor = torch.Tensor(test_inputs).double()
        test_targets_tensor = torch.Tensor(test_targets).double()

        # Define likelihood.
        # likelihood = gpytorch.likelihoods.GaussianLikelihood(
        #     noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        # ).double()
        # self.gaussian_process = GaussianProcessCollection(ZeroMeanIndependentGPModel,
        #                                                   likelihood,
        #                                                   len(self.target_mask),
        #                                                   input_mask=self.input_mask,
        #                                                   target_mask=self.target_mask,
        #                                                   normalize=self.normalize_training_data,
        #                                                   kernel=self.kernel,
        #                                                   parallel=self.parallel
        #                                                   )
        if self.parallel:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([len(self.target_mask)]),
                                                                 noise_constraint=gpytorch.constraints.GreaterThan(1e-6)).double()
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
            ).double()
        self.gaussian_process = GaussianProcessCollection(ZeroMeanIndependentGPModel,
                                                          likelihood,
                                                          len(self.target_mask),
                                                          input_mask=self.input_mask,
                                                          target_mask=self.target_mask,
                                                          normalize=self.normalize_training_data,
                                                          kernel=self.kernel,
                                                          parallel=self.parallel
                                                          )
        if gp_model:
            self.gaussian_process.init_with_hyperparam(train_inputs_tensor,
                                                       train_targets_tensor,
                                                       gp_model)
        else:
            # Train the GP.
            self.gaussian_process.train(train_inputs_tensor,
                                        train_targets_tensor,
                                        test_inputs_tensor,
                                        test_targets_tensor,
                                        n_train=self.optimization_iterations,
                                        learning_rate=self.learning_rate,
                                        gpu=self.use_gpu,
                                        output_dir=self.output_dir)
        self.gaussian_process.plot_trained_gp(train_inputs_tensor,
                                                train_targets_tensor,)
                                                
        self.reset()
        self.prior_ctrl.reset()
        # Collect training results.
        training_results = {}
        training_results['train_targets'] = train_targets
        training_results['train_inputs'] = train_inputs
        try:
            training_results['info'] = train_info
        except UnboundLocalError:
            training_results['info'] = None
        return training_results
    
    def load(self, model_path):
        '''Loads a pretrained batch GP model.        Args:
            model_path (str): Path to the pretrained model.
        '''
        data = np.load(f'{model_path}/data.npz')
        input_data=data['data_inputs']
        target_data=data['data_targets']
        if self.parallel:
            gp_model_path = f'{model_path}/best_model.pth'
            self.train_gp(input_data=data['data_inputs'], target_data=data['data_targets'], gp_model=gp_model_path)
        else:
            self.train_gp(input_data=data['data_inputs'], target_data=data['data_targets'], gp_model=model_path)
        print('================== GP models loaded. =================')
        