"""Template hyperparameter optimization/hyperparameter evaluation script.

"""
import os
from functools import partial

import yaml

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.envs.benchmark_env import Environment, Task

from safe_control_gym.hyperparameters.hpo import HPO
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_device_from_config, set_dir_from_config, set_seed_from_config


def train(config):
    """Training for a given set of hyperparameters.

    Usage:
        * to start training, use with `--func train`.

    """
    # Experiment setup.
    set_seed_from_config(config)
    set_device_from_config(config)

    # Define function to create task/env.
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    # Note:
    # eval_env will take config.seed * 111 as its seed
    # env will take config.seed as its seed
    control_agent = make(config.algo,
                         env_func,
                         training=True,
                         checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                         output_dir=config.output_dir,
                         use_gpu=config.use_gpu,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()

    eval_env = env_func(seed=config.seed * 111)

    experiment = BaseExperiment(eval_env, control_agent)
    if config.algo_config.num_epochs == 1:
        pass
    elif config.algo_config.gp_model_path is not None:
        # if config.algo_config.parallel:
        control_agent.load(config.algo_config.gp_model_path)
        # else:
        #     control_agent.
    else:
        experiment.launch_training()
        print('Training done in the main function.')
    results, metrics = experiment.run_evaluation(n_episodes=config.n_episodes, n_steps=None, done_on_max_steps=True)


    # plotting 
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
        if config.task_config.quad_type == 4:
            system = 'quadrotor_2D'
    else:
        system = config.task


    if control_agent.optimal_reference_path is not None:
        print('Plotting optimal trajectory')
        optimal_traj_load = np.load(control_agent.optimal_reference_path, allow_pickle=True)
        x_optimal = optimal_traj_load['trajs_data']['obs'][0]
        u_optimal = optimal_traj_load['trajs_data']['action'][0]

    if True:
        if system == Environment.CARTPOLE:
            graph1_1 = 2
            graph1_2 = 3
            graph3_1 = 0
            graph3_2 = 1
        elif system == 'quadrotor_2D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
        elif system == 'quadrotor_3D':
            graph1_1 = 6
            graph1_2 = 9
            graph3_1 = 0
            graph3_2 = 4
        
        # plot results['current_physical_action']
        _, ax_act = plt.subplots()
        ax_act.plot(results['current_physical_action'][0][:, 0], 'g', label='Thrust')
        if config.task_config.quad_type == 4:
            ax_act.plot(results['current_physical_action'][0][:, 1], 'r', label='Pitch')
        else:
            ax_act.plot(results['current_physical_action'][0][:, 1], 'r', label='Thrust')
        ax_act.legend()
        ax_act.set_xlabel('Step')
        ax_act.set_ylabel('Input')
        if control_agent.optimal_reference_path is not None:
            ax_act.plot(u_optimal[:, 0], 'b-.', label='Optimal Thrust')
            ax_act.plot(u_optimal[:, 1], 'b--', label='Optimal Pitch')
            ax_act.legend()
        plt.savefig(os.path.join(config.output_dir, 'inputs.png'))

        if config.task_config.quad_type != 4:
            _, ax = plt.subplots()
            ax.plot(results['obs'][0][:, graph1_1], results['obs'][0][:, graph1_2], 'r--', label='Agent Trajectory')
            ax.scatter(results['obs'][0][0, graph1_1], results['obs'][0][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
            ax.set_xlabel(r'$\theta$')
            ax.set_ylabel(r'$\dot{\theta}$')
            ax.set_box_aspect(0.5)
            ax.legend(loc='upper right')
            # save the plot
            plt.savefig(os.path.join(config.output_dir, 'trajectory_theta_theta_dot.png'))

            if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
                _, ax2 = plt.subplots()
                ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), results['obs'][0][:, 0], 'r--', label='Agent Trajectory')
                ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), eval_env.X_GOAL[:, 0], 'b', label='Reference')
                ax2.set_xlabel(r'Time')
                ax2.set_ylabel(r'X')
                ax2.set_box_aspect(0.5)
                ax2.legend(loc='upper right')
                # save the plot
                plt.savefig(os.path.join(config.output_dir, 'trajectory_time_x.png'))
            elif config.task == Environment.QUADROTOR:
                _, ax2 = plt.subplots()
                ax2.plot(results['obs'][0][:, graph3_1 + 1], results['obs'][0][:, graph3_2 + 1], 'r--', label='Agent Trajectory')
                ax2.set_xlabel(r'x_dot')
                ax2.set_ylabel(r'z_dot')
                ax2.set_box_aspect(0.5)
                ax2.legend(loc='upper right')
                # save the plot
                plt.savefig(os.path.join(config.output_dir, 'trajectory_x_dot_z_dot.png'))

        _, ax3 = plt.subplots()
        ax3.plot(results['obs'][0][:, graph3_1], results['obs'][0][:, graph3_2], 'r--', label='Agent Trajectory')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(eval_env.X_GOAL[:, graph3_1], eval_env.X_GOAL[:, graph3_2], 'g--', label='Reference')
        ax3.scatter(results['obs'][0][0, graph3_1], results['obs'][0][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')
        if control_agent.optimal_reference_path is not None:
            ax3.plot(x_optimal[:, graph3_1], x_optimal[:, graph3_2], 'b--', label='Optimal Trajectory')
            ax3.legend(loc='upper right')

        plt.tight_layout()
        # save the plot
        plt.savefig(os.path.join(config.output_dir, 'trajectory_x.png'))

        _, ax4 = plt.subplots(5, 1, figsize=(10, 10))
        ax4[0].plot(results['obs'][0][:, 0], 'r--', label='Agent Trajectory')
        ax4[0].plot(eval_env.X_GOAL[:, 0], 'g--', label='Reference')
        ax4[0].set_xlabel('Step')
        ax4[0].set_ylabel('X')
        ax4[0].legend()
        ax4[1].plot(results['obs'][0][:, 1], 'r--', label='Agent Trajectory')
        ax4[1].plot(eval_env.X_GOAL[:, 1], 'g--', label='Reference')
        ax4[1].set_xlabel('Step')
        ax4[1].set_ylabel('Z')
        ax4[1].legend()
        ax4[2].plot(results['obs'][0][:, 2], 'r--', label='Agent Trajectory')
        ax4[2].plot(eval_env.X_GOAL[:, 2], 'g--', label='Reference')
        ax4[2].set_xlabel('Step')
        ax4[2].set_ylabel('X_dot')
        ax4[2].legend()
        ax4[3].plot(results['obs'][0][:, 3], 'r--', label='Agent Trajectory')
        ax4[3].plot(eval_env.X_GOAL[:, 3], 'g--', label='Reference')
        ax4[3].set_xlabel('Step')
        ax4[3].set_ylabel('Z_dot')
        ax4[3].legend()
        ax4[4].plot(results['obs'][0][:, 4], 'r--', label='Agent Trajectory')
        ax4[4].plot(eval_env.X_GOAL[:, 4], 'g--', label='Reference')
        ax4[4].set_xlabel('Step')
        ax4[4].set_ylabel('Pitch')
        ax4[4].legend()
        if control_agent.optimal_reference_path is not None:
            ax4[0].plot(x_optimal[:, 0], 'b--', label='Optimal Trajectory')
            ax4[1].plot(x_optimal[:, 1], 'b--', label='Optimal Trajectory')
            ax4[2].plot(x_optimal[:, 2], 'b--', label='Optimal Trajectory')
            ax4[3].plot(x_optimal[:, 3], 'b--', label='Optimal Trajectory')
            ax4[4].plot(x_optimal[:, 4], 'b--', label='Optimal Trajectory')
            ax4[0].legend()
            ax4[1].legend()
            ax4[2].legend()
            ax4[3].legend()
            ax4[4].legend()
        plt.tight_layout()
        # save the plot
        plt.savefig(os.path.join(config.output_dir, 'trajectory_all.png'))


    # if control_agent.gaussian_process is not None:
    #     control_agent.line_search_for_CM()
    # else:
    #     control_agent.prior_ctrl.line_search_for_CM()
    control_agent.close()

    # save to pickle
    with open(os.path.join(config.output_dir, 'metrics.pkl'), 'wb') as f:
        import pickle
        pickle.dump(metrics, f)
    
    return eval_env.X_GOAL, results, metrics


# MAIN_FUNCS = {'hpo': hpo, 'train': train}
MAIN_FUNCS = {'train': train}


if __name__ == '__main__':
    import sys
    # Make config.
    # ALGO = 'gp_mpc'
    # ALGO = 'ilqr_c'
    ALGO = 'ilqr_gp'
    # SYS = 'cartpole'
    # TASK = 'stab'
    SYS = 'quadrotor_2D'
    TASK = 'tracking'
    PRIOR = '100'
    agent = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' else SYS
    assert os.path.exists(f'./config_overrides/{SYS}/{SYS}_{TASK}.yaml'), \
                          f'./{ALGO}/config_overrides/{SYS}/{SYS}_{TASK}.yaml does not exist'
    assert os.path.exists(f'./config_overrides/{SYS}/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), \
                          f'./{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
    
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', agent,
                    '--overrides',
                        f'./config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./config_overrides/{SYS}/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                    '--seed', '2',
                    # '--use_gpu', 'True',
                    '--output_dir', f'./{ALGO}/results',
                        ]
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    # merge config
    config = fac.merge()

    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception('Main function {} not supported.'.format(config.func))
    func(config)
