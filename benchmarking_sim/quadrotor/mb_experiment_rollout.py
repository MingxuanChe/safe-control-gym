
import os
import sys
import pickle
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.experiments.epoch_experiments import EpochExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, timing
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.utils.gpmpc_plotting import make_quad_plots
from benchmarking_sim.quadrotor.mb_experiment import plot_quad_eval
from safe_control_gym.controllers.mpc.gpmpc_base import GPMPC

script_path = os.path.dirname(os.path.realpath(__file__))

@timing
def run(gui=False, n_episodes=1, n_steps=None, save_data=True, 
        seed=2, Additional='', ALGO='pid', SYS='quadrotor_2D_attitude',
        noise_factor=1, 
        dw_height=None, dw_height_scale=None, 
        eval_task=None):
    '''The main function running experiments for model-based methods.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''
    # ALGO = 'ilqr'
    # ALGO = 'gp_mpc'
    # ALGO = 'gpmpc_acados'
    # ALGO = 'mpc'
    # ALGO = 'mpc_acados'
    ALGO = ALGO
    # ALGO = 'linear_mpc'
    # ALGO = 'lqr'
    # ALGO = 'lqr_c'
    # ALGO = 'pid'
    SYS = SYS
    TASK = 'tracking'
    # PRIOR = '200'
    PRIOR = '100'
    agent = 'quadrotor' if SYS in ['quadrotor_2D', 'quadrotor_2D_attitude', 'quadrotor_3D_attitude'] else SYS
    # ADDITIONAL = '_fast'
    ADDITIONAL = Additional
    # ADDITIONAL = ''
    SAFETY_FILTER = None
    # SAFETY_FILTER='linear_mpsc'

    # check if the config file exists
    assert os.path.exists(f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml'), f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml does not exist'
    assert os.path.exists(f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
    if SAFETY_FILTER is None:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', agent,
                        '--overrides',
                            f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml',
                            f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                        '--seed', repr(seed),
                        '--use_gpu', 'True',
                        '--output_dir', f'./{ALGO}/results',
                            ]
    else:
        MPSC_COST='one_step_cost'
        assert ALGO != 'gp_mpc', 'Safety filter not supported for gp_mpc'
        assert os.path.exists(f'./config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'), f'./config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', agent,
                        '--safety_filter', SAFETY_FILTER,
                        '--overrides',
                            f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml',
                            f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                            f'./config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml',
                        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}',
                        '--seed', repr(seed),
                        '--use_gpu', 'True',
                        '--output_dir', f'./{ALGO}/results',
                            ]
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    # merge config and create output directory
    config = fac.merge()
    gp_model_path = None
    if ALGO in ['gpmpc_acados', 'gp_mpc', 'gpmpc_acados_TP']:
        gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados_TP/results/100_200/temp'
    elif ALGO in ['gpmpc_acados_TRP']:
        gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados_TRP/results/100_200/temp'
    if gp_model_path is not None:
        # gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados/results/200_300_aggresive'
        # # get all directories in the gp_model_path
        gp_model_dirs = [d for d in os.listdir(gp_model_path) if os.path.isdir(os.path.join(gp_model_path, d))]
        gp_model_dirs = [os.path.join(gp_model_path, d) for d in gp_model_dirs]
        num_data_max = config.algo_config.num_epochs * config.algo_config.num_samples
        config.output_dir = os.path.join(config.output_dir, PRIOR + '_' + repr(num_data_max) + f'_rollout{ADDITIONAL}')
        # if seed%10 == 0:
        #     config.algo_config.gp_model_path = gp_model_dirs[10-1]
        # else:
        #     config.algo_config.gp_model_path = gp_model_dirs[seed%10-1]
    # else:
    if eval_task == 'rollout':
        config.output_dir = config.output_dir + f'_rollout_{SYS}{ADDITIONAL}'
    elif eval_task == 'obs_noise':
        config.output_dir = config.output_dir + f'_obs_noise_{SYS}/' + f'seed_{seed}'
    elif eval_task == 'proc_noise':
        config.output_dir = config.output_dir + f'_proc_noise_{SYS}/' + f'seed_{seed}'
    elif eval_task == 'downwash':
        config.output_dir = config.output_dir + f'_downwash_{SYS}/' + f'seed_{seed}'
    else:
        raise ValueError('eval_task not recognized')
        
    print('output_dir',  config.algo_config.output_dir)
    set_dir_from_config(config)
    config.algo_config.output_dir = config.output_dir
    mkdirs(config.output_dir)

    config.algo_config.gp_model_path = None
    if ALGO in ['gpmpc_acados', 'gp_mpc', 'gpmpc_acados_TP', 'gpmpc_acados_TRP']:
        config.algo_config.gp_model_path = gp_model_dirs[seed-1]
    
    # remove process noise if there is any
    config.task_config.disturbances.dynamics[0]['std'] = 0.0000 \
        if eval_task not in ['proc_noise'] \
        else config.task_config.disturbances.dynamics[0]['std']
    # amplify the observation noise std with a factor 
    if eval_task == 'obs_noise':
        default_noise_std = config.task_config.disturbances.observation[0]['std']
        print(f'Original observation noise std: {default_noise_std}')
        config.task_config.disturbances.observation[0]['std'] = [noise_factor * default_noise_std[i] for i in range(len(default_noise_std))]
        print(f'Amplified observation noise std: {config.task_config.disturbances.observation[0]["std"]}')
    elif eval_task == 'proc_noise':
        default_noise_std = config.task_config.disturbances.dynamics[0]['std']
        print(f'Original process noise std: {default_noise_std}')
        config.task_config.disturbances.dynamics[0]['std'] = noise_factor * default_noise_std 
        print(f'Amplified process noise std: {config.task_config.disturbances.dynamics[0]["std"]}')
    # downwash height scale
    elif eval_task == 'downwash':
        if dw_height is not None:
            config.task_config.disturbances.downwash[0].pos[-1] = dw_height
            print('downwash height: ', config.task_config.disturbances.downwash[0].pos)
        elif dw_height_scale is not None:
            max_dw_height, min_dw_height = 3, 0.5
            dw_height_space = max_dw_height - min_dw_height
            traj_center = config.task_config.task_info.trajectory_position_offset[1] # 1 [m] by default
            config.task_config.disturbances.downwash[0].pos[-1] = traj_center + min_dw_height + \
                                                                dw_height_scale * dw_height_space
            print(f'dw_height_scale: {dw_height_scale:.2f}')
            print('downwash height: ', config.task_config.disturbances.downwash[0].pos[-1])
    
    # Create an environment
    env_func = partial(make,
                       config.task,
                       seed=config.seed,
                       **config.task_config
                       )
    random_env = env_func(gui=False)

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                seed=config.seed,
                **config.algo_config
                )
    
    # Setup safety filter
    if SAFETY_FILTER is not None:
        env_func_filter = partial(make,
                                config.task,
                                seed=config.seed,
                                **config.task_config)
        safety_filter = make(config.safety_filter,
                            env_func_filter,
                            seed=config.seed,
                            **config.sf_config)
        safety_filter.reset()

    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes

    # Run the experiment.
    for _ in range(n_episodes):
        # Get initial state and create environments
        init_state, _ = random_env.reset()
        # init_state = random_env.reset()
        static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

        # Create experiment, train, and run evaluation
        if SAFETY_FILTER is None:  
            if isinstance(ctrl, GPMPC):
                experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
                if config.algo_config.num_epochs == 1:
                    print('Evaluating prior controller')
                elif config.algo_config.gp_model_path is not None:
                    ctrl.load(config.algo_config.gp_model_path)
                else:
                    # manually launch training 
                    # (NOTE: not using launch_training method since calling plotting before eval will break the eval)
                    experiment.reset()
                    train_runs, test_runs = ctrl.learn(env=static_train_env)
            else:   
                experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
                experiment.launch_training()
        else:
            safety_filter.learn(env=static_train_env)
            mkdirs(f'{script_path}/models/')
            safety_filter.save(path=f'{script_path}/models/{config.safety_filter}_{SYS}_{TASK}_{PRIOR}.pkl')
            ctrl.reset()
            experiment = BaseExperiment(env=static_env, ctrl=ctrl, safety_filter=safety_filter)

        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

        # plotting training and evaluation results
        # training
        if isinstance(ctrl, GPMPC) and \
           config.algo_config.gp_model_path is None and \
           config.algo_config.num_epochs > 1:
                if isinstance(static_env, Quadrotor):
                    make_quad_plots(test_runs=test_runs, 
                                    train_runs=train_runs, 
                                    trajectory=ctrl.traj.T,
                                    dir=ctrl.output_dir)
        plot_quad_eval(trajs_data['obs'][0], trajs_data['action'][0], ctrl.env, config.output_dir)


        # Close environments
        experiment.close()
        static_env.close()
        static_train_env.close()

        # Merge in new trajectory data
        for key, value in trajs_data.items():
            all_trajs[key] += value

    ctrl.close()
    random_env.close()
    metrics = experiment.compute_metrics(all_trajs)
    metrics['noise_factor'] = noise_factor
    metrics['dw_height_scale'] = dw_height_scale
    max_dw_force = None
    if hasattr(experiment.env, 'dw_model'):
        dw_force_log = experiment.env.dw_model.get_force_log()
        max_dw_force = np.max(dw_force_log)
    metrics['max_dw_force'] = max_dw_force    
    all_trajs = dict(all_trajs)
    
    if hasattr(experiment.env, 'dw_model'):
        force_log = experiment.env.dw_model.get_force_log()
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(force_log))/60, force_log)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Downwash force [N]')
        ax.set_title('Downwash force')
        fig.savefig(f'./{config.output_dir}/downwash_force.png')


    if save_data:
        results = {'trajs_data': all_trajs, 'metrics': metrics}
        with open(f'./{config.output_dir}/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)
        
        # save rmse to a file
        with open(f'./{config.output_dir}/metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
            print(f'Metrics saved to ./{config.output_dir}/metrics.txt')

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))

if __name__ == '__main__':

    if len(sys.argv) > 1:
        start_seed = int(sys.argv[1])
        num_seed = int(sys.argv[2])
        additional = sys.argv[3]
        if additional == 'none':
            additional = ''
        algo = sys.argv[4]

    else:
        start_seed = 1
    runtime_list = []
    # num_seed = 100
    for seed in range(start_seed, num_seed + start_seed):
        run(seed=seed, Additional=additional, ALGO=algo)
        runtime_list.append(run.elapsed_time)
    print(f'Average runtime for {num_seed} runs: \
          {np.mean(runtime_list):.3f} sec')

