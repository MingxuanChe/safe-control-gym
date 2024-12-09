'''This script tests the MPSC safety filter implementation.'''

import pickle
import shutil
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.experiments.base_experiment import BaseExperiment, MetricExtractor
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(plot=False, model='ppo'):
    '''Main function to run MPSC experiments.

    Returns:
        X_GOAL (np.ndarray): The goal (stabilization or reference trajectory) of the experiment.
        uncert_results (dict): The results of the uncertified experiment.
        uncert_metrics (dict): The metrics of the uncertified experiment.
        cert_results (dict): The results of the certified experiment.
        cert_metrics (dict): The metrics of the certified experiment.
    '''

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    config.algo_config['training'] = False
    config.task_config['done_on_violation'] = False
    config.task_config['randomized_init'] = False

    system = 'quadrotor_2D_attitude'

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func()

    # Tightening constraints beyond tolerance
    config.task_config.constraints[0].upper_bounds = [0.899, 1.99, 1.449, 1.99, 0.749, 2.99]
    config.task_config.constraints[0].lower_bounds = [-0.899, -1.99, 0.551, -1.99, -0.749, -2.99]
    config.task_config.constraints[1].upper_bounds = [0.59, 0.436]
    config.task_config.constraints[1].lower_bounds = [0.113, -0.436]
    env_func = partial(make,
                       config.task,
                       **config.task_config)

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir='./temp')

    if config.algo in ['ppo', 'sac']:
        # Load state_dict from trained.
        ctrl.load(f'./models/rl_models/{model}/model_best.pt')

        # Remove temporary files and directories
        shutil.rmtree('./temp', ignore_errors=True)

    # Run without safety filter
    experiment = BaseExperiment(env, ctrl)
    uncert_results, uncert_metrics = experiment.run_evaluation(n_episodes=1)
    ctrl.reset()

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                         env_func,
                         **config.sf_config)
    safety_filter.reset()

    if config.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl
        safety_filter.cost_function.output_dir = '.'

    safety_filter.load(path=f'./models/mpsc_parameters/{config.safety_filter}_{system}.pkl')

    # Run with safety filter
    experiment = BaseExperiment(env, ctrl, safety_filter=safety_filter)
    cert_results, cert_metrics = experiment.run_evaluation(n_episodes=1)
    experiment.close()
    safety_filter.close()

    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(uncert_results['state'][0][:, 0], uncert_results['state'][0][:, 2], label='Uncertified', color='red')
        ax.plot(cert_results['state'][0][:, 0], cert_results['state'][0][:, 2], label='Certified', color='green')
        ax.plot(env.X_GOAL[:, 0], env.X_GOAL[:, 2], label='Reference', color='black', linestyle='dashdot')
        rec1 = plt.Rectangle((0.9, 0), 2, 2, color='#f1d6d6')
        rec2 = plt.Rectangle((-1.9, 0), 1, 2, color='#f1d6d6')
        ax.add_patch(rec1)
        ax.add_patch(rec2)
        rec3 = plt.Rectangle((-0.9, 1.45), 0.975 * 2, 2, color='#f1d6d6')
        rec4 = plt.Rectangle((-0.9, -0.45), 0.975 * 2, 1, color='#f1d6d6')
        ax.add_patch(rec3)
        ax.add_patch(rec4)
        plt.xlim(-1.1, 1.1)
        plt.ylim(0.45, 1.55)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.legend()
        plt.show()

    elapsed_time_uncert = uncert_results['timestamp'][0][-1] - uncert_results['timestamp'][0][0]
    elapsed_time_cert = cert_results['timestamp'][0][-1] - cert_results['timestamp'][0][0]

    mpsc_results = cert_results['safety_filter_data'][0]
    corrections = mpsc_results['correction'][0] * 10.0 > np.linalg.norm(cert_results['current_physical_action'][0] - safety_filter.U_EQ[0], axis=1)
    corrections = np.append(corrections, False)

    print('Total Uncertified (s):', elapsed_time_uncert)
    print('Total Certified Time (s):', elapsed_time_cert)
    print('Number of Corrections:', np.sum(corrections))
    print('Sum of Corrections:', np.linalg.norm(mpsc_results['correction'][0]))
    print('Max Correction:', np.max(np.abs(mpsc_results['correction'][0])))
    print('Number of Feasible Iterations:', np.sum(mpsc_results['feasible'][0]))
    print('Total Number of Iterations:', uncert_metrics['average_length'])
    print('Total Number of Certified Iterations:', cert_metrics['average_length'])
    print('Number of Violations:', uncert_metrics['average_constraint_violation'])
    print('Number of Certified Violations:', cert_metrics['average_constraint_violation'])

    return env.X_GOAL, uncert_results, uncert_metrics, cert_results, cert_metrics


def run_multiple_models(plot, all_models):
    '''Runs all models at every saved starting point.'''

    fac = ConfigFactory()
    config = fac.merge()

    for model in all_models:
        print(model)
        for i in range(25 if not plot else 1):
            X_GOAL, uncert_results, _, cert_results, _ = run(plot=plot, model=model)
            if i == 0:
                all_uncert_results, all_cert_results = uncert_results, cert_results
            else:
                for key in all_cert_results.keys():
                    if key in all_uncert_results:
                        all_uncert_results[key].append(uncert_results[key][0])
                    all_cert_results[key].append(cert_results[key][0])

        met = MetricExtractor()
        uncert_metrics = met.compute_metrics(data=all_uncert_results, max_steps=660)
        cert_metrics = met.compute_metrics(data=all_cert_results, max_steps=66)

        all_results = {'uncert_results': all_uncert_results,
                       'uncert_metrics': uncert_metrics,
                       'cert_results': all_cert_results,
                       'cert_metrics': cert_metrics,
                       'config': config,
                       'X_GOAL': X_GOAL}

        if not plot:
            with open(f'./results_mpsc/{model}.pkl', 'wb') as f:
                pickle.dump(all_results, f)


if __name__ == '__main__':
    run(plot=True, model='mpsf7')
    # run_multiple_models(plot=True, all_models=['mpsf7'])
