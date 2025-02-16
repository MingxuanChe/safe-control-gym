'''Utility functions for Gaussian Processes.'''

import os.path
from copy import deepcopy

import casadi as ca
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans
from termcolor import colored

from safe_control_gym.utils.utils import mkdirs


torch.manual_seed(0)


def covSEard(x,
             z,
             ell,
             sf2
             ):
    '''GP squared exponential kernel.

    This function is based on the 2018 GP-MPC library by Helge-André Langåker

    Args:
        x (np.array or casadi.MX/SX): First vector.
        z (np.array or casadi.MX/SX): Second vector.
        ell (np.array or casadi.MX/SX): Length scales.
        sf2 (float or casadi.MX/SX): output scale parameter.

    Returns:
        SE kernel (casadi.MX/SX): SE kernel.
    '''
    dist = ca.sum1((x - z)**2 / ell**2)
    return sf2 * ca.SX.exp(-.5 * dist)

def covSE_single(x,
          z,
          ell,
          sf2
          ):
    dist = ca.sum1((x - z) ** 2 / ell ** 2)
    return sf2 * ca.exp(-.5 * dist)

def covMatern52ard(x,
                   z,
                   ell,
                   sf2
                   ):
    '''Matern kernel that takes nu equal to 5/2.

    Args:
        x (np.array or casadi.MX/SX): First vector.
        z (np.array or casadi.MX/SX): Second vector.
        ell (np.array or casadi.MX/SX): Length scales.
        sf2 (float or casadi.MX/SX): output scale parameter.

    Returns:
        Matern52 kernel (casadi.MX/SX): Matern52 kernel.

    '''
    dist = ca.sum1((x - z)**2 / ell**2)
    r_over_l = ca.sqrt(dist)
    return sf2 * (1 + ca.sqrt(5) * r_over_l + 5 / 3 * r_over_l ** 2) * ca.exp(- ca.sqrt(5) * r_over_l)

class ZeroMeanIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    '''Multidimensional Gaussian Process model with zero mean function.

    Or constant mean and radial basis function kernel (SE).

    '''

    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 nx,
                 kernel='RBF'
                 ):
        '''Initialize a multidimensional Gaussian Process model with zero mean function.

        Args:
            train_x (torch.Tensor): input training data (input_dim X N samples).
            train_y (torch.Tensor): output training data (output dim x N samples).
            likelihood (gpytorch.likelihood): Likelihood function (gpytorch.likelihoods.MultitaskGaussianLikelihood).
            nx (int): dimension of the target output (output dim)
        '''
        super().__init__(train_x, train_y, likelihood)
        self.n = nx
        # For Zero mean function.
        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([self.n]))
        # For constant mean function.
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.n]),
                                           ard_num_dims=train_x.shape[1]),
                batch_shape=torch.Size([self.n]),
                ard_num_dims=train_x.shape[1]
            )
        elif kernel == 'RBF_single':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.n])),
                batch_shape=torch.Size([self.n])
            )
        elif kernel == 'Matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(batch_shape=torch.Size([self.n]),
                                              ard_num_dims=train_x.shape[1]),
                batch_shape=torch.Size([self.n]),
                ard_num_dims=train_x.shape[1]
            )
        else:
            raise NotImplementedError

    def forward(self,
                x
                ):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class ZeroMeanIndependentGPModel(gpytorch.models.ExactGP):
    '''Single dimensional output Gaussian Process model with zero mean function.

    Or constant mean and radial basis function kernel (SE).
    '''

    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 kernel='RBF'
                 ):
        '''Initialize a single dimensional Gaussian Process model with zero mean function.

        Args:
            train_x (torch.Tensor): input training data (input_dim X N samples).
            train_y (torch.Tensor): output training data (output dim x N samples).
            likelihood (gpytorch.likelihood): Likelihood function (gpytorch.likelihoods.GaussianLikelihood).
        '''
        super().__init__(train_x, train_y, likelihood)
        # For Zero mean function.
        self.mean_module = gpytorch.means.ZeroMean()
        # For constant mean function.
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]),
                ard_num_dims=train_x.shape[1]
            )
        elif kernel == 'Matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1]),
                ard_num_dims=train_x.shape[1]
            )
        elif kernel == 'RBF_single':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(),
            )

    def forward(self,
                x
                ):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    '''Multidimensional Gaussian Process model with zero mean function.
    '''

    def __init__(self, train_x, train_y, likelihood, kernel='RBF'):
        '''Initialize a multidimensional Gaussian Process model with zero mean function.

        Args:
            train_x (torch.Tensor): input training data (input_dim X N samples).
            train_y (torch.Tensor): output training data (output dim x N samples).
            likelihood (gpytorch.likelihood): Likelihood function (gpytorch.likelihoods.GaussianLikelihood with batch).
        '''
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([train_y.shape[0]]))
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1], batch_shape=torch.Size([train_y.shape[0]])),
                batch_shape=torch.Size([train_y.shape[0]]), ard_num_dims=train_x.shape[-1]
            )
        elif kernel == 'RBF_single':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([train_y.shape[0]])),
                batch_shape=torch.Size([train_y.shape[0]])
            )
        elif kernel == 'Matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[-1], batch_shape=torch.Size([train_y.shape[0]])),
                batch_shape=torch.Size([train_y.shape[0]]), ard_num_dims=train_x.shape[-1]
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessCollection:
    '''Collection of GaussianProcesses for multioutput GPs.'''

    def __init__(self, model_type,
                 likelihood,
                 target_dim,
                 input_mask=None,
                 target_mask=None,
                 kernel='Matern',
                 **kwargs
                 ):
        '''Creates a single GaussianProcess for each output dimension.

        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentGPModel).
            likelihood (gpytorch.likelihood): likelihood function.
            target_dim (int): Dimension of the output (how many GPs to make).
            input_mask (list): Input dimensions to keep. If None, use all input dimensions.
            target_mask (list): Target dimensions to keep. If None, use all target dimensions.
        '''
        self.gp_list = []
        self.model_type = model_type
        self.likelihood = likelihood
        self.optimizer = None
        self.model = None
        self.input_mask = input_mask
        self.target_mask = target_mask
        # perform standardization on the input data and target data.
        self.input_scaler = preprocessing.StandardScaler()
        self.output_scaler = preprocessing.StandardScaler()

        # set prior noise std
        for i in range(target_dim):
            self.gp_list.append(GaussianProcess(model_type,
                                                deepcopy(likelihood),
                                                input_mask=input_mask,
                                                kernel=kernel,
                                                )
                                )

    def _init_properties(self,
                         train_inputs,
                         train_targets
                         ):
        '''Initialize useful properties.

        Args:
            train_inputs, train_targets (torch.tensors): Input and target training data.
        '''
        target_dimension = train_targets.shape[1]
        self.input_dimension = train_inputs.shape[1]
        self.output_dimension = target_dimension
        self.n_training_samples = train_inputs.shape[0]

    def init_with_hyperparam(self,
                             train_inputs,
                             train_targets,
                             path_to_statedicts
                             ):
        '''Load hyperparameters from a state_dict.

        Args:
            train_inputs, train_targets (torch.tensors): Input and target training data.
            path_to_statedicts (str): Path to where the state dicts are saved.
        '''

        self._init_properties(train_inputs, train_targets)
        gp_K_plus_noise_list = []
        gp_K_plus_noise_inv_list = []
        for gp_ind, gp in enumerate(self.gp_list):
            path = os.path.join(path_to_statedicts, f'best_model_{self.target_mask[gp_ind]}.pth')
            print('#########################################')
            print(f'#       Loading GP dimension {self.target_mask[gp_ind]}         #')
            print('#########################################')
            print(f'Path: {path}')
            gp.init_with_hyperparam(train_inputs,
                                    train_targets[:, self.target_mask[gp_ind]],
                                    path)
            gp_K_plus_noise_list.append(gp.model.K_plus_noise.detach())
            gp_K_plus_noise_inv_list.append(gp.model.K_plus_noise_inv.detach())
            print('Loaded!')
        gp_K_plus_noise = torch.stack(gp_K_plus_noise_list)
        gp_K_plus_noise_inv = torch.stack(gp_K_plus_noise_inv_list)
        
        self.K_plus_noise = gp_K_plus_noise
        self.K_plus_noise_inv = gp_K_plus_noise_inv
        self.casadi_predict = self.make_casadi_predict_func()

    def get_hyperparameters(self,
                            as_numpy=False
                            ):
        '''Get the outputscale and lengthscale from the kernel matrices of the GPs.'''
        lengthscale_list = []
        output_scale_list = []
        noise_list = []
        
        for gp in self.gp_list:
            lengthscale_list.append(gp.model.covar_module.base_kernel.lengthscale.detach())
            output_scale_list.append(gp.model.covar_module.outputscale.detach())
            noise_list.append(gp.model.likelihood.noise.detach())
        lengthscale = torch.cat(lengthscale_list)
        outputscale = torch.Tensor(output_scale_list)
        noise = torch.Tensor(noise_list)
    

        if as_numpy:
            return lengthscale.numpy(), outputscale.numpy(), noise.numpy(), self.K_plus_noise.detach().numpy()
        else:
            return lengthscale, outputscale, noise, self.K_plus_noise

    def train(self,
              train_x_raw,
              train_y_raw,
              test_x_raw,
              test_y_raw,
              n_train=[500],
              learning_rate=[0.01],
              gpu=False,
              output_dir='results'
              ):
        '''Train the GP using Train_x and Train_y.

        Args:
            train_x: Torch tensor (N samples [rows] by input dim [cols])
            train_y: Torch tensor (N samples [rows] by target dim [cols])
        '''

        self._init_properties(train_x_raw, train_y_raw)
        self.model_paths = []
        mkdirs(output_dir)

        gp_K_plus_noise_inv_list = []
        gp_K_plus_noise_list = []
        for gp_ind, gp in enumerate(self.gp_list):
            lr = learning_rate[self.target_mask[gp_ind]]
            n_t = n_train[self.target_mask[gp_ind]]
            print('#########################################')
            print(f'#      Training GP dimension {self.target_mask[gp_ind]}         #')
            print('#########################################')
            print(f'Train iterations: {n_t}')
            print(f'Learning Rate: {lr}')
            gp.train(train_x_raw,
                        train_y_raw[:, self.target_mask[gp_ind]],
                        test_x_raw,
                        test_y_raw[:, self.target_mask[gp_ind]],
                        n_train=n_t,
                        learning_rate=lr,
                        gpu=gpu,
                        fname=os.path.join(output_dir, f'best_model_{self.target_mask[gp_ind]}.pth'))
            self.model_paths.append(output_dir)
            gp_K_plus_noise_list.append(gp.model.K_plus_noise)
            gp_K_plus_noise_inv_list.append(gp.model.K_plus_noise_inv)
        gp_K_plus_noise = torch.stack(gp_K_plus_noise_list)
        gp_K_plus_noise_inv = torch.stack(gp_K_plus_noise_inv_list)
        self.K_plus_noise = gp_K_plus_noise
        self.K_plus_noise_inv = gp_K_plus_noise_inv


        self.casadi_predict = self.make_casadi_predict_func()

    def predict(self,
                x,
                requires_grad=False,
                return_pred=True
                ):
        '''
        Args:
            x : torch.Tensor (N_samples x input DIM).

        Return
            Predictions
                means : torch.tensor (N_samples x output DIM).
                covs  : torch.tensor (N_samples x output DIM x output DIM). 
            NOTE: For compatibility with the original implementation, 
            the output will be squeezed when N_samples == 1.
        '''
        num_batch = x.shape[0]
        dim_input = len(self.input_mask)
        dim_output = len(self.target_mask)

        means_list = []
        cov_list = []
        pred_list = []
        for gp in self.gp_list:
            if return_pred:
                mean, cov, pred = gp.predict(x, requires_grad=requires_grad, return_pred=return_pred)
                pred_list.append(pred)
            else:
                mean, cov = gp.predict(x, requires_grad=requires_grad, return_pred=return_pred)
            means_list.append(mean)
            cov_list.append(torch.diag(cov))

        # Stack the means and covariances.
        means = torch.zeros(num_batch, dim_output)
        for i in range(dim_output):
            means[:, i] = means_list[i].squeeze()
        assert means.shape == (num_batch, dim_output), ValueError('Means have wrong shape.')
        covs = torch.zeros(num_batch, dim_output, dim_output)
        for i in range(dim_output):
            covs[:, i, i] = cov_list[i].squeeze()
        assert covs.shape == (num_batch, dim_output, dim_output), ValueError('Covariances have wrong shape.')

        # squeeze the means if num_batch == 1 to retain the original implementation.
        if num_batch == 1:
            means = means.squeeze()
            covs = covs.squeeze()

        if return_pred:
            return means, covs, pred_list
        else:
            return means, covs



    def make_casadi_predict_func(self):
        '''
        Assume train_inputs and train_tergets are already
        '''

        Nz = len(self.input_mask)
        Ny = len(self.target_mask)
        z = ca.SX.sym('z1', Nz)
        y = ca.SX.zeros(Ny)


        for gp_ind, gp in enumerate(self.gp_list):
            y[gp_ind] = gp.casadi_predict(z=z)['mean']
        

        casadi_predict = ca.Function('pred',
                                     [z],
                                     [y],
                                     ['z'],
                                     ['mean'])
        return casadi_predict

    def prediction_jacobian(self,
                            query
                            ):
        '''Return Jacobian.'''
        raise NotImplementedError

    def plot_trained_gp(self,
                        inputs,
                        targets,
                        fig_count=0,
                        output_dir=None,
                        title=None,
                        **kwargs
                        ):
        '''Plot the trained GP given the input and target data.
           NOTE: The result will be saved in the output_dir if provided.
        '''
        inputs = torch.from_numpy(inputs) if type(inputs) is np.ndarray else inputs
        targets = torch.from_numpy(targets) if type(targets) is np.ndarray else targets
        
        if self.target_mask is not None:
            targets = targets[:, self.target_mask]
        num_data = inputs.shape[0]
        num_gps = len(self.gp_list)
        residual_func = kwargs.get('residual_func', None)
        residual = np.zeros((num_data, targets.shape[1]))
        if residual_func is not None:
            for i in range(num_data):
                residual[i, :] = residual_func(inputs[i, :].numpy())[self.target_mask].full().flatten()

        target_label = ['m/s', 'm/s', 'rad/s']

        x = ca.MX.sym('x', self.input_dimension)
        feature = [
            ca.vertcat(ca.sin(x[4]), ca.sin(x[4]) * x[6],),
            ca.vertcat(ca.sin(x[4]), ca.sin(x[4]) * x[6],),
            ca.vertcat(x[4], x[5], x[7]),
        ]
        feature = [ca.Function('feature', [x], [f]) for f in feature]

        t = np.arange(num_data)
        fig, axs = plt.subplots(num_gps, 1, figsize=(10, 3 * num_gps))
        for gp_ind, gp in enumerate(self.gp_list):

            means, _ , preds = gp.predict(inputs, return_pred=True)
            lower, upper = preds.confidence_region()
            num_within_2std = torch.sum((targets[:, gp_ind] > lower) & (targets[:, gp_ind] < upper)).numpy()
            percentage_within_2std = num_within_2std / len(targets[:, gp_ind]) * 100
            print(f'Percentage of test points within 2 std for dim {self.target_mask[gp_ind]}: {percentage_within_2std:.2f}%')
            axs[gp_ind].scatter(t, targets[:, gp_ind], c='r', label='Target', s=5) 
            axs[gp_ind].plot(t, means, 'b', label='GP mean', linewidth=3)
            axs[gp_ind].fill_between(t, lower, upper, alpha=0.3, label='2 std', color='skyblue')
            axs[gp_ind].plot(t, residual[:, gp_ind], 'g', label='Residual', linestyle='--', linewidth=3)
            axs[gp_ind].set_ylabel(target_label[gp_ind])
        
        axs[-1].set_xlabel('Data index')
        axs[0].legend(ncol=2)
        fig.tight_layout()
        if output_dir is not None:
            plt_name = f'gp_validation.png' if title is None else f'gp_validation_{title}.png'
            fig.savefig(os.path.join(output_dir, plt_name))
            print(colored(f'Plot saved to {os.path.join(output_dir, plt_name)}', 'green'))
        plt.close(fig)


    def _kernel_list(self,
                     x1,
                     x2=None
                     ):
        '''Evaluate the kernel given vectors x1 and x2.

        Args:
            x1 (torch.Tensor): First vector.
            x2 (torch.Tensor): Second vector.

        Returns:
            list of LazyTensor Kernels.
        '''
        if x2 is None:
            x2 = x1

        k_list = []
        for gp in self.gp_list:
            k_list.append(gp.model.covar_module(x1, x2))

        return k_list

    def kernel(self,
               x1,
               x2=None
               ):
        '''Evaluate the kernel given vectors x1 and x2.

        Args:
            x1 (torch.Tensor): First vector.
            x2 (torch.Tensor): Second vector.

        Returns:
            Torch tensor of the non-lazy kernel matrices.
        '''
        k_list = self._kernel_list(x1, x2)
        non_lazy_tensors = [k.evaluate() for k in k_list]
        return torch.stack(non_lazy_tensors)

    def kernel_inv(self,
                   x1,
                   x2=None
                   ):
        '''Evaluate the inverse kernel given vectors x1 and x2.

        Only works for square kernel.

        Args:
            x1 (torch.Tensor): First vector.
            x2 (torch.Tensor): Second vector.

        Returns:
            Torch tensor of the non-lazy inverse kernel matrices.
        '''
        if x2 is None:
            x2 = x1
        assert x1.shape == x2.shape, ValueError('x1 and x2 need to have the same shape.')
        k_list = self._kernel_list(x1, x2)
        num_of_points = x1.shape[0]
        # Efficient inversion is performed VIA inv_matmul on the laze tensor with Identity.
        non_lazy_tensors = [k.inv_matmul(torch.eye(num_of_points).double()) for k in k_list]
        return torch.stack(non_lazy_tensors)


class GaussianProcess:
    '''Gaussian Process decorator for gpytorch.'''

    def __init__(self,
                 model_type,
                 likelihood,
                 input_mask=None,
                 target_mask=None,
                 kernel='RBF',
                 noise_prior=None
                 ):
        '''Initialize Gaussian Process.

        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentMultitaskGPModel).
            likelihood (gpytorch.likelihood): likelihood function.
        '''
        self.model_type = model_type
        self.likelihood = likelihood
        self.optimizer = None
        self.model = None
        self.input_mask = input_mask
        self.target_mask = target_mask
        self.kernel = kernel
        self.noise_prior = noise_prior

    def _init_model(self,
                    train_inputs,
                    train_targets
                    ):
        '''Init GP model from train inputs and train_targets.'''
        target_dimension = train_targets.shape[1] if train_targets.ndim > 1 else 1
        input_dimension = train_inputs.shape[1] if train_inputs.ndim > 1 else 1

        if self.model is None:
            self.model = self.model_type(train_inputs,
                                         train_targets,
                                         self.likelihood,
                                         kernel=self.kernel)
        # Extract dimensions for external use.
        self.input_dimension = input_dimension
        self.output_dimension = target_dimension
        self.n_training_samples = train_inputs.shape[0]

    def _compute_GP_covariances(self,
                                train_x
                                ):
        '''Compute K(X,X) + sigma*I and its inverse.'''
        # Pre-compute inverse covariance plus noise to speed-up computation.
        K_lazy = self.model.covar_module(train_x.double())
        K_lazy_plus_noise = K_lazy.add_diag(self.model.likelihood.noise)
        n_samples = train_x.shape[0]
        self.model.K_plus_noise = K_lazy_plus_noise.matmul(torch.eye(n_samples).double())
        self.model.K_plus_noise_inv = K_lazy_plus_noise.inv_matmul(torch.eye(n_samples).double())
        # self.model.K_plus_noise_inv_2 = torch.inverse(self.model.K_plus_noise) # Equivalent to above but slower.
        # compute the condition number of the covariance matrix.
        print(colored(f'Condition number of the covariance matrix: {torch.linalg.cond(self.model.K_plus_noise)}', 'green'))

    def init_with_hyperparam(self,
                             train_inputs,
                             train_targets,
                             path_to_statedict
                             ):
        '''Load hyperparameters from a state_dict.'''
        if self.input_mask is not None:
            train_inputs = train_inputs[:, self.input_mask]
        if self.target_mask is not None:
            train_targets = train_targets[:, self.target_mask]
        device = torch.device('cpu')
        print(colored(f'Loading hyperparameters from {path_to_statedict}', 'green'))
        state_dict = torch.load(path_to_statedict, map_location=device)
        self._init_model(train_inputs, train_targets)
        self.model.load_state_dict(state_dict)
        self.model.double()  # needed otherwise loads state_dict as float32
        self._compute_GP_covariances(train_inputs)
        self.casadi_predict = self.make_casadi_prediction_func(train_inputs, train_targets)
        print(colored(f'outputscale: {self.model.covar_module.outputscale}', 'green'))
        print(colored(f'lengthscale: {self.model.covar_module.base_kernel.lengthscale}', 'green'))
        print(colored(f'noise: {self.model.likelihood.noise}', 'green'))

    def train(self,
              train_input_data,
              train_target_data,
              test_input_data,
              test_target_data,
              n_train=500,
              learning_rate=0.01,
              gpu=False,
              fname='best_model.pth',
              ):
        '''Train the GP using Train_x and Train_y.

        Args:
            train_x: Torch tensor (N samples [rows] by input dim [cols])
            train_y: Torch tensor (N samples [rows] by target dim [cols])
        '''
        train_x_raw = train_input_data
        train_y_raw = train_target_data
        test_x_raw = test_input_data
        test_y_raw = test_target_data
        if self.input_mask is not None:
            train_x_raw = train_x_raw[:, self.input_mask]
            test_x_raw = test_x_raw[:, self.input_mask]
        if self.target_mask is not None:
            train_y_raw = train_y_raw[:, self.target_mask]
            test_y_raw = test_y_raw[:, self.target_mask]
        self._init_model(train_x_raw, train_y_raw)
        train_x = train_x_raw
        train_y = train_y_raw
        test_x = test_x_raw
        test_y = test_y_raw
        if gpu:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            test_x = test_x.cuda()
            test_y = test_y.cuda()
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        if self.input_dimension == 1:
            test_x = test_x.reshape(-1)
            train_x = train_x.reshape(-1)

        self.model.double()
        self.likelihood.double()
        self.model.train()
        self.likelihood.train()

        max_trial = 1
        opti_result = []
        loss_result = []
        for trial_idx in range(max_trial):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
            # init_output_scale = torch.rand(1).requires_grad_(False) * 3
            # init_length_scale = torch.rand(self.input_dimension, 1, 1).requires_grad_(False) * 3
            # init_noise = torch.from_numpy(self.noise_prior)
            # if gpu:
            #     init_output_scale = init_output_scale.cuda()
            #     init_length_scale = init_length_scale.cuda()
                # init_noise = init_noise.cuda()

            # self.model.covar_module.initialize(outputscale=init_output_scale)
            # self.model.covar_module.base_kernel.initialize(lengthscale=init_length_scale)
            # self.model.likelihood.initialize(noise=init_noise)
            print('init outputscale: ', self.model.covar_module.outputscale)
            print('init lengthscale: ', self.model.covar_module.base_kernel.lengthscale)
            print('init noise: ', self.model.likelihood.noise)
            last_loss = 99999999
            best_loss = 99999999
            loss = torch.tensor(0)
            i = 0
            while i < n_train and torch.abs(loss - last_loss) > 1e-2:
                with torch.no_grad():
                    self.model.eval()
                    self.likelihood.eval()
                    test_output = self.model(test_x)
                    test_loss = -mll(test_output, test_y)
                self.model.train()
                self.likelihood.train()
                self.optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                if i % 100 == 0:
                    print('Iter %d/%d - MLL train Loss: %.3f, Posterior Test Loss: %0.3f' % (i + 1, n_train, loss.item(), test_loss.item()))

                self.optimizer.step()
                if test_loss < best_loss:
                    best_loss = test_loss
                    state_dict = self.model.state_dict()
                    torch.save(state_dict, fname)
                    best_epoch = i

                i += 1
            print(f'Lowest epoch: {best_epoch}')
            print(f'Lowest Loss: {best_loss}')
            opti_result.append(state_dict)
            loss_result.append(best_loss.cpu().detach().numpy())
        # find and save the best model among the trials.
        best_idx = np.argmin(loss_result)
        torch.save(opti_result[best_idx], fname)
        print(colored('Training Complete', 'green'))
        print(colored(f'Best loss in {max_trial} trials: {loss_result[best_idx]}', 'green'))
        print(colored(f'final outputscale: {self.model.covar_module.outputscale}', 'green'))
        print(colored(f'final lengthscale: {self.model.covar_module.base_kernel.lengthscale}', 'green'))
        print(colored(f'final noise: {self.model.likelihood.noise}', 'green'))
        self.model = self.model.cpu()
        self.likelihood = self.likelihood.cpu()
        train_x = train_x.cpu()
        train_y = train_y.cpu()
        self.model.load_state_dict(torch.load(fname))
        self._compute_GP_covariances(train_x)
        self.casadi_predict = self.make_casadi_prediction_func(train_x, train_y)


    def predict(self,
                x,
                requires_grad=False,
                return_pred=True
                ):
        '''
        Args:
            x : torch.Tensor (N_samples x input DIM).

        Returns:
            Predictions
                mean : torch.tensor (nx X N_samples).
                lower : torch.tensor (nx X N_samples).
                upper : torch.tensor (nx X N_samples).
        '''
        self.model.eval()
        self.likelihood.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).double()
        if self.input_mask is not None:
            x = x[:, self.input_mask]
        if requires_grad:
            predictions = self.likelihood(self.model(x))
            mean = predictions.mean
            cov = predictions.covariance_matrix
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(state=True), gpytorch.settings.fast_pred_samples(state=True):
                predictions = self.likelihood(self.model(x))
                mean = predictions.mean
                cov = predictions.covariance_matrix
        if return_pred:
            return mean, cov, predictions
        else:
            return mean, cov

    def prediction_jacobian(self,
                            query
                            ):
        mean_der, _ = torch.autograd.functional.jacobian(
            lambda x: self.predict(x, requires_grad=True, return_pred=False),
            query.double())
        return mean_der.detach().squeeze()

    def make_casadi_prediction_func(self, train_inputs, train_targets):
        '''Assumes train_inputs and train_targets are already masked.'''
        train_inputs = train_inputs.numpy()
        train_targets = train_targets.numpy()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
        output_scale = self.model.covar_module.outputscale.detach().numpy()
        # Nx = len(self.input_mask)
        Nx = self.input_dimension
        if train_targets.ndim == 1:
            train_targets = train_targets.reshape(-1, 1)
        if train_inputs.ndim == 1:
            train_inputs = train_inputs.reshape(-1, 1)
        z = ca.SX.sym('z', Nx)
        if self.kernel == 'RBF':
            K_z_ztrain = ca.Function('k_z_ztrain',
                                     [z],
                                     [covSEard(z, train_inputs.T, lengthscale.T, output_scale)],
                                     ['z'],
                                     ['K'])
        elif self.kernel == 'RBF_single':
            K_z_ztrain = ca.Function('k_z_ztrain',
                                        [z],
                                        [covSE_single(z, train_inputs.T, lengthscale.T, output_scale)],
                                        ['z'],
                                        ['K'])
        elif self.kernel == 'Matern':
            K_z_ztrain = ca.Function('k_z_ztrain',
                                     [z],
                                     [covMatern52ard(z, train_inputs.T, lengthscale.T, output_scale)],
                                     ['z'],
                                     ['K'])
        predict = ca.Function('pred',
                              [z],
                              [K_z_ztrain(z=z)['K'] @ self.model.K_plus_noise_inv.detach().numpy() @ train_targets],
                              ['z'],
                              ['mean'])
        return predict


    def plot_trained_gp(self,
                        inputs,
                        targets,
                        output_label,
                        output_dir=None,
                        fig_count=0,
                        title=None,
                        **kwargs
                        ):
        '''Plot the trained GP given the input and target data.
        
        Args:
            inputs (torch.Tensor): Input data (N_samples x input_dim).
            targets (torch.Tensor): Target data (N_samples x 1).
            output_label (str): Label for the output. Usually the index of the output.
            output_dir (str): Directory to save the figure.
        '''
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).double()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).double()

        num_data = inputs.shape[0]
        residual_func = kwargs.get('residual_func', None)
        residual = np.zeros((num_data, targets.shape[1]))
        if residual_func is not None:
            for i in range(num_data):
                residual[i, :] = \
                    residual_func(inputs[i, :].numpy())[output_label]
                
        if self.target_mask is not None:
            targets = targets[:, self.target_mask]
        means, _, preds = self.predict(inputs)
        t = np.arange(inputs.shape[0])
        lower, upper = preds.confidence_region()

        fig_count += 1
        plt.figure(fig_count, figsize=(5, 2))
        # compute the percentage of test points within 2 std
        num_within_2std = torch.sum((targets[:, i] > lower) & (targets[:, i] < upper)).numpy()
        percentage_within_2std = num_within_2std / len(targets[:, i]) * 100
        print(f'Percentage of test points within 2 std for dim {output_label}: {percentage_within_2std:.2f}%')
        plt.fill_between(t, lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='2-$\sigma$')
        plt.plot(t, means, 'b', label='GP mean')
        plt.scatter(t, targets, color='r', label='Target')
        plt.plot(t, residual, 'g', label='Residual',)
        plt.legend(ncol=2)
        plt_title = f'GP validation {output_label}, {percentage_within_2std:.2f}% within 2-$\sigma$'
        if title is not None:
            plt_title += f' {title}'
        plt.title(plt_title)

        if output_dir is not None:
            plt_name = f'gp_validation_{output_label}.png' if title is None else f'gp_validation_{output_label}_{title}.png'
            plt.savefig(f'{output_dir}/{plt_name}')
            print(f'Figure saved at {output_dir}/{plt_name}')
        # clean up the plot
        plt.close(fig_count)

        return fig_count


def kmeans_centriods(n_cent, data, rand_state=0):
    '''kmeans clustering. Useful for finding reasonable inducing points.

    Args:
        n_cent (int): Number of centriods.
        data (np.array): Data to find the centroids of n_samples X n_features.

    Return:
        centriods (np.array): Array of centriods (n_cent X n_features).
    '''
    kmeans = KMeans(n_clusters=n_cent, random_state=rand_state).fit(data)
    return kmeans.cluster_centers_
