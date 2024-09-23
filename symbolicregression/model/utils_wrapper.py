# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from abc import ABC, abstractmethod
import sklearn
from scipy.optimize import minimize
import numpy as np
import time
import torch
from torch.func import grad
from functools import partial
import traceback
from timeout import timeout 
from sklearn.model_selection import KFold  # Import KFold


class TimedFun:
    def __init__(self, fun, verbose=False, stop_after=3):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after
        self.best_fun_value = np.infty
        self.best_x = None
        self.loss_history=[]
        self.verbose = verbose

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            self.loss_history.append(self.best_fun_value)
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(x, *args)
        self.loss_history.append(self.fun_value)
        if self.best_x is None:
            self.best_x = x
        elif self.fun_value < self.best_fun_value:
            self.best_fun_value = self.fun_value
            self.best_x = x
        self.x = x
        return self.fun_value

class Scaler(ABC):
    """
    Base class for scalers
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @timeout(10)
    def rescale_function(self, env, tree, a, b):
        prefix = tree.prefix().split(",")
        idx = 0
        while idx < len(prefix):
            if prefix[idx].startswith("x_"):
                k = int(prefix[idx][-1])
                if k >= len(a): 
                    continue
                a_k, b_k = str(a[k]), str(b[k])
                prefix_to_add = ["add", b_k, "mul", a_k, prefix[idx]]
                prefix = prefix[:idx] + prefix_to_add + prefix[min(idx + 1, len(prefix)):]
                idx += len(prefix_to_add)
            else:
                idx += 1
                continue
        rescaled_tree = env.word_to_infix(prefix, is_float=False, str_array=False)
        return rescaled_tree

class StandardScaler(Scaler):
    def __init__(self):
        """
        transformation is: 
        x' =  (x - mean)/std
        """
        self.scaler = sklearn.preprocessing.StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_X = self.scaler.fit_transform(X)
        return scaled_X
    
    def transform(self, X):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        return (X - m) / s

    def get_params(self):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        a, b = 1 / s, -m / s
        return (a, b)
    
class MinMaxScaler(Scaler):
    def __init__(self):
        """
        transformation is: 
        x' =  2.*(x-xmin)/(xmax-xmin)-1.
        """
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

    def fit(self, X):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_X = self.scaler.fit_transform(X)
        return scaled_X

    def transform(self, X):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        return 2 * (X - val_min) / (val_max - val_min) - 1.

    def get_params(self):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        a, b = 2. / (val_max - val_min), -1. - 2. * val_min / (val_max - val_min)
        return (a, b)

class BFGSRefinement():
    """
    Wrapper around scipy's BFGS solver
    """

    def __init__(self):
        """
        Args:
            func: a PyTorch function that maps dependent variables and
                    parameters to function outputs for all data samples
                    `func(x, coeffs) -> y`
            x, y: problem data as PyTorch tensors. Shape of x is (d, n) and
                    shape of y is (n,)
        """
        super().__init__()        

    def go(
        self, env, tree, coeffs0, X, y, downsample=-1, stop_after=10
    ):
        
        func = env.simplifier.tree_to_torch_module(tree, dtype=torch.float64)
        self.X, self.y = X, y
        if downsample > 0:
            self.X = self.X[:downsample]
            self.y = self.y[:downsample]
        self.X = torch.tensor(self.X, dtype=torch.float64, requires_grad=False)
        self.y = torch.tensor(self.y, dtype=torch.float64, requires_grad=False)
        self.func = partial(func, self.X)

        def objective_torch(coeffs):
            """
            Compute the non-linear least-squares objective value
                objective(coeffs) = (1/2) sum((y - func(coeffs)) ** 2)
            Returns a PyTorch tensor.
            """
            if not isinstance(coeffs, torch.Tensor):
                coeffs = torch.tensor(coeffs, dtype=torch.float64, requires_grad=True)
            y_tilde = self.func(coeffs)
            if y_tilde is None: return None
            mse = (self.y - y_tilde).pow(2).mean().div(2)
            return mse

        def objective_numpy(coeffs):
            """
            Return the objective value as a float (for scipy).
            """
            return objective_torch(coeffs).item()

        def gradient_numpy(coeffs):
            """
            Compute the gradient of the objective at coeffs.
            Returns a numpy array (for scipy)
            """
            if not isinstance(coeffs, torch.Tensor):
                coeffs = torch.tensor(coeffs, dtype=torch.float64, requires_grad=True)
            grad_obj = grad(objective_torch)(coeffs)
            return grad_obj.detach().numpy()
    
        objective_numpy_timed = TimedFun(objective_numpy, stop_after=stop_after)

        try:
            minimize(
                objective_numpy_timed.fun,
                coeffs0,
                method="BFGS",
                jac=gradient_numpy,
                options={"disp": False}
            )
        except ValueError as e:
            traceback.format_exc()
        best_constants = objective_numpy_timed.best_x
        return env.wrap_equation_floats(tree, best_constants)

class SymbolicTransformerRegressor(BaseEstimator):
    def __init__(self,
                model=None,
                max_input_points=10000,
                max_number_bags=-1,
                stop_refinement_after=1,
                n_trees_to_refine=1,
                rescale=True
                ):
        self.max_input_points = max_input_points
        self.max_number_bags = max_number_bags
        self.model = model
        self.stop_refinement_after = stop_refinement_after
        self.n_trees_to_refine = n_trees_to_refine
        self.rescale = rescale

    def set_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self, arg), "{} arg does not exist".format(arg)
            setattr(self, arg, val)

    def fit(
        self,
        X,
        Y,
        verbose=False,
        n_splits=5  # Added parameter for number of splits
    ):
        self.start_fit = time.time()

        if not isinstance(X, list):
            X = [X]
            Y = [Y]
        n_datasets = len(X)

        # Initialize KFold
        kf = KFold(n_splits=n_splits)

        self.top_k_features = [None for _ in range(n_datasets)]
        for i in range(n_datasets):
            self.top_k_features[i] = get_top_k_features(X[i], Y[i], k=self.model.env.params.max_input_dimension)
            X[i] = X[i][:, self.top_k_features[i]]

        # Prepare to store metrics
        metrics = []

        # Cross-validation loop
        for train_index, test_index in kf.split(X[0]):
            X_train, X_test = X[0][train_index], X[0][test_index]
            Y_train, Y_test = Y[0][train_index], Y[0][test_index]

            scaler = utils_wrapper.StandardScaler() if self.rescale else None
            scale_params = {}
            if scaler is not None:
                scaled_X = scaler.fit_transform(X_train)
                scale_params = scaler.get_params()
            else:
                scaled_X = X_train

            inputs = []
            for seq_l in range(len(scaled_X)):
                y_seq = Y_train[seq_l]
                if len(y_seq.shape) == 1:
                    y_seq = np.expand_dims(y_seq, -1)
                if seq_l % self.max_input_points == 0:
                    inputs.append([])
                inputs[-1].append([scaled_X[seq_l], y_seq])

            # Call the model fitting and store performance metrics
            outputs = self.model(inputs)

            # Evaluate the model on X_test and collect metrics
            # Implement your evaluation method to capture metrics here
            metric_value = self.evaluate_model(X_test, Y_test, outputs)  # Placeholder for your evaluation method
            metrics.append(metric_value)

        # Log average metrics across folds
        avg_metric = np.mean(metrics)
        print(f"Average metric across folds: {avg_metric}")

        # Store the final model tree
        self.tree = self.refine_trees(X, Y)

    def evaluate_model(self, X_test, Y_test, outputs):
        # Implement this function to evaluate your model
        # This is a placeholder function
        return np.random.rand()  # Replace with actual evaluation logic

    def refine_trees(self, X, Y):
        # Implement logic to refine trees
        return {}  # Replace with actual tree refinement logic
