import math, time, copy
import numpy as np
import torch
from collections import defaultdict
from symbolicregression.metrics import compute_metrics
from sklearn.base import BaseEstimator
import symbolicregression.model.utils_wrapper as utils_wrapper
from sklearn import feature_selection
from sklearn.model_selection import KFold
import traceback

def corr(X, y, epsilon=1e-10):
    """
    X : shape n*d
    y : shape n
    """
    cov = (y @ X)/len(y) - y.mean()*X.mean(axis=0)
    corr = cov / (epsilon + X.std(axis=0) * y.std())
    return corr

def get_top_k_features(X, y, k=10):
    if y.ndim == 2:
        y = y[:, 0]
    if X.shape[1] <= k:
        return [i for i in range(X.shape[1])]
    else:
        kbest = feature_selection.SelectKBest(feature_selection.r_regression, k=k)
        kbest.fit(X, y)
        scores = kbest.scores_
        top_features = np.argsort(-np.abs(scores))
        print("Keeping only the top-{} features. Order was {}".format(k, top_features))
        return list(top_features[:k])

def exchange_node_values(tree, dico):
    new_tree = copy.deepcopy(tree)
    for (old, new) in dico.items():
        new_tree.replace_node_value(old, new)
    return new_tree

class SymbolicTransformerRegressor(BaseEstimator):

    def __init__(self,
                 model=None,
                 max_input_points=10000,
                 max_number_bags=-1,
                 stop_refinement_after=1,
                 n_trees_to_refine=1,
                 rescale=True):
        super().__init__()  # Ensure BaseEstimator initialization
        self.model = model
        self.max_input_points = max_input_points
        self.max_number_bags = max_number_bags
        self.stop_refinement_after = stop_refinement_after
        self.n_trees_to_refine = n_trees_to_refine
        self.rescale = rescale

    def set_args(self, args={}):
        for arg, val in args.items():
            if hasattr(self, arg):
                setattr(self, arg, val)
            else:
                raise ValueError(f"{arg} arg does not exist")
            
    def fit(self, X, Y, verbose=False, n_splits=5):  # Number of cross-validation folds
        self.start_fit = time.time()

        if not isinstance(X, list):
            X = [X]
            Y = [Y]
        n_datasets = len(X)

        self.top_k_features = [None for _ in range(n_datasets)]
        for i in range(n_datasets):
            self.top_k_features[i] = get_top_k_features(X[i], Y[i], k=self.model.env.params.max_input_dimension)
            X[i] = X[i][:, self.top_k_features[i]]

        scaler = utils_wrapper.StandardScaler() if self.rescale else None
        scale_params = {}
        if scaler is not None:
            scaled_X = []
            for i, x in enumerate(X):
                scaled_X.append(scaler.fit_transform(x))
                scale_params[i] = scaler.get_params()
        else:
            scaled_X = X

        kf = KFold(n_splits=n_splits, shuffle=True)
        candidates_per_dataset = defaultdict(list)

        for input_id in range(n_datasets):
            for train_index, val_index in kf.split(scaled_X[input_id]):
                X_train, X_val = scaled_X[input_id][train_index], scaled_X[input_id][val_index]
                y_train, y_val = Y[input_id][train_index], Y[input_id][val_index]

                # Prepare inputs
                inputs = []
                for seq_l in range(len(X_train)):
                    if seq_l % self.max_input_points == 0:
                        inputs.append([])
                    inputs[-1].append([X_train[seq_l], y_train[seq_l]])
                forward_time = time.time()

                # Run model
                outputs = self.model(inputs)
                if verbose: 
                    print("Finished forward in {} secs".format(time.time() - forward_time))

                # Process candidates
                refined_candidates = self.refine(X_train, y_train, outputs, verbose=verbose)
                for candidate in refined_candidates:
                    candidate["validation_score"] = self.evaluate_tree(candidate["predicted_tree"], X_val, y_val, metric="_mse")
                    candidates_per_dataset[input_id].append(candidate)

        self.tree = {}
        for input_id, candidates in candidates_per_dataset.items():
            if candidates:
                self.tree[input_id] = self.order_candidates(X[input_id], Y[input_id], candidates, metric="_mse", verbose=verbose)

        return self  # Ensuring the estimator returns self for scikit-learn compatibility

    @torch.no_grad()
    def evaluate_tree(self, tree, X, y, metric):
        numexpr_fn = self.model.env.simplifier.tree_to_numexpr_fn(tree)
        y_tilde = numexpr_fn(X)[:, 0]
        metrics = compute_metrics({"true": [y], "predicted": [y_tilde], "predicted_tree": [tree]}, metrics=metric)
        return metrics[metric][0]

    def order_candidates(self, X, y, candidates, metric="_mse", verbose=False):
        scores = []
        for candidate in candidates:
            if metric not in candidate:
                score = self.evaluate_tree(candidate["predicted_tree"], X, y, metric)
                if math.isnan(score): 
                    score = np.inf if metric.startswith("_") else -np.inf
            else:
                score = candidates[metric]
            scores.append(score)
        ordered_idx = np.argsort(scores)  
        if not metric.startswith("_"): 
            ordered_idx = list(reversed(ordered_idx))
        candidates = [candidates[i] for i in ordered_idx]
        return candidates

    def refine(self, X, y, candidates, verbose):
        refined_candidates = []

        for i, candidate in enumerate(candidates):
            candidate_skeleton, candidate_constants = self.model.env.generator.function_to_skeleton(candidate, constants_with_idx=True)
            if "CONSTANT" in candidate_constants:
                candidates[i] = self.model.env.wrap_equation_floats(candidate_skeleton, np.random.randn(len(candidate_constants)))

        candidates = [{"refinement_type": "NoRef", "predicted_tree": candidate, "time": time.time()-self.start_fit} for candidate in candidates]
        candidates = self.order_candidates(X, y, candidates, metric="_mse", verbose=verbose)

        # Remove skeleton duplicates
        skeleton_candidates, candidates_to_remove = {}, []
        for i, candidate in enumerate(candidates):
            skeleton_candidate, _ = self.model.env.generator.function_to_skeleton(candidate["predicted_tree"], constants_with_idx=False)
            if skeleton_candidate.infix() in skeleton_candidates:
                candidates_to_remove.append(i)
            else:
                skeleton_candidates[skeleton_candidate.infix()] = 1
        if verbose: 
            print("Removed {}/{} skeleton duplicates".format(len(candidates_to_remove), len(candidates)))

        candidates = [candidates[i] for i in range(len(candidates)) if i not in candidates_to_remove]
        if self.n_trees_to_refine > 0:
            candidates_to_refine = candidates[:self.n_trees_to_refine]
        else:
            candidates_to_refine = copy.deepcopy(candidates)

        for candidate in candidates_to_refine:
            refinement_strategy = utils_wrapper.BFGSRefinement()
            candidate_skeleton, candidate_constants = self.model.env.generator.function_to_skeleton(candidate["predicted_tree"], constants_with_idx=True)
            try:
                refined_candidate = refinement_strategy.go(env=self.model.env, 
                                                           tree=candidate_skeleton, 
                                                           coeffs0=candidate_constants,
                                                           X=X,
                                                           y=y,
                                                           downsample=1024,
                                                           stop_after=self.stop_refinement_after)
            except Exception as e:
                if verbose: 
                    print(e)
                continue
            
            if refined_candidate is not None:
                refined_candidates.append({ 
                        "refinement_type": "BFGS",
                        "predicted_tree": refined_candidate,
                        })    
        candidates.extend(refined_candidates)  
        candidates = self.order_candidates(X, y, candidates, metric="r2")

        for candidate in candidates:
            if "time" not in candidate:
                candidate["time"] = time.time() - self.start_fit
        return candidates

    def __str__(self):
        if hasattr(self, "tree"):
            for tree_idx in range(len(self.tree)):
                for gen in self.tree[tree_idx]:
                    print(gen)
        return "Transformer"

    def exchange_tree_features(self):
        top_k_features = self.top_k_features
        for dataset_id, candidates in self.tree.items():
            exchanges = {}
            for i, feature in enumerate(top_k_features[dataset_id]):
                exchanges["x_{}".format(i)] = "x_{}".format(feature)
            for candidate in candidates:
                candidate["relabed_predicted_tree"] = exchange_node_values(candidate["predicted_tree"], exchanges)

    def retrieve_tree(self, refinement_type=None, dataset_idx=0, all_trees=False, with_infos=False):
        self.exchange_tree_features()
        if dataset_idx == -1: 
            idxs = [_ for _ in range(len(self.tree))] 
        else: 
            idxs = [dataset_idx]
        best_trees = []
        for idx in idxs:
            best_tree = copy.deepcopy(self.tree