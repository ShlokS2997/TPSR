import time
import json
import os

import torch
import numpy as np
import sympy as sp
from parsers import get_parser

import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer import Trainer
from symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine, respond_to_batch , pred_for_sample, refine_for_sample, pred_for_sample_test, refine_for_sample_test 
from dyna_gym.agents.uct import UCT
from dyna_gym.agents.mcts import update_root, convert_to_json, print_tree
from rl_env import RLEnv
from default_pi import E2EHeuristic, NesymresHeuristic
from symbolicregression.metrics import compute_metrics
from sklearn.model_selection import KFold



from nesymres.src.nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
from functools import partial
from sympy import lambdify
from reward import compute_reward_e2e, compute_reward_nesymres
import omegaconf


def evaluate_metrics(params, y_gt, tree_gt, y_pred, tree_pred):
    metrics = [] # 7 metrics and for all samples to evaluate
    results_fit = compute_metrics(
        {
            "true": [y_gt],
            "predicted": [y_pred],
            "tree": tree_gt,
            "predicted_tree": tree_pred,
        },
        metrics=params.validation_metrics,
    )
    for k, v in results_fit.items():
        print("metric {}: ".format(k), v)
        metrics.append(v[0])
    
    return metrics

def compute_nmse(y_gt , y_pred):
    eps = 1e-9  # For avoiding Nan or Inf
    return np.sqrt(np.mean((y_gt - y_pred)**2) / (np.mean((y_gt)**2) + eps))  # Fixed closing parenthesis

def compute_mse(y_gt , y_pred):
    return np.mean((y_gt - y_pred)**2)

def main_e2e(case, params, equation_env, samples):
    # Prepare for cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=params.random_state)
    all_metrics = []

    # Iterate through each fold
    for train_index, test_index in kf.split(samples['x_to_fit'][0]):
        x_to_fit = samples['x_to_fit'][0][train_index]
        y_to_fit = samples['y_to_fit'][0][train_index]
        x_to_val = samples['x_to_fit'][0][test_index]
        y_to_val = samples['y_to_fit'][0][test_index]

        # Set up the model and environment
        model = Transformer(params=params, env=equation_env, samples={'x_to_fit': [x_to_fit], 'y_to_fit': [y_to_fit]})
        model.to(params.device)
        
        rl_env = RLEnv(params=params, samples={'x_to_fit': [x_to_fit], 'y_to_fit': [y_to_fit]}, equation_env=equation_env, model=model)
        dp = E2EHeuristic(equation_env=equation_env, rl_env=rl_env, model=model, k=params.width, num_beams=params.num_beams, horizon=params.horizon, device=params.device, use_seq_cache=not params.no_seq_cache, use_prefix_cache=not params.no_prefix_cache, length_penalty=params.beam_length_penalty, train_value_mode=params.train_value, debug=params.debug)

        # Same logic for the agent as before
        start = time.time()
        agent = UCT(action_space=[], gamma=1., ucb_constant=1., horizon=params.horizon, rollouts=params.rollout, dp=dp, width=params.width, reuse_tree=True, alg=params.uct_alg, ucb_base=params.ucb_base)
        
        done = False
        s = rl_env.state
        for t in range(params.horizon):
            if len(s) >= params.horizon:
                break
            if done:
                break
            act = agent.act(rl_env, done)
            s, r, done, _ = rl_env.step(act)

        # Evaluate on validation set
        y_val_pred = pred_for_sample_no_refine(model, equation_env, s, x_to_val)
        val_metrics = evaluate_metrics(params, y_to_val, None, y_val_pred, None)  # Adjust this call based on your metric evaluation method
        all_metrics.append(val_metrics)

    # Return aggregated metrics for all folds
    return np.mean(all_metrics, axis=0)

def main_nesymres(case, params, eq_setting, cfg, samples, X, y):
    # Similar cross-validation logic for nesymres
    kf = KFold(n_splits=5, shuffle=True, random_state=params.random_state)
    all_metrics = []

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        # Load model and parameters
        bfgs = BFGSParams(...)
        params_fit = FitParams(...)

        model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()

        # Fit the model
        fitfunc = partial(model.fitfunc, cfg_params=params_fit)
        output_ref = fitfunc(X_train, y_train)

        # Set up RL environment
        rl_env = RLEnv(params=params, samples=samples, model=model, cfg_params=params_fit)
        model.to_encode(X_train, y_train, params_fit)

        dp = NesymresHeuristic(...)
        
        # Same logic for the agent
        start = time.time()
        agent = UCT(action_space=[], gamma=1., ucb_constant=1., horizon=params.horizon, rollouts=params.rollout, dp=dp, width=params.width, reuse_tree=True)

        done = False
        s = rl_env.state
        for t in range(params.horizon):
            if len(s) >= params.horizon:
                break
            if done:
                break
            act = agent.act(rl_env, done)
            s, r, done, _ = rl_env.step(act)

        # Evaluate on validation set
        loss_bfgs, reward_mcts, pred_str = compute_reward_nesymres(model.X, model.y, s, params_fit)
        all_metrics.append((loss_bfgs, reward_mcts))

    # Return aggregated metrics for all folds
    return np.mean(all_metrics, axis=0)

if __name__ == '__main__':

    case = 1
    parser = get_parser()
    params = parser.parse_args()
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    params.debug = True
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if params.backbone_model == 'e2e':
        equation_env = build_env(params)
        modules = build_modules(equation_env, params)
        if not params.cpu:
            assert torch.cuda.is_available()
        symbolicregression.utils.CUDA = not params.cpu
        trainer = Trainer(modules, equation_env, params)
        
        #Example of Equation-Data:
        # x0 = np.random.uniform(-2,2, 200)
        x0 = np.linspace(-2,2, 200)
        # y = (x0 **2) * np.sin(x0)
        y= (x0**2 ) * np.sin(5*x0) + np.exp(-0.5*x0)
        data = np.concatenate((x0.reshape(-1,1),y.reshape(-1,1)), axis=1)
        samples = {'x_to_fit': 0, 'y_to_fit':0,'x_to_pred':0,'y_to_pred':0}
        samples['x_to_fit'] = [data[:,:1]]
        samples['y_to_fit'] = [data[:,1].reshape(-1,1)]
        samples['x_to_pred'] = [data[:,:1]]
        samples['y_to_pred'] = [data[:,1].reshape(-1,1)]
        
        #Main
        main_e2e(case, params, equation_env, samples) 
   
   
            
    if params.backbone_model == 'nesymres':
        with open('nesymres/jupyter/100M/eq_setting.json', 'r') as json_file:
            eq_setting = json.load(json_file)
        cfg = omegaconf.OmegaConf.load("nesymres/jupyter/100M/config.yaml")
        
        #Example of Equation-Data:
        number_of_points = 500
        n_variables = 2
        max_supp = cfg.dataset_train.fun_support["max"] 
        min_supp = cfg.dataset_train.fun_support["min"]
        X = torch.rand(number_of_points,len(list(eq_setting["total_variables"])))*(max_supp-min_supp)+min_supp
        X[:,n_variables:] = 0
        target_eq = "((x_1+0.76)*sin(0.8*exp(x_2))+(0.5*x_2))" #Use x_1,x_2 and x_3 as independent variables
        # target_eq = "((x_1*sin(x_2)+x_3))" #Use x_1,x_2 and x_3 as independent variables
        X_dict = {x:X[:,idx].cpu() for idx, x in enumerate(eq_setting["total_variables"])} 
        y = lambdify(",".join(eq_setting["total_variables"]), target_eq)(**X_dict)
        samples = {'x_to_fit':0, 'y_to_fit':0}
        samples['x_to_fit'] = [X]
        samples['y_to_fit'] = [y]
        
        #Main
        main_nesymres(case,params,eq_setting,cfg,samples,X,y)
