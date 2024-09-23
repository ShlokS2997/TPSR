import numpy as np
import re
import torch
import sys
import sympy as sp
import os
from symbolicregression.metrics import compute_metrics
from nesymres.src.nesymres.architectures import bfgs
from sklearn.model_selection import KFold


def evaluate_metrics(y_gt, tree_gt, y_pred):
    metrics = []
    results_fit = compute_metrics(
        {
            "true": [y_gt],
            "predicted": [y_pred],
            "tree": tree_gt,
            "predicted_tree": tree_gt,
        },
        metrics='accuracy_l1',
    )
    for k, v in results_fit.items():
        metrics.append(v[0])
    
    return metrics

def compute_reward_e2e(params, samples, y_pred, model_str, generations_tree, n_splits=5):  
    # NMSE
    penalty = -2
    if y_pred is None:
        reward = penalty
    else:
        y = samples['y_to_fit'][0].reshape(-1)
        eps = 1e-9
        
        # Cross-Validation Logic
        kf = KFold(n_splits=n_splits)
        NMSEs = []
        
        for train_index, test_index in kf.split(y):
            y_train, y_test = y[train_index], y[test_index]
            y_pred_fold = y_pred[test_index]  # Get predictions for the test fold
            NMSE = np.sqrt(np.mean((y_test - y_pred_fold) ** 2) / (np.mean(y_test ** 2) + eps))
            NMSEs.append(NMSE)

        avg_NMSE = np.mean(NMSEs)

        if not np.isnan(avg_NMSE):
            reward = 1 / (1 + avg_NMSE)
        else:
            reward = penalty

        if generations_tree:
            complexity = len(generations_tree[0].prefix().split(","))
            lam = params.lam
            reward += lam * np.exp(-complexity / 200)

    return reward


def compute_reward_nesymres(X, y, state, cfg_params, n_splits=5):  
    penalty = -2

    cfg_params.id2word[3] = "constant"
    
    # Cross-Validation Logic
    kf = KFold(n_splits=n_splits)
    rewards = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        try:
            pred_w_c, constants, loss_bfgs, exa = bfgs.bfgs(
                state, X_train, y_train, cfg_params
            )
            if np.isnan(loss_bfgs):
                rewards.append(penalty)
            else:
                lam = 0.1
                eps = 1e-9
                nmse = loss_bfgs / (np.mean(y_test.reshape(-1) ** 2) + eps)
                reward = 1 / (1 + nmse) + lam * np.exp(-(len(state) - 2) / 200)
                rewards.append(reward)
        except:
            rewards.append(penalty)

    return np.mean(rewards), str(pred_w_c)
