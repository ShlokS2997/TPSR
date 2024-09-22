import os
import numpy as np
import random
import math
from scipy.optimize import minimize
import types
import click
import marshal
import copyreg
import sys
import ast
import pdb
from torch.utils.data import DataLoader, random_split
import torch
from torch import nn
import torch.nn.functional as F
import sympy as sp
from dataclasses import dataclass
from ..dataset.generator import Generator
from . import data
from typing import Tuple
import time
import re
from ..dataset.sympy_utils import add_multiplicative_constants, add_additive_constants

import numexpr as ne

class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value


def bfgs(pred_str, X, y, cfg):
    # Check dimensions not used, and replace with 1 to avoid numerical issues with BFGS
    y = y.squeeze()
    X = X.clone()
    bool_dim = (X == 0).all(axis=1).squeeze()
    X[:, :, bool_dim] = 1

    if type(pred_str) != list:
        pred_str = pred_str[1:].tolist()
    else:
        pred_str = pred_str[1:]

    raw = data.de_tokenize(pred_str, cfg.id2word)

    # Add constants if needed
    if cfg.bfgs.add_coefficients_if_not_existing and 'constant' not in raw:
        print("No constants in predicted expression. Attaching them everywhere")
        variables = {x: sp.Symbol(x, real=True, nonzero=True) for x in cfg.total_variables}
        infix = Generator.prefix_to_infix(raw, coefficients=cfg.total_coefficients, variables=cfg.total_variables)
        s = Generator.infix_to_sympy(infix, variables, cfg.rewrite_functions)
        placeholder = {x: sp.Symbol(x, real=True, nonzero=True) for x in ["cm", "ca"]}
        s = add_multiplicative_constants(s, placeholder["cm"], unary_operators=cfg.una_ops)
        s = add_additive_constants(s, placeholder, unary_operators=cfg.una_ops)
        s = s.subs(placeholder["cm"], 0.43)
        s = s.subs(placeholder["ca"], 0.421)
        s_simplified = data.constants_to_placeholder(s, symbol="constant")
        prefix = Generator.sympy_to_prefix(s_simplified)
        candidate = Generator.prefix_to_infix(prefix, coefficients=["constant"], variables=cfg.total_variables)
    else:
        candidate = Generator.prefix_to_infix(raw, coefficients=["constant"], variables=cfg.total_variables)

    candidate = candidate.format(constant="constant")
    expr = candidate
    for i in range(candidate.count("constant")):
        expr = expr.replace("constant", f"c{i}", 1)

    if cfg.bfgs.idx_remove:
        bool_con = (X < 200).all(axis=2).squeeze()
        X = X[:, bool_con, :]

    max_y = np.max(np.abs(torch.abs(y).cpu().numpy()))
    if max_y > 300:
        print('Attention, input values are very large. Optimization may fail due to numerical issues')

    # Construct differences for loss calculation
    diffs = []
    for i in range(X.shape[1]):
        curr_expr = expr
        for idx, j in enumerate(cfg.total_variables):
            curr_expr = sp.sympify(curr_expr).subs(j, X[:, i, idx])
        diff = curr_expr - y[i]
        diffs.append(diff)

    if cfg.bfgs.normalization_type == "NMSE":
        mean_y = np.mean(y.numpy())
        if abs(mean_y) < 1e-06:
            print("Normalizing by a small value")
        loss = (np.mean(np.square(diffs))) / mean_y
    elif cfg.bfgs.normalization_type == "MSE":
        loss = (np.mean(np.square(diffs)))
    else:
        raise KeyError

    # Initialize lists to store results from multiple restarts
    F_loss = []
    consts_ = []
    funcs = []
    symbols = {i: sp.Symbol(f'c{i}') for i in range(candidate.count("constant"))}

    # BFGS optimization loop with torch.no_grad()
    with torch.no_grad():
        for i in range(cfg.bfgs.n_restarts):
            x0 = np.random.randn(len(symbols))
            s = list(symbols.values())

            # Timed optimization
            fun_timed = TimedFun(fun=sp.lambdify(s, loss, modules=['numpy']), stop_after=5)
            if len(x0):
                minimize(fun_timed.fun, x0, method='BFGS')
                consts_.append(fun_timed.x)
            else:
                consts_.append([])

            # Update the final expression with optimized constants
            final = expr
            for i in range(len(s)):
                final = sp.sympify(final).replace(s[i], fun_timed.x[i])

            # Simplify the final expression (Optional)
            final = sp.simplify(final)

            if cfg.bfgs.normalization_o:
                funcs.append(max_y * final)
            else:
                funcs.append(final)

            # Evaluate the loss
            values = {x: X[:, :, idx].cpu() for idx, x in enumerate(cfg.total_variables)}
            final = str(final)
            y_found = sp.lambdify(",".join(cfg.total_variables), final)(**values)
            final_loss = np.mean(np.square(y_found - y.cpu()).numpy())
            F_loss.append(final_loss)

    try:
        k_best = np.nanargmin(F_loss)
    except ValueError:
        print("All-Nan slice encountered")
        k_best = 0

    return funcs[k_best], consts_[k_best], F_loss[k_best], expr

            
