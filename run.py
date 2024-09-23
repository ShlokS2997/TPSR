import os
import torch
import numpy as np
import symbolicregression
import requests

from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from parsers import get_parser
from symbolicregression.trainer import Trainer
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from evaluate import evaluate_pmlb, evaluate_pmlb_mcts, evaluate_in_domain, evaluate_mcts_in_domain
from sklearn.model_selection import KFold



# Number of cross-validation folds
n_splits = 5

if __name__ == '__main__':
    # Load model and setup parameters...
    # (existing loading code)

    # Evaluate functions
    if params.eval_in_domain:
        scores_list = []
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(trainer.data):  # Assume `trainer.data` is your dataset
            params.eval_only = True  # Ensure we only evaluate
            # Create a temporary trainer or modify trainer to only use the test index for this fold
            # (This may require additional implementation based on your Trainer's design)
            scores = evaluate_in_domain(
                trainer,
                params,
                model,
                "valid1",
                "functions",
                verbose=True,
                ablation_to_keep=None,
                save=True,
                logger=None,
                save_file=params.save_eval_dic,
            )
            scores_list.append(scores)
        
        # Average scores over all folds
        avg_scores = {key: np.mean([score[key] for score in scores_list]) for key in scores_list[0].keys()}
        print("Average In-Domain Scores: ", avg_scores)

    if params.eval_mcts_in_domain:
        scores_list = []
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(trainer.data):
            scores = evaluate_mcts_in_domain(
                trainer,
                params,
                model,
                "valid1",
                "functions",
                verbose=True,
                ablation_to_keep=None,
                save=True,
                logger=None,
                save_file=None,
                save_suffix="./eval_result/eval_in-domain_tpsr_l{}_b{}_k{}_r{}_cache{}_noise{}.csv".format(params.lam,
                                                                                                        params.num_beams,
                                                                                                        params.width,
                                                                                                        params.rollout,
                                                                                                        params.cache,
                                                                                                        params.eval_noise_gamma), 
            )
            scores_list.append(scores)

        avg_scores = {key: np.mean([score[key] for score in scores_list]) for key in scores_list[0].keys()}
        print("Average MCTS In-Domain Scores: ", avg_scores)

    # Similar logic can be applied for the PMLB evaluations.
    
    if params.eval_on_pmlb:
        target_noise = params.target_noise
        random_state = params.random_state
        data_type = params.pmlb_data_type
        save = params.save_results

        if data_type == "feynman":
            filter_fn = lambda x: x["dataset"].str.contains("feynman")
        elif data_type == "strogatz":
            print("Strogatz data")
            filter_fn = lambda x: x["dataset"].str.contains("strogatz")
        elif data_type == "603_fri_c0_250_50":
            filter_fn = lambda x: x["dataset"].str.contains("603_fri_c0_250_50")
        else:
            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman")
            )

        pmlb_scores = evaluate_pmlb(
            trainer,
            params,
            model,
            target_noise=target_noise,
            verbose=params.eval_verbose_print,
            random_state=random_state,
            save=save,
            filter_fn=filter_fn,
            save_file=None,
            save_suffix="./eval_result/eval_pmlb_pretrained_{}.csv".format(params.beam_type),
        )
        print("Pre-trained E2E scores: ", pmlb_scores)
        
    if params.eval_mcts_on_pmlb:
        target_noise = params.target_noise
        random_state = params.random_state
        data_type = params.pmlb_data_type
        save = params.save_results

        if data_type == "feynman":
            filter_fn = lambda x: x["dataset"].str.contains("feynman")
        elif data_type == "strogatz":
            print("Strogatz data")
            filter_fn = lambda x: x["dataset"].str.contains("strogatz")
        elif data_type == "603_fri_c0_250_50":
            filter_fn = lambda x: x["dataset"].str.contains("603_fri_c0_250_50")
        else:
            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman")
            )

        pmlb_scores = evaluate_pmlb_mcts(
            trainer,
            params,
            model,
            target_noise=target_noise,
            verbose=params.eval_verbose_print,
            random_state=random_state,
            save=save,
            filter_fn=filter_fn,
            save_file=None,
            save_suffix="./eval_result/eval_{}_tpsr_l{}_b{}_k{}_r{}_cache{}_noise{}.csv".format(params.pmlb_data_type,
                                                                                                    params.lam,
                                                                                                    params.num_beams,
                                                                                                    params.width,
                                                                                                    params.rollout,
                                                                                                    params.cache,
                                                                                                    params.target_noise),
        )
        print("TPSR scores: ", pmlb_scores)
