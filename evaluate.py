import numpy as np
import pandas as pd
import os
from collections import OrderedDict, defaultdict
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor , get_top_k_features
from symbolicregression.model.model_wrapper import ModelWrapper
import symbolicregression.model.utils_wrapper as utils_wrapper

from symbolicregression.metrics import compute_metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tpsr import tpsr_fit
import time
from tqdm import tqdm
import copy
import gzip
import shutil
import datasets.pmlb.pmlb.pmlb 

def read_file(filename, label="target", sep=None): 
    print("FILES:", filename)
    input_data = datasets.pmlb.pmlb.pmlb.fetch_data(dataset_name=filename)
    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)
    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values
    assert X.shape[1] == feature_names.shape[0]

    return X, y, feature_names


def evaluate_pmlb(
        trainer,
        params,
        model,
        target_noise=0.0,
        random_state=29910,
        verbose=False,
        save=True,
        filter_fn=None,
        logger=None,
        save_file=None,
        save_suffix="./eval_result/eval_pmlb_feynman_pretrained.csv",
        k_folds=5  # Add this parameter for the number of folds
    ):
    scores = defaultdict(list)
    env = trainer.env
    embedder = model.embedder
    encoder = model.encoder
    decoder = model.decoder
    embedder.eval()
    encoder.eval()
    decoder.eval()

    mw = ModelWrapper(
        env=env,
        embedder=embedder,
        encoder=encoder,
        decoder=decoder,
        beam_length_penalty=params.beam_length_penalty,
        beam_size=params.beam_size,
        max_generated_output_len=params.max_generated_output_len,
        beam_early_stopping=params.beam_early_stopping,
        beam_temperature=params.beam_temperature,
        beam_type=params.beam_type,
    )

    dstr = SymbolicTransformerRegressor(
        model=mw,
        max_input_points=params.max_input_points,
        n_trees_to_refine=params.n_trees_to_refine,
        max_number_bags=params.max_number_bags,
        rescale=params.rescale,
    )

    all_datasets = pd.read_csv(
        "./datasets/pmlb/pmlb/all_summary_stats.tsv",
        sep="\t",
    )
    regression_datasets = all_datasets[all_datasets["task"] == "regression"]
    regression_datasets = regression_datasets[
        regression_datasets["n_categorical_features"] == 0
    ]
    problems = regression_datasets

    if filter_fn is not None:
        problems = problems[filter_fn(problems)]
    problem_names = problems["dataset"].values.tolist()

    rng = np.random.RandomState(random_state)
    pbar = tqdm(total=len(problem_names))

    for problem_name in problem_names:
        # Fetch dataset
        X, y, _ = read_file(f"{problem_name}/{problem_name}.tsv.gz")
        y = np.expand_dims(y, -1)

        # Initialize KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        for train_index, val_index in kf.split(X):
            x_to_fit, x_to_predict = X[train_index], X[val_index]
            y_to_fit, y_to_predict = y[train_index], y[val_index]

            scale = target_noise * np.sqrt(np.mean(np.square(y_to_fit)))
            noise = rng.normal(loc=0.0, scale=scale, size=y_to_fit.shape)
            y_to_fit += noise

            # Fit the model
            dstr.fit(x_to_fit, y_to_fit, verbose=verbose)
            problem_results = defaultdict(list)

            for refinement_type in dstr.retrieve_refinements_types():
                best_gen = copy.deepcopy(
                    dstr.retrieve_tree(refinement_type=refinement_type, with_infos=True)
                )
                predicted_tree = best_gen["predicted_tree"]
                if predicted_tree is None:
                    continue
                del best_gen["predicted_tree"]
                if "metrics" in best_gen:
                    del best_gen["metrics"]

                problem_results["predicted_tree"].append(predicted_tree)
                y_tilde_to_fit = dstr.predict(x_to_fit, refinement_type=refinement_type)
                results_fit = compute_metrics(
                    {
                        "true": [y_to_fit],
                        "predicted": [y_tilde_to_fit],
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    problem_results[k + "_fit"].extend(v)
                    scores[refinement_type + "|" + k + "_fit"].extend(v)

                y_tilde_to_predict = dstr.predict(
                    x_to_predict, refinement_type=refinement_type
                )
                results_predict = compute_metrics(
                    {
                        "true": [y_to_predict],
                        "predicted": [y_tilde_to_predict],
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_predict.items():
                    problem_results[k + "_predict"].extend(v)
                    scores[refinement_type + "|" + k + "_predict"].extend(v)

        # Save results after each problem
        # (Adjust saving logic as needed)

        pbar.update(1)
    
    for k, v in scores.items():
        scores[k] = np.nanmean(v)
    return scores


def evaluate_mcts_in_domain(
        trainer,
        params,
        model,
        data_type,
        task,
        verbose=True,
        ablation_to_keep=None,
        save=False,
        logger=None,
        save_file=None,
        rescale = False,
        save_suffix=None
    ):

        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        scores = OrderedDict({"epoch": trainer.epoch})

        params = params
        embedder =model.embedder
        encoder = model.encoder
        decoder = model.decoder
        embedder.eval()
        encoder.eval()
        decoder.eval()

        env = trainer.env

        eval_size_per_gpu = params.eval_size #old
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=eval_size_per_gpu,
            input_length_modulo=params.eval_input_length_modulo,
            test_env_seed=params.test_env_seed,
        )
        

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            rescale=False,
        )

        first_write = True
        if save:
            save_file = save_suffix

        batch_before_writing_threshold = min(
            2, eval_size_per_gpu // params.batch_size_eval
        )
        batch_before_writing = batch_before_writing_threshold

        if ablation_to_keep is not None:
            ablation_to_keep = list(
                map(lambda x: "info_" + x, ablation_to_keep.split(","))
            )
        else:
            ablation_to_keep = []

        pbar = tqdm(total=eval_size_per_gpu)

        batch_results = defaultdict(list)

        for samples, _ in iterator:
            x_to_fit = samples["x_to_fit"]
            y_to_fit = samples["y_to_fit"]
            infos = samples["infos"]
            tree = samples["tree"]
                        
            #### Scale X 
            X = x_to_fit
            Y = y_to_fit
            if not isinstance(X, list):
                X = [X]
                Y = [Y]
            n_datasets = len(X)
            dstr.top_k_features = [None for _ in range(n_datasets)]
            for i in range(n_datasets):
                dstr.top_k_features[i] = get_top_k_features(X[i], Y[i], k=dstr.model.env.params.max_input_dimension)
                X[i] = X[i][:, dstr.top_k_features[i]]

            scaler = utils_wrapper.StandardScaler() if rescale else None
            scale_params = {}
            if scaler is not None:
                scaled_X = []
                for i, x in enumerate(X):
                    scaled_X.append(scaler.fit_transform(x))
                    scale_params[i]=scaler.get_params()
            else:
                scaled_X = X

            s, time_elapsed, sample_times = tpsr_fit(scaled_X, Y, params,env)
            print("time elapsed for sample: ", time_elapsed)
            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
            generated_tree = list(filter(lambda x: x is not None,
                        [env.idx_to_infix(s[1:], is_float=False, str_array=False)]))
            if generated_tree == []: 
                y = None
                model_str= None
                tree = None
            else:
                dstr.start_fit = time.time()
                dstr.tree = {}             
                refined_candidate = dstr.refine(scaled_X[0], Y[0], generated_tree, verbose=False)
                dstr.tree[0] = refined_candidate

            for k, v in infos.items():
                infos[k] = v.tolist()
                
            for refinement_type in dstr.retrieve_refinements_types():

                best_gen = copy.deepcopy(
                        dstr.retrieve_tree(refinement_type=refinement_type, with_infos=True)
                    )
                predicted_tree = best_gen["predicted_tree"]
                if predicted_tree is None:
                    continue
                del best_gen["predicted_tree"]
                if "metrics" in best_gen:
                    del best_gen["metrics"]

                batch_results["predicted_tree"].append(predicted_tree)
                batch_results["predicted_tree_prefix"].append(
                    predicted_tree.prefix() if predicted_tree is not None else None
                )
                for info, val in best_gen.items():
                    batch_results[info].append(val)
                    
                for k, v in infos.items():
                    batch_results["info_" + k].extend(v)
                        
                y_tilde_to_fit = dstr.predict(x_to_fit, refinement_type=refinement_type)
                
                results_fit = compute_metrics(
                    {
                        "true": y_to_fit,
                        "predicted": [y_tilde_to_fit],
                        "tree": tree,
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    batch_results[k + "_fit"].extend(v)
                del results_fit

                if params.prediction_sigmas is None:
                    prediction_sigmas = []
                else:
                    prediction_sigmas = [
                        float(sigma)
                        for sigma in params.prediction_sigmas.split(",")
                    ]

                for sigma in prediction_sigmas:
                    x_to_predict = samples["x_to_predict_{}".format(sigma)]
                    y_to_predict = samples["y_to_predict_{}".format(sigma)]
                    y_tilde_to_predict = dstr.predict(
                        x_to_predict, refinement_type=refinement_type
                    )
                    results_predict = compute_metrics(
                        {
                            "true": y_to_predict,
                            "predicted": [y_tilde_to_predict],
                            "tree": tree,
                            "predicted_tree": [predicted_tree],
                        },
                        metrics=params.validation_metrics,
                    )
                    for k, v in results_predict.items():
                        batch_results[k + "_predict_{}".format(sigma)].extend(v)
                    del results_predict

                batch_results["tree"].extend(tree)
                batch_results["tree_prefix"].extend([_tree.prefix() for _tree in tree])
                batch_results["time_mcts"].extend([time_elapsed])
                batch_results["sample_times"].extend([sample_times])
                
            if save:

                batch_before_writing -= 1
                if batch_before_writing <= 0:
                    batch_results = pd.DataFrame.from_dict(batch_results)
                    if first_write:
                        batch_results.to_csv(save_file, index=False)
                        first_write = False
                    else:
                        batch_results.to_csv(
                            save_file, mode="a", header=False, index=False
                        )
                    batch_before_writing = batch_before_writing_threshold
                    batch_results = defaultdict(list)

            bs = len(x_to_fit)
            pbar.update(bs)

        try:
            df = pd.read_csv(save_file, na_filter=True)
        except:
            return
        info_columns = filter(lambda x: x.startswith("info_"), df.columns)
        df = df.drop(columns=filter(lambda x: x not in ablation_to_keep, info_columns))

        for refinement_type, df_refinement_type in df.groupby("refinement_type"):
            avg_scores = df_refinement_type.mean().to_dict()
            for k, v in avg_scores.items():
                scores[refinement_type + "|" + k] = v
            for ablation in ablation_to_keep:
                for val, df_ablation in df_refinement_type.groupby(ablation):
                    avg_scores_ablation = df_ablation.mean()
                    for k, v in avg_scores_ablation.items():
                        scores[
                            refinement_type + "|" + k + "_{}_{}".format(ablation, val)
                        ] = v
        return scores
