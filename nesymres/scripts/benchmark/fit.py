import hydra
from nesymres import benchmark
from pathlib import Path
from functools import partial
import pandas as pd
import time
import os
import json



def get_nesymres(cfg):
    from nesymres.architectures import model  
    from nesymres.utils import load_metadata_hdf5
    from nesymres.dclasses import FitParams,BFGSParams

    model = model.Model.load_from_checkpoint(hydra.utils.to_absolute_path(cfg.model.checkpoint_path), cfg=cfg.model.architecture)
    model.eval()
    model.cuda()
    test_data = load_metadata_hdf5(hydra.utils.to_absolute_path(cfg.test_path))
    bfgs = BFGSParams(
        activated= cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )
    params_fit = FitParams(word2id=test_data.word2id, 
                            id2word=test_data.id2word, 
                            una_ops=test_data.una_ops, 
                            bin_ops=test_data.bin_ops, 
                            total_variables=list(test_data.total_variables),  
                            total_coefficients=list(test_data.total_coefficients),
                            rewrite_functions=list(test_data.rewrite_functions),
                            bfgs=bfgs,
                            beam_size=cfg.inference.beam_size
                            )
    fitfunc = partial(model.fitfunc,cfg_params=params_fit)
    model.fit = fitfunc
    return model

def get_model(cfg):
    if cfg.model.model_name == 'brenden':
        """Brenden is not compatible with latest python version"""
        from models.brenden import get_brenden  
        return get_brenden(args.brenden_n_epochs)
    elif cfg.model.model_name == 'nesymres':
        return get_nesymres(cfg)
    elif cfg.model.model_name == 'genetic_prog':
        from models.genetic_prog import get_genetic_prog
        return get_genetic_prog(args.genetic_prog_population_size)
    elif cfg.model.model_name == 'gaussian_proc':
        from models.gaussian_proc import get_gaussian_proc
        return get_gaussian_proc(args.gaussian_proc_n_restarts)
    else:
        raise ValueError(f'Unknown model_name: {args.model_name}')


def evaluate_equation(pred_equation, benchmark_path, equation_idx,
                      num_test_points, pointwise_acc_rtol,
                      pointwise_acc_atol):
    """Evaluate equation `pred_equation` given as a string."""

    def model_predict(X):
        pred_variables = get_variables(pred_equation)
        # assert len(pred_variables) <= X.shape[1]
        return evaluate_func(pred_equation, pred_variables, X)

    return evaluate_model(model_predict, benchmark_path, equation_idx,
                          num_test_points,
                          pointwise_acc_rtol, pointwise_acc_atol)


def evaluate_sklearn(model_path, benchmark_path, equation_idx,
                      num_test_points, pointwise_acc_rtol,
                      pointwise_acc_atol):
    """Evaluate sklearn model at model_path."""

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    def model_predict(X):
        return model.predict(X)

    return evaluate_model(model_predict, benchmark_path, equation_idx,
                          num_test_points,
                          pointwise_acc_rtol, pointwise_acc_atol)


from sklearn.model_selection import KFold

@hydra.main(config_name="fit")
def main(cfg):
    # Load the benchmark equation
    eq = benchmark.load_equation(hydra.utils.to_absolute_path(cfg.benchmark_path), cfg.equation_idx)
    
    # Try to load the data for the benchmark
    try:
        X, y = benchmark.get_robust_data(eq, mode="iid", cfg=cfg)
    except ValueError:
        return None
    
    # Initialize KFold cross-validation
    kf = KFold(n_splits=cfg.num_folds)  # Number of folds specified in the config
    fold_results = []
    
    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Processing fold {fold_idx + 1}/{cfg.num_folds}")
        
        # Split the data into training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Get the model
        model = get_model(cfg)
        
        # Measure the time taken for training
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        duration = time.perf_counter() - start_time
        
        # Predict and evaluate the model on the validation set
        if hasattr(model, 'get_equation'):
            equation = model.get_equation()
            model_path = None
        else:
            equation = None
            model_path = str(Path(f"model_fold_{fold_idx}.pkl"))
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        y_pred = model.fitfunc(X_val)
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        
        # Store results for the current fold
        fold_result = {
            'fold': fold_idx + 1,
            'duration': duration,
            'equation': equation,
            'model_path': model_path,
            'r2': r2,
            'mse': mse
        }
        
        fold_results.append(fold_result)
    
    # Save fold-specific results in a JSON file
    output_data = {
        'fold_results': fold_results,
        'benchmark_path': cfg.benchmark_path,
        'equation_idx': cfg.equation_idx
    }
    
    with open('cross_validation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return fold_results

if __name__ == "__main__":
    main()
