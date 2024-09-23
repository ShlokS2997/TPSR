# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
from sklearn.model_selection import KFold
import numpy as np
import scipy

def compute_metrics(infos, metrics="r2"):
    results = defaultdict(list)
    if metrics == "":
        return {}

    # KFold cross-validation
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(x):
        true, predicted = y[train_index], y[test_index]

        if "true" in infos:
            true, predicted = infos["true"], infos["predicted"]
            assert len(true) == len(predicted), "issue with len, true: {}, predicted: {}".format(len(true), len(predicted))
            for i in range(len(true)):
                if predicted[i] is None: continue
                if len(true[i].shape) == 2:
                    true[i] = true[i][:, 0]
                if len(predicted[i].shape) == 2:
                    predicted[i] = predicted[i][:, 0]
                assert true[i].shape == predicted[i].shape, "Problem with shapes: {}, {}".format(true[i].shape, predicted[i].shape)

        for metric in metrics.split(","):
            if metric == "r2":
                for i in range(len(true)):
                    if predicted[i] is None or np.isnan(np.min(predicted[i])):
                        results[metric].append(np.nan)
                    else:
                        try:
                            results[metric].append(r2_score(true[i], predicted[i]))
                        except Exception as e:
                            results[metric].append(np.nan)
            elif metric == "_mse":
                for i in range(len(true)):
                    if predicted[i] is None or np.isnan(np.min(predicted[i])):
                        results[metric].append(np.nan)
                    else:
                        try:
                            results[metric].append(mean_squared_error(true[i], predicted[i]))
                        except Exception as e:
                            results[metric].append(np.nan)
            # Add similar logic for other metrics you want to compute
    # Average the results for each metric
    for metric in results.keys():
        results[metric] = np.nanmean(results[metric])  # Use np.nanmean to ignore NaNs

    return results
