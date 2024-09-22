"""Simple script that checks that no test equation are in the training dataset"""
from nesymres.utils import load_eq, load_metadata_hdf5
import click
import pandas as pd
from torch.distributions.uniform import Uniform
from nesymres.dataset.data_utils import create_uniform_support, sample_symbolic_constants, evaluate_fun, return_dict_metadata_dummy_constant
import warnings
from sympy import lambdify,sympify
import multiprocessing
import torch
from tqdm import tqdm
import numpy as np
import os
from multiprocessing import Manager



def evaluate_validation_set(validation_eqs: pd.DataFrame, support) -> set:
    res = set()
    for _, row in validation_eqs.iterrows():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            variables = [f"x_{i}" for i in range(1, 1 + support.shape[0])]
            curr = lambdify(variables, row["eq"])(*support).numpy().astype('float16')
            curr = tuple([x if not np.isnan(x) else "nan" for x in curr])
            res.add(curr)
    return res



class Pipeline:
    """"""
    def __init__(self, data_path, metadata, support, target_image: list, validation_eqs: set):
        """
        Args:
            param1: 
            param2: 
            support: 
            target_image: 
            target_image_l:
            validation_eqs: A set containing all the validation equations in a str format and without constant placeholders. 
                            This argument is used for the symbol checking
        """
        self.data_path = data_path
        self.metadata = metadata
        self.support = support
        self.target_image_l = target_image
        self.target_image = set(target_image)
        self.validation_eqs_l = validation_eqs
        self.validation_eqs = set(validation_eqs)
        self.res = Manager().dict()
        
    def is_valid_and_not_in_validation_set(self, idx: int) -> bool:
    eq = load_eq(self.data_path, idx, self.metadata.eqs_per_hdf)
    dict_costs = return_dict_metadata_dummy_constant(self.metadata)
    consts = torch.stack([torch.ones([int(self.support.shape[1])]) * dict_costs[key] for key in dict_costs.keys()])
    input_lambdi = torch.cat([self.support, consts], axis=0)

    const, dummy_const = sample_symbolic_constants(eq)
    eq_str = sympify(eq.expr.format(**dummy_const))
    if str(eq_str) in self.validation_eqs:
        return idx, False  # This equation is in the validation set

    args = [eq.code, input_lambdi]
    y = evaluate_fun(args)

    curr = [x if not np.isnan(x) else "nan" for x in y]
    val = tuple(curr)

    # Check for valid numerical outputs
    if val in self.target_image:
        return idx, False

    return idx, True  # Equation is valid

@click.command()
@click.option("--data_path", default="data/datasets/10000000/")
@click.option("--csv_path", default="data/benchmark/nc_old.csv")
@click.option("--debug/--no-debug", default=False)
def main(data_path,csv_path,debug):
    metatada = load_metadata_hdf5(data_path)
    validation = pd.read_csv(csv_path)
    sampling_distribution = Uniform(-25,25)
    num_p = 400
    support = create_uniform_support(sampling_distribution, len(metatada.total_variables), num_p)
    print("Creating image for validation set")
    target_image = evaluate_validation_set(validation,support)
    pipe = Pipeline(data_path, metatada, support, target_image,  list(validation["eq"]))
    total_eq = int(metatada.total_number_of_eqs)
    res = []
    if not debug:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            chunksize = 10000
            print("Chunksize of {}. Progress bar will update with this resolution".format(chunksize))
            with tqdm(total=total_eq) as pbar:
                for evaled in p.imap_unordered(pipe.is_valid_and_not_in_validation_set, list(range(total_eq)),chunksize=chunksize):
                    pbar.update()
                    res.append(evaled)
    else:
        res = list(map(pipe.is_valid_and_not_in_validation_set, tqdm(range(total_eq))))
    


if __name__=="__main__":
    main()
