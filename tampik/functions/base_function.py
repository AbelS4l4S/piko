from ..data.bench_test_functions import data_functions
import numpy as np


class BaseFunction:
    def __init__(self, name:str, data:dict, dimension:int, parameters:dict):
        self.name = name
        self.dimension = dimension
        self.latex_formula = data['latex_formula']
        self.latex_dimension = data['latex_dimension']
        self.latex_input_domain = data['latex_input_domain']
        self.latex_global_minimum = data['latex_global_minimum'],
        self.np_formula = data['np_formula'],
        self.function_params = data['parameters'] #Corregir
        self.continuous = data['continuous']
        self.convex = data['convex']
        self.separable = data['separable']
        self.differentiable = data['differentiable']
        self.multimodal = data['multimodal']
        self.randomized_term = data['randomized_term']
        self.parametric = data['parametric']

    # def __init__(self, d, m=5, beta=15):
    #     self.input_domain = np.array([[-2 * np.pi, 2 * np.pi] for _ in range(d)])
    #     self.m = m
    #     self.beta = beta

    # def get_param(self):
    #     return {"m": self.m, "beta": self.beta}

    # def get_global_minimum(self, d):
    #     X = np.array([0 for i in range(1, d + 1)])
    #     return (X, self(X))

    # def __call__(self, X):
    #     res = np.exp(-np.sum((X / self.beta) ** (2 * self.m)))
    #     res = res - 2 * np.exp(-np.prod(X**2)) * np.prod(np.cos(X) ** 2)
    #     return res


def get_function_data(name:str) -> dict:
    if name not in data_functions.keys():
        raise ValueError(f"Test bench function '{name}' not founded.")
    return data_functions[name]


def check_dimension(bench_function:str, user_dim:int, bench_dim:str):
    if not (isinstance(user_dim, int) and user_dim >= 0):
        raise ValueError(
            f"Test bench function '{bench_function}' dimension" \
                "must be a positive int")
    
    if bench_dim.isdigit() and (int(bench_dim) != user_dim):
        raise ValueError(
            f"Test bench function '{bench_function}' dimension" \
                "must be {bench_dim}.")
    
    if bench_dim.isdigit() and not user_dim:
        raise ValueError(
            'A dimension must be asigned for the test bench function' \
                f"'{bench_function}'")
    
    if not user_dim:
        user_dim = int(bench_dim)
    
    return user_dim


def create_bench_function(name:str, user_dim:int=0, params:dict={}):
    data = get_function_data(name)
    dimension = check_dimension(name, user_dim, data['latex_dimension'][2:])
        
    if len(params)>0:
        if params.keys() != data['parameters'].keys():
            raise ValueError(
                f"'{name}' only accept {data['parameters'].keys()}"\
                    "parameters")
        for parameter in data['parameters'].keys():
            data['parameters'][parameter] = params[parameter]

    return BaseFunction(name, data, dimension, params)