from ..data.bench_test_functions import data_functions
import numpy as np


class BaseFunction:
    def __init__(self, name:str, dimension:int, data:dict):
        self.name = name
        self.dimension = dimension
        self.latex_formula = data['latex_formula']
        self.latex_dimension = data['latex_dimension']
        self.latex_input_domain = data['latex_input_domain']
        self.latex_global_minimum = data['latex_global_minimum']
        self.function = data['np_formula']
        self.parameters = data['parameters']
        self.continuous = data['continuous']
        self.convex = data['convex']
        self.separable = data['separable']
        self.differentiable = data['differentiable']
        self.multimodal = data['multimodal']
        self.randomized_term = data['randomized_term']
        self.parametric = data['parametric']
        
        if data['parameters'] is not None:
            for key, value in data['parameters'].items():
                setattr(self, key, value)

    def __call__(self, X):
        X = np.atleast_2d(X)
        
        if X.shape[1] != self.dimension:
            raise ValueError(f"Input matrix dimension must be: {self.dimension}")
        
        parameters = self.function.__code__.co_varnames
        
        args = [self.parameters[param]
                for param in parameters 
                if param in self.parameters
        ]
        
        return self.function(*args, X)


def get_function_data(name:str) -> dict:
    if name not in data_functions.keys():
        raise ValueError(f"Test bench function '{name}' not founded.")
    return data_functions[name]


def check_dimension(bench_function:str, user_dim:int, bench_dim:str):
    if not (isinstance(user_dim, int) and user_dim >= 0):
        raise ValueError(
            f"Test bench function '{bench_function}' dimension" \
                " must be a positive int")
    
    if bench_dim and (bench_dim != user_dim):
        raise ValueError(
            f"Test bench function '{bench_function}' dimension" \
                f" must be {bench_dim}.")
    
    return user_dim


def create_bench_function(name:str, user_dim=2, params:dict={}):
    data = get_function_data(name)
    dimension = check_dimension(name, user_dim, data['fix_dimension'])
    # check_parameters()
        
    if len(params)>0:
        if params.keys() != data['parameters'].keys():
            raise ValueError(
                f"'{name}' only accept {data['parameters'].keys()}"\
                    "parameters")
        for parameter in data['parameters'].keys():
            data['parameters'][parameter] = params[parameter]

    return BaseFunction(name, dimension, data)