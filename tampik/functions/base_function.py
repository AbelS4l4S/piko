import numpy as np
import csv


class BaseFunction:
    def __init__(self, data, dimension):
        self.dimension = dimension
        self.name = data['name']
        self.latex_formula = data['latex_formula']
        self.latex_dimension = data['dimension']
        self.latex_input_domain = data['input_domain']
        self.latex_global_minimum = data['global_minimum']
        self.continuous = bool(data['continuous'])
        self.convex = bool(data['convex'])
        self.separable = bool(data['separable'])
        self.differentiable = bool(data['differentiable'])
        self.multimodal = bool(data['multimodal'])
        self.randomized_term = bool(data['randomized_term'])
        self.parametric = bool(data['parametric'])

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


def get_function_data(name:str, file_path: str) -> dict[str,str]:
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['name'] == name:
                return row
    raise ValueError(f"Function with name '{name}' not found in CSV.")


def create_bench_function(name:str, dim:int): 
    if not (isinstance(dim, int) and dim > 0):
        raise ValueError('Dimension must be positive')
    data = get_function_data(name,'data/bench_test_functions.csv')
    return BaseFunction(data, dimension=dim)