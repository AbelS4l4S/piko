import numpy as np


class BaseFunction:
    def __init__(self, name:str, dimension:int, data:dict):
        self.name = name
        self.dimension = dimension
        self.fix_dimension = data['fix_dimension']
        self.latex_formula = data['latex_formula']
        self.latex_dimension = data['latex_dimension']
        self.latex_input_domain = data['latex_input_domain']
        self.global_minimum = data['global_minimun']
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
        
        if not self.fix_dimension:
            self.input_domain = data['input_domain'](self.dimension)
        else:
            self.input_domain = data['input_domain']()
        
        if data['parameters'] is not None:
            for key, value in data['parameters'].items():
                setattr(self, key, value)

    def __call__(self, X):
        X = np.atleast_2d(X)
        
        if X.shape[1] != self.dimension:
            raise ValueError(f"Input matrix dimension must be: {self.dimension}")
        
        boundaries = self.input_domain.T
        X = np.clip(X, boundaries[0,:], boundaries[1,:])
        parameters = self.function.__code__.co_varnames
        
        args = [self.parameters[param]
                for param in parameters 
                if param in self.parameters
        ]
        
        return self.function(*args, X)