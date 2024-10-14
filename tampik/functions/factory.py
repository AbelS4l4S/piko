from abc import ABC, abstractmethod

from .providers import DefaultFunctionProvider, FunctionProvider
from .base_function import BaseFunction


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

def check_parameters(fx_name, user_params, bench_params):  
    if len(user_params)>0:
        if user_params.keys() != bench_params.keys():
            raise ValueError(
                f"'{fx_name}' only accept {bench_params.keys()}"\
                    "parameters")
        for parameter in bench_params.keys():
            bench_params[parameter] = user_params[parameter]

    return bench_params


class FactoryFunction(ABC):
    @abstractmethod
    def build_function(self):
        pass

class BenchFunctionBuilder(FactoryFunction):
    def __init__(self, name:str, user_dim=2, params:dict={}):
        self.name = name
        self.user_dim = user_dim
        self.params = params
        
    def build_function(self):
        default_provider = DefaultFunctionProvider()
        function_provider = FunctionProvider(provider = default_provider)
        data = function_provider.get_function_data(self.name)
        
        dimension = check_dimension(self.name, self.user_dim, data['fix_dimension'])
        data['parameters'] = check_parameters(self.name, self.params, data['parameters'])
        
        return BaseFunction(self.name, dimension, data)
    
class UserFunctionBuilder(FactoryFunction):
    def build_function(self):
        pass