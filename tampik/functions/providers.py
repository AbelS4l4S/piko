from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional

from ..data.bench_test_functions import data_functions


class FunctionDataProvider(ABC):
    @abstractmethod
    def get_data(self, name:str) -> dict:
        pass
    

class DefaultFunctionProvider(FunctionDataProvider):
    def get_data(self, name):
        if name not in data_functions.keys():
            raise ValueError(f"Test bench function '{name}' not founded.")
        return data_functions[name]


class UserInputProvider(FunctionDataProvider):
    def get_data(self, name:str, data:dir)-> dir:
        EXPECTED_DICT = {
            'np_formula': Callable,
            'latex_formula': str,
            'fix_dimension': Optional[int],
            'latex_dimension': str,
            'input_domain': Callable,
            'latex_input_domain': str,
            'global_minimun': Optional[Any],
            'latex_global_minimum': Optional[str],
            'parameters': Dict[str, Any],
            'continuous': bool,
            'convex': bool,
            'separable': bool,
            'differentiable': bool,
            'multimodal': bool,
            'randomized_term': bool,
            'parametric': bool()
            }
        
        if not (EXPECTED_DICT.keys() == data.keys()):
            raise ValueError(
                "Invalid function parametres")
        return data


class FunctionProvider():
    def __init__(self, provider: FunctionDataProvider):
        self.provider = provider
        
    def get_function_data(self, name:str) -> dict:
        return self.provider.get_data(name)