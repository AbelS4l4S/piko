from .functions.base_function import create_bench_function


def bench_function(name:str, dimension:int, params:dict={}):
    return create_bench_function(name, dimension, params)