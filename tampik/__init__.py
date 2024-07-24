from .functions.base_function import create_bench_function


def bench_function(name:str, dimension:int):
    return create_bench_function(name, dimension)