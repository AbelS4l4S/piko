from .functions.factory import BenchFunctionBuilder


def bench_function(name:str, dimension:int, params:dict={}):
    function_buldier = BenchFunctionBuilder(name, dimension, params)
    return function_buldier.build_function()