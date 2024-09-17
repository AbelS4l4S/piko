# Tampik - Optimization algorithms

To use the `tampik` package, follow this convention:
```Python
import tampik as pk
```
The first feature implemented in tampik is the ability to handle functions, which will be used as objective functions for optimization algorithms.

In the field of optimization, there are specific functions known as benchmark functions [[1]](https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective/tree/main?tab=readme-ov-file) used to test the exploration and exploitation capabilities of optimization algorithms. This package includes the following benchmark functions:

|Name|Function|
|-|-|
|Ackley|$f(\mathbf{x}) = -a \cdot exp(-b\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2})-exp(\frac{1}{d}\sum_{i=1}^{d}cos(c \cdot x_i))+ a + exp(1)$|
|Ackley N. 2|$f(x, y) = -200exp(-0.2\sqrt{x^2 + y^2)}$|
|Ackley N. 3|$f(x, y) = -200exp(-0.2\sqrt{x^2 + y^2}) + 5exp(cos(3x) + sin(3y))$|
|Ackley N. 4|$f(\mathbf{x})=\sum_{i=1}^{d-1}\left( e^{-0.2}\sqrt{x_i^2+x_{i+1}^2} + 3\left( cos(2x_i) + sin(2x_{i+1}) \right) \right)$
|Adjiman|$f(x, y)=cos(x)sin(y) - \frac{x}{y^2+1}$|
|Alpine N. 1|$f(\mathbf x) = \sum_{i=1}^{d} abs(x_i sin(x_i)+0.1x_i)$|
|Alpine N. 2|$f(\mathbf x)=- \prod_{i=1}^{d}\sqrt{x_i}sin(x_i)$|
|Thevenot|$f(\mathbf{x}) = exp(-\sum_{i=1}^{d}(x_i / \beta)^{2m}) - 2exp(-\prod_{i=1}^{d}x_i^2) \prod_{i=1}^{d}cos^ 2(x_i)$|

To use any of the functions in the package, call the Python function `pk.bench_function()`. For example, to use the Alpine N. 1 function with 5 dimensions:

```Python
import tampik as pk
import numpy as np

alphine = pk.bench_function('Alpine N. 1', 5)

A = np.array([1, 30, 2, 15, 31])
B = np.array([[1, 30, 2, 15, 31],
              [7, 1, 4, 6, 9],
              [121, 34, 80, 41, 33]])

y_a = alphine(A) # -> [50.2805]
y_b = alphine(B) # -> [50.2805 14.5531 264.557]
```
**Note:** The number of columns in matrices `A` and `B` must match the function's dimension.

The `pk.bench_function()` creates an object from the `BaseFunction` class, which has the following attributes:`continuous`, `convex`, `differentiable`, `dimension`, `function`, `latex_dimension`, `latex_formula`, `latex_global_minimum`, `latex_input_domain`,`multimodal`, `name`, `parameters`, `parametric`, `randomized_term`, `separable`.

This can be useful in specific cases. For example, when a benchmark function has parameters, tampik sets default parameter values. To view the parameters, use the `parameters` attribute:
```Python
import tampik as pk
import numpy as np

thevenot = pk.bench_function('Thevenot', 5)

A = np.array([3, 2, 6, 0.5, 7])
B = np.array([[3, 2, 6, 0.5, 7],
              [7, 1, 4, 6, 9],
              [0.121, 0.34, 5, 0.41, 3]])

y_a = thevenot(A) # -> [0.999405]
y_b = thevenot(B) # -> [0.999405 0.993379 0.890997]

print(thevenot.parameters) # -> {'m': 5, 'beta': 15}
```
If you wish to change the function's parameters, you can provide a dictionary to `pk.bench_function()` with the new values:
```Python
import tampik as pk
import numpy as np

thevenot = pk.bench_function(name = 'Thevenot', 
                            dimension = 5, 
                            params = {'m': 10, 'beta': 25})

A = np.array([3, 2, 6, 0.5, 7])
B = np.array([[3, 2, 6, 0.5, 7],
              [7, 1, 4, 6, 9],
              [0.121, 0.34, 5, 0.41, 3]])

y_a = thevenot(A) # -> [1.]
y_b = thevenot(B) # -> [1. 1. 0.891014]

print(thevenot.parameters) # -> {'m': 10, 'beta': 25}
```
**Note:** The parameters name: `name`, `dimension` and `params` are optional.

## References and Biography
[1] A. Thevenot, "Python_Benchmark_Test_Optimization_Function_Single_Objective", GitHub repository, 2023. [Online]. Available: https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective/tree/main?tab=readme-ov-file