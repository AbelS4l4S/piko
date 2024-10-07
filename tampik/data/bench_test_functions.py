import numpy as np

data_functions = {
    'Ackley':{
        'np_formula':lambda a, b, c, X:(
            -a * np.exp(-b * np.sqrt(np.mean(X**2, axis=1))) 
            - np.exp(np.mean(np.cos(c * X), axis=1)) + a + np.exp(1)
            ),
        'latex_formula':r'f(\mathbf{x}) = -a \cdot exp(-b\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2})-exp(\frac{1}{d}\sum_{i=1}^{d}cos(c \cdot x_i))+ a + exp(1)',
        'fix_dimension':None,
        'latex_dimension':r'd \in \mathbb{N}_{+}^{*}',
        'input_domain':lambda d: np.tile([-2 * np.pi, 2 * np.pi], (d, 1)),
        'latex_input_domain':r'x_i \in [-32, 32], \forall i \in \llbracket 1, d\rrbracket',
        'global_minimun':None,
        'latex_global_minimum':r'f((0, ..., 0)) = 0',
        'parameters':{'a':20, 'b':0.2, 'c':2},
        'continuous':True,
        'convex':False,
        'separable':True,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':True
        },
    'Ackley N. 2':{
        'np_formula':lambda X:(
            -200 * np.exp(-0.2 * np.sqrt(X[:,0]**2 + X[:,1]**2))
            ),
        'latex_formula':r'f(x, y) = -200exp(-0.2\sqrt{x^2 + y^2)}',
        'fix_dimension': 2,
        'latex_dimension':'d=2',
        'input_domain':np.array([[-32, 32], [-32, 32]]),
        'latex_input_domain':r'x \in [-32, 32], y \in [-32, 32]',
        'global_minimun':None,
        'latex_global_minimum':None,
        'parameters':{},
        'continuous':True,
        'convex':False,
        'separable':True,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':True
        },
    'Ackley N. 3':{
        'np_formula':lambda X:(
            -200 * np.exp(-0.2 * np.sqrt(X[:,0]**2 + X[:,1]**2)) 
            + 5 * np.exp(np.cos(3 * X[:,0]) + np.sin(3 * X[:,1]))
            ),
        'latex_formula':r'f(x, y) = -200exp(-0.2\sqrt{x^2 + y^2}) + 5exp(cos(3x) + sin(3y))',
        'fix_dimension': 2,
        'latex_dimension':'d=2',
        'input_domain':np.array([[-32, 32], [-32, 32]]),
        'latex_input_domain':r'x \in [-32, 32], y \in [-32, 32]',
        'global_minimun':None,
        'latex_global_minimum':None,
        'parameters':{},
        'continuous':False,
        'convex':False,
        'separable':False,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Ackley N. 4':{
        'np_formula':lambda X: (
            np.sum(np.exp(-0.2) * np.sqrt(X[:, :-1]**2 + X[:, 1:]**2) 
                   + 3 * (np.cos(2 * X[:, :-1]) + np.sin(2 * X[:, 1:])),
                   axis=1)
            ),
        'latex_formula':r'f(\mathbf{x})=\sum_{i=1}^{d-1}\left( e^{-0.2}\sqrt{x_i^2+x_{i+1}^2} + 3\left( cos(2x_i) + sin(2x_{i+1}) \right) \right)',
        'fix_dimension': None,
        'latex_dimension':r'd \in \mathbb{N}_{+}^{*}',
        'input_domain':lambda d: np.tile([-35, 35], (d, 1)),
        'latex_input_domain':r'x_i \in [-35, 35], \forall i \in \llbracket 1, d\rrbracket',
        'global_minimun':None,
        'latex_global_minimum':None,
        'parameters':{},
        'continuous':False,
        'convex':False,
        'separable':False,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Adjiman':{
        'np_formula':lambda X: (
            np.cos(X[:, 0]) * np.sin(X[:, 1]) - X[:, 0] / (X[:, 1]**2 + 1)
            ),
        'latex_formula':r'f(x, y)=cos(x)sin(y) - \frac{x}{y^2+1}',
        'fix_dimension': 2,
        'latex_dimension':'d=2',
        'input_domain':np.array([[-1, 2], [-1, 1]]),
        'latex_input_domain':r'd \in \mathbb{N}_{+}^{*}',
        'global_minimun':None,
        'latex_global_minimum':None,
        'parameters':{},
        'continuous':True,
        'convex':False,
        'separable':False,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Alpine N. 1':{
        'np_formula':lambda X:(
            np.sum(np.abs(X * np.sin(X) + 0.1 * X), axis=1)
            ),
        'latex_formula':r'f(\mathbf x) = \sum_{i=1}^{d}|x_i sin(x_i)+0.1x_i|',
        'fix_dimension': None,
        'latex_dimension':r'd \in \mathbb{N}_{+}^{*}',
        'input_domain':lambda d: np.tile([0, 10], (d, 1)),
        'latex_input_domain':r'x \in [-1, 2], y \in [-1, 1]',
        'global_minimun':None,
        'latex_global_minimum':None,
        'parameters':{},
        'continuous':False,
        'convex':False,
        'separable':True,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Alpine N. 2':{
        'np_formula':lambda X:(
            -np.prod(np.sqrt(X) * np.sin(X), axis=1)
            ),
        'latex_formula':r'f(\mathbf x)=- \prod_{i=1}^{d}\sqrt{x_i}sin(x_i)',
        'fix_dimension': None,
        'latex_dimension':r'd \in \mathbb{N}_{+}^{*}',
        'input_domain':lambda d: np.tile([0,10], (d,1)),
        'latex_input_domain':r'x_i \in [0, 10], \forall i \in \llbracket 1, d\rrbracket',
        'global_minimun':None,
        'latex_global_minimum':None,
        'parameters':{},
        'continuous':True,
        'convex':False,
        'separable':True,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Bartels':{
        'np_formula':lambda X:(
            np.abs(X[:,0]**2 + X[:,1]**2 + X[:,0] * X[:,1]) 
            + np.abs(np.sin(X[:,0])) 
            + np.abs(np.cos(X[:,1]))
            ),
        'latex_formula':r'f(x,y)=|x^2 + y^2 + xy| + |sin(x)| + |cos(y)|',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-500, 500], [-500, 500]]),
        'latex_input_domain':r'x \in [-500, 500], y \in [-500, 500]',
        'global_minimun':None,
        'latex_global_minimum':r'f(0, 0)=1',
        'parameters':{},
        'continuous':False,
        'convex':False,
        'separable':False,
        'differentiable':False,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Beale':{
        'np_formula':lambda X:(
            (1.5 - X[:,0] + X[:,0] * X[:,1]) ** 2 
            + (2.25 - X[:,0] + X[:,0] * X[:,1]**2) ** 2 
            + (2.625 - X[:,0] + X[:,0] * X[:,1]**3) ** 2
            ),
        'latex_formula':r'f(x, y) = (1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-4.5, 4.5], [-4.5, 4.5]]),
        'latex_input_domain':r'x \in [-4.5, 4.5], y \in [-4.5, 4.5]',
        'global_minimun':None,
        'latex_global_minimum':r'f(3, 0.5)=0',
        'parameters':{},
        'continuous':True,
        'convex':False,
        'separable':False,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Bird':{
        'np_formula':lambda X:(
            np.sin(X[:,0]) * np.exp((1 - np.cos(X[:,1])) ** 2)
            + np.cos(X[:,1]) * np.exp((1 - np.sin(X[:,0])) ** 2) 
            + (X[:,0] - X[:,1]) ** 2
            ),
        'latex_formula':r'f(x, y) = sin(x)exp((1-cos(y))^2)\\+cos(y)exp((1-sin(x))^2)+(x-y)^2',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]]),
        'latex_input_domain':r'x \in [-2\pi, 2\pi], y \in [-2\pi, 2\pi]',
        'global_minimun':None,
        'latex_global_minimum':'f(x, y)\approx-106.764537, at$$ $$(x, y)=(4.70104,3.15294), and$$ $$(x, y)=(-1.58214,-3.13024)',
        'parameters':{},
        'continuous':True,
        'convex':False,
        'separable':False,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'BohachevskyN1':{
        'np_formula':lambda X:(
            X[:,0]**2
            + 2 * X[:,1]**2
            - 0.3 * np.cos(3 * np.pi * X[:,0])
            - 0.4 * np.cos(4 * np.pi * X[:,1])
            + 0.7
            ),
        'latex_formula':r'f(x, y) = x^2 + 2y^2 -0.3cos(3\pi x)-0.4cos(4\pi y)+0.7',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-100, 100], [-100, 100]]),
        'latex_input_domain':r'x \in [-100, 100], y \in [-100, 100]',
        'global_minimun':None,
        'latex_global_minimum':r'f(0, 0)=0',
        'parameters':{},
        'continuous':True,
        'convex':True,
        'separable':True,
        'differentiable':True,
        'multimodal':False,
        'randomized_term':False,
        'parametric':False
        },
    'BohachevskyN2':{
        'np_formula':lambda X:(
            X[:,0]**2
            + 2 * X[:,1]**2
            - 0.3 * np.cos(3 * np.pi * X[:,0]) * np.cos(4 * np.pi * X[:,1])
            + 0.3
            ),
        'latex_formula':r'f(x, y)=x^2 + 2y^2 -0.3cos(3\pi x)cos(4\pi y)+0.3',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-100, 100], [-100, 100]]),
        'latex_input_domain':r'x \in [-100, 100], y \in [-100, 100]',
        'global_minimun':None,
        'latex_global_minimum':r'f(0, 0)=0',
        'parameters':{},
        'continuous':True,
        'convex':False,
        'separable':False,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'BohachevskyN3':{
        'np_formula':lambda X:(
            X[:,0]**2
            + 2 * X[:,1]**2
            - 0.3 * np.cos(3 * np.pi * X[:,0] + 4 * np.pi * X[:,1]) * np.cos(4 * np.pi * X[:,1])
            + 0.3
            ),
        'latex_formula':r'f(x, y)=x^2 + 2y^2 -0.3cos(3\pi x + 4\pi y)cos(4\pi y)+0.3',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-50, 50], [-50, 50]]),
        'latex_input_domain':r'x \in [-50, 50], y \in [-50, 50]',
        'global_minimun':None,
        'latex_global_minimum':r'f(0, 0)=0',
        'parameters':{},
        'continuous':True,
        'convex':False,
        'separable':False,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Booth':{
        'np_formula':lambda X:(
            (X[:,0] + 2 * X[:,1] - 7) ** 2 + (2 * X[:,0] + X[:,1] - 5) ** 2
            ),
        'latex_formula':r'f(x,y)=(x+2y-7)^2+(2x+y-5)^2',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-10, 10], [-10, 10]]),
        'latex_input_domain':r'x \in [-10, 10], y \in [-10, 10]',
        'global_minimun':None,
        'latex_global_minimum':r'f(1, 3)=0',
        'parameters':{},
        'continuous':True,
        'convex':True,
        'separable':False,
        'differentiable':True,
        'multimodal':False,
        'randomized_term':False,
        'parametric':False
        },
    'Branin':{
        'np_formula':lambda a,b,c,r,s,t,X:(
            a * (X[:,1] - b * X[:,0]**2 + c * X[:,0] - r) ** 2
            + s * (1 - t) * np.cos(X[:,0]) + s
            ),
        'latex_formula':r'f(x,y)=a(y - bx^2 + cx - r)^2 + s(1 - t)cos(x) + s',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-5, 10], [0, 15]]),
        'latex_input_domain':r'x \in [-5, 10], y \in [0, 15]',
        'global_minimun':None,
        'latex_global_minimum':r'f(x, y)\approx0.397887, at $$ $$(x, y)=(-\pi, 12.275),$$ $$(x, y)=(\pi, 2.275), and $$ $$(x, y)=(9.42478, 2.475)',
        'parameters':{'a':1,
                      'b':5.1 / (4 * np.pi**2),
                      'c':5 / np.pi,
                      'r':6,
                      's':10,
                      't':1 / (8 * np.pi)},
        'continuous':True,
        'convex':False,
        'separable':False,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Brent':{
        'np_formula':lambda X:(
            (X[:,0] + 10) ** 2 + (X[:,1] + 10) ** 2 + np.exp(-(X[:,0]**2) - X[:,1]**2)
            ),
        'latex_formula':r'f(x, y) = (x + 10)^2 + (y + 10)^2 + exp(-x^2 - y^2)',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-20, 0], [-20, 0]]),
        'latex_input_domain':r'x \in [-20, 0], y \in [-20, 0]',
        'global_minimun':None,
        'latex_global_minimum':r'f(-10, -10)=e^{-200}',
        'parameters':{},
        'continuous':True,
        'convex':True,
        'separable':False,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Brown':{
        'np_formula':lambda X:(
            np.sum((X[:-1]**2) ** (X[1:]**2 + 1) + (X[1:]**2) ** (X[:-1]**2 + 1))
            ),
        'fix_dimension': None,
        'latex_formula':r'f(\mathbf{x}) = \sum_{i=1}^{d-1}(x_i^2)^{(x_{i+1}^{2}+1)}+(x_{i+1}^2)^{(x_{i}^{2}+1)}',
        'latex_dimension':r'd \in \mathbb{N}_{+}^{*}',
        'input_domain':lambda d: np.tile([-1,4], (d,1)),
        'latex_input_domain':r'x_i \in [-1, 4], \forall i \in \llbracket 1, d\rrbracket',
        'global_minimun':None,
        'latex_global_minimum':r'f(0, ..., 0)=0',
        'parameters':{},
        'continuous':True,
        'convex':True,
        'separable':False,
        'differentiable':True,
        'multimodal':False,
        'randomized_term':False,
        'parametric':False
        },
    'BukinN6':{
        'np_formula':lambda X:(
            100 * np.sqrt(np.abs(X[:,1] - 0.01 * X[:,0]**2)) + 0.01 * np.abs(X[:,0] + 10)
            ),
        'latex_formula':r'f(x,y)=100\sqrt{|y-0.01x^2|}+0.01|x+10|',
        'fix_dimension': 2,
        'latex_dimension':r'd=2',
        'input_domain':np.array([[-15,-5],[-3, 3]]),
        'latex_input_domain':r'x \in [-15, -5], y \in [-3, 3]',
        'global_minimun':None,
        'latex_global_minimum':r'f(-10, 1)=0',
        'parameters':{},
        'continuous':True,
        'convex':True,
        'separable':False,
        'differentiable':False,
        'multimodal':True,
        'randomized_term':False,
        'parametric':False
        },
    'Thevenot':{
        'np_formula':lambda m, beta, X: (
            np.exp(-np.sum((X / beta) ** (2 * m), axis=1))
            - 2 * np.exp(-np.prod(X**2, axis=1)) 
            * np.prod(np.cos(X) ** 2, axis=1)),
        'latex_formula':r'f(\mathbf{x}) = exp(-\sum_{i=1}^{d}(x_i / \beta)^{2m}) - 2exp(-\prod_{i=1}^{d}x_i^2) \prod_{i=1}^{d}cos^ 2(x_i)',
        'fix_dimension': None,
        'latex_dimension':r'd \in \mathbb{N}_{+}^{*}',
        'input_domain':lambda d: np.tile([-2 * np.pi, 2 * np.pi], (d, 1)),
        'latex_input_domain':None,
        'global_minimun':None,
        'latex_global_minimum':None,
        'parameters':{'m':5, 'beta':15},
        'continuous':True,
        'convex':True,
        'separable':True,
        'differentiable':True,
        'multimodal':True,
        'randomized_term':False,
        'parametric':True
        }
    }