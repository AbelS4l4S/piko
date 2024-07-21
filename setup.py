from setuptools import setup, find_packages

MAJOR = 1
MINOR = 0
MICRO = 0
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"
DESCRIPTION = 'Python package for metaheuristic optimization algorithms.'

setup(
        name="tampik", 
        version=VERSION,
        author="Abel Salas Leal",
        author_email="<abelsl1999@gmail.com>",
        description=DESCRIPTION,
        long_description=open('README.md').read(),
        long_description_content_type="text/markdown",
        url="https://github.com/AbelS4l4S/piko",
        packages=find_packages(),
        package_data={
            'tampik': ['data/bench_test_functions.csv'],
            },
        install_requires=[],
        keywords=['optimization', 'metaheuristics','operations research', 
                  'artificial intelligence', 'machine learning'],
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ]
)