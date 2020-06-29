import os
import sys

from setuptools import setup
from setuptools import find_packages


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


dependencies = [
    'scikit-optimize==0.5.2',
    'scikit-learn',
    'mpi4py',
    'cython',
    'sphinxcontrib-bibtex'
]

setup(
    name='hyperspaces',
    version="0.4.0",
    packages=find_packages(),
    install_requires=dependencies,
    author="Todd Young",
    author_email="youngmt1@ornl.gov",
    description="Distributed Bayesian model-based optimization",
    license="MIT",
    keywords="parallel optimization smbo",
    url="https://github.com/yngtodd/hyperspace",
)
