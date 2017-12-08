try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name = 'hyperspace',
    packages = ['hyperspace'],
    install_requires=['numpy', 'scikit-optimize', 'scikit-learn'],
    version = '0.2.0',
    description = 'Distributed Bayesian model-based optimization with Scikit-Optimize',
    author = 'Todd Young',
    author_email = 'youngmt1@ornl.gov',
    url = 'https://code-int.ornl.gov/ygx/hyperspace',
    keywords = ['parallel', 'optimization', 'mbo'],
)
