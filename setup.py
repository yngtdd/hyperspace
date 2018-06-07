from setuptools import setup, find_packages
# from os import path

# here = path.abspath(path.dirname(__file__))
#
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name='HyperSpaces',
    version="0.2.0",
    packages=find_packages(),
    install_requires=['numpy', 'scikit-optimize', 'scikit-learn', 'mpi4py'],

    # metadata for upload to PyPI
    author="Todd Young",
    author_email="youngmt1@ornl.gov",
    description="Distributed Bayesian model-based optimization with Scikit-Optimize",
    license="MIT",
    keywords="parallel optimization smbo",
    url="https://github.com/yngtodd/hyperspace",
)
