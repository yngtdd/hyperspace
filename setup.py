from setuptools import setup, find_packages


setup(
    name='hyperspaces',
    version="0.2.3",
    packages=find_packages(),
    install_requires=['mpi4py', 'cython', 'sphinxcontrib-bibtex'],
    author="Todd Young",
    author_email="youngmt1@ornl.gov",
    description="Distributed Bayesian model-based optimization",
    license="MIT",
    keywords="parallel optimization smbo",
    url="https://github.com/yngtodd/hyperspace",
)
