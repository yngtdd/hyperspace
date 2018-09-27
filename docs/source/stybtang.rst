==========================================
Minimal Example: Styblinski-Tang Benchmark
==========================================

In the quickstart guide, we went over the three main pieces we need to 
get up and running with HyperSpace. That gave us a general idea of how
the library can be used for machine learning models, but it didn't give 
us something we can run right out of the box. So this time, let's take 
a look at a complete, minimal example.

For this, we are going to make use of the Styblinski-Tang function, a 
commonly used benchmark objective function for optimization methods.
The Styblinski-Tang function is usually evaluated on the hypercube 
:math:`x_{i} \in [-5., 5.]` for all :math:`i = 1, 2, \dots, D` in :math:`D` 
dimensions. It's global minimum :math:`f(x^{*}) = -39.16599 * D` which
can be found at :math:`x^{*} = (-2.903534, \dots, -2.903534)`. We are going
to use the two dimensional form of the function, which looks like this:

.. raw:: html

    <embed>
        <p align="center">
            <img width="300" src="https://github.com/yngtodd/hyperspace/blob/master/docs/source/_static/img/stybtang.gif">
        </p>
    </embed>

In its two dimensional form, the Styblinski-Tang function has three local
minima in addition to its global minimum. Let's see if we can find the global 
minimum. The following example example contains everything we need to run the 
optimization. There are several benchmark functions built into HyperSpace in
addition to the Styblinski-Tang function. I would encourage you to try out 
several of the functions, changing their number of search dimensions. This 
will give you a sense for how HyperSpace, indeed any optimization library,
behaves in various dimensions. Without further ado, let's do this:

.. code-block:: python

    import argparse
    import numpy as np

    from hyperspace import hyperdrive
    from hyperspace.benchmarks import StyblinskiTang


    def main():
        parser = argparse.ArgumentParser(description='Styblinski-Tang Benchmark')
        parser.add_argument('--ndims', type=int, default=2, help='Dimension of Styblinski-Tang')
        parser.add_argument('--results', type=str, help='Path to save the results.')
        args = parser.parse_args()

        stybtang = StyblinskiTange(args.ndims)
        bounds = np.tile((-5., 5.), (args.ndims, 1))

        hyperdrive(objective=stybtang,
                   hyperparameters=bounds,
                   results_path=args.results,
                   model="GP",
                   n_iterations=50,
                   verbose=True,
                   checkpoints=True,
                   random_state=0)


     if __name__=='__main__':
         main()


Keeping in step with the quickstart guide, we have defined an objective function, `stbtang`.
As we mentioned above, this function is typically evaluated on the interval 
:math:`x_{i} = [-5., 5.] \in D`, and so is initialized to that by default by HyperSpace. We
then define our search bounds using Numpy's tile function, which is a convenient way of 
creating an `ndims` dimensional array of tuples, each of the bounds from `[-5., 5]`. Then 
all we need to do is call HyperSpace's `hyperdrive` function, telling it where to save the 
results (`args.results`). 

Just remmber, HyperSpace initializes :math:`2^{D}` optimizations in parallel. Therefore, if
if we save this as a module called `styblinskitang.py`, we can run this example use 
the following:

.. code-block:: console

    mpirun -n 4 python3 styblinskitang.py --ndims 2 --results </path/to/save/results>

Try experimenting with the dimension of the function. This can be done simply by changing
the `args.ndims` argument and adjusting the number of MPI ranks according to the :math:`2^{D}`
rule. Happy optimizing!
