=========================
Styblinski-Tang Benchmark
=========================

In the quickstart guide, we went over the three main pieces we need to 
get up and running with HyperSpace. That gave us a general idea of how
the library can be used for machine learning models, but it didn't give 
us something we can run right out of the box. So this time, let's take 
a look at a complete, minimal example.

For this, we are going to make use of the Styblinski-Tang function, a 
commonly known benchmark objective function for optimization methods.
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
