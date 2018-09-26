==========
Quickstart
==========

HyperSpace works by parallelizing parameter search spaces, running Bayesian 
model based optimization (SMBO) over each of these spaces in parallel. It was 
designed to be as minimially invasive as possible so that you do not have to
change much of your existing code to get started. It does not mind which 
machine learning libraries you choose to use. In fact, it does not mind if you
are working with machine learning models. You just need an objective function 
to be minimized. 

Here are the basic steps to get you started:
1. Define an objective function.
2. Define a parameter search space.
3. Call one of HyperSpace's minimization functions, passing it the objective function
   and search space.

Step 1: Defining the objective function.
----------------------------------------

Say you have a machine learning model which has two parameters you would like to optimize.
Since in machine learning we care about our models' generalization performance, we would
like to find the optimal setting of these parameters which minimizes some error on a 
validation set. Our objective function then is the result of training our model with some
settings of our hyperparameters, tested on some validation set not used to train the model.
This objective function would look something like this:

.. code-block:: python
    def objective(params: List) -> float:
        param0, param1, param2 = params
        # Instantiate your model with new params
        model = Model(param0, param1, param2)
        train_loss = train(model)
        validation_loss = validate(model)
        return validation_loss

And this is how we setup our objective functions for HyperSpace! Note that the objective 
function will always need one argument; this is a list new hyperparameter settings, one for 
each of your search dimensions. You won't have to worry about setting this list, it is 
controlled by the HyperSpace minimization functions. You will only need to worry about initially
defining the search space.

Step 2: Defining the parameter search space.
--------------------------------------------

Setting the parameter search space is really easy. You just have to define a list of tuples.
Each tuple consists of the lower and upper bounds for each search dimension respectively. Say 
our example machine learning algorithm's three parameters are integer, real, and categorical valued respectively.
We will let `param0` have a lower bound of `5` and and upper bound of `10`, inclusive. Let `param1`
be a float that has a lower bound of `0.01` and an upper bound of `1.0`. Finally, let `param2` be 
categorical with three options: ('cat0', 'cat1', 'cat2'). We would then define
our search space like this:

.. code-block:: python
    params = [(5, 10), (0.01, 1), ('cat0', 'cat1', 'cat2')]

And that's all we need to do. Two quick notes before we move on; Note 1: HyperSpace will do a type check for your parameters.
if both of your parameters are integer valued (as is the case in `param0`), then HyperSpace will treat 
that parameter as integer valued. If either of the parameters are real valued, than that search dimension
will be treated as real valued. If any of your parameters are strings, or if your tuple consists of more than
two values, then the search space will be considered categorical. Note 2; the ordering of the `params` list 
passed to the objective function by the HyperSpace minimization methods will be the same as you initially defined
your search space. So in our example, you will be guaranteed that the `param0` will correspond to the first tuple
in the about `params` list, and so on. All that is left to do then is to call one of HyperSpace's minimization methods!

Step 3: Calling HyperSpace's minimization functions.
----------------------------------------------------

HyperSpace has several methods for running Bayesian optimization. There are two major libraries that handle
the SMBO methods: Scikit-Optimize and RoBO. There is also a method for running a distributed version of 
HyperBand. I will let you take a look into the documentation for these various methods. We have tried to make 
there arguments as similar as possible, though there are some slight differences. For this example, let's use 
Scikit-Optimize:

.. code-block:: python
    from hyperspace import hyperdrive


    hyperdrive(objective=objective,
               hyperparameters=params,
               results_path='/path/to/save/results',
               model="GP",
               n_iterations=100,
               verbose=True,
               random_state=0,
               checkpoints=True)

HyperSpace here runs a distributed SMBO optimization using a Gaussian process (model='GP') to model our objective function.  
This will run for 100 iterations (n_iterations), saving a checkpoint after each iteration (checkpoints=True) to the directory
specified (results_path). When in verbose mode (verbose=True), the optimization will print the progress of our MPI rank 0. 
We can make our results reproducible by setting a random state (here random_state=0). There are several other parameters that
can be passed to this function, including whether you would like to use latin hypercube sampling for the initial random draws 
to warm up the SMBO procedure. You can see more of these in the docstrings of the minimization functions.

Just a quick note on the resources needed to run HyperSpace: we designed this library to takle the exponential scaling problem 
of Bayesian optimization, which states that the number of samples necessary to bound our uncertainty about the optimization 
scales exponentially with the number of search dimensions. If we have :math:`D` dimensions, the number of resources required will
be :math:`2^{D}`. So, for our example, we need :math:`2^{3}=8` MPI ranks. 

And that is all we need to get running with HyperSpace! If we were to save this example in as a python module called `example.py`,
then we would run it using

.. code-block: bash 
    mpirun -n 8 python3 example.py 

I hope this quickstart guide is helpful! If you have any questions or comments, let me know on the HyperSpace's GitHub issues!

-Todd.
