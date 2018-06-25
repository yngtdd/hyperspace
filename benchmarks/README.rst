Benchmarks
----------

These results are obtained by running the distributed optimizers on the set of benchmark
functions provided in Table 2 of http://www.automl.org/papers/13-BayesOpt_EmpiricalFoundation.pdf

All optimizers are run with default values and with `n_calls=50`. Runs are repeated five times.
Here we report the results of the best HyperSpace rank for each of the benchmarks. The results from
all the of the distributed ranks on each benchmark can be found in the respective benchmark directories.

.. role:: python(code)
   :language: python

Branin
------

To reproduce, run :python:`mpirun -n 4 python bench_branin.py --results_dir ./results`.


=========== =============== ============ =================== ================== ======================
Method      Minimum         Best minimum Mean f_calls to min Std f_calls to min Fastest f_calls to min
----------- --------------- ------------ ------------------- ------------------ ----------------------
gp_minimize 0.398 +/- 0.000  0.398        33.1                5.7                27
=========== =============== ============ =================== ================== ======================
