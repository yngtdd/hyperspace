Benchmarks
----------

These results are obtained by running our optimizers on the set of benchmark
functions provided in Table 2 of http://www.automl.org/papers/13-BayesOpt_EmpiricalFoundation.pdf

All optimizers are run with default values and with `n_calls=200`. Runs are
repeated 10 times.

Branin
------

To reproduce, run `python bench_branin.py`

=========== =============== ============ =================== ================== ======================
Method      Minimum         Best minimum Mean f_calls to min Std f_calls to min Fastest f_calls to min
----------- --------------- ------------ ------------------- ------------------ ----------------------
gp_minimize 0.398 +/- 0.000  0.398        33.1                5.7                27
=========== =============== ============ =================== ================== ======================
