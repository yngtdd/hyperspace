Benchmarks
----------

These results are obtained by running our optimizers on the set of benchmark
functions provided in Table 2 of http://www.automl.org/papers/13-BayesOpt_EmpiricalFoundation.pdf

All optimizers are run with default values and with `n_calls=200`. Runs are
repeated 10 times.

Branin
------

To reproduce, run `python bench_branin.py`

| Method | Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
| dummy_minimize | 0.911 +/- 0.294 |0.492 | 27.6 | 14.677 | 4
| gp_minimize | 0.398 +/- 0.000 |0.398 | 33.1 | 5.7 | 27
| forest_minimize| 0.515 +/- 0.15 |0.399 | 163.8 | 33.295 | 83
| gbrt_minimize | 0.580 +/- 0.33 |0.401 | 110.5 | 49.810 | 46
