import os


def get_results(results_dir):
    for rank in ranks:
    results = []
    min_func_calls = []
    time_ = 0.0

    results.append(res)
    func_vals = np.round(res.func_vals, 3)
    min_func_calls.append(np.argmin(func_vals) + 1)

    optimal_values = [result.fun for result in results]
    mean_optimum = np.mean(optimal_values)
    std = np.std(optimal_values)
    best = np.min(optimal_values)

    print(f"Mean optimum: {mean_optimum}")
    print(f"Std of optimal values: {std}")
    print(f"Best optima: {best}")

    mean_fcalls = np.mean(min_func_calls)
    std_fcalls = np.std(min_func_calls)
    best_fcalls = np.min(min_func_calls)

    print("Mean func_calls to reach min: {mean_fcalls}")
    print("Std func_calls to reach min: {std_fcalls}")
    print("Fastest no of func_calls to reach min: {best_fcalls}")
