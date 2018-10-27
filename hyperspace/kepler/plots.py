"""
Plotting functions.

For more plotting functions, see Scikit-Optimize:
https://scikit-optimize.github.io/plots.m.html1
"""
import numpy as np
from scipy.optimize import OptimizeResult

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `true_minimum` [float, optional]:
        The true minimum value of the function, if known.

    * `yscale` [None or string, optional]:
        The scale for the y-axis.

    * `plot_mean` [Bool, default=False]:
        Plots the mean convergence over the distributed runs.

    * `color_map` [string, default='plasma']
        Matplotlib color map to be used.
        - https://matplotlib.org/examples/color/colormaps_reference.html

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    true_maximum = kwargs.get("true_maximum", None)
    yscale = kwargs.get("yscale", None)
    plot_mean = kwargs.get("plot_mean", False)
    color_map = kwargs.get("color_map", "plasma")
    maximize = kwargs.get("maximize", False)

    fig = plt.figure()
    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    if maximize:
        ax.set_ylabel(r"$\max f(x)$ after $n$ calls")
    else:
        ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.coolwarm(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            if maximize:
                results.func_vals = - results.func_vals
                maxes = [np.max(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
                ax.plot(range(1, n_calls + 1), maxes, c=color,
                        marker=".", markersize=12, lw=2, label=name)
            else:
                mins = [np.min(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
                ax.plot(range(1, n_calls + 1), mins, c=color,
                        marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            if maximize:
                mins = [[np.max(r.func_vals[:i]) for i in iterations]
                        for r in results]

                for m in maxes:
                    ax.plot(iterations, m, c=color, alpha=0.2)

                if plot_mean == True:
                    ax.plot(iterations, np.mean(maxes, axis=0), c=color,
                            marker=".", markersize=12, lw=2, label=name)

            else:
                mins = [[np.min(r.func_vals[:i]) for i in iterations]
                        for r in results]

                for m in mins:
                    ax.plot(iterations, m, c=color, alpha=0.2)

                if plot_mean == True:
                    ax.plot(iterations, np.mean(mins, axis=0), c=color,
                            marker=".", markersize=12, lw=2, label=name)
    if maximize:
        if true_minimum:
            ax.axhline(true_minimum, linestyle="--",
                       color="r", lw=1,
                       label="True maximum")

        if true_maximum or name:
            ax.legend(loc="best")

    else:
        if true_minimum:
            ax.axhline(true_minimum, linestyle="--",
                       color="r", lw=1,
                       label="True minimum")

        if true_minimum or name:
            ax.legend(loc="best")

    return fig, ax
