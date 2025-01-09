from matplotlib.collections import LineCollection
from scipy import optimize
import covasim as cv
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings


# colored_line copied from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def plot_result(betas: list, rel_death_probs: list, title: str, file: str) -> None:
    color = np.linspace(0, len(betas), len(betas))
    fig, ax = plt.subplots()
    lines = colored_line(betas, rel_death_probs, color, ax, linewidth=1, cmap="plasma")
    cbar = plt.colorbar(lines)
    cbar.set_label('iteration')

    ax.set_xlim(round(min(betas)-0.1, 1), round(max(betas)+0.1, 1))
    ax.set_ylim(round(min(rel_death_probs)-0.1, 1), round(max(rel_death_probs)+0.1, 1))
    ax.set_xlabel('beta')
    ax.set_ylabel('rel_death_probs')
    ax.set_title(title)

    plt.savefig(file, format='svg')

# objective adapted from https://docs.idmod.org/projects/covasim/en/latest/tutorials/tut_calibration.html
def objective(x, n_runs=10):
    global betas
    global rel_death_probs
    betas.append(x[0])
    rel_death_probs.append(x[1])
    print(f'Running sim for beta={x[0]}, rel_death_prob={x[1]}')

    pars = dict(
        pop_size=209_755,
        start_day='2020-10-01',
        end_day='2020-12-31',
        beta=x[0],
        rel_death_prob=x[1],
        verbose=0,
    )
    sim = cv.Sim(pars=pars, datafile='data.csv', location='Rostock')
    msim = cv.MultiSim(sim)
    msim.run(n_runs=n_runs)
    mean_squared_errors = []
    for sim in msim.sims:
        fit = sim.compute_fit()
        fit.compute_gofs(normalize=False, use_squared=True, as_scalar='mean')
        mean_squared_errors.append(fit.gofs['cum_deaths']+ fit.gofs['cum_diagnoses'])
    mean_squared_error = np.mean(mean_squared_errors)
    return mean_squared_error


def main() -> None:
    global betas
    global rel_death_probs
    rostock_pop = {
        '0-6': 10731,
        '6-15': 14385,
        '15-25': 23378,
        '25-35': 30987,
        '35-45': 27104,
        '45-55': 22790,
        '55-65': 28766,
        '65-75': 22650,
        '75+': 28964
    }
    cv.data.country_age_data.data['Rostock'] = rostock_pop

    print('Start nelder-mead')
    start_time = time.time()
    guess: list = [2,2]
    pars_nelder = optimize.minimize(objective, x0=guess, method='nelder-mead')
    time_nelder = time.time() - start_time
    betas_nelder = betas.copy()
    rel_death_probs_nelder = rel_death_probs.copy()

    betas = []
    rel_death_probs = []

    print('Start basinhopping')
    start_time = time.time()
    pars_basinhopping = optimize.basinhopping(objective, [2,2])
    time_basinhopping = time.time() - start_time
    betas_basinhopping = betas.copy()
    rel_death_probs_basinhopping = rel_death_probs.copy()

    betas = []
    rel_death_probs = []

    plot_result(betas_nelder, rel_death_probs_nelder, 'Nelder-Mead Simplex algorithm', 'images/nelder.svg')
    plot_result(betas_basinhopping, rel_death_probs_basinhopping, 'Basin-hopping', 'images/basinhopping.svg')


    print('\n\n### Results ###')

    print('\nnelder')
    print(f'time: {time_nelder}')
    print(f'pars: {pars_nelder}')
    print(f'beta: {betas_nelder}')
    print(f'rel_death_probs: {rel_death_probs_nelder}')

    print('\nbasinhopping')
    print(f'time: {time_basinhopping}')
    print(f'pars: {pars_basinhopping}')
    print(f'beta: {betas_basinhopping}')
    print(f'rel_death_probs: {rel_death_probs_basinhopping}')


if __name__ == '__main__':
    betas: list = []
    rel_death_probs: list = []

    main()

