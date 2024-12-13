import numpy as np
import covasim as cv
import pandas as pd

"""
The implementation was guided by the section 4.2 and 4.3 of the book "Sensitivity Analysis in Practice" by Andrea Saltelli et al.
"""

def get_orientation_matrix(n_input: int, p: int, ranges: list[list], delta: float) -> tuple[np.array, list]:
    """

    :param n_input: Number of inputs
    :param p:
    :param ranges:
    :param delta:
    :return:    Tuple of the Orientation matrix and a list with changes between following rows. The abs() of the number
                is the index for the changing element and the +/- encodes an increase or decrease.
                !!!CAVE: Index starts at 1!!!
    """
    k: int = n_input
    values: np.array = np.arange(0.0, 1.0 - delta, 1.0 / (p - 1))

    B: np.array = np.tri(k + 1, k, -1)
    J_k: np.array = np.ones((k + 1, k))
    J_1: np.array = np.ones((k + 1, 1))
    I: np.array = np.identity(k)
    P_star: np.array = I[:, np.random.permutation(k)]

    x_star: np.array = np.array([np.random.choice(values, n_input)])
    D_star: np.array = np.diag(np.random.choice([1, -1], k))

    B_star = (J_1 @ x_star + (delta / 2) * ((2 * B - J_k) @ D_star + J_k)) @ P_star

    changes: list = []
    for i in range(k):
        diff: np.array = B_star[i + 1] - B_star[i]
        changes.append(int((np.where(diff != 0.)[0][0] + 1) * np.sign(diff[np.where(diff != 0.)[0]])[0]))

    lower_bound: list = []
    scale: list = []
    for elem in ranges:
        lower_bound.append(elem[0])
        scale.append(elem[1] - elem[0])
    lower_bound_mat = np.array([lower_bound for _ in range(k+1)])
    scale_mat = np.array([scale for _ in range(k+1)])

    return lower_bound_mat + scale_mat * B_star, changes


def morris(pars: dict, n_inputs: int, ranges: list, rs: int, p: int = 4, n_reps=100) -> dict:
    np.random.seed(42)
    delta: float = p / (2 * (p - 1))

    r_orientation_mat: list = []
    r_changes: list = []
    for r in range(rs):
        orientation_mat, changes = get_orientation_matrix(n_inputs, p, ranges, delta)
        r_orientation_mat.append(orientation_mat)
        r_changes.append(changes)

    sims: list = []

    for r in range(rs):
        for row in r_orientation_mat[r]:
            beta, rel_death_prob, rel_symp_prob, rel_severe_prob, mild2rec, asym2rec = row
            pars['beta'] = beta
            pars['rel_death_prob'] = rel_death_prob
            pars['rel_symp_prob'] = rel_symp_prob
            pars['rel_severe_prob'] = rel_severe_prob
            for seed in range(n_reps):
                pars['rand_seed'] = seed

                sim = cv.Sim(pars)
                sim.pars['dur']['mild2rec']['par1'] = mild2rec
                sim.pars['dur']['asym2rec']['par1'] = asym2rec
                sims.append(sim)

    msim = cv.MultiSim(sims)
    msim.run()
    results: list = []

    for i in range(int(len(sims)/n_reps)):
        start_idx = i * n_reps
        end_idx = start_idx + n_reps
        reps = msim.sims[start_idx:end_idx]  # get group of simulations with same parameters
        mean_deaths = np.mean([sim.summary['cum_deaths'] for sim in reps])
        results.append(mean_deaths)

    ds: dict = {'F': {1: [], 2: [], 3: [], 4: [], 5: [], 6: []},
          'G': {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}}
    for r, changes in enumerate(r_changes):
        for i, change in enumerate(changes):
            if change > 0:
                value = (results[r*i+1] - results[r*i])/delta
            else:
                value = (results[r*i] - results[r*i + 1]) / delta
            ds['F'][abs(change)].append(value)
            ds['G'][abs(change)].append(abs(value))
    return ds


def main():
    pars: dict = dict(
        start_day='2020-01-01',
        end_day='2020-12-31',
        pop_type='random',
        pop_size=25_000
    )
    n_inputs: int = 6
    ranges = [
        [0.008, 0.024],  # beta
        [0.5, 1.5],  # rel_death_prob
        [0.5, 1.5],  # rel_symp_prob
        [0.5, 1.5],  # rel_severe_prob
        [4, 12],  # mild2rec
        [4, 12]  # asym2rec
    ]
    ds = morris(pars, n_inputs, ranges, 10, n_reps=100)

    inputs: list = ['beta', 'rel_death_prob', 'rel_symp_prob', 'rel_severe_prob', 'mild2rec', 'asym2rec']
    mu: dict = {}
    mu_star: dict = {}
    sigma: dict = {}
    for i in range(1, n_inputs+1):
        mu[inputs[i-1]] = np.mean(ds["F"][i])
        mu_star[inputs[i-1]] = np.mean(ds["G"][i])
        sigma[inputs[i-1]] = np.std(ds["F"][i])
    result_dict: dict = {'mu': mu, 'mu_star': mu_star, 'sigma': sigma}

    df = pd.DataFrame(result_dict)
    df.to_csv('morris.csv')

if __name__ == "__main__":
    main()
