import covasim as cv
import numpy as np
from SALib.sample.morris import sample
from SALib.analyze.morris import analyze

def main():
    problem = {
        'num_vars': 4,
        'names': ['beta', 'rel_death_prob', 'rel_symp_prob', 'mild2rec'],
        'bounds': [
            [0.01, 0.5],       # beta
            [0.5, 2.0],        # rel_death_prob
            [0.5, 2.0],        # rel_symp_prob
            [5, 20]            # mild2rec
        ]
    }

    param_values = sample(problem, N=25, num_levels=4)  # generate Morris samples
    n_reps = 25
    sims = []

    # create all simulations with Method of Morris parameter values
    for params in param_values:
        beta, rel_death_prob, rel_symp_prob, mild2rec = params

        for seed in range(n_reps):
            pars = dict(
                start_day='2020-01-01',
                end_day='2020-12-31',
                pop_type='hybrid',
                pop_size=10_000,
                beta=beta,
                rel_death_prob=rel_death_prob,
                rel_symp_prob=rel_symp_prob,
                rand_seed=seed
            )

            sim = cv.Sim(pars)
            sim.pars['dur']['mild2rec']['par1'] = mild2rec
            sims.append(sim)

    msim = cv.MultiSim(sims)
    msim.run()

    # group results by parameter set, averaging over repetitions
    results = []
    n_params = len(param_values)
    for i in range(n_params):
        start_idx = i * n_reps
        end_idx = start_idx + n_reps
        reps = msim.sims[start_idx:end_idx] # get group of simulations with same parameters
        mean_deaths = np.mean([sim.summary['cum_deaths'] for sim in reps])
        results.append(mean_deaths)

    results = np.array(results)

    Si = analyze(problem, np.array(param_values), results)
    print("Mu (mean):", Si['mu'])
    print("Mu* (mean absolute):", Si['mu_star'])
    print("Sigma (std):", Si['sigma'])

if __name__ == "__main__":
    main()
