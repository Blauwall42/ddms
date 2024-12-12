import covasim as cv
import numpy as np
from morris import morris
from SALib.analyze.morris import analyze

def main():
    problem = {
        'num_vars': 6,
        'names': ['beta', 'rel_death_prob', 'rel_symp_prob', 'rel_severe_prob', 'mild2rec', 'asym2rec'],
        'bounds': [
            [0.008, 0.024],    # beta
            [0.5, 1.5],        # rel_death_prob
            [0.5, 1.5],        # rel_symp_prob
            [0.5, 1.5],        # rel_severe_prob
            [4, 12],           # mild2rec
            [4, 12]           # asym2rec
        ]
    }

    num_vars = problem['num_vars']
    bounds = np.array(problem['bounds'])
    levels = 4
    n_trajectories = 10
    n_reps = 100

    # discretize each parameter range into specified levels
    values = [np.linspace(bounds[i][0], bounds[i][1], levels) for i in range(num_vars)]

    # create Morris samples
    param_values = []
    np.random.seed(42)
    for _ in range(n_trajectories):
        x_star = np.array([np.random.choice(v) for v in values])  # random starting point
        D_star = np.diag(np.random.choice([1, -1], num_vars))  # random direction matrix
        trajectory = morris(num_vars, values, x_star=x_star, D_star=D_star)
        param_values.extend(trajectory)  # each trajectory contributes `num_vars + 1` samples

    param_values = np.array(param_values)

    with open("output2.txt", "a") as f:
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        print("Trajectory:", file=f)
        print(param_values, file=f)
        print("", file=f)

    print(param_values)
    sims = []

    # create all simulations with Method of Morris parameter values
    for params in param_values:
        beta, rel_death_prob, rel_symp_prob, rel_severe_prob, mild2rec, asym2rec = params

        for seed in range(n_reps):
            pars = dict(
                start_day='2020-01-01',
                end_day='2020-12-31',
                pop_type='random',
                pop_size=100_000,
                beta=beta,
                rel_death_prob=rel_death_prob,
                rel_symp_prob=rel_symp_prob,
                rel_severe_prob=rel_severe_prob,
                rand_seed=seed
            )

            sim = cv.Sim(pars)
            sim.pars['dur']['mild2rec']['par1'] = mild2rec
            sim.pars['dur']['asym2rec']['par1'] = asym2rec
            sims.append(sim)

    msim = cv.MultiSim(sims)
    msim.run()

    # results by parameter set, averaging over repetitions
    results = []
    n_params = len(param_values)
    for i in range(n_params):
        start_idx = i * n_reps
        end_idx = start_idx + n_reps
        reps = msim.sims[start_idx:end_idx]  # group of simulations with same parameters
        mean_deaths = np.mean([sim.summary['cum_deaths'] for sim in reps])
        results.append(mean_deaths)

    results = np.array(results)

    # Morris sensitivity analysis
    Si = analyze(problem, param_values, results)
    print("Trajectory length:", n_trajectories, "Number of reps:", n_reps)
    print("Mu (mean):", Si['mu'])
    print("Mu* (mean absolute):", Si['mu_star'])
    print("Sigma (std):", Si['sigma'])

    with open("output2.txt", "a") as f:
        print("Trajectory length:", n_trajectories, "Number of reps:", n_reps, file=f)
        print("Mu (mean):", Si['mu'], file=f)
        print("Mu* (mean absolute):", Si['mu_star'], file=f)
        print("Sigma (std):", Si['sigma'], file=f)

if __name__ == "__main__":
    main()
