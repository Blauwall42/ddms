import covasim as cv
import time
from cases.base import Case, Baseline
from cases.case1 import ClosedWork, ClosedSchools
from cases.case2 import StrictMasks, LooseMasks
from cases.case3 import LowLockdown, HighLockdown
from cases.case4 import CombinedIntervention1, CombinedIntervention2


def case_simulation(simulations: list[Case], plot_path: str = None, n_sims: int = 100):
    msims: list = []
    for simulation in simulations:
        sims = []
        for seed in range(n_sims):
            sims.append(simulation.sim(seed))
        msim = cv.MultiSim(sims)
        msim.run()
        msim.reduce(use_mean=True, bounds=1.645)
        msims.append(msim)

    if plot_path is not None:
        merged = cv.MultiSim.merge(msims, base=True)
        merged.plot(color_by_sim=True).savefig(plot_path, format='svg')


def experiment(cases: list, pars: dict[str, str], n_sims: int = 100) -> list[dict]:
    times: list = []
    start_time: float = time.time()

    for i in cases:
        if i == 0:
            continue
        elif i == 1:
            case: list = [Baseline(pars), ClosedSchools(pars), ClosedWork(pars)]
        elif i == 2:
            case: list = [Baseline(pars), StrictMasks(pars), LooseMasks(pars)]
        elif i == 3:
            case: list = [Baseline(pars), LowLockdown(pars), HighLockdown(pars)]
        elif i == 4:
            case: list = [Baseline(pars), CombinedIntervention1(pars), CombinedIntervention2(pars)]
        else:
            raise ValueError('Please select a simulation run. Valid runs are: 1, 2, 3, 4')
        start_time_case: float = time.time()
        case_simulation(case, f'images/case{i}.svg', n_sims)
        end_time_case: float = time.time()
        times.append({'name': f'Total time - Case {i}', 'time': round(end_time_case - start_time_case, 2)})

    end_time = time.time()
    times.append({'name': 'Total time - Experiment', 'time': round(end_time - start_time, 2)})
    return times


def avg_scenario_time(cases: list, pars: dict[str, str], n_sims: int = 100) -> list[dict]:
    times = []
    # Baseline
    if 0 in cases:
        start_time_case = time.time()
        case_base = [Baseline(pars)]
        case_simulation(case_base, None, n_sims)
        end_time_case = time.time()
        times.append({'name': 'Avg. time - Baseline', 'time': round((end_time_case - start_time_case) / n_sims, 2)})

    # Case 1
    if 1 in cases:
        start_time_case = time.time()
        case1 = [ClosedSchools(pars)]
        case_simulation(case1, None, n_sims)
        end_time_case = time.time()
        times.append({'name': 'Avg. time - Close Schools', 'time': round((end_time_case - start_time_case) / n_sims, 2)})

        start_time_case = time.time()
        case1 = [ClosedWork(pars)]
        case_simulation(case1, None, n_sims)
        end_time_case = time.time()
        times.append({'name': 'Avg. time - Close Work', 'time': round((end_time_case - start_time_case)/n_sims, 2)})

    # Case 2
    if 2 in cases:
        start_time_case = time.time()
        case2 = [StrictMasks(pars)]
        case_simulation(case2, None, n_sims)
        end_time_case = time.time()
        times.append({'name': 'Avg. time - Strict Mask', 'time': round((end_time_case - start_time_case) / n_sims, 2)})

        start_time_case = time.time()
        case2 = [LooseMasks(pars)]
        case_simulation(case2, None, n_sims)
        end_time_case = time.time()
        times.append({'name': 'Avg. time - time - Loose Mask', 'time': round((end_time_case - start_time_case) / n_sims, 2)})

    # Case 3
    if 3 in cases:
        start_time_case = time.time()
        case = [LowLockdown(pars)]
        case_simulation(case, None, n_sims)
        end_time_case = time.time()
        times.append({'name': 'Avg. time - Low Lockdown', 'time': round((end_time_case - start_time_case) / n_sims, 2)})

        start_time_case = time.time()
        case = [HighLockdown(pars)]
        case_simulation(case, None, n_sims)
        end_time_case = time.time()
        times.append({'name': 'Avg. time - High Lockdown', 'time': round((end_time_case - start_time_case) / n_sims, 2)})

    # Case 4
    if 4 in cases:
        start_time_case = time.time()
        case = [CombinedIntervention1(pars)]
        case_simulation(case, None, n_sims)
        end_time_case = time.time()
        times.append({'name': 'Avg. time - Combined Intervention 1', 'time': round((end_time_case - start_time_case) / n_sims, 2)})

        start_time_case = time.time()
        case = [CombinedIntervention2(pars)]
        case_simulation(case, None, n_sims)
        end_time_case = time.time()
        times.append({'name': 'Avg. time - Combined Intervention 2', 'time': round((end_time_case - start_time_case) / n_sims, 2)})
    
    return times


if __name__ == "__main__":
    # covasim run parameter
    parameter = dict(
        start_day='2020-01-01',
        end_day='2020-12-31',
        pop_type='hybrid',
        pop_size='100_000'
    )

    run_times = []
    cases_to_run = [0, 1, 2, 3, 4]  # Case 0 is only needed for avg_scenario_time(), represents the baseline simulation

    # Simulation
    run_times.extend(experiment(cases_to_run, parameter))

    # uncomment to get the average time per simulation run:
    # run_times.extend(avg_scenario_time(cases_to_run, parameter))

    # print time results
    for time in run_times:
        print(f'{time["name"]}: {time["time"]}s')