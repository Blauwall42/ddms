from cases.base import Case, cv
import numpy as np


# Define the vaccine subtargeting
def vaccinate_by_age(sim):
    young = cv.true(sim.people.age < 50)  # cv.true() returns indices of people matching this condition, i.e. people under 50
    middle = cv.true((sim.people.age >= 50) * (sim.people.age < 75))  # Multiplication means "and" here
    old = cv.true(sim.people.age >= 75)
    inds = sim.people.uid  # Everyone in the population -- equivalent to np.arange(len(sim.people))
    vals = np.ones(len(sim.people))  # Create the array
    vals[young] = 0.1  # 10% probability for people <50
    vals[middle] = 0.5  # 50% probability for people 50-75
    vals[old] = 0.9  # 90% probability for people >75
    output = dict(inds=inds, vals=vals)
    return output


class CombinedIntervention1(Case):
    def __init__(self, pars: dict[str, str]):
        super().__init__('Vaccine + Closed Schools + Loose Mask Mandates', pars)
        self.intervention = []
        self.intervention.append(cv.change_beta(changes=0.85, days='2020-03-01'))
        self.intervention.append(cv.change_beta(changes=0.01, layers='s', days='2020-03-01'))
        self.intervention.append(cv.simple_vaccine(days='2020-03-01', rel_sus=0.8, rel_symp=0.06,
                                                   subtarget=vaccinate_by_age))


class CombinedIntervention2(Case):
    def __init__(self, pars: dict[str, str]):
        super().__init__('Earlier Vaccine + Loose Mask Mandates', pars)
        self.intervention = []
        self.intervention.append(cv.change_beta(changes=0.85, days='2020-03-01'))
        self.intervention.append(cv.simple_vaccine(days='2020-01-01', rel_sus=0.8, rel_symp=0.06,
                                                   subtarget=vaccinate_by_age))