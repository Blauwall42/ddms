import covasim as cv

class Case:
    def __init__(self, label: str, pars: dict[str, str]):
        self.label: str = label
        self.pars: dict[str, str] = pars
        self.intervention = None

    def sim(self, seed: int)-> cv.Sim:
        return cv.Sim(self.pars, interventions=self.intervention, label=self.label, rand_seed=seed)

# class for baseline
class Baseline(Case):
    def __init__(self, pars: dict[str, str]):
        super().__init__('Baseline', pars)