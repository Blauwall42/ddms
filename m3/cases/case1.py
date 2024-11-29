from cases.base import Case, cv

class ClosedSchools(Case):
    def __init__(self, pars: dict[str, str]):
        super().__init__('Closed Schools', pars)
        self.intervention = cv.change_beta(changes=0.01, layers='s', days='2020-03-01')

class ClosedWork(Case):
    def __init__(self, pars: dict[str, str]):
        super().__init__('Closed Workplaces', pars)
        self.intervention = cv.change_beta(changes=0.01, layers='w', days='2020-03-01')