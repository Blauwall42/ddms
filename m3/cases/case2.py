from cases.base import Case, cv


class StrictMasks(Case):
    def __init__(self, pars: dict[str, str]):
        super().__init__('Strict Mask Mandates', pars)
        self.intervention = cv.change_beta(changes=0.5, days='2020-03-01')

class LooseMasks(Case):
    def __init__(self, pars: dict[str, str]):
        super().__init__('Loose Mask Mandates', pars)
        self.intervention = cv.change_beta(changes=0.85, days='2020-03-01')
