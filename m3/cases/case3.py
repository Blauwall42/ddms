from cases.base import Case, cv
import numpy as np


def threshold(self, sim, thresh):
    # Meets threshold, activate
    if sim.people.infectious.sum() > thresh:
        if not self.active:
            self.active = True
            self.t_on = sim.t
    # Does not meet threshold, deactivate
    else:
        if self.active:
            self.active = False
            self.t_off = sim.t
    return [self.t_on, self.t_off]

def low_thresh(self, sim, thresh=250):
    return threshold(self, sim, thresh)


def high_thresh(self, sim, thresh=1000):
    return threshold(self, sim, thresh)



class LowLockdown(Case):
    def __init__(self, pars: dict[str, str]):
        super().__init__('Early Lockdown', pars)
        on = 0.2  # Beta less than 1 -- intervention is on
        off = 1.0  # Beta is 1, i.e. normal -- intervention is off
        changes = [on, off]
        self.intervention = cv.change_beta(days=low_thresh, changes=changes)
        self.intervention.t_on = np.nan
        self.intervention.t_off = np.nan
        self.intervention.active = False
        self.intervention.plot_days = []


class HighLockdown(Case):
    def __init__(self, pars: dict[str, str]):
        super().__init__('Late Lockdown', pars)
        on = 0.2  # Beta less than 1 -- intervention is on
        off = 1.0  # Beta is 1, i.e. normal -- intervention is off
        changes = [on, off]
        self.intervention = cv.change_beta(days=high_thresh, changes=changes)
        self.intervention.t_on = np.nan
        self.intervention.t_off = np.nan
        self.intervention.active = False
        self.intervention.plot_days = []