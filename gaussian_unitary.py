import numpy as np
import scipy as sp
from matrix import *
class GaussianUnitary:
    def __init__(self,generator,gtype,cutoffs):
        self.generator = generator
        self.type = gtype
        self.cutoffs=cutoffs
        self.num_modes = self.generator.shape[0]

    def to_unitary(self):
        if self.type == 'D':
            return displacement(self.generator,self.num_modes,self.cutoffs)
        elif self.type == 'S':
            return squeeze(self.generator,self.num_modes,self.cutoffs)
        elif self.type == 'P':
            return phase(self.generator,self.num_modes,self.cutoffs)

def combine(lst):
    curr = identity(lst[0].cutoffs)
    for gu in lst:
        curr = curr.dot(gu.to_unitary())
    return curr