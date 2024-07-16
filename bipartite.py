import numpy as np
from density import *
from wigner import *
from channel import *

#print(cross_fock(2,2,3).toarray())
#print(bell(2,4).toarray())
# plot_wigner(DensityOperator.bell(3,5).rho,dim=2)
plot_wigner(DensityOperator.noon(1,6).partial_trace(1).rho,dim=2)