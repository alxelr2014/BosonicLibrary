import numpy as np
from density import *
from wigner import *
from channel import *

alpha = 2.5
cutoff = 25
rho = DensityOperator.cat(alpha,cutoff)
plot_wigner(rho.rho,dim=2,file='cat_init')
print('Hi')
rho_out = thermal_channel(rho,1,0.8,cutoff)
plot_wigner(rho_out.rho,dim=2,file='cat_out')
