import numpy as np
from density import *
from wigner import *
from channel import *
from gaussian_unitary import *

def test_squeezed():
    initial = DensityOperator.fock_list([1],[30])
    print(initial.mean())
    print(initial.variance())
    plot_wigner(initial.rho,dim=2,file='graphics//squeezed//initial_1.png')
    U = GaussianUnitary(np.array([[-0.5]]),'S',np.array([30]))
    initial.evolve(combine([U]))
    print(initial.mean())
    print(initial.variance())
    plot_wigner(initial.rho,dim=2,file='graphics//squeezed//squeezed_1.png')

test_squeezed()