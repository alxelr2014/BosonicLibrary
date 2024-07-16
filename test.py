import numpy as np
from density import *
from wigner import *
from channel import *

sigma = 0.01
cutoff_sys = 30
cutoff_env = 15

def test_thermal():
    initial = DensityOperator.fock_list([1],cutoff_sys)
    print(initial.mean())
    print(initial.variance())
    # plot_wigner(initial.rho,dim=2,file='graphics//thermal//initial_3.png')
    ther = thermal_channel(initial,sigma,cutoff_env)
    print(ther.mean())
    print(ther.variance())
    # plot_wigner(ther.rho,dim=2,file='graphics//thermal//out_3.png')
    plot_wigner(ther.rho,dim=2)


def test_entanglement():
    initial = DensityOperator.noon(1,cutoff_sys)
    print(initial.entanglement(0))
    print(initial.mean())
    print(initial.variance())
    ther = partial_thermal_channel(initial,sigma,cutoff_env)
    print(ther.entanglement(0))
    print(ther.mean())
    print(ther.variance())


def actual():
    ther = DensityOperator.thermal_list([sigma],[cutoff])
    print(ther.trace())
    plot_wigner(ther.rho,dim=2,file='actual')

def test_displacement():
    initial = DensityOperator.fock_list([0],cutoff*np.array([1]))
    alpha =np.array([ 1+2j])
    # plot_wigner(initial.rho,dim=2,file='initial')
    displ = displacement(initial,1,alpha,cutoff)
    # plot_wigner(displ.rho,dim=2,file='test')
    print(displ.mean().round(8))
    print(displ.variance().round(8))


def test_fidelity():
    initial1= DensityOperator.coherent([1],cutoff_sys)
    initial2= DensityOperator.coherent([2],cutoff_sys)
    print(initial1.fidelity(initial2))

def test_jpats():
    initial = DensityOperator.jpats([2],[1],[cutoff_sys])
    print(initial.mean())
    print(initial.variance())
    print(np.trace(initial.rho.toarray()))
    plot_wigner(initial.rho,dim=2,file='graphics//thermal//initial_4.png')

def test_noisy_fock():
    sigma = 4
    cutoff=60
    initial = DensityOperator.noisy_fock([1],[sigma],[cutoff])
    print(initial.mean())
    print(initial.variance())
    print(np.trace(initial.rho.toarray()))
    plot_wigner(initial.rho,dim=2)


def test_thermal_theo():
    initial = DensityOperator.fock([2],cutoff_sys)
    print(initial.mean())
    print(initial.variance())
    plot_wigner(initial.rho,dim=2,file='graphics//thermal//initial_6.png')
    ther = thermal_channel(initial,sigma,cutoff_env)
    print(ther.mean())
    print(ther.variance())
    print(ther.trace())
    plot_wigner(ther.rho,dim=2,file='graphics//thermal//out_6.png')
    theory = DensityOperator.noisy_fock([2],[sigma],[cutoff_sys])
    print(theory.mean())
    print(theory.variance())
    print(theory.trace())
    plot_wigner(theory.rho,dim=2,file='graphics//thermal//theory_6.png')
    print(theory.fidelity(ther))


def test_purity():
    theory = DensityOperator.noisy_fock([2],[sigma],[cutoff_sys])
    print(theory.purity())


def test_negative_volume():
    sigma = 2
    cutoff=110
    initial = DensityOperator.noisy_fock([10],[sigma],[cutoff])
    print(initial.mean())
    print(initial.variance())
    print(np.trace(initial.rho.toarray()))
    print(initial.negative_volume(axes_bound=15))
    plot_wigner(initial.rho,dim=2)

# test_displacement()
test_negative_volume()
# test_entanglement()
