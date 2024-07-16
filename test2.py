import numpy as np
from density import *
from wigner import *
from channel import *

sigma = 1
cutoff_sys = 15
cutoff_env = 12


def new_thermal_unitary(sigma,cutoffs):
    t = np.sqrt(2)*sigma
    thermal = np.array([
        [1, 0,  0,  0,  t,  0],
        [0, 1,  0,  t,  0,  0],
        [0, 0,  0,  0, -1,  0],
        [0, t,  0,sigma*sigma,  0,  -1],
        [t, 0,  -1, 0,sigma*sigma,  0],
        [0, 0,  0,  -1, 0,  0]
    ])
    Pi = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    bogo_thermal = transform(Pi,thermal)
    block1 = bogo_thermal[:3,:3]
    block2 = bogo_thermal[:3,3:]
    block3 = bogo_thermal[3:,:3]
    block4 = bogo_thermal[3:,3:]

    A =1/2* (block1 + block4 + 1j*(block3-block2))
    B =1/2* (block1 - block4 + 1j*(block3+block2))

    bogo = np.block([[A,B],[np.conj(B),np.conj(A)]])
    Omega2 = -1j* np.kron(np.array([[-1,0],[0,1]]),np.eye(3))
    hamiltonian = np.matmul(Omega2, sp.linalg.logm(bogo))
    # u,p = sp.linalg.polar(block1,side='right')
    # v,_ = sp.linalg.polar(block2,side='right')
    # r = sp.linalg.logm(p)
    # d,t = np.linalg.eigh(r)
    # U = np.block([[u,np.zeros((3,3))],[np.zeros((3,3)),v]])
    # T = np.block([[t,np.zeros((3,3))],[np.zeros((3,3)),t]])
    # D = np.block([[np.diag(d),np.zeros((3,3))],[np.zeros((3,3)),np.diag(-d)]])

    # xi = transform(t,np.diag(d))
    # phi = passive_operator(transform(Pi.T,U),3)

    # sq = GaussianUnitary(xi,'S',cutoffs)
    # phase = GaussianUnitary(phi,'P',cutoffs)
    return quadratic_hamiltonian(hamiltonian,3,cutoffs)

def new_thermal(rho,sigma,cutoff):
    cutoff_env = cutoff*np.ones(2,dtype=np.intc)
    cutoffs= np.concatenate((rho.form, cutoff_env))

    unitary = new_thermal_unitary(sigma,cutoffs)
    ancilla = DensityOperator.fock_list([0,0],cutoff_env)
    rho_in = DensityOperator.kron(rho,ancilla)
    rho_in.evolve(unitary)
    rho_out = rho_in.partial_trace([1,2])
    return rho_out


def test_thermal():
    initial = DensityOperator.fock_list([1],cutoff_sys)
    print(initial.mean())
    print(initial.variance())
    # plot_wigner(initial.rho,dim=2,file='graphics//thermal//initial_3.png')
    ther = new_thermal(initial,sigma,cutoff_env)
    print(ther.mean())
    print(ther.variance())
    # plot_wigner(ther.rho,dim=2,file='graphics//thermal//out_3.png')
    plot_wigner(ther.rho,dim=2)

test_thermal()