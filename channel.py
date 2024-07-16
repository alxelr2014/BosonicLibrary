import numpy as np
import scipy as sp
from symplectic import *
from gaussian_unitary import *
from matrix import *
from density import *

def thermal_unitary_operator(sigma,cutoffs):
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
    thermal = transform(Pi,thermal)
    block1 = thermal[:3,:3]
    block2 = thermal[3:,3:]

    u,p = sp.linalg.polar(block1,side='right')
    v,_ = sp.linalg.polar(block2,side='right')
    r = sp.linalg.logm(p)
    d,t = np.linalg.eigh(r)
    U = np.block([[u,np.zeros((3,3))],[np.zeros((3,3)),v]])
    T = np.block([[t,np.zeros((3,3))],[np.zeros((3,3)),t]])
    D = np.block([[np.diag(d),np.zeros((3,3))],[np.zeros((3,3)),np.diag(-d)]])

    xi = transform(t,np.diag(d))
    phi = passive_operator(transform(Pi.T,U),3)

    sq = GaussianUnitary(xi,'S',cutoffs)
    phase = GaussianUnitary(phi,'P',cutoffs)
    return combine([phase,sq])

def thermal_channel(rho,sigma,cutoff):
    cutoff_env = cutoff*np.ones(2,dtype=np.intc)
    cutoffs= np.concatenate((rho.form, cutoff_env))

    unitary = thermal_unitary_operator(sigma,cutoffs)
    ancilla = DensityOperator.fock([0,0],cutoff_env)
    rho_in = DensityOperator.kron(rho,ancilla)
    rho_in.evolve(unitary)
    rho_out = rho_in.partial_trace([1,2])
    return rho_out

  
def partial_thermal_channel(rho,sigma,cutoff):
    cutoff_env = cutoff*np.ones(2,dtype=np.intc)
    cutoffs= np.concatenate((np.array([rho.form[1]]), cutoff_env))

    unitary = sp.sparse.kron(np.eye(rho.form[0]), thermal_unitary_operator(sigma,cutoffs))
    ancilla = DensityOperator.fock_list([0,0],cutoff_env)  
    rho_in = DensityOperator.kron(rho,ancilla)
    rho_in.evolve(unitary)
    rho_out = rho_in.partial_trace([2,3])
    return rho_out

def dep_thermal_channel(rho,dim, sigma,cutoff):
    s = np.sqrt(2)*sigma
    thermal = np.array([
        [1,0,0,s,0,0],
        [0,1,0,0,s,0],
        [0,0,0,0,1,0],
        [s,0,0,0,0,1],
        [0,s,1,0,s*s,0],
        [0,0,0,1,0,0]])


    # need to be checked
    omega = np.array([[0,1],[-1,0]])
    omega_curr =np.kron(np.eye(3*dim),omega)
    thermal = np.kron(thermal,np.eye(dim))
    S = SymplecticTransformation(thermal,np.zeros(3*dim),omega_curr)
    d,p2,s,p1 = S.bloch_messiah_operators()

    cutoff_sys = cutoff*np.ones(2*dim,dtype=np.intc)
    cutoffs= np.concatenate((rho.form, cutoff_sys))

    unid = GaussianUnitary(d,'D',cutoffs)
    unip2 = GaussianUnitary(p2,'P',cutoffs)
    unis = GaussianUnitary(s,'S',cutoffs)
    unip1 = GaussianUnitary(p1,'P',cutoffs)
    ancilla = DensityOperator.fock_list([0,0],cutoff_sys)
    rho_in = DensityOperator.kron(rho,ancilla)
    Unitary = combine([unid, unip2, unis, unip1])
    rho_in.evolve(Unitary)
    rho_out = rho_in.partial_trace([1,2])
    return rho_out

def displacement(rho,dim, alpha,cutoff):
    omega = np.array([[0,1],[-1,0]])
    omega_curr =np.kron(np.eye(dim),omega)
    S = SymplecticTransformation(np.eye(2*dim),alpha,omega_curr)
    d,p2,s,p1 = S.bloch_messiah_operators()
    unid = GaussianUnitary(d,'D',cutoff)
    unip2 = GaussianUnitary(p2,'P',cutoff)
    unis = GaussianUnitary(s,'S',cutoff)
    unip1 = GaussianUnitary(p1,'P',cutoff)
    rho_in = rho
    Unitary = combine([unid, unip2, unis, unip1])
    rho_in.evolve(Unitary)
    return rho_in