import numpy as np
import scipy as sp
from matrix import *

def fock_density(i,cutoff=5):
    data = np.zeros(cutoff)
    data[i]=1
    return sp.sparse.spdiags(data,diags=0,m=cutoff,n=cutoff).tocsc()

def thermal_density(sigma, cutoff=5):
    data = 1/(1+sigma**2) *np.logspace(0,cutoff,num=cutoff,base=(sigma**2)/(1+sigma**2),endpoint=False)
    return sp.sparse.spdiags(data,diags=0,m=cutoff,n=cutoff).tocsc()

def jpats_density(j,sigma,cutoff=5):
    un_data = 1/((1+sigma**2)**(j+1)) *np.concatenate((np.zeros(j),
        np.logspace(0,cutoff-j,num=cutoff-j,base=(sigma**2)/(1+sigma**2),endpoint=False)))
    data = un_data*binomials(j,cutoff)
    return sp.sparse.spdiags(data,diags=0,m=cutoff,n=cutoff).tocsc()

def noisy_fock_density(i,sigma,cutoff=5):
    coeff = 1/((1+sigma**2)**(i))*(fbinomials(i,i) * np.logspace(0,i,num=i+1,base=(sigma**2),endpoint=True)[::-1])
    rho = sp.sparse.csc_matrix((cutoff,cutoff))
    for j in range(i+1):
        rho += (coeff[j]*jpats_density(j,sigma,cutoff)).tocsc()
    return rho

def cross_fock(i,j,cutoff=5):
    data = np.zeros(cutoff)
    data[j]=1
    return sp.sparse.spdiags(data,diags=j-i,m=cutoff,n=cutoff).tocsc()


def bell(dim,cutoff=5):
    rho = sp.sparse.csc_matrix((cutoff**2,cutoff**2))
    for i in range(dim):
        for j in range(dim):
            rho += sp.sparse.kron(cross_fock(i,j,cutoff),cross_fock(i,j,cutoff)).tocsc()
    return 1/dim * rho

def noon(n,cutoff=5):
    rho1= sp.sparse.kron(cross_fock(0,0,cutoff), cross_fock(n,n,cutoff))
    rho2= sp.sparse.kron(cross_fock(0,n,cutoff), cross_fock(n,0,cutoff))
    rho3= sp.sparse.kron(cross_fock(n,0,cutoff), cross_fock(0,n,cutoff))
    rho4= sp.sparse.kron(cross_fock(n,n,cutoff), cross_fock(0,0,cutoff))
    rho = 1/2*(rho1+rho2 + rho3+rho4)
    return rho.tocsc()


def factorials(cutoff):
    vec = np.zeros(cutoff)
    vec[0] = 1
    for i in range(cutoff-1):
        vec[i+1]=vec[i]/(i+1)
    return vec

def binomials(j,cutoff): # n choose j for  n = 0 to cutoff-1
    vec = np.zeros(cutoff)
    vec[j] = 1
    for i in range(j,cutoff-1):
        vec[i+1] = (vec[i]*(i+1))/(i+1-j)
    return vec

def fbinomials(n,cutoff): # n choose j for  j = 0 to cutoff
    vec = np.zeros(cutoff + 1)
    vec[0] = 1
    for i in range(cutoff):
        vec[i+1] = (vec[i]*(n-i))/(i+1)
    return vec

def coherent(alpha,cutoff):
    vec = np.exp(-1/2*np.abs(alpha)**2) *np.multiply(np.logspace(0,cutoff,num=cutoff,base=alpha,endpoint=False),np.sqrt(factorials(cutoff)))
    return vec
def cross_coherent(alpha,beta,cutoff):
    return np.outer(coherent(alpha,cutoff), np.conj(coherent(beta,cutoff)))

def cat_state(alpha,cutoff):
    rho1 = cross_coherent(alpha,alpha,cutoff)
    rho2 = cross_coherent(alpha,-alpha,cutoff)
    rho3= cross_coherent(-alpha,alpha,cutoff)
    rho4 = cross_coherent(-alpha,-alpha,cutoff)
    rho = rho1+rho2+rho3+rho4
    rho = 1/(rho.trace()) * rho
    return rho