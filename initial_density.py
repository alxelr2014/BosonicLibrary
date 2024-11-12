import numpy as np
import scipy as sp
from matrix import *
from helper import *

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


def cross_summations(sigma,k,m,n):
    term1 =np.logspace(0,k+1,num=k+1,base=(sigma**2)/(1+sigma**2),endpoint=False)
    term2 = fbinomials(k,k)
    term3 =np.logspace(0,n+1,num=n+1,base=(sigma**2),endpoint=False)
    term4 = fbinomials(m,n)
    term5 = 1/(sigma**2+1)* np.logspace(0,k+1,num=k+1,base=1/(sigma**2 + 1), endpoint=False)
    l = min(k,n)
    kterms = ((term1 * term2)[::-1] * term5)[:l+1]
    nterms = (term3 * term4)[::-1][:l+1]
    return np.dot(kterms,nterms)


def noisy_cross_fock_density(n,m,sigma,cutoff=5):
    if m < n:
        return np.conj(noisy_cross_fock_density(m,n,sigma,cutoff).T)
    coeff = 1/(np.sqrt(div_factorial(n,m-n))*((1+sigma**2)**(m)))
    term1 = np.array([cross_summations(sigma,k,m,n) for k in range(cutoff)])
    term2 = np.sqrt(div_factorials(cutoff,m-n))
    data = coeff*(term1 * term2)
    return sp.sparse.spdiags(data,diags=n-m,m=cutoff,n=cutoff).tocsc()
    # *(fbinomials(i,i) * np.logspace(0,i,num=i+1,base=(sigma**2),endpoint=True)[::-1])    


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
    if n==0:
        rho = sp.sparse.kron(cross_fock(0,0,cutoff),cross_fock(0,0,cutoff))
    else:
        rho1= sp.sparse.kron(cross_fock(0,0,cutoff), cross_fock(n,n,cutoff))
        rho2= sp.sparse.kron(cross_fock(0,n,cutoff), cross_fock(n,0,cutoff))
        rho3= sp.sparse.kron(cross_fock(n,0,cutoff), cross_fock(0,n,cutoff))
        rho4= sp.sparse.kron(cross_fock(n,n,cutoff), cross_fock(0,0,cutoff))
        rho = 1/2*(rho1+rho2 + rho3+rho4)
    return rho.tocsc()

def noisy_noon(n,sigma_1,sigma_2,cutoff_1,cutoff_2):
    if n == 0:
        rho = sp.sparse.kron(noisy_cross_fock_density(0,0,sigma_1,cutoff_1), noisy_cross_fock_density(0,0,sigma_2,cutoff_2))
    else:
        rho1= sp.sparse.kron(noisy_cross_fock_density(0,0,sigma_1,cutoff_1), noisy_cross_fock_density(n,n,sigma_2,cutoff_2))
        rho2= sp.sparse.kron(noisy_cross_fock_density(0,n,sigma_1,cutoff_1), noisy_cross_fock_density(n,0,sigma_2,cutoff_2))
        rho3= sp.sparse.kron(noisy_cross_fock_density(n,0,sigma_1,cutoff_1), noisy_cross_fock_density(0,n,sigma_2,cutoff_2))
        rho4= sp.sparse.kron(noisy_cross_fock_density(n,n,sigma_1,cutoff_1), noisy_cross_fock_density(0,0,sigma_2,cutoff_2))
        rho = 1/2*(rho1+rho2 + rho3+rho4)
    # print(rho.toarray())
    return rho.tocsc()

def factorials(cutoff):
    vec = np.zeros(cutoff)
    vec[0] = 1
    for i in range(cutoff-1):
        vec[i+1]=vec[i]/(i+1)
    return vec

def div_factorial(n,d): # return (n+d)!/n!
    return np.prod(range(n+1,n+d+1))


def div_factorials(n,d): # return (k+d)!/k!  for k = 0 to n-1
    vec = np.zeros(n)
    vec[0] = div_factorial(0,d)
    for i in range(n-1):
        vec[i+1]=vec[i]/(i+1)*(i+d+1)
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

def char_to_density(chi,cutoffs,axis_min=-6,axis_max=6,axis_num = 200):
    d = cutoffs.shape[0]
    rho = np.array([[1]],dtype=np.cdouble)
    coords = []
    for i in range(d):
        rho = np.kron(rho,np.zeros((cutoffs[i],cutoffs[i])))
        coords.append(np.array(range(cutoffs[i])))
    mesh_coord = np.meshgrid(*coords)
    indicies = np.vstack(list(map(np.ravel, mesh_coord))).T
    def integrand(n,m):
        def temp(z):
            return chi(z)*fock_real_displacement(z,n,m)
        return temp
    for v in indicies:
        for u in indicies:
            rho[v,u] = 1/((np.pi)**d)* real_integration(axis_min*np.ones(2*d),axis_max*np.ones(2*d),axis_num*np.ones(2*d),integrand(v,u))
    return rho
