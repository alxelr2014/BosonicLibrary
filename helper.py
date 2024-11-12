import numpy as np
from scipy.special import genlaguerre

def factorial(n):
    return np.prod(np.array(range(1,n+1)))

def ent(t):
    return -t*np.log(t+1e-24)

def complex_integration(axes_min,axes_max,axes_nums,f):
    d = len(axes_min)
    axes_lengths = (axes_max-axes_min)/(axes_nums)
    alpha = []
    for _i in range(d//2):
        xvec = np.linspace(axes_min[2*_i], axes_max[2*_i], int(axes_nums[2*_i]),endpoint=False)
        yvec = np.linspace(axes_min[2*_i+1], axes_max[2*_i+1], int(axes_nums[2*_i+1]),endpoint=False)
        reals,imags = np.meshgrid(xvec,yvec)
        alpha.append((reals + 1j*imags).flatten())
    mesh_coord = np.meshgrid(*alpha)
    positions = np.vstack(list(map(np.ravel, mesh_coord))).T
    vals = np.apply_along_axis(f,1,positions)
    area = np.prod(axes_lengths)
    return area*np.sum(vals)

def real_integration(axes_min,axes_max,axes_nums,f):
    d = len(axes_min)
    axes_lengths = (axes_max-axes_min)/(axes_nums)
    alpha = []
    for _i in range(d):
        xvec = np.linspace(axes_min[_i], axes_max[_i], int(axes_nums[_i]),endpoint=False)
        alpha.append(xvec.flatten())
    mesh_coord = np.meshgrid(*alpha)
    positions = np.vstack(list(map(np.ravel, mesh_coord))).T
    vals = np.apply_along_axis(f,1,positions)
    area = np.prod(axes_lengths)
    return area*np.sum(vals)

def gaussian_characteristic(mean,variance):
    d = mean.shape[0]
    omega = np.kron(np.eye(d//2),np.array([[0,1],[-1,0]]))
    def char(z):
        term0 = np.matmul(omega.T,z)
        term1 = -1/2* np.dot(term0, np.matmul(variance,term0))
        term2 = -1j*np.dot(term0,mean)
        return np.exp(term1 + term2)
    return char



def sqrt_fact(n,m):
    if n >= m:
        return np.sqrt(np.prod(np.array(range(m+1,n+1))))
    return sqrt_fact(m,n)


def g(xi,n,m):
    if n >= m:
        return xi
    else:
        return np.conj(-xi)

def fock_complex_displacement(xi,n,m):
    prod = 1
    d = xi.shape[0]
    for i in range(d):
        term1 = np.exp(- (np.abs(xi[i])**2)/2)
        term2 = sqrt_fact(n[i],m[i])
        term3 = g(xi[i],n[i],m[i])**(np.abs(n[i]-m[i]))
        term4 = genlaguerre(np.min([n[i],m[i]]),np.abs(n[i]-m[i]))(np.abs(xi[i])**2)
        prod *= term1 * term2 * term3 * term4
    return prod

def fock_real_displacement(z,n,m):
    d = z.shape[0]//2
    V_trans = np.kron(np.eye(d),np.array([[1,1j]]))
    xi = np.matmul(V_trans,z)
    return fock_complex_displacement(xi,n,m)
