import numpy as np
import scipy as sp
from matrix import *
from initial_density import *
from entanglement import *
from wigner import wigner
class DensityOperator:
    @classmethod
    def fock(cls,focks,cutoffs):
        rho = sp.sparse.eye(1)
        if isinstance(cutoffs,int):
            cutoffs = np.array([cutoffs],dtype=np.intc)
        for i in range(len(cutoffs)):
            rho = sp.sparse.kron(rho,fock_density(focks[i],cutoffs[i])).tocsc()
        return cls(rho,cutoffs)

    @classmethod
    def thermal(cls,sigmas,cutoffs):
        rho = sp.sparse.eye(1)
        if isinstance(cutoffs,int):
            cutoffs = np.array([cutoffs],dtype=np.intc)
        for i in range(len(cutoffs)):
            rho = sp.sparse.kron(rho,thermal_density(sigmas[i],cutoffs[i])).tocsc()
        return cls(rho,cutoffs)

    @classmethod
    def coherent(cls,alphas,cutoffs):
        rho = sp.sparse.eye(1)
        if isinstance(cutoffs,int):
            cutoffs = np.array([cutoffs],dtype=np.intc)
        for i in range(len(cutoffs)):
            rho = sp.sparse.kron(rho,cross_coherent(alphas[i],alphas[i],cutoffs[i])).tocsc()
        return cls(rho,cutoffs)

    @classmethod
    def jpats(cls,jphoton,sigmas,cutoffs):
        rho = sp.sparse.eye(1)
        if isinstance(cutoffs,int):
            cutoffs = np.array([cutoffs],dtype=np.intc)
        for i in range(len(cutoffs)):
            rho = sp.sparse.kron(rho,jpats_density(jphoton[i],sigmas[i],cutoffs[i])).tocsc()
        return cls(rho,cutoffs)

    @classmethod
    def noisy_fock(cls,focks,sigmas,cutoffs):
        rho = sp.sparse.eye(1)
        if isinstance(cutoffs,int):
            cutoffs = np.array([cutoffs],dtype=np.intc)
        for i in range(len(cutoffs)):
            rho = sp.sparse.kron(rho,noisy_fock_density(focks[i],sigmas[i],cutoffs[i])).tocsc()
        return cls(rho,cutoffs)
    @classmethod
    def kron(cls,rho1,rho2):
        cutoffs = np.concatenate((rho1.form, rho2.form))
        rho = sp.sparse.kron(rho1.rho,rho2.rho)
        return cls(rho,cutoffs)


    @classmethod
    def cat(cls,alpha,cutoff):
        rho = cat_state(alpha,cutoff)
        return cls(rho,[cutoff])
        

    @classmethod
    def bell(cls,dim,cutoff):
        rho = bell(dim,cutoff)
        return cls(rho,[cutoff,cutoff])
    
    @classmethod
    def noon(cls,n,cutoff):
        rho = noon(n,cutoff)
        return cls(rho,np.array([cutoff,cutoff]))

    def __init__(self,rho,cutoffs):
        self.rho = rho
        if isinstance(cutoffs,int):
            self.form = np.array([cutoffs],dtype=np.intc)
        elif isinstance(cutoffs,list):
            self.form = np.array(cutoffs,dtype=np.intc)
        else:
            self.form= cutoffs 
        self.dim = self.form.shape[0]

    def toarray(self):
        if not isinstance(self.rho,np.ndarray):
            return self.rho.toarray()
        return self.rho

    def partial_trace(self, modes):
        if isinstance(modes, int):
            modes = [modes]
        curr_rho = self.toarray()
        curr_form = self.form.copy()
        prod_dim = np.prod(self.form)
        num_dims = self.dim
        modes = np.sort(modes)[::-1] # decreasing
        for mode in modes:
            non_zeros = curr_form[curr_form != 0]
            blockified = curr_rho.reshape(np.concatenate((non_zeros,non_zeros)))
            prod_dim //= curr_form[mode]
            curr_rho = np.trace(blockified,axis1=mode, axis2=mode+num_dims).reshape([prod_dim,prod_dim])
            curr_form[mode] = 0
            num_dims -=1
        return DensityOperator(curr_rho,curr_form[curr_form != 0])

    def trace(self):
        return self.rho.trace()

    def evolve(self,unitary):
        self.rho = unitary.dot(self.rho.dot(unitary.getH()))

    def mean(self):
        mu = np.zeros(2*self.dim,dtype=np.cdouble)
        for i in range(2*self.dim):
            x_op= get_jxop(i,self.dim,self.form)
            mu[i] = self.rho.dot(x_op.toarray()).trace()
        return mu
    
    def variance(self):
        sigma = np.zeros((2*self.dim, 2*self.dim),dtype=np.cdouble)
        x_ops = np.array([get_jxop(i,self.dim,self.form) for i in range(2*self.dim)])
        mu = self.mean()
        for i in range(2*self.dim):
            for j in range(2*self.dim):
                op= 1/2*(x_ops[i].dot(x_ops[j]) + x_ops[j].dot(x_ops[i]))
                sigma[i,j] = self.rho.dot(op.toarray()).trace() - mu[i]*mu[j]
        return sigma

    def entanglement(self,mode):
        rest_modes = [i for i in range(self.dim) if i != mode]
        rho12 = self.partial_trace(rest_modes)
        schmidt = schmidt_coeffs(rho12.rho)
        return entropy(schmidt)

    def fidelity(self,rho):
        op = sp.linalg.sqrtm(np.matmul(self.toarray(), rho.toarray()))
        val = np.real(np.trace(op))
        return val**2
    
    def purity(self):
        return np.trace(self.toarray()**2)

    def negative_volume(self,axes_bound=10,nums=300):
        area = (2*axes_bound/(nums-1))**2
        x,y,z = wigner(self.toarray(),axes_min=-axes_bound,axes_max=axes_bound,axes_steps=nums)
        ans = np.sum(np.abs(z)) *area - 1
        if (ans > 0):
            return ans
        elif (ans > -1e-2):
            return 0
        print('error: low cutoff/low boundry')
        return 0
