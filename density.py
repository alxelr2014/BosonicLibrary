import numpy as np
import scipy as sp
from matrix import *
from initial_density import *
from entanglement import *
from wigner import wigner
from helper import *
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
    def cross_fock(cls,fock1,fock2,cutoff):
        rho = cross_fock(fock1,fock2,cutoff)
        return cls(rho,cutoff)

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
    def gaussian(cls,mean,variance,cutoffs):
        chi = gaussian_characteristic(mean,variance)
        rho = char_to_density(chi,cutoffs)
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

    @classmethod
    def noisy_noon(cls,n,sigma_1,sigma_2,cutoff_1,cutoff_2):
        rho = noisy_noon(n,sigma_1,sigma_2,cutoff_1,cutoff_2)
        return cls(rho,np.array([cutoff_1,cutoff_2]))

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


    def partial_transpose(self,modes):
        "taken from qutip"
        dim = [np.array([k,k]) for k in self.form]
        modes = [int(k in modes) for k in range(self.dim)]
        pt_dims = np.arange(2 * self.dim).reshape(2, self.dim).T
        pt_idx = np.concatenate([[pt_dims[n, modes[n]] for n in range(self.dim)],
                                [pt_dims[n, 1 - modes[n]] for n in range(self.dim)]])
        rho = self.toarray()
        return DensityOperator(rho.reshape(
            np.array(dim).flatten()).transpose(pt_idx).reshape(rho.shape),self.form)



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
    
    def photon_statistics(self):
        def temp(n):
            fock_n = DensityOperator.fock(n,self.form)
            return np.trace(np.matmul(fock_n.toarray(), self.toarray()))
        return temp
    
    def first_photon_statistics(self):
        def temp(n):
            rest_dim = np.prod(self.form[1:])
            sub_block = self.toarray()[rest_dim*n:rest_dim*(n+1), rest_dim*n:rest_dim*(n+1)]
            return np.trace(sub_block)
        return temp
    
    def second_photon_statistics(self):
        def temp(n):
            return self.partial_trace(0).toarray()[n,n]
        return temp


    def purity(self):
        return np.trace(self.toarray()**2)

    def negative_volume(self,axes_bound=10,axes_nums=300):
        area = (2*axes_bound/(axes_nums-1))**2
        x,y,z = wigner(self.toarray(),axis_min=-axes_bound,axis_max=axes_bound,axis_nums=axes_nums)
        ans = np.sum(np.abs(z)) *area - 1
        if (ans > 0):
            return ans
        elif (ans > -1e-2):
            return 0
        print('error: low cutoff/low boundry')
        return 0

    def Q_function(self):
        def temp(alphas):
            if (len(self.form) != len(alphas)):
                print(len(alphas) , len(self.form))
                return 0
            v = np.array([1])
            for i in range(len(self.form)):
                v =np.kron(v,coherent(alphas[i],self.form[i]).T)
            return (1/np.pi)**(self.dim)*np.real(np.dot(np.conj(v),np.matmul(self.toarray(),v)))
        return temp
    
    def dep_Wehrl_entropy(self,axes_min= -6, axes_max= 6,axes_nums= 200):
        axes_area = (((axes_max-axes_min)/(axes_nums))**2)**(self.dim)
        xvec = np.linspace(axes_min, axes_max, axes_nums,endpoint=False)
        reals,imags = np.meshgrid(xvec,xvec)
        alpha = reals + 1j*imags
        alphas = np.stack([alpha for i in range(self.dim)],axis=-1)
        sh = np.array(list(alphas.shape)[0:-1])
        alphas = np.reshape(alphas,(np.prod(sh),1))
        q = self.Q_function()
        # for a in range(len(alphas)):
        #     for b in range(len(alphas[a])):
        #         t = q(alphas[a][b])
        #         if t < 1e-15: 
        #             q_vals[a][b] =0
        #         else:
        #             q_vals[a][b]= -t*np.log(t)
        def went(alpha):
            pass 
        q_vals = np.apply_along_axis(lambda alpha: q(alpha) ,1,alphas)
        q_vals = -q_vals*np.log(q_vals+1e-20*np.ones(q_vals.shape))
        return np.sum(q_vals)*axes_area 
    
    def Wehrl_entropy(self,axis_min= -6, axis_max= 6,axis_num= 200):
        axes_min = axis_min*np.ones(2*self.dim)
        axes_max = axis_max*np.ones(2*self.dim)
        axes_num = axis_num*np.ones(2*self.dim)
        q = self.Q_function()        
        return complex_integration(axes_min,axes_max,axes_num,lambda alpha:ent(q(alpha)))


    def gaussianity(self,axes_bound= 6,axes_nums= 200):
        det_var = 1/2*np.log(np.real(np.linalg.det(self.variance()+ np.eye(2*self.dim))))
        constant = self.dim * np.log(np.pi*np.e/2)
        entropy = self.Wehrl_entropy(axis_min=-axes_bound,axis_max=axes_bound,axis_num=axes_nums)
        return det_var + constant - entropy

    def entangle_negativity(self,modes):
        return (np.linalg.norm(self.partial_transpose(modes).toarray(),ord='nuc')-1)/2


    def entangle_log_negativity(self,modes):
        return np.log2(np.linalg.norm(self.partial_transpose(modes).toarray(),ord='nuc'))

    def entangle_relent(self,num_tests = 5000,ksep = 18):
        return relative_ent(self.toarray(),self.form[0],self.form[1],num_tests,ksep)

    def entangle_wang(self):
        return wang_entangle(self.toarray(),self.form)
