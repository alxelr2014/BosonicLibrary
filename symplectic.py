import numpy as np
import scipy as sp
from matrix import *

class SymplecticTransformation:
    def __init__(self,symplec,displace,commutation):
        self.symplec = symplec
        self.commutation = commutation
        self.dim = symplec.shape[0]
        self.half_dim = self.dim//2
        self.displace =displace

    def original_commutation(self):
        om = np.array([[0,1],[-1,0]])
        return np.kron(om,np.eye(self.half_dim))


    def bloch_messiah_decomposition(self):
        q = commutation_correspondence(self.commutation,self.original_commutation()).real
        T = transform(q,self.symplec)
        u,p = sp.linalg.polar(T,side='right')
        o,d= symplectic_diagonalization(p,self.half_dim)
        op = np.matmul(o.T,u)
        return transform(q.T,o),transform(q.T,d), transform(q.T,op)

    def bloch_messiah_operators(self):
        o,d,op = self.bloch_messiah_decomposition()
        phi2 = passive_operator(o,self.half_dim)        
        xi = active_operator(d,self.half_dim)
        phi1 = passive_operator(op, self.half_dim)
        return self.displace,phi2,xi,phi1
    
    def test_block_messiah_decomposition(self):
        phi2,xi,phi1 = self.bloch_messiah_decomposition()
        return np.all(np.isclose(np.matmul(phi2,np.matmul(xi,phi1)),self.symplec))
