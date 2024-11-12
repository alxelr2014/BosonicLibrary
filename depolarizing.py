import numpy as np


def pauli_i():
    return np.array([[1,0],[0,1]])
def pauli_x():
    return np.array([[0,1],[1,0]])

def pauli_y():
    return np.array([[0,-1j],[1j,0]])

def pauli_z():
    return np.array([[1,0],[0,-1]])

def apply_kraus(rho,k):
    return np.matmul(k,np.matmul(rho, np.conj(k.T)))

def channel(rho,kraus):
    ans = np.zeros(rho.shape,dtype=np.cdouble)
    for k in kraus:
        ans += apply_kraus(rho,k)
    return ans

def depolarizing_kraus(p):
    return [np.sqrt(1-3*p/4)*pauli_i(), np.sqrt(p/4)*pauli_x(),np.sqrt(p/4)*pauli_y(),np.sqrt(p/4)*pauli_z()]

def identity_kraus():
    return pauli_i()

def bipartite_gate(kraus1,kraus2):
    bipartite_kraus = []
    for k1 in kraus1:
        for k2 in kraus2:
            bipartite_kraus.append(np.kron(k1,k2))
    return bipartite_kraus

# rho = 1/2*np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
# dep_channel = bipartite_gate(depolarizing_kraus(0.5), depolarizing_kraus(0.5))
# print(channel(rho,dep_channel))
# print(np.trace(channel(rho,dep_channel)))

rho = 1/2*np.array([[1,1],[1,1]])
dep_channel = depolarizing_kraus(0.5)
print(channel(rho,dep_channel))