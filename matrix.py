import numpy as np
import scipy as sp

def commutation_correspondence(omega1,omega2):
    _,v1 = sp.linalg.eig(omega1)
    _,v2 = sp.linalg.eig(omega2)
    return np.matmul(v2,np.conj(v1).T)

def transform(similarity,matrix):
    return np.matmul(similarity,np.matmul(matrix,np.conj(similarity.T)))

def partitions(matrix,half_dim):
    A = matrix[:half_dim,:half_dim]
    B = matrix[:half_dim,half_dim:]
    C= matrix[half_dim:,half_dim:]
    return A,B,C

def takagi_decomposition(matrix):
    u,d,vh = np.linalg.svd(matrix)
    temp = sp.linalg.sqrtm(np.matmul(np.conj(u.T),vh.T))
    return np.diag(d), np.matmul(u,temp)


def symplectic_diagonalization(symplec,half_dim):
    a,b,c = partitions(symplec,half_dim)
    M = 1/2*(a-c+1j*(b+b.T))
    l,w = takagi_decomposition(M)
    o = np.block([[w.real,-w.imag],[w.imag,w.real]])
    g =  sp.linalg.sqrtm(np.eye(half_dim)+l*l)
    d = np.block([[  l+g ,np.zeros((half_dim,half_dim))],[np.zeros((half_dim,half_dim)),g-l]])
    return o,d
def is_symplectic(symplec,omega):
    temp= np.matmul(symplec,np.matmul(omega,symplec.T))-omega
    return np.all(np.isclose(temp,np.zeros(temp.shape)))

def V_matrix(half_dim):
    a = np.zeros(2*half_dim,dtype=np.cdouble)
    a[0]=1
    a[1] = 1j
    return np.array([np.roll(a,2*k) for k in range(half_dim)])

def passive_operator(ortho,half_dim):
    V = V_matrix(half_dim)
    pi = np.block([[V.real],[V.imag]])
    E = 1/2*  transform(V,ortho)
    phi= -1j*sp.linalg.logm(E)
    return phi

def active_operator(diags,half_dim):
    V = V_matrix(half_dim)
    pi = np.block([[V.real],[V.imag]])
    S = transform(pi,diags)
    r = sp.linalg.logm(S[:half_dim,:half_dim])
    return r


def is_unitary(uni):
    temp=  uni.dot(uni.getH())
    return np.abs(temp- np.eye(temp.shape[0])).max() < 1e-1

def get_aop(cutoff=5):
    data = np.sqrt(range(cutoff))
    return sp.sparse.spdiags(
        data=data, diags=[1], m=len(data), n=len(data)).tocsc()

def get_jaop(j,ndim,cutoffs):
    a_op = sp.sparse.eye(1).tocsc()
    for i in range(ndim):
        if i == j:
            a_op = sp.sparse.kron(a_op, get_aop(cutoffs[j])).tocsc()
        else:
            a_op = sp.sparse.kron(a_op, sp.sparse.eye(cutoffs[i])).tocsc()
    return a_op

def get_jxop(j,ndim,cutoffs):
    a_op = get_jaop(j//2,ndim,cutoffs)
    c_op = a_op.getH().tocsc()
    if j%2==0:
        return a_op+c_op
    return -1j*(a_op - c_op)

def phase(phi,ndim,cutoffs):
    an_ops = np.array([get_jaop(j,ndim,cutoffs) for j in range(ndim)])
    cr_ops = np.array([an_ops[j].getH().tocsc() for j in range(ndim)])
    param = np.matmul(cr_ops.T,np.matmul(phi,an_ops))
    return sp.sparse.linalg.expm(1j*param).tocsc()

def squeeze(xi,ndim,cutoffs):
    an_ops = np.array([get_jaop(j,ndim,cutoffs) for j in range(ndim)])
    cr_ops = np.array([an_ops[j].getH().tocsc() for j in range(ndim)])
    param = 1/2*(np.matmul(cr_ops.T,np.matmul(xi,cr_ops))-np.matmul(an_ops.T,np.matmul(np.conj(xi.T),an_ops)))
    return sp.sparse.linalg.expm(param)


def quadratic_hamiltonian(H,ndim,cutoffs):
    an_ops = np.array([get_jaop(j,ndim,cutoffs) for j in range(ndim)])
    cr_ops = np.array([an_ops[j].getH().tocsc() for j in range(ndim)])
    b_ops = np.concatenate((an_ops,cr_ops))
    bd_ops = np.concatenate((cr_ops,an_ops))
    return  sp.sparse.linalg.expm(-1j/2*(np.matmul(bd_ops.T,np.matmul(H,b_ops))))

def displacement(alpha,ndim,cutoffs):
    an_ops = np.array([get_jaop(j,ndim,cutoffs) for j in range(ndim)])
    cr_ops = np.array([an_ops[j].getH().tocsc() for j in range(ndim)])
    param = np.matmul(alpha.T,cr_ops) - np.matmul(np.conj(alpha.T), an_ops)
    return sp.sparse.linalg.expm(param).tocsc()

def identity(cutoffs):
    return sp.sparse.eye(np.prod(cutoffs)).tocsc()
