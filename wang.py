import cvxpy as cp
import numpy as np

def wang_entangle(rho,form):
    dims = form[0]*form[1]
    X = cp.Variable((dims,dims), complex=True)
    # The operator >> denotes matrix inequality.
    constraints = [X - rho >> 0]
    prob = cp.Problem(cp.Minimize(cp.normNuc(cp.partial_transpose(X,form,1))),
                    constraints)
    prob.solve()
    return np.log2(prob.value)


def bra(vec):
    return np.conj(vec.T)

def ketone(d):
    result = np.zeros((d,1))
    result[1] = 1
    return result
    
def ketzero(d):
    result = np.zeros((d,1))
    result[0] = 1
    return result


di = [10,10]
rho1 = np.kron(np.matmul(ketzero(di[0]),bra(ketzero(di[0]))),np.matmul(ketone(di[1]),bra(ketone(di[1]))))
bell = 1/np.sqrt(2)*(np.kron(ketzero(di[0]),ketone(di[1])) + np.kron(ketone(di[0]),ketzero(di[1])))
rho2 = np.matmul(bell,bra(bell))
print(wang_entangle(rho2,di))