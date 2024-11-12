import numpy as np
import scipy as sp
import cvxpy as cp


def schmidt_coeffs(rho):
    u,s,vh = np.linalg.svd(rho,full_matrices=True)
    return s


def f(x):
    if x <= 1e-5:
        return 0
    return -x*np.log2(x)

def entropy(probs):
    return sum(np.vectorize(f)(probs))


def random_density(dim):
    lambdas = np.random.rand(dim)
    lambdas = np.diag(1/np.sum(lambdas)*lambdas)
    u = sp.stats.unitary_group.rvs(dim)
    return np.matmul(u,np.matmul(lambdas,np.conj(u.T)))

def random_bipartite(dim_a, dim_b,k):
    probs = np.random.rand(k)
    probs = 1/np.sum(probs)*probs
    rho = np.zeros((dim_a*dim_b,dim_a*dim_b),dtype=np.cdouble)
    for i in range(k):
        rho += probs[i]* np.kron(random_density(dim_a), random_density(dim_b))
    return rho


def relative_entropy(rho,sigma):
    rho_vals,rho_vec = np.linalg.eigh(rho)
    sigma_vals,sigma_vec = np.linalg.eigh(sigma)
    rho_vals = np.diag(np.log(np.abs(rho_vals) + 1e-20*np.ones(rho_vals.shape)))
    sigma_vals = np.diag(np.log(np.abs(sigma_vals) + 1e-20*np.ones(sigma_vals.shape)))
    log_rho = np.matmul(rho_vec,np.matmul(rho_vals,np.conj(rho_vec.T)))
    log_sigma = np.matmul(sigma_vec,np.matmul(sigma_vals,np.conj(sigma_vec.T)))
    return np.real(np.trace(np.matmul(rho,log_rho-log_sigma)))


def relative_ent(rho,dim_a,dim_b,num_tests,k):
    vals = np.zeros(num_tests)
    for i in range(num_tests):
        sigma = random_bipartite(dim_a,dim_b,k)
        vals = relative_entropy(rho,sigma)
    return np.min(vals)

    rho_vals,rho_vec = np.linalg.eigh(rho)


def expected(pr,sup,f):
    # print(np.vectorize(f)(sup))
    return np.sum(np.multiply(pr,np.vectorize(f)(sup)))


def correlation(pr,suppx,suppy):
    suppxy = np.array([np.array([_*__ for _ in suppx]) for __ in suppy])
    prxy = np.array([np.array([pr([_,__]) for _ in suppx]) for __ in suppy])
    prx = np.sum(prxy,axis=0)
    pry = np.sum(prxy,axis=1)
    Exy = expected(prxy,suppxy,lambda x:x)
    Ex = expected(prx,suppx,lambda x:x)
    Ey = expected(pry, suppy, lambda x:x)
    Ex2 = expected(prx, suppx, lambda x:x**2)
    Ey2 = expected(pry, suppy, lambda x:x**2)
    return (Exy - Ex*Ey)/(np.sqrt((Ex2 - Ex**2)*(Ey2 - Ey**2)))
    # Exy = np.sum(np.array([suppxy[i][0]*suppxy[i][1]*prxy[i] for i in range(len(suppxy))]))
    # Ex = 


def wang_entangle(rho,form):
    dims = form[0]*form[1]
    X = cp.Variable((dims,dims), complex=True)
    # The operator >> denotes matrix inequality.
    constraints = [X - rho >> 0]
    prob = cp.Problem(cp.Minimize(cp.normNuc(cp.partial_transpose(X,form,1))),
                    constraints)
    prob.solve()
    return np.log2(prob.value)

