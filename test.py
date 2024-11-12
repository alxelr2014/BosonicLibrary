import numpy as np
from density import *
from wigner import *
from channel import *
import time

sigma = 2
cutoff_sys = 30
cutoff_env = 20

def test_thermal():
    initial = DensityOperator.noisy_fock([1],[sigma],cutoff_sys)
    print(initial.mean())
    print(initial.variance())
    plot_wigner(initial.rho,dim=2)
    # ther = thermal_channel(initial,sigma,cutoff_env)
    # print(ther.mean())
    # print(ther.variance())
    # # plot_wigner(ther.rho,dim=2,file='graphics//thermal//out_3.png')
    # plot_wigner(ther.rho,dim=2)

def test_noon_thermal():
    sigma = 1
    cutoff_sys =40
    n = 10
    initial = DensityOperator.noisy_noon(n,sigma,0,cutoff_sys,cutoff_sys)
    # initial_2 = DensityOperator.noon(n,cutoff_sys)
    # print(np.sum(initial_1.toarray()-initial_2.toarray()))
    print(initial.entangle_negativity([0]))
    # print(noisy_fock_density(1,sigma,cutoff_sys))
    # print((noisy_fock_density(n,sigma,cutoff_sys)-noisy_cross_fock_density(n,n,sigma,cutoff_sys)).toarray())

def test_entanglement():
    initial = DensityOperator.noon(1,cutoff_sys)
    print(initial.entangle_negativity([0]))
    print(initial.mean())
    print(initial.variance())
    ther = partial_thermal_channel(initial,sigma,cutoff_env)
    print(ther.entangle_negativity([0]))
    print(ther.mean())
    print(ther.variance())


def actual():
    ther = DensityOperator.thermal_list([sigma],[cutoff])
    print(ther.trace())
    plot_wigner(ther.rho,dim=2,file='actual')

def test_displacement():
    initial = DensityOperator.fock_list([0],cutoff*np.array([1]))
    alpha =np.array([ 1+2j])
    # plot_wigner(initial.rho,dim=2,file='initial')
    displ = displacement(initial,1,alpha,cutoff)
    # plot_wigner(displ.rho,dim=2,file='test')
    print(displ.mean().round(8))
    print(displ.variance().round(8))


def test_fidelity():
    initial1= DensityOperator.coherent([1],cutoff_sys)
    initial2= DensityOperator.coherent([2],cutoff_sys)
    print(initial1.fidelity(initial2))

def test_jpats():
    initial = DensityOperator.jpats([2],[1],[cutoff_sys])
    print(initial.mean())
    print(initial.variance())
    print(np.trace(initial.rho.toarray()))
    plot_wigner(initial.rho,dim=2,file='graphics//thermal//initial_4.png')

def test_noisy_fock():
    sigma = 1
    cutoff=60
    initial = DensityOperator.noisy_fock([50],[sigma],[cutoff])
    print(initial.mean())
    print(initial.variance())
    print(np.trace(initial.rho.toarray()))
    plot_wigner(initial.rho,dim=2)


def test_thermal_theo():
    initial = DensityOperator.fock([2],cutoff_sys)
    print(initial.mean())
    print(initial.variance())
    plot_wigner(initial.rho,dim=2,file='graphics//thermal//initial_6.png')
    ther = thermal_channel(initial,sigma,cutoff_env)
    print(ther.mean())
    print(ther.variance())
    print(ther.trace())
    plot_wigner(ther.rho,dim=2,file='graphics//thermal//out_6.png')
    theory = DensityOperator.noisy_fock([2],[sigma],[cutoff_sys])
    print(theory.mean())
    print(theory.variance())
    print(theory.trace())
    plot_wigner(theory.rho,dim=2,file='graphics//thermal//theory_6.png')
    print(theory.fidelity(ther))


def test_purity():
    theory = DensityOperator.noisy_fock([2],[sigma],[cutoff_sys])
    print(theory.purity())


def test_negative_volume():
    sigma = 0.8
    fock = 150
    cutoff= int(fock*1.5)
    axes_bound = 18
    initial = DensityOperator.noisy_fock([fock],[sigma],[cutoff])
    print(initial.mean())
    print(initial.variance())
    print(np.trace(initial.rho.toarray()))
    print(initial.negative_volume(axes_bound=axes_bound))
    # plot_wigner(initial.rho,dim=2,axis_min=-axes_bound,axis_max=axes_bound,axis_nums=300)


def test_q_function():
    cutoff=110
    initial = DensityOperator.fock([3],[cutoff])
    axes_min= -6
    axes_max= 6
    axes_steps= 200
    xvec = np.linspace(axes_min, axes_max, axes_steps)
    reals,imags = np.meshgrid(xvec,xvec)
    alphas = reals + 1j*imags
    q = initial.Q_function()
    q_vals = np.zeros(alphas.shape)
    for a in range(len(alphas)):
        for b in range(len(alphas[a])):
            q_vals[a][b] = q(np.array([alphas[a][b]],dtype = complex))
    plot3d(xvec,xvec,q_vals)



def test_q_function1():
    cutoff=30
    initial = DensityOperator.fock([2],[cutoff])
    dim = 1
    axes_min= -6
    axes_max= 6
    axes_nums= 100
    axes_area = (1/np.pi*((axes_max-axes_min)/(axes_nums-1))**2)**(dim)
    xvec = np.linspace(axes_min, axes_max, axes_nums)
    reals,imags = np.meshgrid(xvec,xvec)
    alpha = reals + 1j*imags
    alphas = np.stack([alpha for i in range(dim)],axis=-1)
    q = initial.Q_function()
    q_vals = np.zeros(alphas.shape[0:-1])
    for a in range(len(alphas)):
        for b in range(len(alphas[a])):
            t = q(alphas[a][b])
            q_vals[a][b] = -t*np.log(t)
        
    sw = np.sum(q_vals)*axes_area
    
def fact(n):
    if n==0:
        return 1
    else:
        return n*fact(n-1)

def psi(n):
    gamma = 0.577215664901532860606512090082
    harm = 0
    for _i in range(1,n):
        harm += 1/_i
    return  np.log(np.e)+gamma -harm

def test_wehrl_entropy():
    cutoff=30
    n = 3
    initial = DensityOperator.fock([n],[cutoff])
    print((np.log(np.e*np.pi*fact(n))) + n*psi(n+1))
    start1 = time.time()
    print(initial.Wehrl_entropy())
    end1 = time.time()
    start2 = time.time()
    print(initial.dep_Wehrl_entropy())
    end2 = time.time()
    print(end1-start1)
    print(end2-start2)


def test_wehrl_entropy_coherent():
    cutoff=30
    n = 1
    initial = DensityOperator.fock([0],[cutoff])
    # print(initial.toarray())
    print(np.log(np.pi*np.e))
    start1 = time.time()
    print(initial.Wehrl_entropy(axis_min= -8, axis_max= 8,axis_num= 300))
    end1 = time.time()
    start2 = time.time()
    print(initial.dep_Wehrl_entropy())
    end2 = time.time()
    print(end1-start1)
    print(end2-start2)

def test_stack():
    a = np.array([[1,1,1],[1,1,1],[1,1,1]])
    b = np.array([[2,2,2],[2,2,2],[2,2,2]])
    l = [a for i in range(3)]
    print(np.stack(l,-1))


# def test_gaussian():
#     cutoff= 120
#     # initial = DensityOperator.gaussian(np.array([2,2]),np.array([[1,0],[0,1]]),np.array([cutoff]))
#     coherent = DensityOperator.coherent(np.array([1+1j]),np.array([cutoff]))
#     print(coherent.gaussianity())


def test_gaussian():
    cutoff= 30
    # initial = DensityOperator.gaussian(np.array([2,2]),np.array([[1,0],[0,1]]),np.array([cutoff]))
    coherent = DensityOperator.noon(0,10)
    print('Hi')
    print(coherent.gaussianity(axes_bound=3,axes_nums=20))
    print('ENd')

def test_coherent():
    cutoff= 6
    coherent = DensityOperator.coherent(np.array([1+1j]),np.array([cutoff]))
    print(coherent.toarray())

def test_entangle_negativity():
    cutoff = 8
    bell = DensityOperator.noon(2,cutoff)
    print(bell.entangle_negativity([0]))


def test_relent():
    # initial = DensityOperator.noon(2,cutoff_sys)
    # print(initial.entangle_relent(2000,100))
    coherent = DensityOperator.coherent([0.5+0.5j,0.3+0.2j], [cutoff_sys,cutoff_sys])
    print(coherent.entangle_relent())
    # ther = partial_thermal_channel(initial,sigma,cutoff_env)
    # print(ther.entangle_relent())

# test_entangle_negativity()
# test_coherent()
# test_wehrl_entropy()
# test_wehrl_entropy_coherent()
# test_gaussian()
# test_displacement()
# test_negative_volume()
# test_entanglement()
# test_entanglement()
# test_relent()
# test_thermal()
test_gaussian()