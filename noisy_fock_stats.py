import numpy as np
from density import *
from wigner import *
from channel import *
import matplotlib.pyplot as plt


def label(num):
    return r'$ | {{{n}}} \rangle\langle {{{n}}} | $'.format(n=num)

def purity_graph(sigma_min,sigma_max,sigma_num, fock,cutoff):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def purity_val(number):
        def temp(sigma):
            initial = DensityOperator.noisy_fock([number],[sigma],[cutoff])
            return initial.purity()
        return temp
    for f in fock:
        purities = np.vectorize(purity_val(f))(sigmas)
        plt.plot(sigmas,purities,label=label(f))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def negative_volume_graph(sigma_min,sigma_max,sigma_num, fock,cutoff):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def neg_vol_val(number):
        def temp(sigma):
            initial = DensityOperator.noisy_fock([number],[sigma],[cutoff])
            return initial.negative_volume(axes_bound=15)
        return temp
    for f in fock:
        neg_vols = np.vectorize(neg_vol_val(f))(sigmas)
        plt.plot(sigmas,neg_vols,label=label(f))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

negative_volume_graph(0,1.5,20,[0,1,2,10],80)