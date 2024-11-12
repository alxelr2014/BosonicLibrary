import numpy as np
from density import *
from wigner import *
from channel import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def label(num):
    return r'$ | {{{n}}} \rangle\langle {{{n}}} | $'.format(n=num)

def noon_label(num):
    return r'$  N = {{{n}}} $'.format(n=num)

def ncolor(num):
    if num==0:
        return 'steelblue'
    elif num == 1:
        return 'crimson'
    elif num == 2:
        return 'goldenrod'
    elif num == 5:
        return 'seagreen'
    elif num == 10:
        return 'darkviolet'
    elif num == 3:
        return 'indigo'
    elif num == 4:
        return 'lavender'
    else:
        return 'indigo'


def plotting(x,ys,focks,xlabel,ylabel,filepath,lfunc = label):
    for f in range(len(focks)):
        plt.plot(x,ys[f],label=lfunc(focks[f]),color=ncolor(focks[f]),linewidth=1.6)
    plt.xlim(0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.grid(linestyle='--', linewidth=0.8,color='grey')
    plt.savefig(filepath)
    plt.show()

def purity_graph(sigma_min,sigma_max,sigma_num, fock,cutoff):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def purity_val(number):
        def temp(sigma):
            initial = DensityOperator.noisy_fock([number],[(sigma)],[cutoff])
            return initial.purity()
        return temp
    purities = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        purities[f] = np.vectorize(purity_val(fock[f]))(sigmas)
    plotting(sigmas,purities,fock,r'$\sigma$',"Purity",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\purity')



def negative_volume_graph(sigma_min,sigma_max,sigma_num, fock,cutoffs,axes_bound=15,axes_nums=300):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def neg_vol_val(number,cutoff):
        def temp(sigma):
            initial = DensityOperator.noisy_fock([number],[sigma],[cutoff])
            return initial.negative_volume(axes_bound=axes_bound,axes_nums=axes_nums)
        return temp
    neg_vols = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        neg_vols[f] = np.vectorize(neg_vol_val(fock[f],cutoffs[f]))(sigmas)
    plotting(sigmas,neg_vols,fock,r'$\sigma$',"Negative Volume",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\quantumness')


def gaussianity_graph(sigma_min,sigma_max,sigma_num, fock,cutoffs,axes_bound=15,axes_nums=300):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def gaussianity(number,cutoff):
        def temp(sigma):
            initial = DensityOperator.noisy_fock([number],[sigma],[cutoff])
            return initial.gaussianity(axes_bound=axes_bound,axes_nums=axes_nums)
        return temp
    gaussians = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        gaussians[f] = np.vectorize(gaussianity(fock[f],cutoffs[f]))(sigmas)
    plotting(sigmas,gaussians,fock,r'$\sigma$',"Non-Gaussianity",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\gaussian')




def noon_gaussianity_graph(sigma_min,sigma_max,sigma_num, fock,cutoffs,axes_bound=15,axes_nums=300):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def gaussianity(number,cutoff):
        def temp(sigma):
            print(sigma)
            initial = DensityOperator.noisy_noon(number,sigma,0,cutoff,cutoff)
            return initial.gaussianity(axes_bound=axes_bound,axes_nums=axes_nums)
        return temp
    gaussians = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        print(f)
        gaussians[f] = np.vectorize(gaussianity(fock[f],cutoffs[f]))(sigmas)
    plotting(sigmas,gaussians,fock,r'$\sigma$',"Non-Gaussianity",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\gaussian_noon')


def negativity_graph(sigma_min,sigma_max,sigma_num, fock,cutoffs,axes_bound=15,axes_nums=300):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def negativitiy(number,cutoff):
        def temp(sigma):
            initial = DensityOperator.noisy_noon(number,sigma,0,cutoff,cutoff)
            return initial.entangle_negativity([1])
        return temp
    negativities = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        negativities[f] = np.vectorize(negativitiy(fock[f],cutoffs[f]))(sigmas)
    plotting(sigmas,negativities,fock,r'$\sigma$',"Negativity",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\negativity2')


def wang_ent_graph(sigma_min,sigma_max,sigma_num, fock,cutoffs):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    print(sigmas)
    def negativitiy(number,cutoff):
        def temp(sigma):
            print(sigma,number,cutoff)
            initial = DensityOperator.noisy_noon(number,sigma,0,cutoff,number+1)
            return initial.entangle_wang()
        return temp
    negativities = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        negativities[f] = np.vectorize(negativitiy(fock[f],cutoffs[f]))(sigmas)
    plotting(sigmas,negativities,fock,r'$\sigma$',"Wang Entanglement",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\wang')



def log_negativity_graph(sigma_min,sigma_max,sigma_num, fock,cutoffs,axes_bound=15,axes_nums=300):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def negativitiy(number,cutoff):
        def temp(sigma):
            initial = DensityOperator.noisy_noon(number,sigma,0,cutoff,cutoff)
            return initial.entangle_log_negativity([0])
        return temp
    negativities = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        negativities[f] = np.vectorize(negativitiy(fock[f],cutoffs[f]))(sigmas)
    plotting(sigmas,negativities,fock,r'$\sigma$',"Log-Negativity",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\log-negativity',lfunc=noon_label)
    
  

def wigner_graph(sigma,fock,cutoff,axes_bound=10,axes_nums=250):
    initial_1 = DensityOperator.fock(np.array([fock]),np.array([cutoff]))
    initial_2 = DensityOperator.noisy_fock(np.array([fock]),np.array([sigma]),np.array([cutoff]))
    plot_wigner(initial_1.toarray(),axis_min=-axes_bound,axis_max=axes_bound,axis_nums=axes_nums, file= 'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\clean')
    plot_wigner(initial_2.toarray(),axis_min=-axes_bound,axis_max=axes_bound,axis_nums=axes_nums, file= 'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\noisy')




def photon_ent_graph(sigma_min,sigma_max,sigma_num, fock,cutoffs,axes_bound=15,axes_nums=300):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def negativitiy(number,cutoff):
        def temp(sigma):
            initial = DensityOperator.noisy_noon(number,sigma,0,cutoff,cutoff)
            p = initial.second_photon_statistics()
            probs = np.array([p(_) for _ in range(cutoff)])
            return entropy(probs)
        return temp
    negativities = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        negativities[f] = np.vectorize(negativitiy(fock[f],cutoffs[f]))(sigmas)
    plotting(sigmas,negativities,fock,r'$\sigma$',"Photon Entanglement",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\photonent')
    

def photon_statistics(sigma_min,sigma_max,sigma_num,fock,cutoff):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    sigma_step = (sigma_max - sigma_min)/(sigma_num - 1)

    _x = np.arange(cutoff)
    _y = np.arange(cutoff)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    def probs(sigma):
        initial = DensityOperator.noisy_fock(fock,sigma,0,cutoff,cutoff)
        p = initial.photon_statistics()
        return np.array([p([_,__]) for _ in _x for __ in _y])
    top = {s:probs(s) for s in sigmas}
    # top = x + y
    
    bottom = np.zeros_like(top[0.0])
    width = depth = 0.8

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot( projection='3d')
    bar = ax.bar3d(x, y, bottom, width, depth, top[sigma_min], shade=True)

    # Choose the Slider color
    axcolor = "White"
    
    # Set the frequency and amplitude axis
    sigma_axis = plt.axes([0.19, 0.05, 0.65, 0.03], facecolor = axcolor)
    
    # Set the slider for frequency and amplitude
    sigma_slider = Slider(sigma_axis, r'$\sigma$',sigma_min, sigma_max, valinit = sigma_min,valstep=sigma_step)

    def update(val):
        sigm = sigma_slider.val
        ax.clear()
        ax.bar3d(x, y, bottom, width, depth, top[sigm], shade=True)
        fig.canvas.draw_idle()

     
# update function called using on_changed() function 
# for both frequency and amplitude
    sigma_slider.on_changed(update)

    plt.show()



def photon_fock_statistics(sigma_min,sigma_max,sigma_num,fock,cutoff):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    sigma_step = (sigma_max - sigma_min)/(sigma_num - 1)

    x = np.arange(cutoff)
    # _y = np.arange(cutoff)
    # _xx, _yy = np.meshgrid(_x, _y)
    # x, y = _xx.ravel(), _yy.ravel()
    def probs(sigma):
        initial = DensityOperator.noisy_fock([fock],[sigma],[cutoff])
        p = initial.photon_statistics()
        return np.array([p([_]) for _ in x])
    top = {s:probs(s) for s in sigmas}
    # top = x + y
    
    bottom = np.zeros_like(top[0.0])
    width = depth = 0.8

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    bar = ax.bar(x,top[sigma_min],bottom= bottom,width= width,tick_label=[str(d) for d in x])

    # Choose the Slider color
    axcolor = "White"
    
    # Set the frequency and amplitude axis
    sigma_axis = plt.axes([0.19, 0.05, 0.65, 0.03], facecolor = axcolor)
    
    # Set the slider for frequency and amplitude
    sigma_slider = Slider(sigma_axis, r'$\sigma$',sigma_min, sigma_max, valinit = sigma_min,valstep=sigma_step)

    def update(val):
        sigm = sigma_slider.val
        ax.clear()
        bar = ax.bar(x,top[sigm],bottom= bottom,width= width,tick_label=[str(d) for d in x])
        fig.canvas.draw_idle()

     
# update function called using on_changed() function 
# for both frequency and amplitude
    sigma_slider.on_changed(update)

    plt.show()


def correlation_graph(sigma_min,sigma_max,sigma_num, fock,cutoffs):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def negativitiy(number,cutoff):
        def temp(sigma):
            initial = DensityOperator.noisy_noon(number,sigma,0,cutoff,cutoff)
            p = initial.photon_statistics()
            return correlation(p,range(cutoff),range(cutoff))
        return temp
    negativities = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        print(f)
        negativities[f] = np.vectorize(negativitiy(fock[f],cutoffs[f]))(sigmas)
    plotting(sigmas,negativities,fock,r'$\sigma$',"Log-Negativity",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\covariance')
    

def correlation_test(sigma,number,cutoff):
    initial = DensityOperator.noisy_noon(number,sigma,0,cutoff,cutoff)
    p = initial.photon_statistics()
    print(correlation(p,range(cutoff),range(cutoff)))

def variance_test(sigma,number,cutoff):
    initial = DensityOperator.noisy_noon(number,sigma,0,cutoff,cutoff)
    print(initial.variance())





def photon_stat_graph(sigma_min,sigma_max,sigma_num, fock,cutoffs,axes_bound=15,axes_nums=300):
    sigmas = np.linspace(sigma_min,sigma_max,num=sigma_num)
    def negativitiy(number,cutoff):
        def temp(sigma):
            initial = DensityOperator.noisy_fock([number],[sigma],[cutoff])
            p = initial.photon_statistics()
            probs = np.array([p(_) for _ in range(cutoff)])
            return probs
        return temp
    negativities = [0 for _ in range(len(fock))]
    for f in range(len(fock)):
        negativities[f] = np.vectorize(negativitiy(fock[f],cutoffs[f]))(sigmas)
    plotting(sigmas,negativities,fock,r'$\sigma$',"Photon Entanglement",'D:\\School\\Courses\\Quantum Information and Communication\\Quantum Communication\\Project\\graphics\\photonent')
    

# purity_graph(0,2,50,[0,1,2,5,10],50)
# negative_volume_graph(0,0.8,30,[0,1,2,5,10],[30,30,30,40,50],15,400)
# negativity_graph(0,1,20,[0,1,2,3,4,5,6],[30,40,40,40,40,50,50],15,400)
# correlation_graph(0,1,10,[1,2,3,4,5],[20,20,20,25,25,50,50])
# log_negativity_graph(0,1,20,[0,1,2,3,5,10],[30,40,40,40,40,50],15,400)
# photon_ent_graph(0,1,20,[0,1,2,3,4,5,6],[30,40,40,40,40,50,50],15,400)
# gaussianity_graph(0,1.5,40,[0,1,2,5,10],[30,30,30,40,50],15,400)
# wigner_graph(0.5,2,30,axes_bound=5)
# photon_statistics(0,1.1,20,2,20)
# correlation_test(1,2,10)
# noon_gaussianity_graph(0,1,8,[0,1,2,5],[20,20,20,20,50],5,20)
# photon_fock_statistics(0,1.5,40,10,50)
wang_ent_graph(0,0.8,4,[1,2,3,5],[20,20,20,40,50,50])