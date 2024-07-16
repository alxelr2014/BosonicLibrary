import qiskit as qs
import numpy as np
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num_states = 4
shape = 2,2
states = {'|0\u27E9':[1,0],'|1\u27E9':[0,1],'|+\u27E9':[1/np.sqrt(2), 1/np.sqrt(2)],'|-\u27E9':[1/np.sqrt(2),-1/np.sqrt(2)]}

def ket_to_spherical(qubit):
    r = 1
    alpha,beta = qubit[0],qubit[1]
    phase_alpha = np.exp(1j*np.angle(alpha))
    alpha /= phase_alpha
    beta /= phase_alpha
    phi = np.angle(beta)
    theta =2* np.arccos(np.real(alpha))
    return [r,theta,phi]

fig = plt.figure(figsize=[2*shape[0], 2*shape[1]])
hdist = 1/shape[0]
vdist = 1/shape[1]
indx,indy = 0,0
for st in states:
    ax = fig.add_axes([indx*hdist,indy*vdist, hdist, vdist], axes_class = Axes3D)
    indx += 1
    if indx == shape[0]:
        indx=0
        indy += 1
    plot_bloch_vector(ket_to_spherical(states[st]),coord_type='spherical',title="state " + st,ax=ax)
plt.show()
