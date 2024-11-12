import qutip as qtp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.colors as mcolors 
def plot_wigner(rho,sparse=False,dim=2,file: str = None,axis_min= -6, axis_max= 6, axis_nums= 200):
    x,y,z = wigner(rho,sparse,axis_min, axis_max, axis_nums)
    if dim==2:
        plot2d(x,y,z,file=file)
    if dim==3:
        plot3d(x,y,z,file=file)


def wigner(rho,sparse=False, axis_min= -6, axis_max= 6, axis_nums= 200):
    ''' method : string {'clenshaw', 'iterative', 'laguerre', 'fft'}
        Select method 'clenshaw' 'iterative', 'laguerre', or 'fft', where 'clenshaw' 
        and 'iterative' use an iterative method to evaluate the Wigner functions for density
        matrices :math:`|m><n|`, while 'laguerre' uses the Laguerre polynomials
        in scipy for the same task. The 'fft' method evaluates the Fourier
        transform of the density matrix. The 'iterative' method is default, and
        in general recommended, but the 'laguerre' method is more efficient for
        very sparse density matrices (e.g., superpositions of Fock states in a
        large Hilbert space). The 'clenshaw' method is the preferred method for
        dealing with density matrices that have a large number of excitations
        (>~50). 'clenshaw' is a fast and numerically stable method.'''


    xvec = np.linspace(axis_min, axis_max, axis_nums)
    wigner_data = qtp.wigner(psi=qtp.Qobj(rho),xvec= xvec,yvec= xvec,g=2,method='clenshaw',sparse=sparse)
    return xvec,xvec, wigner_data



def plot2d(xvec,yvec,data,file: str = None):
    num_colors= 100
    dpi = 100
    """Contour plot the given data array"""

    amax = np.amax(data)
    amin = np.amin(data)
    if amax == 0 and amin == 0:
        amax = 1
        amin = -1
    abs_max = max(amax, abs(amin))
    color_levels = np.linspace(-abs_max, abs_max, num_colors)

    fig, ax = plt.subplots(constrained_layout=True)
    cont = ax.contourf(xvec, yvec, data, color_levels, cmap="RdBu")

    # xvec_int = [int(x) for x in xvec]
    # xvec_int = sorted(set(xvec_int))
    ax.set_xlabel(r"$x$")
    # ax.set_xticks(xvec_int)
    ax.set_ylabel(r"$p$")
    # ax.set_yticks(xvec_int)
    ax.set_aspect('equal', 'box')
    # if draw_grid:
    #     ax.grid()


    cb = fig.colorbar(cont, ax=ax, format=tick.FormatStrFormatter('%.2f'))
    cb.set_label(r"$W(x,p)$",rotation=270,labelpad=25)

    if file:
        plt.savefig(file, dpi=dpi)
    else:
        plt.show()




def plot3d(xvec,yvec,data, file: str = None,  num_overlays=1):
    """Contour plot the given data array"""
    num_colors = 100
    dpi = 100
    x,y=np.meshgrid(xvec,yvec)
    amax = np.amax(data)
    amin = np.amin(data)
    if amax == 0 and amin == 0:
        amax = 1
        amin = -1
    abs_max = max(amax, abs(amin))

    fig, ax = plt.subplots(constrained_layout=True,subplot_kw={"projection": "3d"})
    downsample = 1
    lightsource = mcolors.LightSource(azdeg=50, altdeg=10)
    for _ in range(num_overlays):
        surf = ax.plot_surface(
        x, y, data, antialiased=True,
        cmap= 'RdBu',
        lightsource=lightsource, linewidth=0, alpha=1, rstride=downsample, cstride=downsample,
        norm=mcolors.TwoSlopeNorm(0)
    )

    xvec_int = [int(x) for x in xvec]
    xvec_int = sorted(set(xvec_int))
    ax.set_xlabel(r"$x$")
    # ax.set_xticks(xvec_int)
    ax.set_ylabel(r"$p$")
    # ax.set_yticks(xvec_int)
    # ax.set_aspect('equal', 'box')
    ax.set_zlim3d(amin, amax)
    ax.zaxis.set_major_locator(tick.LinearLocator(10))
    ax.zaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    # if draw_grid:
    #     ax.grid()

    cb = fig.colorbar(surf, ax=ax, format=tick.FormatStrFormatter('%.2f'))
    cb.set_label(r"$W(x,p)$",rotation=270,labelpad=25)

    if file:
        plt.savefig(file, dpi=dpi)
    else:
        plt.show()