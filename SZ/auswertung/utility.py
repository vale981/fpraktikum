import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn
from matplotlib import rc

rc('font', **{'family':'serif', 'sans-serif':['Times']})
rc('text', usetex=False)

def parse_ccurve(path, compliance=.98):
    """Parses the characteristic curve data from the data logs.

    :param path: path to the log
    :returns: voltage, current
    """
    data = np.loadtxt(path, skiprows=11, encoding='latin1')[:, :2]
    data = data[data[:,1] < compliance]
    return data

def plot_ccurve(ccurve, log=False, compliance=.99, median=False, save=False,
              **pyplot_args):
    """Plots the characteristic curve.

    :param ccurve: a numpy array with the ccurve data
    :returns: a figure with the plot

    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(y=0, color='gray')
    ax.axvline(x=0, color='gray')


    if median:
        compliance = np.median(ccurve[:, 0])

    plot_ccurve_line(ax, ccurve, compliance=compliance, **pyplot_args)
    if log:
        ax.set_yscale('log')

    if save:
        save_fig(fig, save)
    return fig, ax

def plot_ccurve_line(ax, ccurve, marker='.', compliance=.99, **pyplot_args):
    v, c = ccurve[ccurve[:,1] < compliance].T
    ax.errorbar(v, c, linestyle='None', marker=marker, markersize=1.5, alpha=1,
            **pyplot_args)
    ax.set_xlabel("Spannung V [V]")
    ax.set_ylabel("Stromstaerke I [A]")
    ax.grid(True, which='both')
    ax.set_xlim(v[0], v[-1])

def save_fig(fig, name):
    fig.savefig('./figs/' + name, tranparent=True, dpi=300)

def parse_and_plot_ccurve(path, *args, **kwargs):
    return plot_ccurve(parse_ccurve(path), *args, **kwargs)
