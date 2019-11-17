import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn
from matplotlib import rc
from scipy import interpolate
from scipy import optimize

# rc('font', **{'family':'serif', 'sans-serif':['Times']})
# rc('text', usetex=False)

def parse_ccurve(path, compliance=.98):
    """Parses the characteristic curve data from the data logs.

    :param path: path to the log
    :returns: voltage, current
    """
    data = np.loadtxt(path, skiprows=11, encoding='latin1')[:, :2]
    data = data[data[:,1] < compliance]
    return data

def plot_ccurve(ccurve, log=False, area=None, compliance=.99, median=False,
              save=False, **pyplot_args):
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

def plot_ccurve_line(ax, ccurve, area=None, marker='.', compliance=.99, **pyplot_args):
    v, c = ccurve[ccurve[:,1] < compliance].T

    if area:
        c /= area

    ax.errorbar(v, c, linestyle='None', marker=marker, markersize=1.5, alpha=1,
                **pyplot_args)
    ax.set_xlabel("Spannung V [V]")
    ax.set_ylabel("Stromstaerke I [A]" \
                  if area else r"Stromdichte j [$\frac{A}{cm^2}$]")
    ax.grid(True, which='both')
    ax.set_xlim(v[0], v[-1])

def save_fig(fig, name):
    fig.savefig('./figs/' + name, tranparent=True, dpi=300)

def parse_and_plot_ccurve(path, *args, **kwargs):
    return plot_ccurve(parse_ccurve(path), *args, **kwargs)

def analyze_ccurve(ccurve, area, int_ein):
    """Calculates characteristic values from the char.  curve by
    linear interpolation.

    :param ccurve: char. curve
    :param area: area of the solar cell
    :param int_ein: lighting intesity

    :returns: j_c, u_cc, u_mlp, p_mlp, ff, eta
    """

    interpolated = interpolate.interp1d(*ccurve.T)
    i_c = interpolated(0)
    j_c = i_c / area
    u_cc = optimize.root_scalar(interpolated, bracket=[0, ccurve[:,0][-1]], method='brentq').root
    u_mlp = optimize.minimize_scalar(lambda u: u * interpolated(u),
                                     bracket=(0, u_cc), bounds=(0, u_cc),
                                     method='bounded').x
    p_mlp = -interpolated(u_mlp)*u_mlp
    ff = -p_mlp / i_c * u_cc

    eta = p_mlp / (int_ein * area)
    return -j_c, u_cc, u_mlp, p_mlp, ff, eta
