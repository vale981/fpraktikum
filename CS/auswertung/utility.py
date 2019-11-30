import matplotlib.pyplot as plt
import numpy as np
from SecondaryValue import SecondaryValue
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.sparse as sparse
import matplotlib.ticker as ticker
import matplotlib

import os


mu = SecondaryValue('cos(theta)')
kappa = SecondaryValue('E_in/er', defaults=dict(er=510.99895000))
E = SecondaryValue('E_in/(1+kappa*(1-mu))',
                   dependencies=dict(mu=mu, kappa=kappa))

t_opt = SecondaryValue('(N_g + N_0)/(prec^2*(N_g-N_0)^2)')
d_t_opt = SecondaryValue('sqrt((N_g+3*N_0)^2*N_g/t+(3*N_g+N_0)^2*N_0/t)/(prec^2*(N_g-N_0)^2)')

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
def klein_nisha(mu, E_in):
    kappa = E_in/510.99895000
    return const.physical_constants['classical electron radius'][0]**2/2*(1/(1+kappa*(1-mu)))**2*(kappa*(1-mu)+1/(1+kappa*(1-mu))+mu**2)
def load_spectrum(path, absolute=True):
    """Parses the characteristic curve data from the data logs.

    :param path: path to the log
    :returns: channel, events
    """
    data = np.loadtxt(path, skiprows=1, encoding='latin1')[:,1]

    return data if absolute else data/data.max()

def channels(spec):
    return np.arange(1, len(spec) + 1)


def pinmp_ticks(axis, ticks):
    axis.set_major_locator(ticker.MaxNLocator(ticks))
    axis.set_minor_locator(ticker.MaxNLocator(ticks*10))
    return axis

def set_up_angle_plot(ticks=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def mu_to_deg(mu):
        return np.arccos(mu)*180/np.pi

    def deg_to_mu(deg):
        return np.cos(deg*np.pi/180)

    deg_ax = ax.secondary_xaxis('top', functions=(mu_to_deg, deg_to_mu))
    deg_ax.set_xlabel('Winkel')
    ax.set_xlabel('$\mu$')

    pinmp_ticks(ax.xaxis, ticks)
    pinmp_ticks(ax.yaxis, ticks)

    ax.grid(which='minor', alpha=.3)
    ax.grid(which='major', alpha=.8)

    ax.tick_params(right=True, which='both')

    return fig, ax

def plot_spec(spec, is_relative=False, offset=0, ticks=10, save=None, **pyplot_args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    chans = channels(spec) + offset

    pinmp_ticks(ax.xaxis, ticks)
    pinmp_ticks(ax.yaxis, ticks)

    ax.tick_params(top=True, right=True, which='both')
    ax.step(chans, spec, **pyplot_args)
    ax.grid(which='minor', alpha=.3)
    ax.grid(which='major', alpha=.8)
    ax.set_xlabel('Kanal')
    ax.set_ylabel(('Normierte' if is_relative else '') + 'Ereignisszahl')

    ax.set_xlim(chans[0], chans[-1])
    ax.set_ylim(spec.min(), spec.max())
    fig.tight_layout()

    if save:
        save_fig(fig, *save)

    return fig, ax

def gauss(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))

def gauss_offset(x, mu, sigma, A, O):
    return np.exp(-(x-mu)**2/(2*sigma**2))*A + O

def calibrate_peak(spec, start, end, save=None, ret_sigma=False):
    slc = spec[start:end]

    chan = channels(spec)[start:end]
    opt, d_opt = curve_fit(gauss_offset, chan,
                           slc, p0=((start+end)/2, (end-start)/2, 1*slc.max(), 0),
                           sigma=np.sqrt(np.abs(slc))+1,
                           bounds=([start, 0, 1/2*slc.max(), 0], [end, np.inf, 1*slc.max(), 1/2*slc.max()]))
    d_opt = np.sqrt(np.diag(d_opt))


    fig, ax = plot_spec(slc, offset=start, label='Gemessen')
    ax.plot(chan, gauss_offset(chan, *opt), label='Peak Fit', linestyle='--')
    if ret_sigma:
        ax.axvspan(opt[0]-3*opt[1], opt[0]+3*opt[1], alpha=.2,
                   label='$3\cdot\sigma$', color='gray', zorder=-10)
    ax.legend()

    if save:
        save_fig(fig, *save)

    if ret_sigma:
        return opt[0], d_opt[0] + d_opt[1], opt[1], d_opt[1], opt[2], d_opt[2], opt[3], d_opt[3]
    return opt[0], d_opt[0] + d_opt[1]

def save_fig(fig, title, folder='unsorted', size=(5, 4)):
    fig.set_size_inches(*size)
    fig.tight_layout()
    try:
        os.makedirs(f'./figs/{folder}/')
    except OSError as exc:
        pass
    fig.savefig(f'./figs/{folder}/{title}.pdf')
    fig.savefig(f'./figs/{folder}/{title}.pgf')

def find_peak(spec, under, interval, width, height=.5, distance=100, save=None):
    "Find the peak roughly and fit a gauss curve."
    corrected = (spec - under)[interval[0]:interval[1]]
    corrected = corrected/corrected.max()
    peak = find_peaks(corrected, height=height, distance=distance)[0]

    return calibrate_peak(spec-under, int(interval[0]+peak-width), int(interval[0]+peak+width), ret_sigma=True, save=save)

area = SecondaryValue('sqrt(2*p)*s*A+O', defaults=dict(p=np.pi))

def find_rates(spec, null, t, mu, sigma):
    "3 sigma"
    mu = int(np.round(mu))
    sigma = int(np.round(sigma))

    N = np.sum(spec[mu - 3*sigma:mu + 3*sigma])
    N0 = np.sum(null[mu - 3*sigma:mu + 3*sigma])
    N_f = (N - N0)/t
    d_N_f = 2*np.sqrt(np.sum((np.sqrt([N, N0])/t)**2))
    return N_f, d_N_f


import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z
