import matplotlib.pyplot as plt
import numpy as np
from SecondaryValue import SecondaryValue
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.sparse as sparse
import matplotlib.ticker as ticker
mu = SecondaryValue('cos(theta)')
kappa = SecondaryValue('E_in/er', defaults=dict(er=510.99895000))
E = SecondaryValue('E_in/(1+kappa*(1-mu))',
                   dependencies=dict(mu=mu, kappa=kappa))

t_opt = SecondaryValue('(N_g + N_0)/(prec^2*(N_g-N_0)^2)')
d_t_opt = SecondaryValue('sqrt((N_g+3*N_0)^2*N_g/t+(3*N_g+N_0)^2*N_0/t)/(prec^2*(N_g-N_0)^2)')

def load_spectrum(path, absolute=False):
    """Parses the characteristic curve data from the data logs.

    :param path: path to the log
    :returns: channel, events
    """
    data = np.loadtxt(path, skiprows=1, encoding='latin1')[:,1]

    return data if absolute else data/data.max()

def channels(spec):
    return np.arange(1, len(spec) + 1)


def pinmp_ticks(axis):
    axis.set_major_locator(plt.LinearLocator(numticks=10))
    axis.set_minor_locator(plt.LinearLocator(numticks=100))
    return axis


def plot_spec(spec, is_relative=False, offset=0, **pyplot_args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    chans = channels(spec) + offset

    pinmp_ticks(ax.xaxis)
    pinmp_ticks(ax.yaxis)

    ax.tick_params(top=True, right=True, which='both')
    ax.step(chans, spec, **pyplot_args)
    ax.grid(which='minor', alpha=.3)
    ax.grid(which='major', alpha=.8)
    ax.set_xlabel('Kanal')
    ax.set_ylabel(('Normierte' if is_relative else '') + 'Zaehlrate')

    ax.set_xlim(chans[0], chans[-1])
    ax.set_ylim(spec.min(), spec.max())
    fig.tight_layout()

    return fig, ax

def gauss(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))

def calibrate_peak(spec, start, end):
    slc = spec[start:end]
    slc = slc-slc.min()
    slc = slc/slc.max()
    chan = channels(spec)[start:end]
    opt, d_opt = curve_fit(gauss, chan,
                           slc, p0=((start+end)/2, (end-start)/2),
                           sigma=1/2*np.ones_like(chan), absolute_sigma=True)
    d_opt = np.sqrt(np.diag(d_opt))


    fig, ax = plot_spec(slc, is_relative=True, offset=start, label='Gemessen')
    ax.plot(chan, gauss(chan, *opt), label='Fit')
    ax.legend()
    fig.show()
    return opt[0], d_opt[0]

def find_peak(spec, under, interval, width, height=.5, distance=100):
    corrected = (spec - under)[interval[0]:interval[1]]
    corrected = corrected/corrected.max()
    peak = find_peaks(corrected, height=height, distance=distance)[0]

    return calibrate_peak(spec-under, int(interval[0]+peak-width), int(interval[0]+peak+width))

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
