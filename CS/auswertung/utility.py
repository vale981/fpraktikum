import matplotlib.pyplot as plt
import numpy as np
from SecondaryValue import SecondaryValue
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.sparse as sparse
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

def plot_spec(spec):
    plt.clf()
    plt.plot(channels(spec), spec)
    plt.xlabel('Channel')
    plt.ylabel('Relative Counts')


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

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(chan, slc)
    ax.plot(chan, gauss(chan, *opt))
    ax.set_xlabel('Kanalnummer')
    ax.set_ylabel('Relative Ereignisszahl')
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
