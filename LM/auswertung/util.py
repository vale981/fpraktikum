"""
Units:
  * Time: nanosec

Conventions:
  * Channels have a zero based index: i.e. Channel 1 = Channel 0 here
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root_scalar
from SecondaryValue import SecondaryValue
import matplotlib.ticker as ticker
from typing import Dict, Tuple, Sequence
from scipy.stats import binned_statistic
import os
from scipy.optimize import curve_fit
import pylandau
import matplotlib
from scipy import constants

###############################################################################
#                                  Auxiliary                                  #
###############################################################################

def load_counts(path):
    """Parses the TDC data from the text format.

    :param path: path to the log
    :returns: counts
    """

    skip = 0
    with open(path, 'r') as f:
        for line in f:
            if 'Summenspektrum' in line:
                break
            skip += 1

    data = np.loadtxt(path, skiprows=skip+1, encoding='latin1',
                      usecols=[1], dtype=np.integer)

    return data



def chan_to_time(channel, tick=1000/24, center=False, offset=0):
    """Convert channel number to time.

    :param channel: channel Number
    :param tick: width of the channels
    :param center: wether to take the time at the channel center
    :param offset: start channel

    :returns: mean time of channel (center)

    :rtype: float
    """

    return (channel + offset) * tick + (tick/2 if center else 0)

def calculate_peak_uncertainty(peak):
    """Gives the uncertainty of the peak measurements with the osciloscope.

    :param peak: peak value
    :returns: uncertainty
    """

    uncertainties = np.ones_like(peak) * 0.01
    uncertainties[peak > 1] = 0.1
    return uncertainties

m_mu, _, d_m_mu = constants.physical_constants['muon mass energy equivalent in MeV']
h, _, d_h = constants.physical_constants['Planck constant over 2 pi in eV s']
weak_constant = SecondaryValue('sqrt(192*pi^3/((1e-9*tau/(h*1e-9))*(1e-3*m)^5))',
                               defaults=dict(pi=np.pi, h=(h, d_h),
                    m=(m_mu, d_m_mu)))

###############################################################################
#                                   Binning                                   #
###############################################################################

def calculate_bins(peaks):
    return int(np.log2(peaks.size) + 1) + 1

def plot_hist(peaks, bins, scale=1, save=None):
    fig, ax = set_up_plot()
    peaks = peaks*scale
    ax.hist(peaks, bins, density=True, label='Peakhoehenhistogram')

    hist, edges = np.histogram(peaks, bins, density=True)
    mids = (edges[:-1] + edges[1:])/2
    popt, _ = curve_fit(boltzmann, mids, hist, bounds=([1, .2], np.inf))

    points = np.linspace(0, edges[-1], 100)
    ax.plot(points, boltzmann(points, *popt),
            label=fr'Landauverteilung $\mu={popt[1]:.1f},\; \eta={popt[0]:.1f}$')
    ax.legend()
    ax.set_ylabel('Warscheinlichkeitsdichte')
    ax.set_xlabel(fr'Peakhoehe [Volt$\cdot {scale}$]')
    if save:
        save_fig(fig, *save)
    return fig, ax


def boltzmann(x, c, mu):
    return pylandau.landau_pdf(x, mu=mu, eta=c)

###############################################################################
#                                  Plot Porn                                  #
###############################################################################

matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
})

def pinmp_ticks(axis, ticks):
    axis.set_major_locator(ticker.MaxNLocator(ticks))
    axis.set_minor_locator(ticker.MaxNLocator(ticks*10))
    return axis

def set_up_plot(ticks=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    pinmp_ticks(ax.xaxis, ticks)
    pinmp_ticks(ax.yaxis, ticks)

    ax.grid(which='minor', alpha=.3)
    ax.grid(which='major', alpha=.8)


    ax.tick_params(right=True, top=True, which='both')

    return fig, ax

def save_fig(fig, title, folder='unsorted', size=(5, 4)):
    fig.set_size_inches(*size)
    fig.tight_layout()
    try:
        os.makedirs(f'./figs/{folder}/')
    except OSError as exc:
        pass
    fig.savefig(f'./figs/{folder}/{title}.pdf')
    fig.savefig(f'./figs/{folder}/{title}.pgf')

    with open('./out/figlist.txt', 'a') as f:
        f.write(r'''
\begin{figure}[H]\centering
  \input{../auswertung/figs/'''
  + f'{folder}/{title}.pgf' +
  r'''}
  \caption{}
  \label{fig:''' + folder + '-' + title + '''}
\end{figure}
    ''')

def plot_spectrum(counts, offset=1, errorevery=False, save=None, **pyplot_args):
    fig, ax = set_up_plot()

    channels = np.arange(0, counts.size) + offset
    ax.step(channels, counts, **pyplot_args)

    if errorevery:
        ax.errorbar(channels, counts, yerr=np.sqrt(counts), linestyle='None',
                    errorevery=errorevery)

    ax.set_xlabel('Kanal')
    ax.set_ylabel('Counts')
    ax.set_xlim([channels[0], channels[-1]])
    ax.set_ylim(0)

    if save:
        save_fig(fig, *save)

    return fig, ax

###############################################################################
#                              Maximum Likelihood                             #
###############################################################################

def continous(counts: np.ndarray, interval: Tuple[float, float], epsilon: float=1e-3) \
  -> (float, float, int, float):
    """Maximizes the likelihood for the continous propability model.
    (Method 1)

    :param array counts: the time-spectrum
    :param tuple interval: the interval of the time spectrum used
        (inclusive, exclusive)
    :param float epsilon: the desired precision

    :returns: the lifetime, the deviation of the same, total counts,
              total time
    """

    cts = counts[interval[0]:interval[1]]  # constrained counts
    channels = np.arange(0, interval[1] - interval[0])  # channel indices
    times = chan_to_time(channels, center=True)  # mean time of channels

    T = (times[1] - times[0])*len(cts)  # time interval
    N = cts.sum()  # total count
    tau_0 = np.sum(cts*times)/N  # initial guess

    def model(tau):
        return tau_0 + T/(np.exp(T/tau) - 1)

    taus = [tau_0]
    def optimize(tau):
        next_tau = model(tau)
        taus.append(next_tau)

        if np.abs(tau - next_tau) < epsilon:
            return next_tau
        return optimize(next_tau)

    tau = optimize(tau_0)

    correction_factor = 1/(1-(T/((np.exp(T/tau)-1)*tau))**2*np.exp(T/tau))
    delta_tau = np.sqrt(np.sum(cts*times**2))/N*correction_factor

    return tau, delta_tau, N, T, taus, correction_factor


def binned_likelihood(counts, interval):
    """Generates the -2*ln(likelihood) functions as closures under the
    assumption of poisson and gaussian statistics.

    :param array counts: the time-spectrum
    :param tuple interval: the interval of the time spectrum used
        (inclusive, exclusive)

    :returns: poisson, gauss likelihood funcions, total N
    """

    cts = counts[interval[0]:interval[1]][:, None]  # constrained counts
    channels = np.arange(interval[0], interval[1])  # channel indices
    times = chan_to_time(channels)[:, None]
    tick = times[1][0] - times[0][0]
    N = cts.sum()

    def N_0(tau):
        return N/(np.exp(-1*times[0][0]/tau) - np.exp(-1*(times[-1][0] + tick)/tau))

    def f(tau):
        return N_0(tau)/tau*np.exp(-(times + tick/2)/tau)*tick

    def convert_tau(tau):
        """Move tau to the second first dimension."""
        return tau[None,:] if isinstance(tau, np.ndarray) else np.array([[tau]])

    def ln_poisson_likelihood(tau):
        tau = convert_tau(tau)
        return -2*np.sum(cts*np.log(f(tau)), axis=0)

    def ln_gauss_likelihood(tau):
        tau = convert_tau(tau)
        fi = f(tau)
        return np.sum((cts-fi)**2/fi, axis=0)

    # for cross checking
    def ln_exp_likelihood(tau):
        tau = convert_tau(tau)
        mod_times = times - times[0][0] + tick/2
        T = tick * len(cts)
        return 2*np.sum(cts*(mod_times/tau+np.log(tau)+np.log(1-np.exp(-T/tau))), axis=0)

    return ln_poisson_likelihood, ln_gauss_likelihood, ln_exp_likelihood, N, N_0

def maximize_likelihood(likelihood, tau_range, epsilon=1e-3):
    """Minizizes the -2ln(likelihood) function thus maximizing the
    likelihood.

    :param likelihood: the -2*ln(likelihood) function
    :param tau_range: the range in which to look for the lifetime
    :param epsilon: the desired precision

    :returns: tau, (sigma minus, sigma plus), -2*ln(likelihood) at minimum,
              (gf, delta_gf)
    """

    result = minimize_scalar(likelihood, bounds=tau_range, bracket=tau_range,
                             tol=epsilon)
    tau = result.x[0]
    l = likelihood(tau)[0]

    # one should check for errors here, but this is a one-off, so i
    # dont bother

    def find_sigma(right=False):
        return np.abs(tau - root_scalar(lambda t: likelihood(t) - (l + 1),
                                 bracket=[tau, tau_range[1]] \
                                 if right else [tau_range[0], tau],
                                 x0=tau,
                                 x1=tau + 100 if right else tau - 100).root)
    sigma_minus = find_sigma()
    sigma_plus = find_sigma(right=True)
    max_sigma = np.max([sigma_plus, sigma_minus])
    return tau, (sigma_minus, sigma_plus), l, weak_constant(tau=(tau, max_sigma))


# i could just use *args **kwargs for this wrapper, but i won't for
# the sake of clarity
def maximize_and_plot_likelihood(likelihood, tau_range, epsilon=1e-5, save=None):
    """Minizizes the -2ln(likelihood) function thus maximizing the
    likelihood.

    :param likelihood: the -2*ln(likelihood) function
    :param tau_range: the range in which to look for the lifetime
    :param epsilon: the desired precision

    :returns: (fig, ax), tau, (sigma minus, sigma plus),
              -2*ln(likelihood) at minimum, (gf, delta_gf)
    """

    tau, sigma, l, gf = maximize_likelihood(likelihood,
                                      tau_range, epsilon)

    rng = np.max(sigma) * 3
    taus = np.linspace(tau - rng, tau + rng, 1000)
    fig, ax = set_up_plot()

    ax.plot(taus, likelihood(taus), label='$-2\ln(L)$')
    ax.plot([(tau - sigma[0]), (tau + sigma[1])], [(l + 1), (l + 1)],
            linestyle='dotted', color='gray', label='Deviation from +1 Unit')
    ax.plot([(tau - sigma[0]), (tau - sigma[0])], [(l + 1), l],
            linestyle='dotted', color='gray')
    ax.plot([(tau + sigma[1]), (tau + sigma[1])], [(l + 1), l],
            linestyle='dotted', color='gray')

    ax.set_xlim((taus[0], taus[-1]))
    ax.axvline(tau, linestyle='--', color='black', label=r'$\tau$')
    ax.errorbar(tau, l, marker='x', markersize=10, xerr=np.array([sigma]).T,
                label='Minimum', capsize=10)

    ax.set_xlabel(r'$\tau$ [ns]')
    ax.set_ylabel(r'$-2\ln(L)$')
    ax.legend()


    if save:
        save_fig(fig, *save)
    return (fig, ax), tau, sigma, l, gf
