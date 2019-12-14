"""
Units:
  * Time: nanosec

Conventions:
  * Channels have a zero based index: i.e. Channel 1 = Channel 0 here
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root_scalar
import matplotlib.ticker as ticker
from typing import Dict, Tuple, Sequence

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



def chan_to_time(channel, tick=41.67, center=False, offset=0):
    """Convert channel number to time.

    :param channel: channel Number
    :param tick: width of the channels
    :param center: wether to take the time at the channel center
    :param offset: start channel

    :returns: mean time of channel (center)

    :rtype: float
    """

    return (channel + offset) * tick + (tick/2 if center else 0)


###############################################################################
#                              Maximum Likelihood                             #
###############################################################################

def continous(counts: np.ndarray, interval: Tuple[float, float], epsilon: float=1e-9) \
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

    T = times[-1] - times[0]  # time interval
    N = cts.sum()  # total count
    tau_0 = np.sum(cts*times)/N  # initial guess
    delta_tau = np.sqrt(np.sum(cts*times**2))/N

    def model(tau):
        return tau_0 + T/(np.exp(T/tau) - 1)

    def optimize(tau):
        next_tau = model(tau)

        if np.abs(tau - next_tau) < epsilon:
            return next_tau
        return optimize(next_tau)

    return optimize(tau_0), delta_tau, N, T


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
        return tau[None,:] if isinstance(tau, np.ndarray) else np.array([[tau]])

    def ln_poisson_likelihood(tau):
        tau = convert_tau(tau)
        return -2*np.sum(cts*np.log(f(tau)), axis=0)

    def ln_gauss_likelihood(tau):
        tau = convert_tau(tau)
        fi = f(tau)
        return np.sum((cts-fi)**2/fi, axis=0)

    return ln_poisson_likelihood, ln_gauss_likelihood, N

def maximize_likelihood(likelihood, tau_range, epsilon=1e-5):
    """Minizizes the -2ln(likelihood) function thus maximizing the
    likelihood.

    :param likelihood: the -2*ln(likelihood) function
    :param tau_range: the range in which to look for the lifetime
    :param epsilon: the desired precision

    :returns: tau, (sigma minus, sigma plus), -2*ln(likelihood) at minimum

    :rtype:
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
    return tau, (sigma_minus, sigma_plus), l


###############################################################################
#                                  Plot Porn                                  #
###############################################################################

def pinmp_ticks(axis, ticks):
    axis.set_major_locator(ticker.MaxNLocator(ticks))
    axis.set_minor_locator(ticker.MaxNLocator(ticks*10))
    return axis

def set_up_plot(ticks=10, pimp_top=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    pinmp_ticks(ax.xaxis, ticks)
    pinmp_ticks(ax.yaxis, ticks)

    ax.grid(which='minor', alpha=.3)
    ax.grid(which='major', alpha=.8)


    if pimp_top:
        ax.tick_params(right=True, top=True, which='both')
    else:
        ax.tick_params(right=True, which='both')

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

def plot_spectrum(counts, offset=1, save=None, **pyplot_args):
    fig, ax = set_up_plot()

    channels = np.arange(0, counts.size) + offset
    ax.step(channels, counts, **pyplot_args)

    ax.set_xlabel('Kanal')
    ax.set_ylabel('Counts')
    ax.set_xlim([channels[0], channels[-1]])
    ax.set_ylim(0)

    if save:
        save_fig(fig, *save)

    return fig, ax
