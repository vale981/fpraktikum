import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import matplotlib
import matplotlib.ticker as ticker
import os
import re
from SecondaryValue import SecondaryValue
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

###############################################################################
#                                  Auxiliary                                  #
###############################################################################


def normalize(array):
    tmp = array.copy()
    tmp = tmp - tmp.min()
    return tmp/tmp.max()

def load_profiles(path):
    """Parses the measured Gold profiles.

    :param path: path to the log
    :returns: numpy array, (x, amplitude)
    """

    skip = 0
    with open(path, 'r') as f:
        for line in f:
            if 'Values' in line:
                break
            skip += 1

    data = np.loadtxt(path, encoding='latin1', dtype=np.float, skiprows=skip + 1)

    data[:, 1] = normalize(data[:, 1])
    return data

def scientific_round(val, *err):
    """Scientifically rounds the values to the given errors."""
    val, err = np.asarray(val), np.asarray(err)
    if len(err.shape) == 1:
        err = np.array([err])
        err = err.T
    err = err.T

    if err.size == 1 and val.size > 1:
        err = np.ones_like(val)*err

    if len(err.shape) == 0:
        err = np.array([err])

    if val.size == 1 and err.shape[0] > 1:
        val = np.ones_like(err)*val

    i = np.floor(np.log10(err))
    first_digit = (err // 10**i).astype(int)
    prec = (-i + np.ones_like(err) * (first_digit <= 3)).astype(int)
    prec = np.max(prec, axis=1)

    def smart_round(value, precision):
        value = np.round(value, precision)
        if precision <= 0:
            value = value.astype(int)
        return value

    if val.size > 1:
        rounded = np.empty_like(val)
        rounded_err = np.empty_like(err)
        for n, (value, error, precision) in enumerate(zip(val, err, prec)):
            rounded[n] = smart_round(value, precision)
            rounded_err[n] = smart_round(error, precision)

        return rounded, rounded_err
    else:
        prec = prec[0]
        return smart_round(val, prec), *smart_round(err, prec)[0]

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

def set_up_plot(ticks=10, pimp_top=True, subplot=111, fig=None):
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot)

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
  \label{fig:''' + folder + '-' + title + r'''}
\end{figure}
    ''')

def plot_profile_common(profile, **pyplot_args):
    x, amp = profile.T
    fig, ax = set_up_plot()

    ax.step(x, amp, label='Intensität', **pyplot_args)
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylabel('relative Intensit\"a')
    ax.legend()

    return fig, ax


def plot_profile(profile, **pyplot_args):
    fig, ax = plot_profile_common(profile, **pyplot_args)

    ax.set_xlabel('x [nm]')

    return fig, ax

def plot_diffr_profile(profile, **pyplot_args):
    fig, ax = plot_profile_common(profile, **pyplot_args)

    ax.set_xlabel('x [1/nm]')

    return fig, ax

def analyze_diffr_profile(profile, limits, save=None, **peak_args):
    x, amp = profile.T
    fig, ax = plot_diffr_profile(profile)

    peaks, peak_info = find_peaks(amp[limits[0]:limits[1]], width=0, **peak_args)

    peaks += limits[0]

    ax.plot(x[peaks], amp[peaks], "x", label='Peaks')
    ax.axvspan(x[limits[0]], x[limits[1]], color='gray', zorder=-1, alpha=.2,
               label='Auswertungsbereich')
    ax.legend()

    candidates = 1/x[peaks]
    d_candidates = candidates**2*(x[1]-x[0])
    sigma_candidates = candidates**2*peak_info['widths']*(x[1]-x[0])
    return candidates, d_candidates, sigma_candidates

def analyze_profile(profile, limits=(0, -1), save=None, **peak_args):
    x, amp = profile.T
    fig, ax = plot_profile(profile)

    peaks, _ = find_peaks(amp[limits[0]:limits[1]], **peak_args)
    peaks += limits[0]

    ax.plot(x[peaks], amp[peaks], "x", label='Peaks')
    ax.axvspan(x[limits[0]], x[limits[1]], color='gray', zorder=-1, alpha=.2,
               label='Auswertungsbereich')
    ax.legend()

    dx = (x[1] - x[0])
    numpeaks = peaks.size - 1
    sigma = (x[peaks[1:]] - x[peaks[:-1]]).std()/np.sqrt(numpeaks)
    l = (x[peaks[-1]] - x[peaks[0]])/numpeaks
    dl = np.sqrt(2)*dx/numpeaks

    if save:
        save_fig(fig, *save)

    return l, dl, sigma

def find_miller_indices(squares):
    squares = np.asarray(squares)
    if squares.size > 1:
        return np.array([find_miller_indices(x) for x in squares])
    square = squares
    return np.array([(a, b, c) for (a, b, c) \
                in np.ndindex((square+1, square+1, square+1)) \
       if a**2 + b**2 + c**2 == square and a >= b >= c])

def can_be_sum_of_squares(square):
    for a, b, c in np.ndindex((square+1, square+1, square+1)):
        if a**2 + b**2 + c**2 == square:
            return True

    return False

def generate_miller_table(squares):
    squares = np.unique(squares)
    inds = find_miller_indices(squares)
    out = ''
    for i, ind_list in zip(squares, inds):
        out += f'{i + 1} & '
        for ind in ind_list:
            out += r'\mqty{' + ' & '.join(ind.astype(str)) + '}, '
        out = out[:-2]

        out += r' \\' + '\n'
    return out

def evaluate_hypothesis(analyzed, maximum=10, gold=.4078):
    diffs = np.empty((maximum, analyzed.shape[0]))

    squared_ds = np.array([x for x in np.arange(1, maximum + 1, 1) \
                           if can_be_sum_of_squares(x)])
    ds = np.sqrt(squared_ds)
    a = analyzed[:,0][:, None] * ds[None, :]
    diff = np.abs(a - gold)
    mindiff = np.argmin(diff, axis=1)
    return squared_ds[mindiff], analyzed[:]*ds[mindiff,None],  diff.min(axis=1)

def generate_hypethsesis_table(squared, analyzed, residues):
    out = ''
    for square, value, residue in zip(squared, analyzed, residues):
        value = np.array(scientific_round(*value))

        out += rf'\(\sqrt{{{square}}}\) & ' \
        + ' & '.join(value.astype(str)) + f' & {residue:.3f} \\\\\n'

    return out

def determine_lattice_constant(hypothesis):
    """
    Calculate the weighted mean and standard deviation by using the
    combined deviation as weights.

    The systemic deviation is calculated by error propagation.
    """
    a_s = hypothesis[1][:,0]
    syst_err = hypothesis[1][:,1] + hypothesis[1][:,2]
    weights = 1/syst_err**2
    a = np.average(a_s, weights=weights)
    d_a = np.sqrt(1/np.sum(weights))
    sigma_a = np.sqrt(np.average((a_s-a)**2, weights=weights))

    return a, d_a, sigma_a
