import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import matplotlib
import matplotlib.ticker as ticker
import os
from SecondaryValue import SecondaryValue
from scipy.optimize import curve_fit

###############################################################################
#                                  Auxiliary                                  #
###############################################################################

def load_counts(path):
    """Parses the TDC data from the text format.

    :param path: path to the log
    :returns: counts
    """


    data = np.loadtxt(path, encoding='latin1',
                      usecols=[0], dtype=np.integer, delimiter=',')

    return data

def energy_a(channel):
    """Channel to Energy, one based index."""
    return 81 + (.179)*channel


def energy_b(channel):
    """Channel to Energy, one based index."""
    return 100 + .220*channel

channel_to_time_val = SecondaryValue('a + b*(k-1/2)', defaults=dict(a=(-.014, .0192), b=(.0483, 2e-5)))
def channel_to_time(channel, d=0):
    return channel_to_time_val(k=(channel, d))
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

def set_up_plot(ticks=10, pimp_top=True):
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
    ax.set_ylabel('Ereignisszahl')
    ax.set_xlim([channels[0], channels[-1]])
    ax.set_ylim(0)

    if save:
        save_fig(fig, *save)

    return fig, ax

###############################################################################
#                           Time of Flight Analysiss                          #
###############################################################################

def gauss(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def find_and_plot_peak(counts, ax, label):
    channels = np.arange(0, counts.size) + 1
    counts = counts.copy()/counts.max()
    times, d_times = channel_to_time(channels)

    splot = ax.step(times, counts, label=label, alpha=.4)
    opt, cov = curve_fit(gauss, channels, counts, p0=(1, counts.argmax(), 100))
    cov = np.sqrt(np.diag(cov))
    gplot = ax.plot(times, gauss(channels,*opt), label=f"Fit {label}", color=splot[0].get_color())
    ax.axvline(channel_to_time(opt[1])[0], color=gplot[0].get_color())
    t, dt = channel_to_time(opt[1], opt[2]/10)
    ax.axvspan(t-dt, t+dt, alpha=.2, color=gplot[0].get_color())
    return channel_to_time(opt[1], opt[2]/10)


###############################################################################
#                            Fake PET Reconstrucion                           #
###############################################################################

def calc_projection_matrix(source, weights):
    proj = np.zeros((4, 5))
    N = source.shape[0]

    # 0
    proj[0,:] = np.sum(source, axis=0)

    # 90
    proj[2,:] = np.flip(np.sum(source, axis=1))

    # diag
    proj[1,:] = np.array(np.array([np.sum(source.diagonal(n)) for n in range(-N//2 - 1, N//2 + 3)])) @ weights
    source = np.flip(source, 0)

    proj[3,:] = np.flip([np.sum(source.diagonal(n)) for n in range(-N//2 - 1, N//2 + 3)] @ weights)
    return proj

def convolute_projection(proj):
    filt = np.array([-.1, .25, -.1])
    conv = np.zeros_like(proj)
    for row in range(0, proj.shape[0]):
        conv[row] = np.convolve(proj[row], filt)[1:-1]
    return conv

def reconstruct_diags(proj, weights):
   return proj[1] @ weights, proj[3] @ weights

def reproject_diagonal(diag):
    N = diag.size
    d = int((diag.size + 1)/2)
    rep = np.zeros((d, d))

    for di, elem in enumerate(diag):
        offset = int(d - di) - 1
        for i in range(0, d - int(np.abs(offset))):
            if offset >= 0:
                rep[i + offset][i] = elem
            else:
                rep[i][i - offset] = elem

    return rep

def reproject_straight(row):
    d = row.size
    rep = np.empty((d, d))

    for i, elem in enumerate(row):
        rep[:, i] = elem
    return rep

def reconstruct(proj, backweights):
    p0, p90 = proj[0], proj[2]  # uuuugly
    p45, p135 = reconstruct_diags(proj, backweights)
    print('Reconstructed Diagonals')
    print('45', bmatrix(p45))
    print('135', bmatrix(p135))

    r0 = reproject_straight(p0)
    r90 = np.flip(reproject_straight(p90).T)
    r45 = reproject_diagonal(p45)
    r135 = np.flip(reproject_diagonal(p135).T, 1)

    print('0', bmatrix(r0))
    print('45', bmatrix(r45))
    print('90', bmatrix(r90))
    print('135', bmatrix(r135))
    return r0 + r90 + 1/2*(r45 + r135)

def bmatrix(a, prec=3):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    with np.printoptions(precision=prec, suppress=True):
        if len(a.shape) > 2:
            raise ValueError('bmatrix can at most display two dimensions')
        lines = str(a).replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{pmatrix}']
        rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        rv +=  [r'\end{pmatrix}']
        return '\n'.join(rv)

def create_matrix_image(mat, show_latex=True, save=None):
    fig, ax = plt.subplots()
    ax.matshow(mat, cmap='Greys')

    name = None
    if save:
        name = save[0]
        save_fig(fig, *save, size=(2,2))

    if show_latex:
        print(name, bmatrix(mat))

    return fig, ax
