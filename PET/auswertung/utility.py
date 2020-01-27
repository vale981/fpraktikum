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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.ndimage as ndimage
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

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

channel_to_time_val = SecondaryValue('a + b*(k-1/2)',
                                     defaults=dict(a=(-.014, .0192), b=(.0483, 2e-5)))
def channel_to_time(channel, d=0):
    return channel_to_time_val(k=(channel, d))

def time_to_channel(time):
    a = -.014
    b = .0483
    return ((time - a)/b).astype(int)

def scientific_round(val, err):
    """Scientifically rounds the values to the given errors."""
    val, err = np.asarray(val), np.asarray(err)

    if err.size == 1 and val.size > 1:
        err = np.ones_like(val)*err

    if val.size == 1 and err.size > 1:
        val = np.ones_like(err)*val

    i = np.floor(np.log10(err))
    first_digit = err // 10**i
    i = i.astype(int)
    prec = (-i + np.ones_like(val) *  (first_digit <= 3)).astype(int)

    def smart_round(value, precision):
        value = np.round(value, precision)
        if precision <= 0:
            value = int(value)
        return value


    if val.size > 1:
        rounded = np.empty_like(val)
        rounded_err = np.empty_like(val)
        for n, (value, error, precision) in enumerate(zip(val, err, prec)):
            rounded[n] = smart_round(value, precision)
            rounded_err[n] = smart_round(error, precision)

        return rounded, rounded_err
    else:
        return smart_round(val, prec), smart_round(err, prec)


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
  \label{fig:''' + folder + '-' + title + '''}
\end{figure}
    ''')
def plot_all(axes, counts, label):
    axt, axa, axb = axes
    c_t, c_a, c_b = counts


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

def gauss(x, a, mu, sigma, o):
    return a*np.exp(-(x-mu)**2/(2*sigma**2)) + o

def find_and_plot_peak(counts, ax, label, time_interval=[40, 70]):
    channels = np.arange(0, counts.size) + 1
    counts = counts.copy()/counts.max()
    times, d_times = channel_to_time(channels)
    interval = time_to_channel(np.array(time_interval))
    splot = ax.step(times, counts, label=label, alpha=.4)
    opt, cov = curve_fit(gauss, channels[interval[0]:interval[1]],
                         counts[interval[0]:interval[1]],
                         sigma=np.sqrt(counts[interval[0]:interval[1]]),
                         absolute_sigma=True,
                         p0=(1, counts.argmax(), 100, .3))
    cov = np.sqrt(np.diag(cov))
    gplot = ax.plot(times, gauss(channels,*opt), label=f"Fit {label}",
                    color=splot[0].get_color())
    ax.axvline(channel_to_time(opt[1])[0], color=gplot[0].get_color())
    t, dt = channel_to_time(opt[1], cov[1])
    ax.axvspan(t-dt, t+dt, alpha=.2, color=gplot[0].get_color())
    return t, dt


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
    r135 = np.flip(reproject_diagonal(p135), 1)

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

###############################################################################
#                             3D Tomography Stuff                             #
###############################################################################

def load_matrix(path):
    """Parses the reconstructed matrix from the text format.

    :param path: path to the matrix
    :returns: matrix
    """


    data = np.loadtxt(path, encoding='latin1')

    return data

def coordinates_to_position(x, y=None, reco_raster=3.375, absolute=False):
    pos = x*reco_raster
    if not absolute:
        pos += reco_raster/2

    return (pos, coordinates_to_position(y, reco_raster=reco_raster,
                                   absolute=absolute)) \
        if y is not None else pos

def coordinates_error_to_position_error(x, y=None, reco_raster=3.375):
    err = x*reco_raster
    return (err, coordinates_error_to_position_error(y, reco_raster=reco_raster)) \
        if y is not None else err

def plot_reconstruction(reconstruction, fig=None, subplot=111, title=None,
                      azim=-45, elev=50, save=None, **pyplot_args):
    fig = fig if fig else plt.figure()
    ax = fig.add_subplot(subplot, projection='3d')

    ax.view_init(azim=azim, elev=elev)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')

    if title:
        ax.set_title(title)

    x, y = coordinates_to_position(x=np.arange(0, reconstruction.shape[0]),
                                   y=np.arange(0, reconstruction.shape[1]))
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, reconstruction, cmap=cm.plasma,
                    shade=True, **pyplot_args)
    fig.tight_layout()

    if save:
        save_fig(fig, *save)
    return fig, ax

def gauss2d(xy, A, mx, my, sigma):
    x, y = xy
    return np.ravel(A*np.exp(-((x-mx)**2 + (y-my)**2)/(2*sigma**2)))

def normalize(array):
    tmp = array.copy()
    tmp = tmp - tmp.min()
    return tmp/tmp.max()


def find_peak_positions(reconstruction, threshold=.4, save=None):
    rec = reconstruction.copy()

    # normalize
    rec_normalized = normalize(rec)

    # create grid with real coordinates
    x, y = np.arange(0, rec.shape[0]), np.arange(0, rec.shape[1])
    x, y = np.meshgrid(x, y)

    # foodprint
    neighborhood = generate_binary_structure(2,4)
    local_max = maximum_filter(rec_normalized, footprint=neighborhood) > threshold

    # extract peaks
    features, num_labels = label(local_max)
    sliced = ndimage.find_objects(features)

    peaks = []

    plotdim = 200 + num_labels*10
    fig = plt.figure()

    for i in range(1, num_labels + 1):
        slc = sliced[i-1]

        # mask input to only show one peak
        masked = rec*(features == i)

        # guess the peak position
        guess = [1, (slc[1].start + slc[1].stop)/2, (slc[0].start + slc[0].stop)/2, 1]

        # fit gauss
        opt, cov = curve_fit(gauss2d, (x, y), np.ravel(masked), p0=guess)
        cov = np.sqrt(np.diag(cov))

        plot_reconstruction(gauss2d((x, y), *opt).reshape(*rec.shape),
                            fig=fig, subplot=plotdim + i, elev=30, title=f"Fit Peak {i}")
        plot_reconstruction(masked, fig=fig, subplot=plotdim + num_labels + i, elev=30, title=f"Peak {i}")

        peaks += [([opt[0]] + list(coordinates_to_position(*opt[1:3])) + [coordinates_to_position(opt[3], absolute=True)],
                   [cov[0]] + list(coordinates_error_to_position_error(*cov[1:3]))  + [coordinates_error_to_position_error(cov[3])])]

    save_fig(fig, save[0], save[1], size=(12, 4))
    return peaks

def peaks_to_table(peaks):
    res = ""
    for i, (peak, err_peak) in enumerate(peaks):
        peak, err_peak = scientific_round(peak, err_peak)
        out = np.empty(2*np.asarray(peak).size)
        out[::2] = peak
        out[1::2] = err_peak
        res += f'Peak {i} & ' + ' & '.join(out.astype(str)) + ' \\\\\n'
    return res

###############################################################################
#                                     Tom2                                    #
###############################################################################

def parse_sinogramm_from_ps(path):
    """Parse a Sinogramm from the given postscript file."""

    mat = []
    dim = None
    res_regex = \
        re.compile(r'gsave \/picstr [0-9]+ string def .*? [0-9]+ scale ([0-9]+) ([0-9]+) [0-9]+ \[.*\]')

    with open(path, 'r') as f:
        for line in f:
            match = res_regex.match(line)
            if match:
              dim = np.flip(np.array(match.groups()).astype(int))
            if line.strip() == "{J} image":
                break

        for line in f.readlines():
            stripped = line.strip()
            if stripped == 'grestore':
                break

            mat += [int(digit, 16) for digit in stripped]

    return np.array(mat).reshape(*dim)

def correction_function(x, a, b, omega, phi, o):
    return a*np.sin(x/180*np.pi*omega + phi) + b*np.sin(2*x/180*np.pi*omega + phi) + o



def get_lowest_order_fft(line, orders=3):
    coeff = fft.rfft(line)
    return (coeff[:orders])

def reconstruct_fft(coeff, n, m):
    return (1/n*(coeff[0].real + np.sum([(coeff[k]*np.exp(2*np.pi*1j*m*k/n)).real*2 for k in range(1, len(coeff))], axis=0))).real

def get_coeff_table(coeff):
    out = r'\(c_i\) & ' + ' & '.join(np.arange(0, len(coeff)).astype(str)) + '\\\\\n &'
    return out + (' & '.join(np.round(coeff, 2).astype(str))).replace('j', 'i').replace('(', '').replace(')', '')

def plot_sinogram(ax, sino, title, degrange=(0, 178), transpose=True):
    ax.imshow(sino.T if transpose else sino, interpolation='none',
              extent=(*degrange, 0, sino.shape[1]))
    ax.set_title(title)
    if transpose:
        ax.set_xlabel(r'Winkel [$\circ$]')
        ax.set_ylabel(r'Bin')
    else:
        ax.set_ylabel(r'Winkel [$\circ$]')
        ax.set_xlabel(r'Bin')

    return ax
