import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.ticker as ticker
import os
from SecondaryValue import SecondaryValue
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.ndimage as ndimage
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
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
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if title:
        ax.set_title(title)

    x, y = np.arange(0, reconstruction.shape[0]),np.arange(0, reconstruction.shape[1])
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


def find_peak_positions(reconstruction, threshold=.1, save=None):
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

        peaks += [cov]

    if save:
        save_fig(fig, save[0], save[1], size=(2,4))
    return peaks
