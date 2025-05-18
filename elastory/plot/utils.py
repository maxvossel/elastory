import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, to_rgba
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def colors_light_to_dark(color, num_colors, resolution=100):
    """
    Returns an array of colors with the hue of the given color,
    the colors are adapted to range from lighter to darker.
    """

    if (type(color) is tuple) and (len(color) == 4):
        color = color[:-1]
    elif (type(color) is tuple) and (len(color) == 3):
        color = color
    elif isinstance(color, str):
        color = to_rgba(color)

    hue = rgb_to_hsv(color[:-1])[0]
    my_hue = myHSVcmap(hue, rgba=True)
    assert isinstance(my_hue, dict)
    colors = plt.cm.get_cmap(LinearSegmentedColormap("mycmap", my_hue))

    colors = colors(np.linspace(0.2, 0.8, resolution))
    colors = colors[np.linspace(0, resolution - 1, num_colors, dtype=int)]
    return colors


def myHSVcmap(H, rgba=True) -> dict | np.ndarray:
    """Returns a hsv color map around the hue given
    Args:
        H (float): color hue value

    Returns:
        (dict): Color Dict for matlab"""
    resolution = 2
    lowerbound = 0.0
    upperbound = 1.0

    half = np.linspace(lowerbound, upperbound, 10**resolution)
    ones = np.ones(10**resolution) * upperbound

    lins = np.linspace(0, 1, 2 * 10**resolution - 1)
    lins = np.append(lins, 1)

    HSV_array = np.c_[
        np.r_[ones * H, ones * H],
        np.r_[half, ones],
        np.r_[0.95 * ones, -0.05 + lowerbound + upperbound - (half * 0.5)],
    ]
    RGBA_array = mplcolors.hsv_to_rgb(HSV_array)

    if not rgba:
        arr: np.ndarray = np.c_[lins, HSV_array]
        return arr

    else:
        arr = np.c_[lins, RGBA_array]
        r = g = b = []
        for i in np.arange(3):
            r = list(map(tuple, np.c_[arr[:, 0], arr[:, 1], arr[:, 1]]))
            g = list(map(tuple, np.c_[arr[:, 0], arr[:, 2], arr[:, 2]]))
            b = list(map(tuple, np.c_[arr[:, 0], arr[:, 3], arr[:, 3]]))

        cdict: dict[str, list] = {"red": r, "green": g, "blue": b}
        return cdict


def fill_between_steps(x, y1, y2=0, h_align="right", ax=None, **kwargs):
    """Fills a hole in matplotlib: fill_between for step plots.
    Parameters :
    ------------
    x : array-like
        Array/vector of index values. These are assumed to be equally-spaced.
        If not, the result will probably look weird...
    y1 : array-like
        Array/vector of values to be filled under.
    y2 : array-Like
        Array/vector or bottom values for filled area. Default is 0.
    **kwargs will be passed to the matplotlib fill_between() function.
    """
    # If no Axes opject given, grab the current one:
    if ax is None:
        ax = plt.gca()
    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1])

    # Make it possible to chenge step alignment.
    if h_align == "mid":
        xx -= xstep / 2.0
    elif h_align == "right":
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)  # [:-1]
    if type(y2) is np.ndarray:
        y2 = y2.repeat(2)  # [:-1]

    # now to the plotting part:
    ax.fill_between(xx, y1, y2=y2, **kwargs)

    return ax


def colorbar(ax, mat):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mat, cax=cax)


def axes_formatter(axs, inside=True):
    for ax in axs:
        plt.setp(ax.spines.values(), color="darkgrey")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.tick_params(axis="both", which="major", labelsize=6)
        # ax.tick_params(axis = 'both', which = 'minor', labelsize = 3)
        ax.tick_params(which="minor", width=0.25)
        ax.tick_params(which="major", width=0.75)
        if inside:
            ax.tick_params(axis="both", which="both", direction="in")  # , pad=-22)
        else:
            ax.tick_params(axis="both", which="both", direction="out")  # , pad=-22)


def cmap_map(function, cmap):
    """Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ("red", "green", "blue"):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))

    # Then compute the LUT, and apply the function to the LUT
    def reduced_cmap(step):
        return np.array(cmap(step)[0:3])

    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(["red", "green", "blue"]):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1],), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return LinearSegmentedColormap("colormap", cdict, 1024)


def plot_labels(axs, fig):
    for label, ax in axs.items():
        trans = mtransforms.ScaledTranslation(5 / 72, -5 / 72, fig.dpi_scale_trans)
        ax.text(
            0.03,  # offset from left
            0.99,  # offset from top
            label,
            transform=ax.transAxes + trans,
            fontsize="7",
            verticalalignment="top",
            horizontalalignment="left",
            fontfamily="sans",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="1",
                alpha=0.85,
                edgecolor="none",
                pad=0.8,  # padding
            ),
        )
