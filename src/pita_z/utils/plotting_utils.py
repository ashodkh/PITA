import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap
from pita_z.utils.metrics import bias_nmad_outliers

def get_cmap_white(cmap):
    mycmap = plt.get_cmap(cmap, 256)
    newcolors = mycmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 0])
    newcolors[:1, :] = white
    newcolors[:1, :] = white
    mycmap_white = ListedColormap(newcolors, name=f"{mycmap.name}_white")
    return mycmap_white
    
def photoz_plot_2d_hist(fig=None, ax=None, x=None, y=None, cmap=None,  bins=100, vmin=0, vmax=100,\
                        range=[[0,4], [0,4]], lines_color='m', lw=2, outlier_threshold=0.15, remove_outliers_for_bias=True):

    x_plot = np.linspace(0, 4, 100)
    y_p = outlier_threshold*(1+x_plot)+x_plot
    y_m = -outlier_threshold*(1+x_plot)+x_plot
    ax.plot(x_plot,x_plot, ls='-', lw=lw, c=lines_color)
    ax.plot(x_plot, y_p, ls='--', lw=lw, c=lines_color)
    ax.plot(x_plot, y_m, ls='--', lw=lw, c=lines_color)
    
    sd = ax.hist2d(x, y, cmap=cmap, bins=bins, range=range, vmin=vmin, vmax=vmax)
    bias, nmad, outlier_fraction, _, _ = bias_nmad_outliers(x, y, outlier_threshold, remove_outliers_for_bias)
    ax.annotate(f'Bias: {bias:.4f}\nNMAD: {nmad:.4f}\nOutliers : {outlier_fraction*100:.2f}%', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)

    return fig, ax, sd[3]
    