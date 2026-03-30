import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap

def calculate_stats(y, y_pred, outlier_f=0.15):
    delta = (y_pred - y)/(1+y)
    bias = np.mean(delta)
    nmad = 1.4826*np.median(np.abs(delta - np.median(delta)))
    outlier_fraction = np.sum(np.abs(delta)>outlier_f)/len(y)
    nmad_err = nmad/np.sqrt(2*len(y))
    outlier_f_error = np.sqrt(outlier_fraction*(1-outlier_fraction)/len(y))
    return bias, nmad, outlier_fraction, nmad_err, outlier_f_error

def get_cmap_white(cmap):
    mycmap = plt.get_cmap(cmap, 256)
    newcolors = mycmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 0])
    newcolors[:1, :] = white
    newcolors[:1, :] = white
    mycmap_white = ListedColormap(newcolors, name=f"{mycmap.name}_white")
    return mycmap_white
    
def photoz_plot_2d_hist(fig=None, ax=None, x=None, y=None, cmap=None,  bins=100, cmin=0, cmax=100,\
                        range=[[0,4], [0,4]], lines_color='m', lw=2, outlier_f=0.15):

    x_plot = np.linspace(0, 4, 100)
    y_p = outlier_f*(1+x_plot)+x_plot
    y_m = -outlier_f*(1+x_plot)+x_plot
    ax.plot(x_plot,x_plot, ls='-', lw=lw, c=lines_color)
    ax.plot(x_plot, y_p, ls='--', lw=lw, c=lines_color)
    ax.plot(x_plot, y_m, ls='--', lw=lw, c=lines_color)
    
    sd = ax.hist2d(x, y, cmap=cmap, bins=bins, range=range, cmin=cmin, cmax=cmax)
    bias, nmad, outlier_fraction, _, _ = calculate_stats(x, y, outlier_f=outlier_f)
    ax.annotate(f'Bias: {bias:.3f}\nNMAD: {nmad:.3f}\nOutliers : {outlier_fraction*100:.1f}%', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)

    return fig, ax, sd[3]

    