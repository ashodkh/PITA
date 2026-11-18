import numpy as np
import pylab as plt
from pita_z.utils import plotting_utils as pu
from matplotlib.backends.backend_pdf import PdfPages

def cde_plots(
    cdes=None,
    cdfs=None,
    z_grid=np.linspace(0,4,201),
    z_ref=None,
    config_file=None,
    run=None,
    load_epoch=None,
    out_dir="/pscratch/sd/a/ashodkh/calpit/plots/"
):
    pp = PdfPages(out_dir + f'cde_plots_{config_file}_{run}_{load_epoch}.pdf')
    d_z = z_grid[1] - z_grid[0]
    if cdfs is None:
        cdfs = np.cumsum(cdes, axis=1) * d_z

    max_idxs = np.argmax(cdes, axis=1)
    z_max = z_grid[max_idxs]

    fig, ax = plt.subplots()
    
    fig, ax, sd = pu.photoz_plot_2d_hist(
        fig=fig,
        ax=ax,
        x=z_ref,
        y=z_max,
        cmap=pu.get_cmap_white('inferno'),
        cmax=20
    )

    fig.colorbar(sd, ax=ax)
    ax.set_xlabel(r'$z_\mathrm{ref}$')
    ax.set_ylabel(r'z mode')
    pp.savefig(fig, bbox_inches='tight')

    z_ref_idxs = np.searchsorted(z_grid, z_ref)
    z_ref_idxs[z_ref_idxs == len(z_grid)] = len(z_grid) - 1

    fig, ax = plt.subplots()

    ax.hist(cdfs[np.arange(len(cdfs)), z_ref_idxs], histtype='step', bins=100, range=(0,1), label=f"")
    

    