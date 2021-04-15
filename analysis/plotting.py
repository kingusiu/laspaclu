import os
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm


def make_pairwise_heatmaps(data, num_dim):
    heatmaps = []
    xedges = []
    yedges = []

    for i in range(0, num_dim, 2):
        heatmap, xedge, yedge = np.histogram2d(data[:, i], data[:, i+1], bins=70)
        heatmaps.append(heatmap)
        xedges.append(xedge)
        yedges.append(yedge)

    return np.asarray(heatmaps), np.asarray(xedges), np.asarray(yedges)


def plot_latent_space_2D(latent_coords, title_suffix="data", filename_suffix="data", fig_dir='fig'): # ndarray [M x K] 
    '''
        plot latent space coordinates by slices of 2D
        latent_coords ... ndarray of sim [M x K] containing M samples with each K latent coordinates
    '''

    latent_dim_n = latent_coords.shape[1] - 1 if latent_coords.shape[1] % 2 else latent_coords.shape[1] # if num latent dims is odd, slice off last dim

    heatmaps, xedges, yedges = make_pairwise_heatmaps(latent_coords, latent_dim_n)    

    min_hist_val = np.min(heatmaps[heatmaps > 0])  # find min value for log color bar clipping zero values
    max_hist_val = np.max(heatmaps)

    extent = [np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)]

    # calculate number of subplots on canvas
    nrows = int(round(math.sqrt(latent_dim_n/2)))
    ncols = math.ceil(math.sqrt(latent_dim_n/2))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 9))

    for d, ax in zip(range(len(heatmaps)), axs.flat[:latent_dim_n]):
        im = ax.imshow(heatmaps[d].T, extent=extent, origin='lower',
                       norm=colors.LogNorm(vmin=min_hist_val, vmax=max_hist_val))
        ax.set_title('dims {} & {}'.format(d*2, d*2+1), fontsize='small')
        # fig.colorbar(im, ax=ax)

    for a in axs.flat: a.set_xticks(a.get_xticks()[::2])
    if axs.size > latent_dim_n/2:
        for a in axs.flat[int(latent_dim_n/2):]: a.axis('off')

    plt.suptitle('latent space ' + title_suffix)
    cb = fig.colorbar(im)
    cb.set_label('count')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'latent_space_2D_hist_' + filename_suffix + '.png'))
    plt.close(fig)


def plot_latent_space_2D_bg_vs_sig(latent_bg, latent_sig, title_suffix="data", filename_suffix="data", fig_dir='fig'):
    '''
        plot latent space coordinates by slices of 2D
        latent_coords ... ndarray of sim [M x K] containing M samples with each K latent coordinates
    '''

    latent_dim_n = latent_bg.shape[1] - 1 if latent_bg.shape[1] % 2 else latent_bg.shape[1] # if num latent dims is odd, slice off last dim

    heatmaps_bg, xedges_bg, yedges_bg = make_pairwise_heatmaps(latent_bg, latent_dim_n)
    heatmaps_sig, xedges_sig, yedges_sig = make_pairwise_heatmaps(latent_sig, latent_dim_n)

    min_hist_val = min(np.min(heatmaps_bg[heatmaps_bg > 0]), np.min(heatmaps_sig[heatmaps_sig > 0]))  # find min value for log color bar clipping zero values
    max_hist_val = max(np.max(heatmaps_bg), np.max(heatmaps_sig))

    extent = [min(np.min(xedges_bg), np.min(xedges_sig)), max(np.max(xedges_bg), np.max(xedges_sig)), \
              min(np.min(yedges_bg), np.min(yedges_sig)), max(np.max(yedges_bg), np.max(yedges_sig))]

    # calculate number of subplots on canvas
    nrows = int(round(math.sqrt(latent_dim_n/2)))
    ncols = math.ceil(math.sqrt(latent_dim_n/2))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 9))
    # make dark controur lines
    levels_n = 6
    colors_bg = [mpl.colors.rgb2hex(cm.get_cmap('Blues')(int(i))) for i in np.linspace(60, 350, levels_n)]
    colors_sig = [mpl.colors.rgb2hex(cm.get_cmap('Oranges')(int(i))) for i in np.linspace(60, 350, levels_n)]

    for d, (heat_bg, heat_sig, ax) in enumerate(zip(heatmaps_bg, heatmaps_sig, axs.flat[:latent_dim_n])):
        cont_bg = ax.contour(heat_bg.T, extent=extent, colors=colors_bg, levels=levels_n)
        cont_sig = ax.contour(heat_sig.T, extent=extent, colors=colors_sig, levels=levels_n)
        ax.set_title('dims {} & {}'.format(d*2, d*2+1), fontsize='small')
        # fig.colorbar(im, ax=ax)

    for a in axs.flat: a.set_xticks(a.get_xticks()[::2])
    if axs.size > latent_dim_n/2:
        for a in axs.flat[int(latent_dim_n/2):]: a.axis('off')

    plt.legend([cont_bg.collections[0], cont_sig.collections[0]], ['Background', 'Signal'], loc='center')
    plt.suptitle('latent space ' + title_suffix)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'latent_space_2D_contour_' + filename_suffix + '.png'))
    plt.close(fig)


def plot_kmeans(data, cluster_assignemnts, cluster_centers):

    latent_dim_n = data.shape[1] - 1 if data.shape[1] % 2 else data.shape[1] # if num latent dims is odd, slice off last dim

    for i in range(0, latent_dim_n, 2):
        plt.scatter(data[:,i], data[:,i+1], c=cluster_assignemnts, cmap='')