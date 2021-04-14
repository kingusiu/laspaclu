import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm


def plot_latent_space_2D(latent_coords, title_suffix="data", filename_suffix="data", fig_dir='fig'): # ndarray [M x K] 
    '''
        plot latent space coordinates by slices of 2D
        latent_coords ... ndarray of sim [M x K] containing M samples with each K latent coordinates
    '''

    latent_dim_n = latent_coords.shape[1] - 1 if latent_coords.shape[1] % 2 else latent_coords.shape[1] # if num latent dims is odd, slice off last dim

    heatmaps = []
    xedges = []
    yedges = []

    for i in range(0, latent_dim_n, 2):
        heatmap, xedge, yedge = np.histogram2d(latent_coords[:, i], latent_coords[:, i+1], bins=70)
        heatmaps.append(heatmap)
        xedges.append(xedge)
        yedges.append(yedge)

    heatmaps = np.asarray(heatmaps)
    xedges = np.asarray(xedges)
    yedges = np.asarray(yedges)

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


def plot_latent_space_2D_bg_vs_sig(latent_bg, sig_latent, title_suffix="data", filename_suffix="data", fig_dir='fig'):
    '''
        plot latent space coordinates by slices of 2D
        latent_coords ... ndarray of sim [M x K] containing M samples with each K latent coordinates
    '''

    latent_dim_n = latent_bg.shape[1] - 1 if latent_bg.shape[1] % 2 else latent_bg.shape[1] # if num latent dims is odd, slice off last dim

    heatmaps_bg = []
    xedges_bg = []
    yedges_bg = []
    heatmaps_sig = []
    xedges_sig = []
    yedges_sig = []

    for i in range(0, latent_dim_n, 2):
        heatmap, xedge, yedge = np.histogram2d(latent_bg[:, i], latent_bg[:, i+1], bins=70)
        heatmaps_bg.append(heatmap)
        xedges_bg.append(xedge)
        yedges_bg.append(yedge)

    heatmaps = np.asarray(heatmaps)
    xedges = np.asarray(xedges)
    yedges = np.asarray(yedges)

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
    fig.savefig(os.path.join(fig_dir, 'latent_space_2D_contour_' + filename_suffix + '.png'))
    plt.close(fig)