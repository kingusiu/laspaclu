import os

def plot_latent_space_2D(mu, log_sigma, title_suffix="data", filename_suffix="data", fig_dir='fig'):

    num_latent_dim = mu.shape[1]

    heatmaps = []
    xedges = []
    yedges = []

    for i in np.arange(num_latent_dim):
        heatmap, xedge, yedge = np.histogram2d(np.array(mu[:, i]), np.array(log_sigma[:, i]), bins=70)
        heatmaps.append(heatmap)
        xedges.append(xedge)
        yedges.append(yedge)

    heatmaps = np.asarray(heatmaps)
    xedges = np.asarray(xedges)
    yedges = np.asarray(yedges)

    min_hist_val = np.min(heatmaps[heatmaps > 0])  # find min value for log color bar clipping zero values
    max_hist_val = np.max(heatmaps)

    extent = [np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)]

    nrows = int(round(math.sqrt(num_latent_dim)))
    ncols = math.ceil(math.sqrt(num_latent_dim))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 9))

    for d, ax in zip(np.arange(num_latent_dim), axs.flat[:num_latent_dim]):
        im = ax.imshow(heatmaps[d].T, extent=extent, origin='lower',
                       norm=colors.LogNorm(vmin=min_hist_val, vmax=max_hist_val))
        ax.set_title('dim ' + str(d), fontsize='small')
        # fig.colorbar(im, ax=ax)

    for a in axs[:, 0]: a.set_ylabel('log sigma')
    for a in axs[-1, :]: a.set_xlabel('mu')
    for a in axs.flat: a.set_xticks(a.get_xticks()[::2])
    if axs.size > num_latent_dim:
        for a in axs.flat[num_latent_dim:]: a.axis('off')

    plt.suptitle('mu vs sigma ' + title_suffix)
    # sns.jointplot(x=mu[:,0], y=log_sigma[:,0], kind='hex')
    cb = fig.colorbar(im)
    cb.set_label('count')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'latent_space_2D_hist_mu_vs_sig_' + filename_suffix + '.png'))
    plt.close(fig)
