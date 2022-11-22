

logger.info('plotting classic cluster assignments')

plot.plot_clusters_pairplot(latent_coords, cluster_assign, cluster_centers, filename_suffix=params.cluster_alg+'_'+sample_id, fig_dir=fig_dir)


logger.info('plotting quantum cluster assignments')
plot.plot_clusters_pairplot(latent_coords, cluster_assign_q, cluster_centers, filename_suffix='qmeans_'+str(params.run_n)+'_'+sample_id, fig_dir=fig_dir)
