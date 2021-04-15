import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import pathlib
import tensorflow as tf
import numpy as np
from collections import namedtuple

import inference.train_autoencoder as train
import inference.predict_autoencoder as pred
import inference.kmeans as kmeans
import analysis.plotting as plot
import anpofah.util.plotting_util as pu
import anpofah.model_analysis.roc_analysis as roc


def make_model_path(date=None):
    date_str = ''
    if date is None:
        date = datetime.date.today()
        date = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    path = os.path.join('models/saved', 'AEmodel_{}'.format(date))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


euclidian_dist = lambda a,b: np.sqrt(np.sum((a-b)**2,axis=1))


#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'load_ae epochs read_n sample_id_train')
params = Parameters(load_ae=False, epochs=300, read_n=int(3e6), sample_id_train='qcdSide')


model_path = make_model_path()


#****************************************#
#           Autoencoder
#****************************************#


if params.load_ae:
    ae_model = tf.saved_model.load(model_path)

else:
    # train AE model
    ae_model = train.train(params.sample_id_train, epochs=params.epochs, read_n=params.read_n)

    # model save /load
    tf.saved_model.save(ae_model, model_path) 

# apply AE model
latent_coords_qcd = pred.map_to_latent_space(params.sample_id_train, ae_model, read_n=params.read_n)

#****************************************#
#               K-MEANS
#****************************************#


# train k-means model
print('>>> training kmeans')
km_model = kmeans.train(latent_coords_qcd)

# apply to data -> obtain cluster assignment
cluster_centers = km_model.cluster_centers_
# qcdSide
cluster_assignemnts_qcd = km_model.predict(latent_coords_qcd) # latent coords of background obtained from AE
 

# plot kmeans clustering
plot.plot_kmeans_clusters(latent_coords_qcd, cluster_assignemnts_qcd, cluster_centers, title_suffix=params.sample_id_train, filename_suffix=params.sample_id_train)


#******************************************#
#         AE + KMEANS ON TESTSET
#******************************************#


sample_id_qcd_test = 'qcdSideExt'
sample_id_sig = 'GtoWW35na'

# apply autoencoder
latent_coords_qcd_test = pred.map_to_latent_space(sample_id_qcd_test, ae_model, read_n=params.read_n)
latent_coords_sig = pred.map_to_latent_space(sample_id_sig, ae_model, read_n=params.read_n)

# plot latent space
plot.plot_latent_space_2D_bg_vs_sig(latent_coords_qcd, latent_coords_sig, title_suffix=' '.join([params.sample_id_train, 'vs', sample_id_sig]), filename_suffix='_'.join([params.sample_id_train, 'vs', sample_id_sig]), fig_dir='fig')
plot.plot_latent_space_2D_bg_vs_sig(latent_coords_qcd_test, latent_coords_sig, title_suffix=' '.join([sample_id_qcd_test, 'vs', sample_id_sig]), filename_suffix='_'.join([sample_id_qcd_test, 'vs', sample_id_sig]), fig_dir='fig')

# apply kmeans
cluster_assignemnts_sig = km_model.predict(latent_coords_sig) # latent coords of signal obtained from AE
cluster_assignemnts_qcd_test = km_model.predict(latent_coords_qcd_test) # latent coords of signal obtained from AE
plot.plot_kmeans_clusters(latent_coords_sig, cluster_assignemnts_sig, cluster_centers, title_suffix=sample_id_sig, filename_suffix=sample_id_sig)
plot.plot_kmeans_clusters(latent_coords_qcd_test, cluster_assignemnts_qcd_test, cluster_centers, title_suffix=sample_id_qcd_test, filename_suffix=sample_id_qcd_test)



#****************************************#
#               METRIC
#****************************************#

dist_qcd = euclidian_dist(latent_coords_qcd, cluster_centers[cluster_assignemnts_qcd])
dist_qcd_test = euclidian_dist(latent_coords_qcd_test, cluster_centers[cluster_assignemnts_qcd_test])
dist_sig = euclidian_dist(latent_coords_sig, cluster_centers[cluster_assignemnts_sig])

pu.plot_bg_vs_sig([dist_qcd, dist_sig], legend=[params.sample_id_train,sample_id_sig], xlabel='distance to closest cluster', title='euclidian distance distribution qcd vs sig', plot_name='loss_qcd_vs_sig', fig_dir='fig', ylogscale=True, fig_format='.png')
pu.plot_bg_vs_sig([dist_qcd_test, dist_sig], legend=[sample_id_qcd_test, sample_id_sig], xlabel='distance to closest cluster', title='euclidian distance distribution qcd vs sig', plot_name='loss_qcd_vs_sig', fig_dir='fig', ylogscale=True, fig_format='.png')
roc.plot_roc([dist_qcd], [dist_sig], legend=[params.sample_id_train,sample_id_sig], title=' '.join([params.sample_id_train, 'vs', sample_id_sig]), plot_name='_'.join(['ROC', params.sample_id_train, 'vs', sample_id_sig]), fig_dir='fig')
roc.plot_roc([dist_qcd_test], [dist_sig], legend=[sample_id_qcd_test, sample_id_sig], title=' '.join([sample_id_qcd_test, 'vs', sample_id_sig]), plot_name='_'.join(['ROC', sample_id_qcd_test, 'vs', sample_id_sig]), fig_dir='fig')
