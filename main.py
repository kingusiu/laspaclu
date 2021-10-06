import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import tensorflow as tf
import numpy as np
from collections import namedtuple
from sklearn import svm
import joblib as jli

import data.data_sample as dasa
import inference.predict_autoencoder as pred
import inference.clustering_quantum as cluster_q
import inference.metrics as metr
import analysis.plotting as plot
import util.persistence as pers
import anpofah.util.plotting_util as pu
import anpofah.model_analysis.roc_analysis as roc




#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'load_km latent_dim read_n sample_id_train cluster_alg')
params = Parameters(load_km=False, latent_dim=8, read_n=int(1e4), sample_id_train='qcdSide', cluster_alg='kmeans')

data_sample = dasa.DataSample(params.sample_id_train)


#****************************************#
#           Autoencoder
#****************************************#

model_path_ae = pers.make_model_path(date='20211004', prefix='AE')

print('[main] >>> loading autoencoder ' + model_path_ae)
ae_model = tf.saved_model.load(model_path_ae)

# apply AE model
latent_coords_qcd = pred.map_to_latent_space(data_sample=data_sample, sample_id=params.sample_id_train, model=ae_model, read_n=params.read_n)


#****************************************#
#               CLUSTERING
#****************************************#

#****************************************#
#               KMEANS

if params.cluster_alg == 'kmeans':

    model_path_km = pers.make_model_path(date='20211006', prefix='KM')
    print('[main] >>> loading clustering model ' + model_path_km)

    cluster_model = jli.load(model_path_km+'.joblib')    
    cluster_centers = cluster_model.cluster_centers_


#****************************************#
#           ONE CLASS SVM

else:

    model_path_svm = pers.make_model_path(date='20211006', prefix='SVM')
    print('[main] >>> loading clustering model ' + model_path_svm)

    cluster_model = jli.load(model_path_svm+'.joblib')    
    cluster_centers = None # no cluster centers for one class svm


# apply to qcd training set -> obtain cluster assignment
cluster_assignemnts_qcd = cluster_model.predict(latent_coords_qcd) # latent coords of background obtained from AE
 
# plot kmeans clustering
plot.plot_clusters(latent_coords_qcd, cluster_assignemnts_qcd, cluster_centers, title_suffix=params.cluster_alg+' '+params.sample_id_train, filename_suffix=params.cluster_alg+'_'+params.sample_id_train)



#******************************************#
#         AE + CLUSTER ALGO ON TESTSET
#******************************************#


sample_id_qcd_test = 'qcdSideExt'
sample_id_sig = 'GtoWW35na'

# apply autoencoder
latent_coords_qcd_test = pred.map_to_latent_space(data_sample=data_sample, sample_id=sample_id_qcd_test, model=ae_model, read_n=params.read_n)
latent_coords_sig = pred.map_to_latent_space(data_sample=data_sample, sample_id=sample_id_sig, model=ae_model, read_n=params.read_n)

# plot latent space
plot.plot_latent_space_1D_bg_vs_sig(latent_coords_qcd, latent_coords_sig, title_suffix=' '.join([params.sample_id_train, 'vs', sample_id_sig]), filename_suffix='_'.join([params.sample_id_train, 'vs', sample_id_sig]), fig_dir='fig')
plot.plot_latent_space_1D_bg_vs_sig(latent_coords_qcd_test, latent_coords_sig, title_suffix=' '.join([sample_id_qcd_test, 'vs', sample_id_sig]), filename_suffix='_'.join([sample_id_qcd_test, 'vs', sample_id_sig]), fig_dir='fig')
plot.plot_latent_space_2D_bg_vs_sig(latent_coords_qcd, latent_coords_sig, title_suffix=' '.join([params.sample_id_train, 'vs', sample_id_sig]), filename_suffix='_'.join([params.sample_id_train, 'vs', sample_id_sig]), fig_dir='fig')
plot.plot_latent_space_2D_bg_vs_sig(latent_coords_qcd_test, latent_coords_sig, title_suffix=' '.join([sample_id_qcd_test, 'vs', sample_id_sig]), filename_suffix='_'.join([sample_id_qcd_test, 'vs', sample_id_sig]), fig_dir='fig')

# apply clustering algo
cluster_assign_sig = cluster_model.predict(latent_coords_sig) # latent coords of signal obtained from AE
cluster_assign_qcd_test = cluster_model.predict(latent_coords_qcd_test) # latent coords of signal obtained from AE
plot.plot_clusters(latent_coords_sig, cluster_assign_sig, cluster_centers, title_suffix=params.cluster_alg+' '+sample_id_sig, filename_suffix=params.cluster_alg+'_'+sample_id_sig)
plot.plot_clusters(latent_coords_qcd_test, cluster_assign_qcd_test, cluster_centers, title_suffix=params.cluster_alg+' '+sample_id_qcd_test, filename_suffix=params.cluster_alg+'_'+sample_id_qcd_test)



#****************************************#
#               METRIC
#****************************************#

dist_qcd = metr.compute_metric_score(algo_str=params.cluster_alg, coords=latent_coords_qcd, model=cluster_model)
dist_qcd_test = metr.compute_metric_score(algo_str=params.cluster_alg, coords=latent_coords_qcd_test, model=cluster_model)
dist_sig = metr.compute_metric_score(algo_str=params.cluster_alg, coords=latent_coords_sig, model=cluster_model)

if params.cluster_alg == 'kmeans':
    xlabel = 'distance to closest cluster'
    title = 'euclidian distance distribution qcd vs sig'
else:
    xlabel = 'distance to border'
    title = 'distance to decision border distribution qcd vs sig'

pu.plot_bg_vs_sig([dist_qcd, dist_sig], legend=[params.sample_id_train,sample_id_sig], xlabel=xlabel, title=title, plot_name='loss_qcd_vs_sig_'+params.cluster_alg, fig_dir='fig', ylogscale=True, fig_format='.png')
pu.plot_bg_vs_sig([dist_qcd_test, dist_sig], legend=[sample_id_qcd_test, sample_id_sig], xlabel=xlabel, title=title, plot_name='loss_qcd_vs_sig_'+params.cluster_alg, fig_dir='fig', ylogscale=True, fig_format='.png')
roc.plot_roc([dist_qcd], [dist_sig], legend=[params.sample_id_train,sample_id_sig], title=' '.join([params.sample_id_train, 'vs', sample_id_sig, params.cluster_alg]), plot_name='_'.join(['ROC', params.sample_id_train, 'vs', sample_id_sig, params.cluster_alg]), fig_dir='fig')
roc.plot_roc([dist_qcd_test], [dist_sig], legend=[sample_id_qcd_test, sample_id_sig], title=' '.join([sample_id_qcd_test, 'vs', sample_id_sig, params.cluster_alg]), plot_name='_'.join(['ROC', sample_id_qcd_test, 'vs', sample_id_sig, params.cluster_alg]), fig_dir='fig')


#****************************************#
#               QUANTUM CLUSTERING
#****************************************#

print('>>> loading qmeans')
model_path_qm = make_model_path(date='20211006', prefix='QM') + '.npy'
with open(model_path_qm, 'rb') as f:
    cluster_q_centers = np.load(f, cluster_q_centers)


# apply clustering algo
cluster_q_assign_qcd, q_dist_qcd = cluster_q.assign_clusters(latent_coords_qcd, cluster_q_centers) # latent coords of qcd train obtained from AE
cluster_q_assign_qcd_test, q_dist_qcd_test = cluster_q.assign_clusters(latent_coords_qcd_test, cluster_q_centers) # latent coords of qcd test obtained from AE
cluster_q_assign_sig, q_dist_sig = cluster_q.assign_clusters(latent_coords_sig, cluster_q_centers) # latent coords of signal obtained from AE

plot.plot_clusters(latent_coords_qcd, cluster_q_assign_sig, cluster_centers, title_suffix='quantum_'+params.cluster_alg+' '+params.sample_id_train, filename_suffix='quantum_'+params.cluster_alg+'_'+params.sample_id_train)
plot.plot_clusters(latent_coords_qcd_test, cluster_q_assign_qcd_test, cluster_centers, title_suffix='quantum_'+params.cluster_alg+' '+sample_id_qcd_test, filename_suffix='quantum_'+params.cluster_alg+'_'+sample_id_qcd_test)
plot.plot_clusters(latent_coords_sig, cluster_q_assign_sig, cluster_centers, title_suffix='quantum_'+params.cluster_alg+' '+sample_id_sig, filename_suffix='quantum_'+params.cluster_alg+'_'+sample_id_sig)

dist_q_qcd = metr.compute_quantum_metric_score(q_dist_qcd, cluster_q_assign_qcd)
dist_q_qcd_test = metr.compute_quantum_metric_score(q_dist_qcd_test, cluster_q_assign_qcd_test)
dist_q_sig = metr.compute_quantum_metric_score(q_dist_sig, cluster_q_assign_sig)

title = 'quantum ' + title

pu.plot_bg_vs_sig([dist_q_qcd, dist_q_sig], legend=[params.sample_id_train,sample_id_sig], xlabel=xlabel, title=title, plot_name='quantum_loss_qcd_vs_sig_'+params.cluster_alg, fig_dir='fig', ylogscale=True, fig_format='.png')
pu.plot_bg_vs_sig([dist_q_qcd_test, dist_q_sig], legend=[sample_id_qcd_test, sample_id_sig], xlabel=xlabel, title=title, plot_name='quantum_loss_qcd_vs_sig_'+params.cluster_alg, fig_dir='fig', ylogscale=True, fig_format='.png')
roc.plot_roc([dist_q_qcd], [dist_q_sig], legend=[params.sample_id_train,sample_id_sig], title=' '.join(['quantum', params.sample_id_train, 'vs', sample_id_sig, params.cluster_alg]), plot_name='_'.join(['quantum', 'ROC', params.sample_id_train, 'vs', sample_id_sig, params.cluster_alg]), fig_dir='fig')
roc.plot_roc([dist_q_qcd_test], [dist_q_sig], legend=[sample_id_qcd_test, sample_id_sig], title=' '.join(['quantum', sample_id_qcd_test, 'vs', sample_id_sig, params.cluster_alg]), plot_name='_'.join(['quantum', 'ROC', sample_id_qcd_test, 'vs', sample_id_sig, params.cluster_alg]), fig_dir='fig')
