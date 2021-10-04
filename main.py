import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import pathlib
import tensorflow as tf
import numpy as np
from collections import namedtuple
from sklearn import svm
import joblib as jli

import data.data_sample as dasa
import inference.train_autoencoder as train
import inference.predict_autoencoder as pred
import inference.clustering_classic as cluster
import inference.clustering_quantum as cluster_q
import analysis.plotting as plot
import anpofah.util.plotting_util as pu
import anpofah.model_analysis.roc_analysis as roc


def make_model_path(date=None, prefix='AE'):
    date_str = ''
    if date is None:
        date = datetime.date.today()
        date = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    path = os.path.join('models/saved', prefix+'model_{}'.format(date))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path

# euclidian distance to cluster center
dist_to_cluster_center = lambda a,b: np.sum((a-b)**2,axis=1)


def compute_metric_score(algo_str, coords, model):
    # compute euclidian distance to closest cluster center for kmeans
    if algo_str == 'kmeans':
        return np.sum(model.transform(coords)**2, axis=1)
    # compute squared distance to separating hyperplane (shifting to all < 0) => Losing info of inlier vs outlier in this way (???)
    elif algo_str == 'one_class_svm':
        distances = model.decision_function(coords)
        distances = distances - np.max(distances)
        return distances**2


def compute_quantum_metric_score(sample_dist, cluster_assign, metric_type='sum_all_dist'):
    # compute squared sum of all distances
    if metric_type == 'sum_all_dist':
        return np.sum(sample_dist**2, axis=1)
    # compute squared dist to closest cluster
    else: 
        return sample_dist[range(len(sample_dist)), cluster_assign]**2



#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'load_ae load_km epochs latent_dim read_n sample_id_train cluster_alg')
params = Parameters(load_ae=True, load_km=False, epochs=200, latent_dim=8, read_n=int(1e3), sample_id_train='qcdSide', cluster_alg='kmeans')

model_path_ae = make_model_path(prefix='AE')
data_sample = dasa.DataSample(params.sample_id_train)


#****************************************#
#           Autoencoder
#****************************************#

if params.load_ae:
    ae_model = tf.saved_model.load(model_path_ae)

else:
    # train AE model
    print('>>> training autoencoder')
    ae_model = train.train(data_sample, epochs=params.epochs, latent_dim=params.latent_dim, read_n=params.read_n)

    # model save /load
    tf.saved_model.save(ae_model, model_path_ae) 

# apply AE model
latent_coords_qcd = pred.map_to_latent_space(data_sample=data_sample, sample_id=params.sample_id_train, model=ae_model, read_n=params.read_n)

#****************************************#
#               CLUSTERING
#****************************************#

#****************************************#
#               KMEANS

if params.cluster_alg == 'kmeans':

    model_path_km = make_model_path(prefix='KM')

    if params.load_km:

        cluster_model = jli.load(model_path_km+'.joblib')

    else:
        print('>>> training kmeans')
        cluster_model = cluster.train_kmeans(latent_coords_qcd)
        # save
        jli.dump(cluster_model, model_path_km+'.joblib') 
    
    cluster_centers = cluster_model.cluster_centers_


#****************************************#
#           ONE CLASS SVM

else:

    print('>>> training one class svm')
    cluster_model = cluster.train_one_class_svm(latent_coords_qcd)
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

dist_qcd = compute_metric_score(algo_str=params.cluster_alg, coords=latent_coords_qcd, model=cluster_model)
dist_qcd_test = compute_metric_score(algo_str=params.cluster_alg, coords=latent_coords_qcd_test, model=cluster_model)
dist_sig = compute_metric_score(algo_str=params.cluster_alg, coords=latent_coords_sig, model=cluster_model)

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

## train quantum kmeans

print('>>> training qmeans')
cluster_q_centers = cluster_q.train_qmeans(latent_coords_qcd)

# apply clustering algo
cluster_q_assign_qcd, q_dist_qcd = cluster_q.assign_clusters(latent_coords_qcd, cluster_q_centers) # latent coords of qcd train obtained from AE
cluster_q_assign_qcd_test, q_dist_qcd_test = cluster_q.assign_clusters(latent_coords_qcd_test, cluster_q_centers) # latent coords of qcd test obtained from AE
cluster_q_assign_sig, q_dist_sig = cluster_q.assign_clusters(latent_coords_sig, cluster_q_centers) # latent coords of signal obtained from AE
plot.plot_clusters(latent_coords_qcd, cluster_q_assign_sig, cluster_centers, title_suffix='quantum_'+params.cluster_alg+' '+sample_id_train, filename_suffix='quantum_'+params.cluster_alg+'_'+sample_id_train)
plot.plot_clusters(latent_coords_qcd_test, cluster_q_assign_qcd_test, cluster_centers, title_suffix='quantum_'+params.cluster_alg+' '+sample_id_qcd_test, filename_suffix='quantum_'+params.cluster_alg+'_'+sample_id_qcd_test)
plot.plot_clusters(latent_coords_sig, cluster_q_assign_sig, cluster_centers, title_suffix='quantum_'+params.cluster_alg+' '+sample_id_sig, filename_suffix='quantum_'+params.cluster_alg+'_'+sample_id_sig)

dist_q_qcd = compute_quantum_metric_score(q_dist_qcd, cluster_q_assign_qcd)
dist_q_qcd_test = compute_quantum_metric_score(q_dist_qcd_test, cluster_q_assign_qcd_test)
dist_q_sig = compute_quantum_metric_score(q_dist_sig, cluster_q_assign_sig)

title = 'quantum ' + title

pu.plot_bg_vs_sig([dist_q_qcd, dist_q_sig], legend=[params.sample_id_train,sample_id_sig], xlabel=xlabel, title=title, plot_name='quantum_loss_qcd_vs_sig_'+params.cluster_alg, fig_dir='fig', ylogscale=True, fig_format='.png')
pu.plot_bg_vs_sig([dist_q_qcd_test, dist_q_sig], legend=[sample_id_qcd_test, sample_id_sig], xlabel=xlabel, title=title, plot_name='quantum_loss_qcd_vs_sig_'+params.cluster_alg, fig_dir='fig', ylogscale=True, fig_format='.png')
roc.plot_roc([dist_q_qcd], [dist_q_sig], legend=[params.sample_id_train,sample_id_sig], title=' '.join(['quantum', params.sample_id_train, 'vs', sample_id_sig, params.cluster_alg]), plot_name='_'.join(['quantum', 'ROC', params.sample_id_train, 'vs', sample_id_sig, params.cluster_alg]), fig_dir='fig')
roc.plot_roc([dist_q_qcd_test], [dist_q_sig], legend=[sample_id_qcd_test, sample_id_sig], title=' '.join(['quantum', sample_id_qcd_test, 'vs', sample_id_sig, params.cluster_alg]), plot_name='_'.join(['quantum', 'ROC', sample_id_qcd_test, 'vs', sample_id_sig, params.cluster_alg]), fig_dir='fig')
