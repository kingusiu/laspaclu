#import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import tensorflow as tf
import numpy as np
from collections import namedtuple
from sklearn import svm
import joblib as jli
import pathlib

import data.data_sample as dasa
import inference.predict_autoencoder as pred
import inference.clustering_quantum as cluster_q
import inference.metrics as metr
import analysis.plotting as plot
import util.persistence as pers
import anpofah.util.plotting_util as pu
import anpofah.model_analysis.roc_analysis as roc
import pofah.path_constants.sample_dict_file_parts_input as sdi
import pofah.util.event_sample as evsa
import pofah.jet_sample as jesa
import dadrah.selection.loss_strategy as losa
import pofah.util.sample_factory as safa
import util.preprocessing as prep
import util.logging as log


def combine_loss_min(loss):
    loss_j1, loss_j2 = np.split(loss, 2)
    return np.minimum(loss_j1, loss_j2)


#****************************************#
#           Runtime Params
#****************************************#

mG = 3500
Parameters = namedtuple('Parameters', 'run_n date_model ae_run_n read_n sample_id_qcd sample_id_sig cluster_alg normalize quantum_min')
params = Parameters(run_n=23, 
                    date_model='20220303',
                    ae_run_n=50, 
                    read_n=None, 
                    sample_id_qcd='qcdSigExt', 
                    sample_id_sig='GtoWW35na', 
                    cluster_alg='kmeans', 
                    normalize=False,
                    quantum_min=True)
fig_dir = 'fig/run_'+str(params.run_n)
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*70+'\n'+'\t\t\t PREDICTION RUN \n'+str(params)+'\n'+'*'*70)

#****************************************#
#           load CLUSTERING model
#****************************************#

#****************************************#
#               KMEANS

if params.cluster_alg == 'kmeans':

    model_path_km = pers.make_model_path(date=params.date_model, prefix='KM', run_n=params.run_n)
    logger.info('loading clustering model ' + model_path_km)

    cluster_model = jli.load(model_path_km+'.joblib')    
    cluster_centers = cluster_model.cluster_centers_
    logger.info('classic cluster centers: ')
    logger.info(cluster_centers)


#****************************************#
#           ONE CLASS SVM

else:

    model_path_svm = pers.make_model_path(date='20211006', prefix='SVM')
    logger.info('loading clustering model ' + model_path_svm)

    cluster_model = jli.load(model_path_svm+'.joblib')    
    cluster_centers = None # no cluster centers for one class svm


#****************************************#
#      load data latent representation
#****************************************#

input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.ae_run_n)
sample_qcd = pers.read_latent_jet_sample(input_dir, params.sample_id_qcd, read_n=params.read_n) 
latent_coords_qcd = pers.read_latent_representation(sample_qcd, shuffle=False) # do not shuffle, as loss is later combined assuming first half=j1 and second half=j2
sample_sig = pers.read_latent_jet_sample(input_dir, params.sample_id_sig, read_n=params.read_n) 
latent_coords_sig = pers.read_latent_representation(sample_sig, shuffle=False) # do not shuffle, as loss is later combined assuming first half=j1 and second half=j2
if params.normalize:
    latent_coords_qcd, latent_coords_sig = prep.min_max_normalize_all_data(latent_coords_qcd, latent_coords_sig)

logger.info('read {} {} events and {} {} events'.format(len(sample_qcd), params.sample_id_qcd, len(sample_sig), params.sample_id_sig))


#****************************************#
#           apply clustering
#****************************************#

logger.info('applying classic clustering model')

cluster_assign_qcd = cluster_model.predict(latent_coords_qcd) # latent coords of qcd obtained from AE
cluster_assign_sig = cluster_model.predict(latent_coords_sig) # latent coords of signal obtained from AE

logger.info('plotting classic cluster assignments')

plot.plot_clusters_pairplot(latent_coords_qcd, cluster_assign_qcd, cluster_centers, filename_suffix=params.cluster_alg+'_'+params.sample_id_qcd, fig_dir=fig_dir)
plot.plot_clusters_pairplot(latent_coords_sig, cluster_assign_sig, cluster_centers, filename_suffix=params.cluster_alg+'_'+params.sample_id_sig, fig_dir=fig_dir)


#****************************************#
#               METRIC
#****************************************#

logger.info('computing classic clustering metrics')

dist_qcd = metr.compute_metric_score(algo_str=params.cluster_alg, coords=latent_coords_qcd, model=cluster_model)
dist_sig = metr.compute_metric_score(algo_str=params.cluster_alg, coords=latent_coords_sig, model=cluster_model)

if params.cluster_alg == 'kmeans':
    xlabel = 'sum distances to clusters'
    title = 'euclidian distance distribution qcd vs sig'
else:
    xlabel = 'distance to border'
    title = 'distance to decision border distribution qcd vs sig'



#****************************************#
#               QUANTUM CLUSTERING
#****************************************#

logger.info('loading qmeans')
model_path_qm = pers.make_model_path(date=params.date_model, prefix='QM', run_n=params.run_n) + '.npy'
with open(model_path_qm, 'rb') as f:
    cluster_q_centers = np.load(f)
logger.info('quantum cluster centers: ')
logger.info(cluster_q_centers)


# apply clustering algo
logger.info('applying quantum clustering model')
cluster_q_assign_qcd, q_dist_qcd = cluster_q.assign_clusters(latent_coords_qcd, cluster_q_centers, quantum_min=params.quantum_min) # latent coords of qcd train obtained from AE
cluster_q_assign_sig, q_dist_sig = cluster_q.assign_clusters(latent_coords_sig, cluster_q_centers, quantum_min=params.quantum_min) # latent coords of signal obtained from AE
logger.info('plotting quantum cluster assignments')
plot.plot_clusters_pairplot(latent_coords_qcd, cluster_q_assign_sig, cluster_centers, filename_suffix='quantum_'+params.cluster_alg+'_'+params.sample_id_qcd, fig_dir=fig_dir)
plot.plot_clusters_pairplot(latent_coords_sig, cluster_q_assign_sig, cluster_centers, filename_suffix='quantum_'+params.cluster_alg+'_'+params.sample_id_sig, fig_dir=fig_dir)

logger.info('computing quantum clustering metrics')
dist_q_qcd = metr.compute_quantum_metric_score(q_dist_qcd, cluster_q_assign_qcd)
dist_q_sig = metr.compute_quantum_metric_score(q_dist_sig, cluster_q_assign_sig)



#****************************************#
#               WRITE RESULTS
#****************************************#

output_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/run_"+str(params.run_n)
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
logger.info('writing results to ' + output_dir)

# qcd results
sample_qcd_out = jesa.JetSample.from_latent_jet_sample(sample_qcd)
sample_qcd_out.add_feature('classic_loss', combine_loss_min(dist_qcd))
sample_qcd_out.add_feature('quantum_loss', combine_loss_min(dist_q_qcd))
sample_qcd_out.dump(os.path.join(output_dir, sample_qcd_out.name+'.h5'))
# signal results
sample_sig_out = jesa.JetSample.from_latent_jet_sample(sample_sig)
sample_sig_out.add_feature('classic_loss', combine_loss_min(dist_sig))
sample_sig_out.add_feature('quantum_loss', combine_loss_min(dist_q_sig))
sample_sig_out.dump(os.path.join(output_dir, sample_sig_out.name+'.h5'))
