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
import pandas as pd

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

seed = 12345

mG = 3500
Parameters = namedtuple('Parameters', 'run_n latent_dim date_model ae_run_n read_n sample_ids cluster_alg normalize quantum_min raw_format kfold_n')
params = Parameters(run_n=32, 
                    latent_dim=16,
                    date_model='20220512', #'20220209'
                    ae_run_n=50, 
                    read_n=int(2e4), # test on 20K events in 10 fold 
                    sample_ids=['qcdSigExt', 'GtoWW35na', 'GtoWW15br', 'AtoHZ35'], 
                    cluster_alg='kmeans', 
                    normalize=False,
                    quantum_min=True,
                    raw_format=True,
                    kfold_n=10)

# path setup
fig_dir = 'fig/run_'+str(params.run_n)
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

# input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.ae_run_n)
input_dir = '/eos/home-e/epuljak/private/epuljak/public/diJet/'+str(params.latent_dim)
output_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/run_"+str(params.run_n)
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

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
#               load QUANTUM model
#****************************************#

logger.info('loading qmeans')
model_path_qm = pers.make_model_path(date=params.date_model, prefix='QM', run_n=params.run_n) + '.npy'
with open(model_path_qm, 'rb') as f:
    cluster_q_centers = np.load(f)
logger.info('quantum cluster centers: ')
logger.info(cluster_q_centers)



for sample_id in params.sample_ids:

    #****************************************#
    #      load data latent representation
    #****************************************#

    if params.raw_format: # workaround for format without jet features
        latent_coords = pers.read_latent_rep_from_file(input_dir, sample_id, read_n=params.read_n, raw_format=params.raw_format, shuffle=False)
        latent_coords_reshaped = np.stack(np.split(latent_coords, 2),axis=1) # reshape to N x 2 x z_dim
        sample_in = jesa.JetSampleLatent(name=sample_id, features=pd.DataFrame()).add_latent_representation(latent_coords_reshaped)
    else:
        sample_in = pers.read_latent_jet_sample(input_dir, sample_id, read_n=params.read_n) 
        latent_coords = pers.read_latent_representation(sample_in, shuffle=False) # do not shuffle, as loss is later combined assuming first half=j1 and second half=j2
    # if params.normalize: # not using normalization at the moment, if to use again -> adapt for multiple signal samples
    #     latent_coords_qcd, latent_coords_sig = prep.min_max_normalize_all_data(latent_coords_qcd, latent_coords_sig)

    logger.info('read {} {} events'.format(len(sample_in), sample_id))


    #****************************************#
    #           apply clustering
    #****************************************#

    logger.info('applying classic clustering model')

    # import ipdb; ipdb.set_trace()

    cluster_assign = cluster_model.predict(latent_coords) # latent coords obtained from AE

    logger.info('plotting classic cluster assignments')

    plot.plot_clusters_pairplot(latent_coords, cluster_assign, cluster_centers, filename_suffix=params.cluster_alg+'_'+sample_id, fig_dir=fig_dir)


    #****************************************#
    #               METRIC
    #****************************************#

    logger.info('computing classic clustering metrics')

    metric_c = metr.compute_metric_score(algo_str=params.cluster_alg, coords=latent_coords, model=cluster_model)

    if params.cluster_alg == 'kmeans':
        xlabel = 'sum distances to clusters'
        title = 'euclidian distance distribution qcd vs sig'
    else:
        xlabel = 'distance to border'
        title = 'distance to decision border distribution qcd vs sig'


    #****************************************#
    #          apply QUANTUM CLUSTERING
    #****************************************#

    # apply clustering algo
    logger.info('applying quantum clustering model')
    cluster_assign_q, distances_q = cluster_q.assign_clusters(latent_coords, cluster_q_centers, quantum_min=params.quantum_min) # latent coords of qcd train obtained from AE
    logger.info('plotting quantum cluster assignments')
    plot.plot_clusters_pairplot(latent_coords, cluster_assign_q, cluster_centers, filename_suffix='quantum_'+params.cluster_alg+'_'+sample_id, fig_dir=fig_dir)

    logger.info('computing quantum clustering metrics')
    metric_q = metr.compute_quantum_metric_score(distances_q, cluster_assign_q)


    #****************************************#
    #               WRITE RESULTS
    #****************************************#

    logger.info('writing results to ' + output_dir)

    sample_out = jesa.JetSample.from_latent_jet_sample(sample_in)
    sample_out.add_feature('classic_loss', combine_loss_min(metric_c))
    sample_out.add_feature('quantum_loss', combine_loss_min(metric_q))
    sample_out.dump(os.path.join(output_dir, sample_out.name+'.h5'))
