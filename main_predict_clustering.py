#import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import tensorflow as tf
import numpy as np
from collections import namedtuple
import argparse
from sklearn import svm
import joblib as jli
import pathlib
import pandas as pd

import laspaclu.src.ml.clustering_quantum as cluster_q
import laspaclu.src.ml.metrics as metr
import laspaclu.src.util.persistence as pers
import laspaclu.src.util.preprocessing as prep
import laspaclu.src.util.logging as log
import laspaclu.src.util.string_constants as stco
import pofah.jet_sample as jesa



#****************************************#
#           Clustering prediction
#****************************************#

# task: for a sample x with latent coordinates in R^Z and a previously trained model M, find closest cluster c for x in M, calculate distance to c and store cluster assignment and distance-value in x_out
# inputs: latent space coordinates qcd & signals, kmeans model, qkmeans model
# outputs: out_sample qcd & signal with latent coords, classic loss, quantum loss, classic cluster assignment, quantum cluster assignment

### --------------------------------- ### 


def combine_loss_min(loss):
    loss_j1, loss_j2 = np.split(loss, 2)
    return np.minimum(loss_j1, loss_j2)


logger = log.get_logger(__name__)

#****************************************#
#           Runtime Params
#****************************************#

# command line
parser = argparse.ArgumentParser(description='read arguments for qkmeans prediction')
parser.add_argument('-r', dest='run_n', type=int, help='experiment run number')
parser.add_argument('-z', dest='lat_dim', type=int, help='latent dimensionality')
parser.add_argument('-n', dest='read_n', type=int, help='training data size', default=int(5e4))
parser.add_argument('--classicopt', dest='quantum_min', action="store_false", help='do optimization classically')
args = parser.parse_args()

logger.info('\n'+'*'*60+'\n'+'\t\t\t PREDICTION RUN \n'+str(args)+'\n'+'*'*60)


# fixed 
Parameters = namedtuple('Parameters', 'ae_run_n sample_ids cluster_alg normalize raw_format')
params = Parameters(ae_run_n=50, 
                    sample_ids=['qcdSigExt', 'GtoWW35na', 'GtoWW15br', 'AtoHZ35'], 
                    cluster_alg='kmeans', 
                    normalize=False,
                    raw_format=True)

# path setup
fig_dir = os.path.join(stco.reporting_fig_base_dir,'qkmeans_run_'+str(args.run_n))
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

# input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.ae_run_n)
input_dir = stco.cluster_in_data_dir+str(args.lat_dim)
output_dir = stco.cluster_out_data_dir+'/run_'+str(args.run_n)
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# logging
logger.info('\n'+'*'*70+'\n'+'\t\t\t params \n'+str(params)+'\n'+'*'*70)

#****************************************#
#           load CLUSTERING model
#****************************************#

#****************************************#
#               KMEANS

if params.cluster_alg == 'kmeans':

    model_path_km = pers.make_model_path(prefix='KM', run_n=args.run_n)
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
model_path_qm = pers.make_model_path(prefix='QM', run_n=args.run_n) + '.npy'
with open(model_path_qm, 'rb') as f:
    cluster_q_centers = np.load(f)
logger.info('quantum cluster centers: ')
logger.info(cluster_q_centers)



for sample_id in params.sample_ids:

    #****************************************#
    #      load data latent representation
    #****************************************#

    if params.raw_format: # workaround for format without jet features
        latent_coords = pers.read_latent_rep_from_file(input_dir, sample_id, read_n=args.read_n, raw_format=params.raw_format, shuffle=False)
        latent_coords_reshaped = np.stack(np.split(latent_coords, 2),axis=1) # reshape to N x 2 x z_dim
        sample_in = jesa.JetSampleLatent(name=sample_id, features=pd.DataFrame()).add_latent_representation(latent_coords_reshaped)
    else:
        sample_in = pers.read_latent_jet_sample(input_dir, sample_id, read_n=args.read_n) 
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
    cluster_assign_q, distances_q = cluster_q.assign_clusters(latent_coords, cluster_q_centers, quantum_min=args.quantum_min) # latent coords of qcd train obtained from AE
   
    logger.info('computing quantum clustering metrics')
    metric_q = metr.compute_quantum_metric_score(distances_q, cluster_assign_q)


    #****************************************#
    #               WRITE RESULTS
    #****************************************#
    #import ipdb; ipdb.set_trace()

    sample_out = sample_in.copy()
    sample_out.add_feature('classic_loss', combine_loss_min(metric_c))
    sample_out.add_feature('quantum_loss', combine_loss_min(metric_q))
    cluster_assign_j1, cluster_assign_j2 = np.split(cluster_assign, 2)
    cluster_assign_q_j1, cluster_assign_q_j2 = np.split(cluster_assign_q, 2)
    sample_out.add_feature('classic_assign_j1', cluster_assign_j1)
    sample_out.add_feature('classic_assign_j2', cluster_assign_j2)
    sample_out.add_feature('quantum_assign_j1', cluster_assign_q_j1)
    sample_out.add_feature('quantum_assign_j2', cluster_assign_q_j2)
    
    sample_out_full_path = os.path.join(output_dir, sample_out.name+'.h5')
    logger.info('writing results to ' + sample_out_full_path)
    sample_out.dump(sample_out_full_path)
