#import setGPU
from collections import namedtuple
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import joblib as jli
import pathlib
from qiskit.utils import algorithm_globals

import pofah.jet_sample as jesa
import inference.clustering_classic as cluster
import inference.clustering_quantum as cluster_q
import inference.predict_autoencoder as pred
import laspaclu.data.data_sample as dasa
import util.persistence as pers
import util.preprocessing as prep
import util.logging as log


#****************************************#
#           Runtime Params
#****************************************#


# inputs: latent space coordinates qcd
# outputs: k-means model, q-means model

### -------------------------------- ### 

Parameters = namedtuple('Parameters', 'run_n ae_run_n lat_dim read_n sample_id_train cluster_alg cluster_n, max_iter normalize quantum_min rtol mjj_center raw_format')
params = Parameters(run_n=44,
                    ae_run_n=50,
                    lat_dim=4,
                    read_n=int(600),
                    sample_id_train='qcdSig',
                    cluster_alg='kmeans',
                    cluster_n=2,
                    max_iter=100,
                    normalize=False,
                    quantum_min=True,
                    rtol=1e-2,
                    mjj_center=False,
                    raw_format=True)

# set seed
seed = 12345
algorithm_globals.random_seed = seed

# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*60+'\n'+'\t\t\t TRAINING RUN \n'+str(params)+'\n'+'*'*60)

#****************************************#
#      load data latent representation
#****************************************#

# input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.ae_run_n)
input_dir = '/eos/home-e/epuljak/private/epuljak/public/diJet/'+str(int(params.lat_dim))
latent_coords_qcd = pers.read_latent_rep_from_file(input_dir, sample_id=params.sample_id_train, read_n=params.read_n, raw_format=params.raw_format, shuffle=True, seed=seed)
logger.info('read {} training samples ({} jets)'.format(len(latent_coords_qcd)/2, len(latent_coords_qcd))) # stacked j1 & j2


#****************************************#
#          CLUSTERING CLASSIC
#****************************************#

#****************************************#
#               KMEANS

# init cluster centers randomly
idx = np.random.choice(len(latent_coords_qcd), size=params.cluster_n, replace=False)
cluster_centers_ini = latent_coords_qcd[idx]

if params.cluster_alg == 'kmeans':

    model_path = pers.make_model_path(prefix='KM', run_n=params.run_n)

    logger.info('>>> training kmeans')
    cluster_model = cluster.train_kmeans(latent_coords_qcd, cluster_centers_ini, params.cluster_n)
    logger.info('>>> centers {}'.format(cluster_model.cluster_centers_))

    

#****************************************#
#           ONE CLASS SVM

else:

    model_path = pers.make_model_path(prefix='SVM', run_n=params.run_n)

    logger.info('>>> training one class svm')
    cluster_model = cluster.train_one_class_svm(latent_coords_qcd)

# save
logger.info('>>> saving classic clustering model to ' + model_path)
jli.dump(cluster_model, model_path+'.joblib') 


#****************************************#
#            QUANTUM CLUSTERING
#****************************************#

## train quantum kmeans

print('>>> training qmeans')
#cluster_q_centers = cluster_q.train_qmeans(latent_coords_qcd, quantum_min=params.quantum_min, rtol=params.rtol)
gif_dir = 'gif/qkmeans_run_'+str(int(params.run_n))
pathlib.Path(gif_dir).mkdir(parents=True, exist_ok=True)
cluster_q_centers = cluster_q.train_qmeans_animated(latent_coords_qcd, cluster_centers_ini, cluster_n=params.cluster_n, quantum_min=True, rtol=1e-2, max_iter=params.max_iter, gif_dir=gif_dir)

model_path_qm = pers.make_model_path(prefix='QM', run_n=params.run_n) + '.npy'
with open(model_path_qm, 'wb') as f:
    logger.info('>>> saving qmeans model to ' + model_path_qm)
    np.save(f, cluster_q_centers)
