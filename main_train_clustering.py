#import setGPU
from collections import namedtuple
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import joblib as jli

import pofah.jet_sample as jesa
import inference.clustering_classic as cluster
import inference.clustering_quantum as cluster_q
import inference.predict_autoencoder as pred
import data.data_sample as dasa
import util.persistence as pers
import util.preprocessing as prep
import util.logging as log


#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'run_n ae_run_n read_n sample_id_train cluster_alg normalize quantum_min rtol mjj_center')
params = Parameters(run_n=22,
                    ae_run_n=50,
                    read_n=int(1e3),
                    sample_id_train='qcdSig',
                    cluster_alg='kmeans',
                    normalize=False,
                    quantum_min=True,
                    rtol=3e-2,
                    mjj_center=True)

# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*60+'\n'+'\t\t\t TRAINING RUN \n'+str(params)+'\n'+'*'*60)

#****************************************#
#      load data latent representation
#****************************************#

input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.ae_run_n)
sample_qcd = pers.read_latent_jet_sample(input_dir, params.sample_id_train, read_n=params.read_n)
if mjj_center:
    sample_qcd = jesa.get_mjj_binned_sample_center_bin(sample_qcd, mjj_peak=3500) 
latent_coords_qcd = pers.read_latent_representation(sample_qcd)
logger.info('read {} training samples ({} jets)'.format(len(latent_coords_qcd)/2, len(latent_coords_qcd))) # stacked j1 & j2


#****************************************#
#          CLUSTERING CLASSIC
#****************************************#

#****************************************#
#               KMEANS

if params.cluster_alg == 'kmeans':

    model_path = pers.make_model_path(prefix='KM', run_n=params.run_n)

    print('>>> training kmeans')
    cluster_model = cluster.train_kmeans(latent_coords_qcd)
    print('[main_train_clustering] >>> centers {}'.format(cluster_model.cluster_centers_))

    

#****************************************#
#           ONE CLASS SVM

else:

    model_path = pers.make_model_path(prefix='SVM', run_n=params.run_n)

    print('>>> training one class svm')
    cluster_model = cluster.train_one_class_svm(latent_coords_qcd)

# save
print('>>> saving classic clustering model to ' + model_path)
jli.dump(cluster_model, model_path+'.joblib') 


#****************************************#
#            QUANTUM CLUSTERING
#****************************************#

## train quantum kmeans

print('>>> training qmeans')
cluster_q_centers = cluster_q.train_qmeans(latent_coords_qcd, quantum_min=params.quantum_min, rtol=params.rtol)

model_path_qm = pers.make_model_path(prefix='QM', run_n=params.run_n) + '.npy'
with open(model_path_qm, 'wb') as f:
    print('>>> saving qmeans model to ' + model_path_qm)
    np.save(f, cluster_q_centers)
