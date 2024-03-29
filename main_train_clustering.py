#import setGPU
from collections import namedtuple
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib as jli
import pathlib
import argparse
from qiskit.utils import algorithm_globals

import pofah.jet_sample as jesa
import laspaclu.src.ml.clustering_classic as cluster
import laspaclu.src.ml.clustering_quantum as cluster_q
import laspaclu.src.data.data_sample as dasa
import laspaclu.src.util.persistence as pers
import laspaclu.src.util.preprocessing as prep
import laspaclu.src.util.logging as log
import laspaclu.src.util.string_constants as stco


# inputs: latent space coordinates qcd
# outputs: k-means model, q-means model

### -------------------------------- ### 

# logging
logger = log.get_logger(__name__)

#****************************************#
#           Runtime Params
#****************************************#

# command line
parser = argparse.ArgumentParser(description='read arguments for quantum kmeans training')
parser.add_argument('-r', dest='run_n', type=int, help='experiment run number')
parser.add_argument('-z', dest='lat_dim', type=int, help='latent dimensionality')
parser.add_argument('-n', dest='read_n', type=int, help='training data size')
parser.add_argument('--classicopt', dest='quantum_min', action="store_false", help='do optimization classically')
args = parser.parse_args()

logger.info('\n'+'*'*60+'\n'+'\t\t\t TRAINING RUN \n'+str(args)+'\n'+'*'*60)

# fixed
Parameters = namedtuple('Parameters', 'ae_run_n sample_id_train cluster_alg cluster_n, max_iter normalize rtol mjj_center raw_format')
params = Parameters(ae_run_n=50,
                    sample_id_train='qcdSig',
                    cluster_alg='kmeans',
                    cluster_n=2,
                    max_iter=100,
                    normalize=False,
                    rtol=1e-2,
                    mjj_center=False,
                    raw_format=True)

logger.info('\n'+'*'*60+'\n'+'\t\t\t params \n'+str(params)+'\n'+'*'*60)

# set seed
seed = 12345
algorithm_globals.random_seed = seed

#****************************************#
#      load data latent representation
#****************************************#

# input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.ae_run_n)
input_dir = os.path.join(stco.cluster_in_data_dir,str(int(args.lat_dim)))
latent_coords_qcd = pers.read_latent_rep_from_file(input_dir, sample_id=params.sample_id_train, read_n=args.read_n, raw_format=params.raw_format, shuffle=True, seed=seed)
logger.info('read {} training samples ({} jets)'.format(len(latent_coords_qcd)/2, len(latent_coords_qcd))) # stacked j1 & j2


#****************************************#
#          CLASSIC CLUSTERING
#****************************************#

#****************************************#
#               KMEANS

# init cluster centers randomly
idx = np.random.choice(len(latent_coords_qcd), size=params.cluster_n, replace=False)
cluster_centers_ini = latent_coords_qcd[idx]

if params.cluster_alg == 'kmeans':

    model_path = pers.make_model_path(prefix='KM', run_n=args.run_n)

    logger.info('>>> training kmeans')
    cluster_model = cluster.train_kmeans(latent_coords_qcd, cluster_centers_ini, params.cluster_n)
    logger.info('>>> centers {}'.format(cluster_model.cluster_centers_))

    

#****************************************#
#           ONE CLASS SVM

else:

    model_path = pers.make_model_path(prefix='SVM', run_n=args.run_n)

    logger.info('>>> training one class svm')
    cluster_model = cluster.train_one_class_svm(latent_coords_qcd)

# save
logger.info('>>> saving classic clustering model to ' + model_path)
jli.dump(cluster_model, model_path+'.joblib')


#****************************************#
#            QUANTUM CLUSTERING
#****************************************#

## train quantum kmeans

logger.info('>>> training qmeans')

gif_dir = os.path.join(stco.reporting_gif_base_dir,'qkmeans_run_'+str(int(args.run_n)))
pathlib.Path(gif_dir).mkdir(parents=True, exist_ok=True)

cluster_q_centers = cluster_q.train_qmeans(latent_coords_qcd, cluster_centers_ini, cluster_n=params.cluster_n, quantum_min=args.quantum_min, rtol=params.rtol, max_iter=params.max_iter)
#cluster_q_centers = cluster_q.train_qmeans_animated(latent_coords_qcd, cluster_centers_ini, cluster_n=params.cluster_n, quantum_min=args.quantum_min, rtol=params.rtol, max_iter=params.max_iter, gif_dir=gif_dir)

model_path_qm = pers.make_model_path(prefix='QM', run_n=args.run_n) + '.npy'
with open(model_path_qm, 'wb') as f:
    logger.info('>>> saving qmeans model to ' + model_path_qm)
    np.save(f, cluster_q_centers)
