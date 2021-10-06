from collections import namedtuple
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import joblib as jli

import inference.clustering_classic as cluster
import inference.clustering_quantum as cluster_q
import inference.predict_autoencoder as pred
import data.data_sample as dasa
import util.persistence as pers



#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'read_n sample_id_train cluster_alg')
params = Parameters(read_n=int(1e4), sample_id_train='qcdSide', cluster_alg='kmeans')


#****************************************#
#           Data Sample -> AE -> latent
#****************************************#

data_sample = dasa.DataSample(params.sample_id_train)
model_path_ae = pers.make_model_path(date='20211004', prefix='AE')

print('[main] >>> loading autoencoder ' + model_path_ae)
ae_model = tf.saved_model.load(model_path_ae)

# apply AE model
latent_coords_qcd = pred.map_to_latent_space(data_sample=data_sample, sample_id=params.sample_id_train, model=ae_model, read_n=params.read_n)




#****************************************#
#               CLUSTERING CLASSIC
#****************************************#

#****************************************#
#               KMEANS

if params.cluster_alg == 'kmeans':

    model_path = pers.make_model_path(prefix='KM')

    print('>>> training kmeans')
    cluster_model = cluster.train_kmeans(latent_coords_qcd)
    

#****************************************#
#           ONE CLASS SVM

else:

    model_path = pers.make_model_path(prefix='SVM')

    print('>>> training one class svm')
    cluster_model = cluster.train_one_class_svm(latent_coords_qcd)

# save
jli.dump(cluster_model, model_path+'.joblib') 



#****************************************#
#               QUANTUM CLUSTERING
#****************************************#

## train quantum kmeans

print('>>> training qmeans')
cluster_q_centers = cluster_q.train_qmeans(latent_coords_qcd)

model_path_qm = pers.make_model_path(prefix='QM') + '.npy'
with open(model_path_qm, 'wb') as f:
    print('>>> saving qmeans model to ' + model_path_qm)
    np.save(f, cluster_q_centers)
