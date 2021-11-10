from collections import namedtuple
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import joblib as jli
from sklearn.decomposition import PCA

import inference.clustering_classic as cluster
import inference.clustering_quantum as cluster_q
import inference.predict_autoencoder as pred
import data.data_sample as dasa
import util.persistence as pers
import util.preprocessing as prep



#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', ' run_n read_n sample_id_train cluster_alg normalize')
params = Parameters(run_n=70, read_n=int(1e4), sample_id_train='qcdSide', cluster_alg='kmeans', normalize=False)


#****************************************#
#           Data Sample -> AE -> latent
#****************************************#

data_sample = dasa.DataSample(params.sample_id_train)
model_path_ae = pers.make_model_path(run_n=50, date='20211110', prefix='AE')

print('[main] >>> loading autoencoder ' + model_path_ae)
ae_model = tf.saved_model.load(model_path_ae)

# apply AE model
latent_coords_qcd = pred.map_to_latent_space(data_sample=data_sample, model=ae_model, read_n=params.read_n)
if params.normalize:
    latent_coords_qcd = prep.min_max_normalize(latent_coords_qcd)

#****************************************#
#               PCA CLASSIC
#****************************************#

pca = PCA(n_components=2)
qcd_proj = pca.fit(latent_coords_qcd)

model_path = pers.make_model_path(prefix='PCA', run_n=params.run_n)
# save
print('>>> saving classic PCA model to ' + model_path)
jli.dump(pca, model_path+'.joblib') 
