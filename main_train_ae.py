import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from collections import namedtuple

import util.persistence as pers
import inference.train_autoencoder as train


#****************************************#
#           Runtime Params
#****************************************#

do_clustering = True

Parameters = namedtuple('Parameters', 'load_ae load_km epochs latent_dim read_n sample_id_train cluster_alg')
params = Parameters(load_ae=True, load_km=False, epochs=200, latent_dim=8, read_n=int(1e3), sample_id_train='qcdSide', cluster_alg='kmeans')

model_path_ae = pers.make_model_path(date='20211004', prefix='AE', mkdir=True)
data_sample = dasa.DataSample(params.sample_id_train)

#****************************************#
#           Autoencoder
#****************************************#

# train AE model
print('>>> training autoencoder')
ae_model = train.train(data_sample, epochs=params.epochs, latent_dim=params.latent_dim, read_n=params.read_n)

# model save /load
tf.saved_model.save(ae_model, model_path_ae)

