import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np

import util.persistence as pers
import inference.train_autoencoder as train
import models.autoencoder as auen
import anpofah.sample_analysis.sample_converter as saco
import pofah.util.utility_fun as utfu
import vande.vae.losses as loss



def compare_jet_images(test_ds, model):
    print('>>> plotting jet image comparison original vs predicted')
    batch = next(test_ds.as_numpy_iterator())
    for i in np.random.choice(len(batch), 3):
        particles = batch[i]
        img = saco.convert_jet_particles_to_jet_image(particles)
        plt.imshow(np.squeeze(img), cmap='viridis')
        plt.savefig('fig/img_orig_'+str(i)+'.png')
        plt.clf()
        particles_pred = model.predict(particles[np.newaxis,:,:])
        img_pred = saco.convert_jet_particles_to_jet_image(particles_pred)
        plt.imshow(np.squeeze(img_pred), cmap='viridis')
        plt.savefig('fig/img_pred_'+str(i)+'.png')


def train(data_sample, input_shape=(100,3), latent_dim=6, epochs=10, read_n=int(1e4)):

    # get data
    train_ds, valid_ds = data_sample.get_datasets_for_training(read_n=read_n, test_dataset=False)
    model = auen.ParticleAutoencoder(input_shape=input_shape, latent_dim=latent_dim, x_mean_stdev=data_sample.get_mean_and_stdev())
    model.compile(optimizer=tf.keras.optimizers.Adam(), reco_loss=loss.threeD_loss)
    # print(model.summary())

    model.fit(train_ds, epochs=epochs, shuffle=True, validation_data=valid_ds, \
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)])
        #callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=70, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.8, patience=7, verbose=1)])

    return model



#****************************************#
#           Runtime Params
#****************************************#

do_clustering = True

Parameters = namedtuple('Parameters', 'load_ae load_km epochs latent_dim read_n sample_id_train cluster_alg')
params = Parameters(load_ae=True, load_km=False, epochs=200, latent_dim=8, read_n=int(1e3), sample_id_train='qcdSide', cluster_alg='kmeans')

model_path = pers.make_model_path(date='20211004', prefix='AE', mkdir=True)
data_sample = dasa.DataSample(params.sample_id_train)

#****************************************#
#           Autoencoder
#****************************************#

# train AE model
print('>>> training autoencoder')
ae_model = train.train(data_sample, epochs=params.epochs, latent_dim=params.latent_dim, read_n=params.read_n)

# model save
print('>>> saving autoencoder to ' + model_path)
tf.saved_model.save(ae_model, model_path)

