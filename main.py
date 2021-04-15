import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import pathlib
import tensorflow as tf

import inference.train_autoencoder as train
import inference.predict_autoencoder as pred
import inference.kmeans as kmeans
import analysis.plotting as plot


def make_model_path(date=None):
    date_str = ''
    if date is None:
        date = datetime.date.today()
        date = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    path = os.path.join('models/saved', 'AEmodel_{}'.format(date))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path

load = True
model_path = make_model_path('20210414')

if load:
    ae_model = tf.saved_model.load(model_path)

else:
    # train AE model
    ae_model = train.train('qcdSide', epochs=20)

    # model save /load
    tf.saved_model.save(ae_model, model_path) 

# apply AE model
sample_id = 'qcdSideExt'
latent_coords_qcd = pred.map_to_latent_space(sample_id, ae_model)
sample_id = 'GtoWW35na'
latent_coords_sig = pred.map_to_latent_space(sample_id, ae_model)

# plot latent space
plot.plot_latent_space_2D_bg_vs_sig(latent_coords_qcd, latent_coords_sig, title_suffix="qcd vs G_RS", filename_suffix="qcd_vs_sig", fig_dir='fig')

# train k-means model
print('>>> training kmeans')
km_model = kmeans.train(latent_coords_qcd)

# apply to data -> obtain cluster assignment
centroids = km_model.labels_
# qcdSide
# signal
cluster_assignemnts = km_model.predict(latent_coords_sig) # latent coords of signal obtained from AE
# 

