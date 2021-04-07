import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import inference.train_autoencoder as train
import inference.predict_autoencoder as pred

# train AE model
model = train.train()

# model save /load 
# ...

# apply AE model
sample_id = 'qcdSideExt'
latent_coords = pred.map_to_latent_space(sample_id, model)

# train k-means model


