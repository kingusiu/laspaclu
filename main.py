import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import inference.train_autoencoder as train
import inference.predict_autoencoder as pred
import inference.kmeans as kmeans

# train AE model
ae_model = train.train('qcdSide')

# model save /load 
# ...

# apply AE model
sample_id = 'qcdSideExt'
latent_coords = pred.map_to_latent_space(sample_id, ae_model)
sample_id = 'GtoWW35na'
latent_coords_sig = pred.map_to_latent_space(sample_id, ae_model)


# train k-means model
print('>>> training kmeans')
km_model = kmeans.train(latent_coords)

# apply to data
# qcdSide
km_model.labels_
# qcdSig
km_model.predict(latent_coords_sig) # latent coords of signal obtained from AE
# 

