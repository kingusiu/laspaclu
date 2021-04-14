import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import inference.train_autoencoder as train
import inference.predict_autoencoder as pred
import inference.kmeans as kmeans
import analysis.plotting as plot

# train AE model
ae_model = train.train('qcdSide', epochs=20)

# model save /load 
# ae_model.save(...) # calls native keras function
# ae_model.load(...) # calls overriding load fun with custom objects


# apply AE model
sample_id = 'qcdSideExt'
latent_coords_qcd = pred.map_to_latent_space(sample_id, ae_model)
sample_id = 'GtoWW35na'
latent_coords_sig = pred.map_to_latent_space(sample_id, ae_model)

# plot latent space
plot.plot_latent_space_2D(latent_coords_qcd, title_suffix="qcd", filename_suffix="qcd", fig_dir='fig')
plot.plot_latent_space_2D(latent_coords_sig, title_suffix="sig", filename_suffix="sig", fig_dir='fig')

# train k-means model
print('>>> training kmeans')
km_model = kmeans.train(latent_coords_qcd)

# apply to data -> obtain cluster assignment
centroids = km_model.labels_
# qcdSide
# signal
cluster_assignemnts = km_model.predict(latent_coords_sig) # latent coords of signal obtained from AE
# 

