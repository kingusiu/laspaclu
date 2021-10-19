import numpy as np
import tensorflow as tf
import data.data_sample as dasa


def map_to_latent_space(data_sample, model, read_n=int(1e5)) -> np.ndarray: # [N x Z]
    
    # convert data to tf-batched dataset
    if isinstance(data_sample, dasa.DataSample):
        data_sample = data_sample.get_dataset_for_inference(read_n=read_n)
    else:
        data_sample = tf.data.Dataset.from_tensor_slices(data_sample).batch(2048)

    latent_coords = []

    for batch in data_sample:
        # run encoder
        coords = model.encoder(batch)
        latent_coords.append(coords)

    # return latent (per jet?)
    return np.concatenate(latent_coords, axis=0)
    