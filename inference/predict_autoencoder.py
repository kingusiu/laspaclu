import numpy as np
import data.data_sample as dasa


def map_to_latent_space(data_sample, model, read_n=int(1e5)) -> np.ndarray: # [N x Z]
    
    # read data
    jets = data_sample.get_dataset_for_inference(read_n=read_n)

    latent_coords = []

    for batch in jets:
        # run encoder
        coords = model.encoder(batch)
        latent_coords.append(coords)

    # return latent (per jet?)
    return np.concatenate(latent_coords, axis=0)
    