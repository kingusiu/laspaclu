import numpy as np
import data.data_sample as dasa


def map_to_latent_space(data_sample, sample_id, model, read_n=int(1e5)):
    
    # read data
    jets = data_sample.get_dataset_for_inference(sample_id=sample_id, read_n=read_n)

    latent_coords = []

    for batch in jets:
        # run encoder
        coords = model.encoder(batch)
        latent_coords.append(coords)

    # return latent (per jet?)
    return np.concatenate(latent_coords, axis=0)
    