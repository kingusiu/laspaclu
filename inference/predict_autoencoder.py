import data.data_sample as dasa


def map_to_latent_space(sample_id, model, read_n=int(1e5)):
    
    # read data
    sample = dasa.DataSample(sample_id)
    jets = sample.get_dataset_for_inference(sample_id=sample_id, read_n=read_n)

    # run encoder
    latent_coords = model.encoder(jets)

    # return latent (per jet?)
    return latent_coords
    