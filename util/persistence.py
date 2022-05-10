import os
import datetime
import pathlib
import h5py
import numpy as np
import pofah.jet_sample as jesa


def make_model_path(date=None, prefix='AE', run_n=0, mkdir=False):

    if date is None:
        date = datetime.date.today()
        date = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    path = os.path.join('models/saved', prefix+'model_run{}_{}'.format(str(run_n), date))
    if mkdir:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def read_latent_representation(sample, normalize=False, shuffle=True):

    l1, l2 = sample.get_latent_representation()

    # stack jets
    latent_coords = np.vstack([l1, l2])
    
    if shuffle:
        np.random.shuffle(latent_coords)
    if normalize:
        latent_coords = prep.min_max_normalize(latent_coords)

    return latent_coords


def read_latent_jet_sample(input_dir, sample_id, read_n=None, mJJ_binned=False):

    file_name = os.path.join(input_dir, sample_id+'.h5')
    print('>>> reading {} events from {}'.format(str(read_n),file_name))

    return jesa.JetSampleLatent.from_input_file(name=sample_id, path=file_name, read_n=read_n)


# read data from external format
def read_latent_representation_raw(input_dir, sample_id, read_n=None, shuffle=True):

    file_name = os.path.join(input_dir, sample_id+'.h5')
    print('>>> reading {} events from {}'.format(str(read_n),file_name))

    # different keys for each datafile...
    latent_dat_key = {
        'latentrep_QCD_sig' : 'latent_space',
        'latentrep_RSGraviton_WW_NA' : 'latent_space_NA_RSGraviton_WW_NA_3.5',
        'latentrep_RSGraviton_WW_BR' : 'latent_space_BR_RSGraviton_WW_BR_1.5',
        'latentrep_AtoHZ_to_ZZZ' : 'latent_space_AtoHZ_to_ZZZ_3.5'
    }

    ff = h5py.File(file_name,'r')

    # where are the two jets??
    latent_coords = np.array(ff.get(latent_dat_key[sample_id]))[:read_n]

    if shuffle:
        np.random.shuffle(latent_coords)

    return latent_coords


