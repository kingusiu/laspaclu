import os
import datetime
import pathlib
import h5py
import numpy as np
import pofah.jet_sample as jesa
import laspaclu.src.util.string_constants as stco


def make_model_path(prefix='KM', run_n=0, mkdir=False):

    path = os.path.join(stco.cluster_out_model_dir, prefix+'model_run{}'.format(str(run_n)))
    if mkdir:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def read_latent_representation(sample, normalize=False, shuffle=True, stacked=True):

    l1, l2 = sample.get_latent_representation()
    if stacked == False:
        return l1, l2

    # stack (and shuffle)
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
def read_latent_representation_raw(input_dir, sample_id, read_n=None, shuffle=True, seed=None):

    file_name_dict = {
        'qcdSig' : 'latentrep_QCD_sig.h5',
        'qcdSigExt' : 'latentrep_QCD_sig_testclustering.h5',
        'GtoWW35na' : 'latentrep_RSGraviton_WW_NA_35.h5',
        'GtoWW15br' : 'latentrep_RSGraviton_WW_BR_15.h5',
        'AtoHZ35' : 'latentrep_AtoHZ_to_ZZZ_35.h5'
    }

    file_name = os.path.join(input_dir, file_name_dict[sample_id])
    print('>>> reading {} events from {}'.format(str(read_n),file_name))

    ff = h5py.File(file_name,'r')

    latent_coords = np.array(ff.get('latent_space'))[:read_n]
    latent_coords = np.vstack([latent_coords[:,0,:], latent_coords[:,1,:]])

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(latent_coords)

    return latent_coords


def read_latent_rep_from_file(input_dir, sample_id, read_n=None, raw_format=False, shuffle=False, mjj_center=False, seed=None):

    if raw_format: # add raw format option if data not saved in JetSampleLatent structure
        return read_latent_representation_raw(input_dir, sample_id, read_n=read_n, shuffle=shuffle, seed=seed)

    sample = read_latent_jet_sample(input_dir, sample_id, read_n=read_n)
    if mjj_center:
        sample = jesa.get_mjj_binned_sample_center_bin(sample, mjj_peak=3500) 
    return pers.read_latent_representation(sample, shuffle=shuffle)
