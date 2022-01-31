import os
import datetime
import pathlib
import numpy as np
import pofah.jet_sample as jesa


def make_model_path(date=None, prefix='AE', run_n=0, mkdir=False):
    date_str = ''
    if date is None:
        date = datetime.date.today()
        date = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    path = os.path.join('models/saved', prefix+'model_run{}_{}'.format(str(run_n), date))
    if mkdir:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def read_latent_representation(sample, normalize=False, shuffle=True):

    l1, l2 = sample.get_latent_representation()

    latent_coords = np.vstack([l1, l2])
    if shuffle:
        np.random.shuffle(latent_coords)
    if normalize:
        latent_coords = prep.min_max_normalize(latent_coords)

    return latent_coords


def read_latent_jet_sample(input_dir, sample_id, read_n=None):

    file_name = os.path.join(input_dir, sample_id+'.h5')
    print('>>> reading {} events from {}'.format(str(read_n),file_name))

    return jesa.JetSampleLatent.from_input_file(name=sample_id, path=file_name, read_n=read_n)

