import tensorflow as tf
import numpy as np

import pofah.path_constants.sample_dict_file_parts_input as sdi
import pofah.util.sample_factory as safa
import pofah.util.event_sample as evsa
import pofah.phase_space.cut_constants as cuco
import pofah.util.utility_fun as utfu


class DataSample():

    def __init__(self, sample_id):
        self.sample_id = sample_id
        self.jets = None


    def read_event_sample(self, sample_id, read_n=None, **cuts):

        paths = safa.SamplePathDirFactory(sdi.path_dict)
        # TODO: add jet-pt > 200 cuts !!!
        return evsa.EventSample.from_input_dir(name=sample_id, path=paths.sample_dir_path(sample_id), read_n=read_n, **cuts) 


    def read_particles(self, sample_id, read_n=None, **cuts):

        sample = self.read_event_sample(sample_id=sample_id, read_n=read_n, **cuts)
        particles_j1, particles_j2 = sample.get_particles()
        return particles_j1, particles_j2


    def read_particles_dijet(self, sample_id, read_n=None, shuffle=True): # -> nd.array [jet1_n+jet2_n, 100, 3]

        """ main method to read training data for AE 
            returns stacked J1 & J2 dataset
        """

        cuts = cuco.sideband_cuts if 'Side' in sample_id else cuco.signalregion_cuts
        particles_j1, particles_j2 = self.read_particles(sample_id=sample_id, read_n=read_n, **cuts)
        jets = np.vstack([particles_j1, particles_j2])
        if shuffle:
            np.random.shuffle(jets)
        return jets


    def get_datasets_for_training(self, batch_sz=256, read_n=int(1e5), test_dataset=True):

        self.jets = self.read_particles_dijet(self.sample_id, read_n=read_n)

        train_valid_split = int(len(self.jets)*0.8)
        train_dataset = tf.data.Dataset.from_tensor_slices(self.jets[:train_valid_split]).batch(batch_sz, drop_remainder=True)
        valid_dataset = tf.data.Dataset.from_tensor_slices(self.jets[train_valid_split:]).batch(batch_sz, drop_remainder=True)

        if not test_dataset:
            return train_dataset, valid_dataset

        # test dataset
        jets_test = self.read_particles_dijet(self.sample_id+'Ext', int(read_n/10))
        test_dataset = tf.data.Dataset.from_tensor_slices(jets_test).batch(batch_sz)

        return train_dataset, valid_dataset, test_dataset  


    def get_dataset_for_inference(self, read_n=None, batch_sz=256):
        """ 
            returns batched dataset of stacked J1 & J2 samples (single-jet input)
        """
        
        jets = self.read_particles_dijet(self.sample_id, read_n)
        return tf.data.Dataset.from_tensor_slices(jets).batch(batch_sz)


    def get_mean_and_stdev(self, sample_id=None, read_n=None):
        # read data if none was read yet
        if self.jets is None:
            sample_id = sample_id or self.sample_id
            self.jets = self.read_particles_dijet(sample_id=sample_id, read_n=read_n) # TODO: problematic to pass SB cuts here!
        
        return utfu.get_mean_and_stdev(self.jets)
