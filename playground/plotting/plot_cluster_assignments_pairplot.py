from collections import namedtuple
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
import pandas as pd

import pofah.jet_sample as jesa
import inference.clustering_classic as cluster
import inference.clustering_quantum as cluster_q
import inference.predict_autoencoder as pred
import laspaclu.data.data_sample as dasa
import laspaclu.analysis.plotting as plot
import util.persistence as pers
import util.preprocessing as prep
import util.logging as log



#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'run_n, lat_dim read_n sample_ids raw_format')
params = Parameters(run_n=42,
                    lat_dim=16,
                    read_n=int(1e3),
                    sample_ids=['qcdSigExt', 'GtoWW35na', 'GtoWW15br', 'AtoHZ35'],
                    raw_format=True)

fig_sizes = {
    4: (8,8),
    8: (12,12),
    16: (16,16),
    32: (20,20)
}

sample_name_dict = {

    'qcdSig': 'QCD signal-region',
    'qcdSigExt' : 'QCD signal-region ext',
    'GtoWW35na': r'$G(3.5 TeV)\to WW$ narrow',
    'GtoWW15br': r'$G(1.5 TeV)\to WW$ broad',
    'AtoHZ35': r'$A (3.5 TeV) \to H \to ZZZ$'
}

# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*60+'\n'+'\t\t\t plotting AE latent distribution of training data \n'+str(params)+'\n'+'*'*60)

# data input dir
input_dir = '/eos/home-e/epuljak/private/epuljak/public/diJet/'+str(int(params.lat_dim))

fig_dir = 'fig/qkmeans_run_'+str(int(params.run_n))
pathlib.Path(fig_dir).mkdir(parents=True,exist_ok=True)


#****************************************#
#               load QUANTUM model
#****************************************#

logger.info('loading qmeans')
model_path_qm = pers.make_model_path(prefix='QM', run_n=params.run_n) + '.npy'
with open(model_path_qm, 'rb') as f:
    cluster_q_centers = np.load(f)
logger.info('quantum cluster centers: ')
logger.info(cluster_q_centers)



for sample_id in params.sample_ids:

    #****************************************#
    #      load data latent representation
    #****************************************#

    latent_coords = pers.read_latent_rep_from_file(input_dir, sample_id, read_n=params.read_n, raw_format=True, shuffle=False)
    latent_coords_reshaped = np.stack(np.split(latent_coords, 2),axis=1) # reshape to N x 2 x z_dim
    sample_in = jesa.JetSampleLatent(name=sample_id, features=pd.DataFrame()).add_latent_representation(latent_coords_reshaped)

    logger.info('read {} {} jets'.format(len(latent_coords), sample_id))

    #****************************************#
    #          apply QUANTUM CLUSTERING
    #****************************************#

    # apply clustering algo
    logger.info('applying quantum clustering model')
    cluster_assign_q, distances_q = cluster_q.assign_clusters(latent_coords, cluster_q_centers, quantum_min=True) # latent coords of qcd train obtained from AE
    logger.info('plotting quantum cluster assignments')
    plot.plot_clusters_pairplot(latent_coords, cluster_assign_q, cluster_q_centers, filename_suffix='qmeans_'+str(params.run_n)+'_'+sample_id, fig_dir=fig_dir)
