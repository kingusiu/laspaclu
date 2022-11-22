import anpofah.util.plotting_util as pu
import analysis.roc_for_paper as roc
import dadrah.selection.loss_strategy as losa
import pofah.jet_sample as jesa
import util.logging as log

from collections import namedtuple
import pathlib
import pandas as pd
import os
import h5py
import numpy as np


sample_name_dict = {

    'qcdSigExt': 'QCD signal-region',
    'GtoWW35na': r'$G(3.5 TeV)\to WW$ narrow',
}


# setup

data_source = 'Vasilis' # 'Ema'
plot_loss_distr = False

mG = 3500
cluster_alg = 'kmedians' if data_source == 'Ema' else 'qsvm'
classic_loss_key = 'classic_loss_' if data_source == 'Ema' else 'classical_loss_'
Parameters = namedtuple('Parameters', 'read_n sample_id_qcd sample_id_sig cluster_alg normalize n_train')
params = Parameters(read_n=int(1e4), sample_id_qcd='qcdSigExt', sample_id_sig='GtoWW35na', cluster_alg=cluster_alg, \
    normalize=False, n_train=2e6)
fig_dir = 'fig/kmedians' if data_source == 'Ema' else 'fig/qsvm'
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)


default_latent_dim = 8

file_name_dict = {
    'GtoWW15br' : 'RSGraviton_WW15BR.h5', 
    'GtoWW35na' : 'RSGraviton_WW35NA.h5', 
    'AtoHZ35' : 'AtoHZ_to_ZZZ35.h5'
}

#****************************************#
#               READ DATA
#****************************************#
if data_source == 'Ema':
    input_dir = "/eos/user/e/epuljak/private/epuljak/public/iML_data/"+str(default_latent_dim)
else:
    input_dir = "/eos/user/v/vabelis/mpp_collab/iML_qsvm_plots/"+str(default_latent_dim)
file_name_default = 'Latent_8_trainsize_600_RSGraviton_WW35NA.h5'


#****************************************#
#               ANALYSIS LOSS
#****************************************#

if plot_loss_distr:

    ff = h5py.File(os.path.join(input_dir,file_name_default),'r')

    dist_qcd = np.asarray(ff.get(classic_loss_key+'qcd'))
    dist_sig = np.asarray(ff.get(classic_loss_key+'sig'))
    dist_q_qcd = np.asarray(ff.get('quantum_loss_qcd'))
    dist_q_sig = np.asarray(ff.get('quantum_loss_sig'))

    xlabel = 'sum distances to clusters'

    # get shared xlimits
    xmin, xmax = min(min(min(dist_qcd), min(dist_sig)), min(min(dist_q_qcd), min(dist_q_sig))), max(max(max(dist_qcd), max(dist_sig)), max(max(dist_q_qcd), max(dist_q_sig))) 

    # plot classic distances
    pu.plot_feature_for_n_samples([dist_qcd, dist_sig], sample_names=[sample_name_dict[params.sample_id_qcd], sample_name_dict[params.sample_id_sig]], \
        xlabel=xlabel, xlim=[xmin, xmax], plot_name='loss_qcd_vs_sig_'+params.cluster_alg, fig_dir=fig_dir, bg_name=sample_name_dict[params.sample_id_qcd], legend_outside=False)

    # plot quantum distances
    pu.plot_feature_for_n_samples([dist_q_qcd, dist_q_sig], sample_names=[sample_name_dict[params.sample_id_qcd], sample_name_dict[params.sample_id_sig]], \
        xlabel='quantum '+xlabel, xlim=[xmin, xmax], plot_name='quantum_loss_qcd_vs_sig_'+params.cluster_alg, fig_dir=fig_dir, bg_name=sample_name_dict[params.sample_id_qcd], legend_outside=False)


#****************************************#
#               ANALYSIS ROC
#****************************************#

var_sig = False
var_train_sz = False
var_latent_dim = True

test_n = int(1e4)
default_train_n = int(6e2)
bg_id = 'qcdSigExt'

#****************************************#
#               signal variation

if var_sig:

    sig_ids = ['GtoWW35na', 'AtoHZ35', 'GtoWW15br']
    # sig_ids = ['AtoHZ35']
    read_n = int(1e4)

    # logging
    logger = log.get_logger(__name__)
    logger.info('\n'+'*'*70+'\n'+'\t\t\t plotting roc_analysis for run '+input_dir+'\n'+'*'*70)

    #****************************************#
    #               READ DATA
    #****************************************#

    # import ipdb; ipdb.set_trace()
    losses_sig = []
    for sig_id in sig_ids:
        file_name = 'Latent_'+str(int(default_latent_dim))+'_trainsize_'+str(default_train_n)+'_'+file_name_dict[sig_id]
        file = h5py.File(os.path.join(input_dir,file_name),'r')
        dist_q_sig = np.asarray(file.get('quantum_loss_sig'))
        dist_sig = np.asarray(file.get(classic_loss_key+'sig'))
        losses_sig.append([dist_q_sig, dist_sig])

    dist_qcd = np.asarray(file.get(classic_loss_key+'qcd'))
    dist_q_qcd = np.asarray(file.get('quantum_loss_qcd'))
    loss_qcd = [dist_q_qcd, dist_qcd]


    class_labels, losses = roc.prepare_labels_and_losses_signal_comparison(loss_qcd, losses_sig)
    legend_colors = [roc.sig_name_dict[sig_id] for sig_id in sig_ids]
    title = ' '.join(filter(None, [r"$N^{train}=$", '{:.0E}'.format(default_train_n).replace("E+0", "E"), r"$N^{test}=$", '{:.0E}'.format(test_n).replace("E+0", "E")]))

    aucs = roc.plot_roc(class_labels, losses, legend_colors=legend_colors, legend_colors_title='signals', title=title, plot_name='ROC_'+cluster_alg+'_allSig', fig_dir=fig_dir)

    print(aucs)


#****************************************#
#               train size variation


if var_train_sz:
    
    train_szs = [10, 600, 6000]

    sig_id = 'GtoWW35na'
    read_n = int(1e4)

    #****************************************#
    #               READ DATA
    #****************************************#
    losses_qcd = []
    losses_sig = []

    for train_sz in train_szs:

        file_name = 'Latent_'+str(int(default_latent_dim))+'_trainsize_'+str(train_sz)+'_'+file_name_dict[sig_id]
        file = h5py.File(os.path.join(input_dir,file_name),'r')
        # qcd loss
        dist_qcd = np.asarray(file.get(classic_loss_key+'qcd'))
        dist_q_qcd = np.asarray(file.get('quantum_loss_qcd'))
        losses_qcd.append([dist_q_qcd, dist_qcd])
        # sig loss
        dist_q_sig = np.asarray(file.get('quantum_loss_sig'))
        dist_sig = np.asarray(file.get(classic_loss_key+'sig'))
        losses_sig.append([dist_q_sig, dist_sig])

    class_labels, losses = roc.prepare_labels_and_losses_train_sz_comparison(losses_qcd, losses_sig)
    title = r'$N^{train}=$var, ' + r'$N^{test}=$' + '{:.0E}'.format(test_n).replace("E+0", "E")

    aucs = roc.plot_roc(class_labels, losses, legend_colors=[str(s) for s in train_szs], legend_colors_title='N train', \
        title=title, plot_name='ROC_multi_train_sz_compare_allSig', fig_dir=fig_dir)

    print(aucs)



#****************************************#
#               latent dim variation


if var_latent_dim:

    latent_dim_dict = {
     4 : r'$z \in \mathbb{R}^4$',
     8 : r'$z \in \mathbb{R}^8$',
     16 : r'$z \in \mathbb{R}^{16}$',
     24 : r'$z \in \mathbb{R}^{24}$',
     32 : r'$z \in \mathbb{R}^{32}$',
    }
    
    latent_dims = [4, 16, 32]

    default_sig_id = 'GtoWW35na'
    read_n = int(1e4)

    #****************************************#
    #               READ DATA
    #****************************************#
    losses_qcd = []
    losses_sig = []

    for latent_dim in latent_dims:

        input_dir = "/eos/user/e/epuljak/private/epuljak/public/iML_data/"+str(latent_dim)

        file_name = 'Latent_'+str(int(latent_dim))+'_trainsize_'+str(default_train_n)+'_'+file_name_dict[default_sig_id]
        file = h5py.File(os.path.join(input_dir,file_name),'r')
        # qcd loss
        dist_qcd = np.asarray(file.get(classic_loss_key+'qcd'))
        dist_q_qcd = np.asarray(file.get('quantum_loss_qcd'))
        losses_qcd.append([dist_q_qcd, dist_qcd])
        # sig loss
        dist_q_sig = np.asarray(file.get('quantum_loss_sig'))
        dist_sig = np.asarray(file.get(classic_loss_key+'sig'))
        losses_sig.append([dist_q_sig, dist_sig])

    class_labels, losses = roc.prepare_labels_and_losses_train_sz_comparison(losses_qcd, losses_sig)
    title = ' '.join(filter(None, [r"$N^{train}=$", '{:.0E}'.format(default_train_n).replace("E+0", "E"), r"$N^{test}=$", '{:.0E}'.format(test_n).replace("E+0", "E")]))

    aucs = roc.plot_roc(class_labels, losses, legend_colors=[str(latent_dim_dict[d]) for d in latent_dims], legend_colors_title='N train', \
        title=title, plot_name='ROC_multi_latent_dim_compare_allSig_'+'_'.join(str(d) for d in latent_dims), fig_dir=fig_dir, test_n=int(read_n/100))

    print(aucs)
