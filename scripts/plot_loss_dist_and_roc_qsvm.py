import numpy as np
import os
import matplotlib.pyplot as plt
import anpofah.util.plotting_util as pu
import anpofah.model_analysis.roc_analysis as roc
import dadrah.selection.loss_strategy as losa
import pofah.jet_sample as jesa

from collections import namedtuple


sample_name_dict = {
    
    'qcdSigExt': 'QCD signal-region',
    'GtoWW35na': r'$G(3.5 TeV)\to WW$ narrow',
}


# setup
Parameters = namedtuple('Parameters', 'run_n read_n sample_id_qcd sample_id_sig cluster_alg normalize')
params = Parameters(run_n=14, read_n=int(5e4), sample_id_qcd='qcdSigExt', sample_id_sig='GtoWW35na', cluster_alg='svm', normalize=False)
fig_dir = 'fig/run_'+str(params.run_n)

#****************************************#
#               READ DATA
#****************************************#

def read_data(input_dir, fname):
    with open(os.path.join(input_dir,fname),'rb') as f:
        return np.load(f)

input_dir_classic = '/eos/user/v/vabelis/kinga_colab/kinga_classical_10k_c=1.0_gamma=scale/'
input_dir_quantum = '/eos/user/v/vabelis/kinga_colab/kinga_quantum_10k_c=1.0_gamma=scale/'

tpr_classic = read_data(input_dir_classic, 'tpr.npy').flatten()
fpr_classic = read_data(input_dir_classic, 'fpr.npy').flatten()
dist_classic = read_data(input_dir_classic, 'y_score_list.npy')

tpr_quantum = read_data(input_dir_quantum, 'tpr.npy').flatten()
fpr_quantum = read_data(input_dir_quantum, 'fpr.npy').flatten()
dist_quantum = read_data(input_dir_quantum, 'y_score_list.npy')

# distance values
bg_sig_split = int(len(dist_classic.flatten())/2)
print('reading first {} elements as bg and last {} elements as signal'.format(bg_sig_split,bg_sig_split))
dist_classic_qcd = dist_classic.flatten()[:bg_sig_split]
dist_quantum_qcd = dist_quantum.flatten()[:bg_sig_split]
dist_classic_sig = dist_classic.flatten()[bg_sig_split:]
dist_quantum_sig = dist_quantum.flatten()[bg_sig_split:]



# import ipdb; ipdb.set_trace()

#****************************************#
#               ANALYSIS
#****************************************#


xlabel = 'distance to boundary'

# get shared xlimits
xmin, xmax = min(min(min(dist_classic_qcd), min(dist_classic_sig)), min(min(dist_quantum_qcd), min(dist_quantum_sig))), max(max(max(dist_classic_qcd), max(dist_classic_sig)), max(max(dist_quantum_qcd), max(dist_quantum_sig))) 

# plot classic distances
pu.plot_feature_for_n_samples([dist_classic_qcd, dist_classic_sig], sample_names=[sample_name_dict[params.sample_id_qcd], sample_name_dict[params.sample_id_sig]], \
    xlabel=xlabel, plot_name='loss_qcd_vs_sig_'+params.cluster_alg, fig_dir=fig_dir, bg_name=sample_name_dict[params.sample_id_qcd], legend_outside=False)

# plot quantum distances
pu.plot_feature_for_n_samples([dist_quantum_qcd, dist_quantum_sig], sample_names=[sample_name_dict[params.sample_id_qcd], sample_name_dict[params.sample_id_sig]], \
    xlabel='quantum '+xlabel, plot_name='quantum_loss_qcd_vs_sig_'+params.cluster_alg, fig_dir=fig_dir, bg_name=sample_name_dict[params.sample_id_qcd], legend_outside=False)

# inclusive roc
roc.plot_roc([dist_classic_sig, dist_quantum_sig], [dist_classic_qcd, dist_quantum_qcd], legend=['classic svm', 'quantum svm'], \
    plot_name='_'.join(['ROC', params.sample_id_qcd, 'vs', params.sample_id_sig, params.cluster_alg]), fig_dir=fig_dir, \
    n_train=600, n_rand_class=int(1e2), fig_format='.pdf')

# plot_roc(fpr_classic, tpr_classic, fpr_quantum, tpr_quantum,  legend=['classic svm', 'quantum svm'], \
#     plot_name='_'.join(['ROC', params.sample_id_qcd, 'vs', params.sample_id_sig, params.cluster_alg]), fig_dir=fig_dir, n_train=600, n_rand_class=10)


# no binned roc
