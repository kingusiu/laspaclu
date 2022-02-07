import numpy as np
import os
import matplotlib.pyplot as plt
import anpofah.util.plotting_util as pu
import anpofah.model_analysis.roc_analysis as roc
import dadrah.selection.loss_strategy as losa
import pofah.jet_sample as jesa

from collections import namedtuple


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

input_dir_classic = '/eos/user/v/vabelis/kinga_colab/kinga_classical_c=1.0_gamma=scale/'
input_dir_quantum = '/eos/user/v/vabelis/kinga_colab/kinga_quantum_c=1.0_gamma=scale/'

tpr_classic = read_data(input_dir_classic, 'tpr.npy').flatten()
fpr_classic = read_data(input_dir_classic, 'fpr.npy').flatten()
dist_classic = read_data(input_dir_classic, 'y_score_list.npy')

tpr_quantum = read_data(input_dir_quantum, 'tpr.npy').flatten()
fpr_quantum = read_data(input_dir_quantum, 'fpr.npy').flatten()
dist_quantum = read_data(input_dir_quantum, 'y_score_list.npy')

# distance values
dist_classic_qcd = dist_classic.flatten()[:500]
dist_quantum_qcd = dist_quantum.flatten()[:500]
dist_classic_sig = dist_classic.flatten()[500:]
dist_quantum_sig = dist_quantum.flatten()[500:]



# import ipdb; ipdb.set_trace()

#****************************************#
#               ANALYSIS
#****************************************#


def plot_roc(fpr_classic, tpr_classic, fpr_quantum, tpr_quantum, legend, title_suffix=None, legend_loc='best', plot_name='ROC', fig_dir=None, xlim=None, ylim=None, log_x=True, fig_format='.png', show_plt=False, n_train=1e6, n_rand_class=None):

    n_test = 1000

    palette = ['#D62828', '#74A57F', '#FF9505', '#713E5A', '#3E96A1', '#EC4E20', ]
    fig = plt.figure(figsize=(8, 8))

    # plot classic & quantum
    if log_x:
        plt.loglog(tpr_classic, 1./fpr_classic, label='classic svm', color=palette[0])
        plt.loglog(tpr_quantum, 1./fpr_quantum, label='quantum svm', color=palette[1])
    else:
        plt.semilogy(tpr_classic, 1./fpr_classic, label='classic svm', color=palette[0])
        plt.semilogy(tpr_quantum, 1./fpr_quantum, label='quantum svm', color=palette[1])

    # add random decision line
    n_rand_class = 50 or n_test
    if log_x:
        plt.loglog(np.linspace(0, 1, num=n_rand_class), 1./np.linspace(0, 1, num=n_rand_class), linewidth=1.2, linestyle='--', color='slategrey')
    else:
        plt.semilogy(np.linspace(0, 1, num=n_rand_class), 1./np.linspace(0, 1, num=n_rand_class), linewidth=1.2, linestyle='--', color='slategrey')

    plt.grid()
    if xlim:
        plt.xlim(left=xlim)
    if ylim:
        plt.ylim(top=ylim)
    plt.xlabel('True positive rate',fontsize=20)
    plt.ylabel('1 / False positive rate', fontsize=20)
    legend = plt.legend(loc=legend_loc, handlelength=1.5,fontsize=20)
    for leg in legend.legendHandles:
        leg.set_linewidth(2.2)
    title = ' '.join(filter(None, [r"$N^{train}=$", '{:.1E}'.format(n_train).replace("E+0", "E"), r"$N^{test}=$", '{:.1E}'.format(n_test).replace("E+0", "E"), title_suffix]))
    plt.title(title , loc="right",fontsize=13)
    plt.tight_layout()
    if show_plt:
        plt.show()
    if fig_dir:
        print('writing ROC plot to {}'.format(fig_dir))
        fig.savefig(os.path.join(fig_dir, plot_name + fig_format), bbox_inches='tight')
    plt.close(fig)


xlabel = 'distance to boundary'

# plot classic distances
pu.plot_feature_for_n_samples([dist_classic_qcd, dist_classic_sig], sample_names=[params.sample_id_qcd, params.sample_id_sig], \
    xlabel=xlabel, plot_name='loss_qcd_vs_sig_'+params.cluster_alg, fig_dir=fig_dir, bg_name=params.sample_id_qcd)

# plot quantum distances
pu.plot_feature_for_n_samples([dist_quantum_qcd, dist_quantum_sig], sample_names=[params.sample_id_qcd, params.sample_id_sig], \
    xlabel='quantum '+xlabel, plot_name='quantum_loss_qcd_vs_sig_'+params.cluster_alg, fig_dir=fig_dir, bg_name=params.sample_id_qcd)

# inclusive roc
roc.plot_roc([dist_classic_sig, dist_quantum_sig], [dist_classic_qcd, dist_quantum_qcd], legend=['classic svm', 'quantum svm'], \
    plot_name='_'.join(['ROC', params.sample_id_qcd, 'vs', params.sample_id_sig, params.cluster_alg]), fig_dir=fig_dir, n_train=600, n_rand_class=0)

# plot_roc(fpr_classic, tpr_classic, fpr_quantum, tpr_quantum,  legend=['classic svm', 'quantum svm'], \
#     plot_name='_'.join(['ROC', params.sample_id_qcd, 'vs', params.sample_id_sig, params.cluster_alg]), fig_dir=fig_dir, n_train=600, n_rand_class=10)


# no binned roc
