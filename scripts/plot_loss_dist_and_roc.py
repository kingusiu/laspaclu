import anpofah.util.plotting_util as pu
import anpofah.model_analysis.roc_analysis as roc
import dadrah.selection.loss_strategy as losa
import pofah.jet_sample as jesa

from collections import namedtuple


# setup
mG = 3500
Parameters = namedtuple('Parameters', 'run_n read_n sample_id_qcd sample_id_sig cluster_alg normalize')
params = Parameters(run_n=14, read_n=int(5e4), sample_id_qcd='qcdSigExt', sample_id_sig='GtoWW35na', cluster_alg='kmeans', normalize=False)
fig_dir = 'fig/run_'+str(params.run_n)

#****************************************#
#               READ DATA
#****************************************#
input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/run_"+str(params.run_n)

sample_qcd = jesa.JetSample.from_input_file(name=params.sample_id_qcd, path=input_dir+'/'+params.sample_id_qcd+'.h5')
sample_sig = jesa.JetSample.from_input_file(name=params.sample_id_sig, path=input_dir+'/'+params.sample_id_sig+'.h5')

dist_qcd = sample_qcd['classic_loss']
dist_sig = sample_sig['classic_loss']
dist_q_qcd = sample_qcd['quantum_loss']
dist_q_sig = sample_sig['quantum_loss']


#****************************************#
#               ANALYSIS
#****************************************#

xlabel = 'sum distances to clusters'

# plot classic distances
pu.plot_feature_for_n_samples([dist_qcd, dist_sig], sample_names=[params.sample_id_qcd, params.sample_id_sig], \
    xlabel=xlabel, plot_name='loss_qcd_vs_sig_'+params.cluster_alg, fig_dir=fig_dir, bg_name=params.sample_id_qcd)

# plot quantum distances
pu.plot_feature_for_n_samples([dist_q_qcd, dist_q_sig], sample_names=[params.sample_id_qcd, params.sample_id_sig], \
    xlabel='quantum '+xlabel, plot_name='quantum_loss_qcd_vs_sig_'+params.cluster_alg, fig_dir=fig_dir, bg_name=params.sample_id_qcd)

# inclusive roc
roc.plot_roc([dist_qcd, dist_q_qcd], [dist_sig, dist_q_sig], legend=['classic kmeans', 'quantum kmeans'], plot_name='_'.join(['ROC', params.sample_id_qcd, 'vs', params.sample_id_sig, params.cluster_alg]), fig_dir=fig_dir)
# binned roc
loss_dict = {
    'classic' : losa.LossStrategy(loss_fun=(lambda x : x['classic_loss']), title_str='classic kmeans', file_str='classic_kmeans'),
    'quantum' : losa.LossStrategy(loss_fun=(lambda x : x['quantum_loss']), title_str='quantum kmeans', file_str='quantum_kmeans')
    }
roc.plot_binned_ROC_loss_strategy(sample_qcd, sample_sig, mass_center=mG, strategy_ids=loss_dict.keys(), loss_dict=loss_dict, fig_dir=fig_dir)
