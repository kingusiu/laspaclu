import numpy as np
import os
import pathlib
from collections import namedtuple
from collections import OrderedDict

import laspaclu.src.analysis.roc as roc
import laspaclu.src.util.logging as log
import laspaclu.src.util.string_constants as stco
import pofah.jet_sample as jesa



def read_scores_from_multiple_runs(ll_run_n, sample_id, read_n):

    ll_scores_c = []; ll_scores_q = []

    for run_n in ll_run_n:

        input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

        sample = jesa.JetSample.from_input_file(name=sample_id, path=input_data_dir+'/'+sample_id+'.h5').filter(slice(read_n))

        ll_scores_c.append(sample['classic_loss'])
        ll_scores_q.append(sample['quantum_loss'])

    return ll_scores_c, ll_scores_q


#****************************************#
#           Runtime Params
#****************************************#


Parameters = namedtuple('Parameters', 'sample_id_qcd sample_id_sigs read_n')
params = Parameters(sample_id_qcd='qcdSigExt',
                    sample_id_sigs=['GtoWW35na', 'GtoWW15br', 'AtoHZ35'], 
                    read_n=int(5e4))


#****************************************************************#
#    PLOT ALL SIGNALS, FIXED DIM=8, FIXED TRAIN-N=600 (run 45)
#****************************************************************#

#****************************************#
#               READ DATA

run_n = 49
dim_z = 8
train_n = 6000

# path setup
fig_dir = os.path.join(stco.reporting_fig_base_dir,'roc','allSig','r'+str(run_n))
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

sample_qcd = jesa.JetSample.from_input_file(name=params.sample_id_qcd, path=input_data_dir+'/'+params.sample_id_qcd+'.h5').filter(slice(params.read_n))

scores_qcd_c = sample_qcd['classic_loss']
scores_qcd_q = sample_qcd['quantum_loss']

ll_scores_qcd_c = []; ll_scores_qcd_q = []
ll_scores_sig_c = []; ll_scores_sig_q = []
for sample_id_sig in params.sample_id_sigs:
    ll_scores_qcd_c.append(scores_qcd_c)
    ll_scores_qcd_q.append(scores_qcd_q)
    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))
    ll_scores_sig_c.append(sample_sig['classic_loss'])
    ll_scores_sig_q.append(sample_sig['quantum_loss'])


legend_labels = [stco.sample_name_dict[id_sig] for id_sig in params.sample_id_sigs]
legend_title = r'anomaly signature'

roc.plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(1e4), plot_name='roc_allSig_r'+str(run_n), fig_dir=fig_dir)


#**********************************************************#
#    PLOT ALL TRAIN-N, FIXED DIM=8, FIXED SIG (runs 
#**********************************************************#

run_n_dict = OrderedDict([
    (10,41),
    (600,45),
    (6000,49),
])

dim_z = 8

# path setup
fig_dir = os.path.join(stco.reporting_fig_base_dir,'roc','allTrainN','z'+str(dim_z)+'_'+'_'.join(['r'+str(nn) for nn in run_n_dict.values()]))
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

# main legend setup

legend_labels=list(run_n_dict.keys())
legend_title = r"training size"


#****************************************#
#               QCD

ll_scores_qcd_c, ll_scores_qcd_q = read_scores_from_multiple_runs(run_n_dict.values(), params.sample_id_qcd, params.read_n)

#****************************************#
#               A to HZ

sample_id_sig = 'AtoHZ35'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

roc.plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(7e2), auc_legend_offset=-0.12, plot_name='roc_allTrainN_z'+str(dim_z)+'_'+str(sample_id_sig), fig_dir=fig_dir)


#****************************************#
#               G_RS 1.5TeV broad

sample_id_sig = 'GtoWW15br'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

roc.plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(1e4), auc_legend_offset=-0.12, plot_name='roc_allTrainN_z'+str(dim_z)+'_'+str(sample_id_sig), fig_dir=fig_dir)


#****************************************#
#               G_RS 3.5TeV na

sample_id_sig = 'GtoWW35na'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

roc.plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(1e2), auc_legend_offset=-0.12, plot_name='roc_allTrainN_z'+str(dim_z)+'_'+str(sample_id_sig), fig_dir=fig_dir)


#**********************************************************#
#    PLOT ALL Z-D, FIXED N=?, FIXED SIG (runs 
#**********************************************************#

run_n_dict = OrderedDict([
    (4,40),
    (8,41),
    (16,42),
    (32,43)
])

train_n = 10

# path setup
fig_dir = os.path.join(stco.reporting_fig_base_dir,'roc','allZDim','trainN'+str(train_n)+'_'+'_'.join(['r'+str(nn) for nn in run_n_dict.values()]))
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

# legend setup
# legend_labels=[r"$z \in \mathbb{R}^4$",r"$z \in \mathbb{R}^8$", r"$z \in \mathbb{R}^{16}$", r"$z \in \mathbb{R}^{32}$"])
legend_labels=[r"$z \in R^4$",r"$z \in R^8$", r"$z \in R^{16}$", r"$z \in R^{32}$"]
legend_title = r"latent dimension"

#****************************************#
#               QCD

ll_scores_qcd_c, ll_scores_qcd_q = read_scores_from_multiple_runs(run_n_dict.values(), params.sample_id_qcd, params.read_n)

#****************************************#
#               A to HZ

sample_id_sig = 'AtoHZ35'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

roc.plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(7e2), auc_legend_offset=-0.12, plot_name='roc_allZDim_trainN'+str(train_n)+'_'+str(sample_id_sig), fig_dir=fig_dir)


#****************************************#
#               G_RS 1.5TeV broad

sample_id_sig = 'GtoWW15br'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

roc.plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(5e3), auc_legend_offset=-0.12, plot_name='roc_allZDim_trainN'+str(train_n)+'_'+str(sample_id_sig), fig_dir=fig_dir)


#****************************************#
#               G_RS 3.5TeV na

sample_id_sig = 'GtoWW35na'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

roc.plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(1e3), auc_legend_offset=-0.1, plot_name='roc_allZDim_trainN'+str(train_n)+'_'+str(sample_id_sig), fig_dir=fig_dir)

