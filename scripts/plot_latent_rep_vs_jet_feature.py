import os
import pandas as pd
import numpy as np
from collections import namedtuple

import pofah.jet_sample as jesa
import anpofah.util.plotting_util as pu


#****************************************#
#               RUNTIME PARAMS
#****************************************#

sample_name_dict = {

    'qcdSig': 'QCD signal-region',
    'GtoWW35na': r'$G(3.5 TeV)\to WW$ narrow',
}

sample_id_qcd = 'qcdSig'
sample_id_sig = 'GtoWW35na'

Parameters = namedtuple('Parameters', ' run_n read_n dims_n')
params = Parameters(run_n=50, read_n=int(1e5), dims_n=None)

input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.run_n)
fig_dir = 'fig/ae_run_'+str(params.run_n)+'/corr_latent_space_jet_features'

#****************************************#
#               READ DATA
#****************************************#

# read qcd
file_name = os.path.join(input_dir, sample_id_qcd+'.h5')
print('>>> reading {} events from {}'.format(str(params.read_n), file_name))
sample_qcd = jesa.JetSampleLatent.from_input_file(name=sample_id_qcd, path=file_name, read_n=params.read_n)
l1, l2 = sample_qcd.get_latent_representation()
df_l1_qcd = pd.DataFrame(l1).iloc[:,:params.dims_n]
df_l2_qcd = pd.DataFrame(l2).iloc[:,:params.dims_n]

# read signal
file_name = os.path.join(input_dir, sample_id_sig+'.h5')
print('>>> reading {} events from {}'.format(str(params.read_n), file_name))
sample_sig = jesa.JetSampleLatent.from_input_file(name=sample_id_sig, path=file_name, read_n=params.read_n)
l1, l2 = sample_sig.get_latent_representation()
df_l1_sig = pd.DataFrame(l1).iloc[:,:params.dims_n]
df_l2_sig = pd.DataFrame(l2).iloc[:,:params.dims_n]


#****************************************#
#               PLOT
#****************************************#

feature_name_2 = 'mJJ'

### background

sample_name = 'qcd'
# qcd J1
for dim_n, dim_col in enumerate(df_l1_qcd):
    feature_name_1 = 'z'+str(dim_n)
    suffix = 'J1'
    title = ' '.join(['distribution', sample_name, feature_name_1, 'vs', feature_name_2, suffix])
    plot_name = '_'.join(['hist_2D', feature_name_1, feature_name_2, sample_name, suffix])
    pu.plot_hist_2d(df_l1_qcd[dim_col].to_numpy(), sample_qcd['mJJ'], xlabel=feature_name_1, ylabel=feature_name_2, title=title, plot_name=plot_name, fig_dir=fig_dir, clip_outlier=False)

# qcd J2
for dim_n, dim_col in enumerate(df_l2_qcd):
    feature_name_1 = 'z'+str(dim_n)
    suffix = 'J2'
    title = ' '.join(['distribution', sample_name, feature_name_1, 'vs', feature_name_2, suffix])
    plot_name = '_'.join(['hist_2D', feature_name_1, feature_name_2, sample_name, suffix])
    pu.plot_hist_2d(df_l2_qcd[dim_col].to_numpy(), sample_qcd['mJJ'], xlabel=feature_name_1, ylabel=feature_name_2, title=title, plot_name=plot_name, fig_dir=fig_dir, clip_outlier=False)

### signal

sample_name = 'sig'
# signal J1
for dim_n, dim_col in enumerate(df_l1_sig):
    feature_name_1 = 'z'+str(dim_n)
    suffix = 'J1'
    title = ' '.join(['distribution', sample_name, feature_name_1, 'vs', feature_name_2, suffix])
    plot_name = '_'.join(['hist_2D', feature_name_1, feature_name_2, sample_name, suffix])
    pu.plot_hist_2d(df_l1_sig[dim_col].to_numpy(), sample_sig['mJJ'], xlabel=feature_name_1, ylabel=feature_name_2, title=title, plot_name=plot_name, fig_dir=fig_dir, clip_outlier=False)

# signal J2
for dim_n, dim_col in enumerate(df_l2_sig):
    feature_name_1 = 'z'+str(dim_n)
    suffix = 'J2'
    title = ' '.join(['distribution', sample_name, feature_name_1, 'vs', feature_name_2, suffix])
    plot_name = '_'.join(['hist_2D', feature_name_1, feature_name_2, sample_name, suffix])
    pu.plot_hist_2d(df_l2_sig[dim_col].to_numpy(), sample_sig['mJJ'], xlabel=feature_name_1, ylabel=feature_name_2, title=title, plot_name=plot_name, fig_dir=fig_dir, clip_outlier=False)

