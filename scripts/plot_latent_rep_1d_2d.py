#import setGPU
from collections import namedtuple
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib as jli
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
import anpofah.util.plotting_util as plut



def plot_m_features_for_n_samples(data, feature_names, sample_names, bins=100, suptitle=None, clip_outlier=False, normed=True, \
        ylogscale=True, single_row=False, plot_name='multi_feature_hist', fig_dir='fig', fig_format='.png', fig_size=(7,7), bg_name=None, histtype_bg='stepfilled'):
    '''
        plot multiple features for multiple samples as 1D histograms in one figure
        :param data: list of J ndarrays of K features with each N values
        :param bg_name: if not None, one sample will be treated as background and plotted in histtype_bg style
    '''

    # if one sample is to be treated as background sample
    bg_idx = 0
    
    rows_n, cols_n = plut.subplots_rows_cols(len(feature_names), single_row=single_row)
    fig, axs = plt.subplots(nrows=rows_n, ncols=cols_n, figsize=fig_size)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # for each feature
    for k, (ax, xlabel) in enumerate(zip(axs.flat, feature_names)):
        # loop through datasets
        for i, (dat, col) in enumerate(zip(data,plut.palette)): 
            if i == bg_idx:
                ax.hist(dat[k], bins=bins, density=True, alpha=0.5, histtype=histtype_bg, label=sample_names[i], color=col)
            else:
                ax.hist(dat[k], bins=bins, density=True, alpha=1.0, histtype='step', linewidth=1.3, label=sample_names[i], color=col)
        if ylogscale:
            ax.set_yscale('log', nonpositive='clip')
        ax.set_xlabel(xlabel, fontsize=12)
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
    
    if len(axs.shape) == 1: # single row
        axs[0].set_ylabel('fraction events',fontsize=12)
    else: # first subplot in each row
        for ax in axs[:,0]: ax.set_ylabel('fraction events',fontsize=12)
    #plt.legend(bbox_to_anchor=(0.5,-0.1), loc="upper center", mode='expand', ncol=len(data))
    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.1), loc="lower center", ncol=len(data))
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    print('writing figure to ' + os.path.join(fig_dir, plot_name + fig_format))
    fig.savefig(os.path.join(fig_dir, plot_name + fig_format), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)


def plot_latent_space_1D_bg_vs_sig(latent_bg, latent_sig, sample_names, fig_size, fig_dir='fig'):
    
    feature_names = [r'$z_{' + str(d+1) +'}$' for d in range(latent_bg.shape[1])]
    plot_name_suffix = '_'.join([s for s in sample_names])
    suptitle = 'latent space distributions BG vs SIG'
    plot_name = '_'.join(['latent_space_1D_hist', plot_name_suffix])

    plot_m_features_for_n_samples([latent_bg.T, latent_sig.T], feature_names, sample_names, plot_name=plot_name, fig_dir=fig_dir, fig_size=fig_size, bg_name='qcd')


#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'lat_dim read_n bg_sample_id sig_sample_ids raw_format')
params = Parameters(lat_dim=4,
                    read_n=int(1e3),
                    bg_sample_id='qcdSig',
                    sig_sample_ids=['qcdSigExt', 'GtoWW35na', 'GtoWW15br', 'AtoHZ35'],
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

fig_dir = 'fig/ae_extern/'+str(int(params.lat_dim))
pathlib.Path(fig_dir).mkdir(parents=True,exist_ok=True)

#****************************************#
#      load data latent representation
#****************************************#

# input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.ae_run_n)
input_dir = '/eos/home-e/epuljak/private/epuljak/public/diJet/'+str(int(params.lat_dim))
latent_bg = pers.read_latent_rep_from_file(input_dir, sample_id=params.bg_sample_id, read_n=params.read_n, raw_format=params.raw_format, shuffle=False, seed=None)
logger.info('read {} training samples ({} jets)'.format(len(latent_bg)/2, len(latent_bg))) # stacked j1 & j2

# plot single hist
#import ipdb; ipdb.set_trace()
feature_names = [r'$z_{' + str(d+1) +'}$' for d in range(latent_bg.shape[1])]
plot_m_features_for_n_samples([latent_bg.T], feature_names, params.bg_sample_id, plot_name='latent_hist_single_'+params.bg_sample_id, fig_dir=fig_dir, fig_size=fig_sizes[params.lat_dim], bg_name='qcd')


latent_sig_dict = {}
for sig_sample_id in params.sig_sample_ids:

    #****************************************#
    #      load data latent representation
    #****************************************#

    latent_sig = pers.read_latent_rep_from_file(input_dir, sig_sample_id, read_n=params.read_n, raw_format=params.raw_format, shuffle=False)
    latent_sig_dict[sig_sample_id] = np.stack(np.split(latent_sig, 2),axis=1)
    logger.info('plotting {} vs {} latent distribution to {}'.format(params.bg_sample_id, sig_sample_id, fig_dir))
    plot_latent_space_1D_bg_vs_sig(latent_bg, latent_sig, [params.bg_sample_id, sig_sample_id], fig_size=fig_sizes[params.lat_dim], fig_dir=fig_dir)

    # plot single hist
    plot_m_features_for_n_samples([latent_sig.T], feature_names, sig_sample_id, plot_name='latent_hist_single_'+sig_sample_id, fig_dir=fig_dir, fig_size=fig_sizes[params.lat_dim], bg_name='qcd')


#******************************************
#               2D scatter plots
#******************************************

# qcd
latent_bg_reshaped = np.stack(np.split(latent_bg, 2),axis=1) # reshape to N x 2 x z_dim

for sig_sample_id in params.sig_sample_ids:

    #****************************************#
    #      load data latent representation
    #****************************************#

    # read new qcd dataframe for each signal
    df_l1_qcd = pd.DataFrame(latent_bg_reshaped[:,0,:]).iloc[:,:params.lat_dim]
    df_l2_qcd = pd.DataFrame(latent_bg_reshaped[:,1,:]).iloc[:,:params.lat_dim]
    df_l1_qcd['sample_id'] = params.bg_sample_id
    df_l2_qcd['sample_id'] = params.bg_sample_id


    latent_sig = latent_sig_dict[sig_sample_id]

    df_l1_sig = pd.DataFrame(latent_sig[:,0,:]).iloc[:,:params.lat_dim]
    df_l2_sig = pd.DataFrame(latent_sig[:,0,:]).iloc[:,:params.lat_dim]
    df_l1_sig['sample_id'] = sig_sample_id
    df_l2_sig['sample_id'] = sig_sample_id

    df_j1 = df_l1_qcd.append(df_l1_sig, ignore_index=True)
    df_j2 = df_l2_qcd.append(df_l2_sig, ignore_index=True)

    df_j1 = df_j1.assign(sample_label=df_j1.sample_id.map({params.bg_sample_id: sample_name_dict[params.bg_sample_id], sig_sample_id: sample_name_dict[sig_sample_id]}))
    df_j2 = df_j2.assign(sample_label=df_j2.sample_id.map({params.bg_sample_id: sample_name_dict[params.bg_sample_id], sig_sample_id: sample_name_dict[sig_sample_id]}))


    #****************************************#
    #               PLOT SCATTER
    #****************************************#

    sns.set_style(hep.style.CMS)

    print('plotting latent space pair scatter plot to {}'.format(fig_dir))

    plot1 = sns.pairplot(df_j1, hue='sample_label', kind='kde')
    sns.move_legend(plot1, bbox_to_anchor=(0.5, -0.1), loc="lower center", ncol=2, labelspacing=0.8, fontsize=18, title='Samples', title_fontsize='18')
    plt.tight_layout()
    plot1.savefig(fig_dir+'/latent_pair_scatter_'+params.bg_sample_id+'_vs_'+sig_sample_id+'_j1_'+str(params.lat_dim)+'dims.png')

    plot2 = sns.pairplot(df_j2, hue='sample_label', kind='kde')
    sns.move_legend(plot1, bbox_to_anchor=(0.5, -0.1), loc="lower center", ncol=2, labelspacing=0.8, fontsize=18, title='Samples', title_fontsize='18')
    plt.tight_layout()
    plot2.savefig(fig_dir+'/latent_pair_scatter_'+params.bg_sample_id+'_vs_'+sig_sample_id+'_j2_'+str(params.lat_dim)+'dims.png')




