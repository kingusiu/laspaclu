import os
from collections import namedtuple
import pathlib
import mplhep as hep
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jli
import numpy as np
import pandas as pd

import laspaclu.util.logging as log
import laspaclu.util.string_constants as stco
import laspaclu.util.persistence as pers
import laspaclu.analysis.plotting as plott
import pofah.jet_sample as jesa




def plot_loss_distributions(loss_qcd, loss_signals, xlim=None, xlabel_prefix='', plot_name='loss_distribution', fig_dir='fig', fig_format='.pdf'):
    
    fig = plt.figure(figsize=(7,5))

    sns.kdeplot(data=loss_qcd, color=stco.bg_blue, fill=True, alpha=0.7, lw=3, label='QCD sig')

    for (sig_id,loss_sig),c in zip(loss_signals.items(),stco.multi_sig_palette):
        sns.kdeplot(data=loss_sig, color=c, fill=False, lw=3, label=stco.sample_name_dict[sig_id])

    plt.legend(fontsize=17)
    plt.xlabel(xlabel_prefix + r' $\sum$ distance$^2$ to cluster',fontsize=18)
    plt.ylabel('density',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(xlim)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, plot_name + fig_format)
    print('writing figure to ' + fig_path)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)



def plot_cluster_centers_classic_vs_quantum(centers_c, centers_q, plot_name, fig_dir, fig_format='.pdf'):

    Z = centers_c.shape[1]
    columns = [r'$z_{}$'.format(i) for i in range(1,Z+1)]
    centroids = pd.DataFrame(centers_c, columns=columns)
    centroids_quantum = pd.DataFrame(centers_q, columns=columns)
    centroids = centroids.append(centroids_quantum, ignore_index=True)
    centroids['algorithm'] = ['classic']*2 +['quantum']*2

    palette = ['#6C56B3', '#5AD871']

    gg = sns.pairplot(centroids, hue='algorithm', markers='X', plot_kws=dict(alpha=0.9, s=500), diag_kind="hist", diag_kws=dict(alpha=0.7, shrink=0.6, edgecolor=None), palette=palette)

    for ax in gg.figure.axes:
        ax.tick_params(labelsize=15)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
    for ax in gg.diag_axes:
        ax.tick_params(labelsize=15)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)

    sns.move_legend(gg, bbox_to_anchor=(0.44, -0.1), loc="lower center", ncol=2, labelspacing=0.8, fontsize=14, markerscale=2, title='Algorithm')
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, plot_name + fig_format)
    print('writing figure to ' + fig_path)
    fig = gg.figure
    fig.savefig(fig_path, bbox_extra_artists=(gg._legend,), bbox_inches="tight")
    plt.close(fig)



#****************************************#
#           Runtime Params
#****************************************#


Parameters = namedtuple('Parameters', 'run_n latent_dim ae_run_n read_n sample_id_qcd sample_id_sigs raw_format')
params = Parameters(run_n=45, 
                    latent_dim=8,
                    ae_run_n=50, 
                    read_n=int(2e4), # test on 20K events in 10 fold (10x2000)
                    sample_id_qcd='qcdSigExt',
                    sample_id_sigs=['GtoWW35na', 'GtoWW15br', 'AtoHZ35'], 
                    raw_format=True)

# path setup
fig_dir = 'fig/qkmeans_run_'+str(params.run_n)
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*70+'\n'+'\t\t\t PLOTING RUN \n'+str(params)+'\n'+'*'*70)


######      what to plot        ######
do_loss_distributions = True
do_cluster_assignments = True
do_cluster_centers = True


plt.style.use(hep.style.CMS)

#****************************************#
#               READ DATA
#****************************************#
input_data_dir = stco.cluster_out_data_dir+'/run_'+str(params.run_n)

sample_qcd = jesa.JetSampleLatent.from_input_file(name=params.sample_id_qcd, path=input_data_dir+'/'+params.sample_id_qcd+'.h5').filter(slice(params.read_n))
sample_sigs = {}
for sample_id_sig in params.sample_id_sigs:
    sample_sigs[sample_id_sig] = jesa.JetSampleLatent.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))

dist_qcd = sample_qcd['classic_loss']
dist_q_qcd = sample_qcd['quantum_loss']

dist_sigs = {}
dist_q_sigs = {}
for sample_id_sig in params.sample_id_sigs:
    dist_sigs[sample_id_sig] = sample_sigs[sample_id_sig]['classic_loss']
    dist_q_sigs[sample_id_sig] = sample_sigs[sample_id_sig]['quantum_loss']


#****************************************#
#           Read in Models
#****************************************#

model_path_km = pers.make_model_path(prefix='KM', run_n=params.run_n)
logger.info('loading clustering model ' + model_path_km)

cluster_model = jli.load(model_path_km+'.joblib')    
centers_c = cluster_model.cluster_centers_
logger.info('classic cluster centers: ')
logger.info(centers_c)


## quantum model (distance calc and minimization = quantum)

logger.info('loading qmeans')
model_path_qm = pers.make_model_path(prefix='QM', run_n=params.run_n) + '.npy'
with open(model_path_qm, 'rb') as f:
    centers_q = np.load(f)
logger.info('quantum cluster centers: ')
logger.info(centers_q)



#****************************************#
#        PLOT LOSS DISTRIBUTIONS
#****************************************#

if do_loss_distributions:

    # get shared xlimits
    xmin, xmax = min(min(min(dist_qcd), min(dist_sigs['GtoWW35na'])), min(min(dist_q_qcd), min(dist_q_sigs['GtoWW35na']))), max(max(max(dist_qcd), max(dist_sigs['GtoWW35na'])), max(max(dist_q_qcd), max(dist_q_sigs['GtoWW35na']))) 

    logger.info('plotting loss distributions')

    # classic loss
    plot_name = 'loss_distributions_classic_run_'+str(params.run_n)
    plot_loss_distributions(dist_qcd, dist_sigs, xlim=[xmin,xmax], xlabel_prefix='classic', plot_name=plot_name, fig_dir=fig_dir)

    # quantum loss
    plot_name = 'loss_distributions_quantum_run_'+str(params.run_n)
    plot_loss_distributions(dist_q_qcd, dist_q_sigs, xlim=[xmin,xmax], xlabel_prefix='quantum', plot_name=plot_name, fig_dir=fig_dir)


#************************************************#
#           PLOT CLUSTER CENTERS
#************************************************#


if do_cluster_centers:
     
    plot_cluster_centers_classic_vs_quantum(centers_c, centers_q, plot_name='cluster_centers_classic_vs_quantum_run_'+str(params.run_n), fig_dir=fig_dir)


#************************************************#
#           PLOT CLUSTER ASSIGNMENTS
#************************************************#

if do_cluster_assignments:

    latent_coords = {}
    assign_c = {}
    assign_q = {}

    # qcd
    latent_coords[params.sample_id_qcd] = pers.read_latent_representation(sample_qcd, shuffle=False)
    assign_c[params.sample_id_qcd] = np.vstack((sample_qcd['classic_assign_j1'], sample_qcd['classic_assign_j2']))
    assign_q[params.sample_id_qcd] = np.vstack((sample_qcd['quantum_assign_j1'], sample_qcd['quantum_assign_j2']))
    
    # signals
    for sample_id in params.sample_id_sigs:

        latent_coords[sample_id] = pers.read_latent_representation(sample_sigs[sample_id], shuffle=False) # do not shuffle, as loss is later combined assuming first half=j1 and second half=j2

        assign_c[sample_id] = np.vstack((sample_sigs[sample_id]['classic_assign_j1'], sample_sigs[sample_id]['classic_assign_j2']))
        assign_q[sample_id] = np.vstack((sample_sigs[sample_id]['quantum_assign_j1'], sample_sigs[sample_id]['quantum_assign_j2']))

    # plot classic results
    for sample_id in [params.sample_id_qcd]+params.sample_id_sigs:
        plott.plot_clusters_pairplot(latent_coords[sample_id], assign_c[sample_id], cluster_centers_c, filename_suffix='cluster_assignments_classic_'+str(sample_id)+'_run_'+str(params.run_n), fig_dir=fig_dir)

    # plot quantum results
    for sample_id in [params.sample_id_qcd]+params.sample_id_sigs:
        plott.plot_clusters_pairplot(latent_coords[sample_id], assign_q[sample_id], cluster_centers_q, filename_suffix='cluster_assignments_quantum_'+str(sample_id)+'_run_'+str(params.run_n), fig_dir=fig_dir)

