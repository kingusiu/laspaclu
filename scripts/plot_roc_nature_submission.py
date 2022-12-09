import os
from collections import namedtuple
from collections import OrderedDict
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import roc_curve, auc
import sklearn.metrics as skl
import mplhep as hep
from matplotlib.lines import Line2D

import laspaclu.src.util.logging as log
import laspaclu.src.util.string_constants as stco
import pofah.jet_sample as jesa



def get_roc_data(qcd, bsm, fix_tpr=False):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = skl.roc_curve(true_val, pred_val, drop_intermediate=False)
    if fix_tpr: return fpr_loss, tpr_loss, threshold_loss, true_val, pred_val
    return fpr_loss, tpr_loss

def get_fpr_mean_and_std_for_tpr_fixpoint(tpr_fixpoint, tprs, fprs, window=0.001):
    idx = np.where((tprs>=tpr_fixpoint-tpr_fixpoint*window) & (tprs<=tpr_fixpoint+tpr_fixpoint*window))[0]
    return np.mean(fprs[0][idx]), np.mean(fprs[1][idx])


def get_mean_and_error(data):
    return [np.mean(data, axis=0), np.std(data, axis=0)]


def compute_fpr_mean_and_std_for_fixpoint_tpr(quantum_loss_qcd, quantum_loss_sig, classic_loss_qcd, classic_loss_sig, ids, n_folds, save_dir=None):

    # base fpr to interpolate for varying size fprs returned by scikit (omitting repeated values)
    base_tpr = np.linspace(0, 1, 10001)

    # todo: for both tpr fixpoints in (0.6,0.8) 
    tpr_fixpoint = 0.8

    # collecting mean and error of fpr for tpr fixpoint
    fpr_fix_mu_qs = []; fpr_fix_mu_cs = []
    fpr_fix_std_qs = []; fpr_fix_std_cs = [] 

    for i, id_name in enumerate(ids): # for each latent space or train size
        
        fpr_q=[]; fpr_c=[]
        tpr_q=[]; tpr_c=[]
        # import ipdb; ipdb.set_trace()
        
        for j in range(n_folds):
        
            # quantum data
            fq, tq = get_roc_data(quantum_loss_qcd[i][j], quantum_loss_sig[i][j])
        
            # classic data
            fc, tc = get_roc_data(classic_loss_qcd[i][j], classic_loss_sig[i][j])
            
            # interpolate
            fc = np.interp(base_tpr, tc, fc)
            fq = np.interp(base_tpr, tq, fq)
                                    
            fpr_q.append(fq); fpr_c.append(fc)
            tpr_q.append(base_tpr); tpr_c.append(base_tpr)
        
        fpr_data_q = np.nan_to_num(get_mean_and_error(1.0/np.array(fpr_q)))
        fpr_data_c = np.nan_to_num(get_mean_and_error(1.0/np.array(fpr_c)))
        
        tpr_mean_q = np.mean(np.array(tpr_q), axis=0)
        tpr_mean_c = np.mean(np.array(tpr_c), axis=0)

        fpr_fix_mu_q, fpr_fix_std_q = get_fpr_mean_and_std_for_tpr_fixpoint(tpr_fixpoint, tprs=tpr_mean_q, fprs=fpr_data_q)
        fpr_fix_mu_c, fpr_fix_std_c = get_fpr_mean_and_std_for_tpr_fixpoint(tpr_fixpoint, tprs=tpr_mean_c, fprs=fpr_data_c)

        fpr_fix_mu_qs.append(fpr_fix_mu_q); fpr_fix_mu_cs.append(fpr_fix_mu_c)
        fpr_fix_std_qs.append(fpr_fix_std_q); fpr_fix_std_cs.append(fpr_fix_std_c)

    # print results:
    for i, id_name in enumerate(ids):
        print(f'{id_name}: quantum {fpr_fix_mu_qs[i]} +/- {fpr_fix_std_qs[i]} | classic {fpr_fix_mu_cs[i]} +/- {fpr_fix_std_cs[i]}')



def plot_ROC_kfold_mean(quantum_loss_qcd, quantum_loss_sig, classic_loss_qcd, classic_loss_sig, ids, n_folds, 
                       pic_id=None, xlabel='TPR', ylabel=r'FPR$^{-1}$', legend_loc='center right', legend_title='$ROC$', save_dir=None,
                       palette=['#3E96A1', '#EC4E20', '#FF9505', '#6C56B3']):

    lines_n = len(quantum_loss_qcd)
    palette = palette[:lines_n]
    
    styles = ['solid', 'dashed']
    plt.style.use(hep.style.CMS)   
    fig = plt.figure(figsize=(8, 8))
    anomaly_auc_legend = []; study_legend=[]

    # base fpr to interpolate for varying size fprs returned by scikit (omitting repeated values)
    base_tpr = np.linspace(0, 1, 10001)

    # collecting mean and error of fpr for tpr fixpoint
    fpr_fix_mu_q = []; fpr_fix_mu_c = []
    fpr_fix_std_q = []

    for i, id_name in enumerate(ids): # for each latent space or train size
        
        fpr_q=[]; fpr_c=[]
        auc_q=[]; auc_c=[]
        tpr_q=[]; tpr_c=[]
        # import ipdb; ipdb.set_trace()
        
        for j in range(n_folds):
        
            # quantum data
            fq, tq = get_roc_data(quantum_loss_qcd[i][j], quantum_loss_sig[i][j])
        
            # classic data
            fc, tc = get_roc_data(classic_loss_qcd[i][j], classic_loss_sig[i][j])
            
            auc_q.append(skl.auc(fq, tq)); auc_c.append(skl.auc(fc, tc))

            # interpolate
            fc = np.interp(base_tpr, tc, fc)
            fq = np.interp(base_tpr, tq, fq)
                                    
            fpr_q.append(fq); fpr_c.append(fc)
            tpr_q.append(base_tpr); tpr_c.append(base_tpr)
        
        auc_data_q = get_mean_and_error(np.array(auc_q))
        auc_data_c = get_mean_and_error(np.array(auc_c))
        
        fpr_data_q = np.nan_to_num(get_mean_and_error(1.0/np.array(fpr_q)))
        fpr_data_c = np.nan_to_num(get_mean_and_error(1.0/np.array(fpr_c)))
        
        tpr_mean_q = np.mean(np.array(tpr_q), axis=0)
        tpr_mean_c = np.mean(np.array(tpr_c), axis=0)

        fpr_fixpoint_mean_q, fpr_fixpoint_std_q = get_fpr_mean_and_std_for_tpr_fixpoint(tpr_fixpoint, tprs=tpr_mean_q, fprs=fpr_data_q)
        fpr_fixpoint_mean_c, fpr_fixpoint_std_c = get_fpr_mean_and_std_for_tpr_fixpoint(tpr_fixpoint, tprs=tpr_mean_c, fprs=fpr_data_c)
        
        # import ipdb; ipdb.set_trace()
        if 'GtoWW35na' in pic_id or 'Narrow' in str(id_name): # uncertainties are bigger for G_NA
            band_ind = np.where(tpr_mean_q > 0.5)[0] 
        else:
            band_ind = np.where(tpr_mean_q > 0.2)[0]
        
        plt.plot(tpr_mean_q, fpr_data_q[0], linewidth=1.5, color=palette[i])
        plt.plot(tpr_mean_c, fpr_data_c[0], linewidth=1.5, color=palette[i], linestyle='dashed')
        plt.fill_between(tpr_mean_q[band_ind], fpr_data_q[0][band_ind]-fpr_data_q[1][band_ind], fpr_data_q[0][band_ind]+fpr_data_q[1][band_ind], alpha=0.2, color=palette[i])
        plt.fill_between(tpr_mean_c[band_ind], fpr_data_c[0][band_ind]-fpr_data_c[1][band_ind], fpr_data_c[0][band_ind]+fpr_data_c[1][band_ind], alpha=0.2, color=palette[i])
        anomaly_auc_legend.append(f" {auc_data_q[0]*100:.2f}"f"± {auc_data_q[1]*100:.2f} "
                                  f"| {auc_data_c[0]*100:.2f}"f"± {auc_data_c[1]*100:.2f}")
        study_legend.append(id_name)                    
    dummy_res_lines = [Line2D([0,1],[0,1],linestyle=s, color='black') for s in styles[:2]]
    lines = plt.gca().get_lines()
    plt.semilogy(np.linspace(0, 1, num=int(1e4)), 1./np.linspace(0, 1, num=int(1e4)), linewidth=1.5, linestyle='--', color='0.75')
    legend1 = plt.legend(dummy_res_lines, [r'Quantum', r'Classical'], frameon=False, loc='upper right', \
            handlelength=1.5, fontsize=16, title_fontsize=14)#, bbox_to_anchor=(0.01,0.65)) # bbox_to_anchor=(0.97,0.78) -> except for latent study
    legend2 = plt.legend([lines[i*2] for i in range(len(palette))], anomaly_auc_legend, loc='lower left', \
            frameon=True, title=r'AUC $\;\quad$Quantum $\quad\;\;\;$ Classical', fontsize=15, title_fontsize=14,markerscale=0.5)
    bb =  {} if 'Anomaly' in legend_title else {'bbox_to_anchor': (0.95,0.75)}
    legend3 = plt.legend([lines[i*2] for i in range(len(palette))], study_legend, markerscale=0.5, loc=legend_loc, 
                         frameon=True, title=legend_title, fontsize=14, title_fontsize=15, **bb)
    legend3.get_frame().set_alpha(0.35)
    
    legend1._legend_box.align = "left"
    legend2._legend_box.align = "left"
    legend3._legend_box.align = "center"
    
    for leg in legend1.legendHandles:
        leg.set_linewidth(2.2)
        leg.set_color('gray')
    for leg in legend2.legendHandles:
        leg.set_linewidth(2.2) 
        
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.gca().add_artist(legend3)
    plt.ylabel(ylabel, fontsize=24)
    plt.xlabel(xlabel, fontsize=24)
    plt.yscale('log')
    plt.xlim(0.0, 1.05)
    #plt.title(title)
    fig.tight_layout()
    if save_dir:
        fig_full_path = f'{save_dir}/ROC_{pic_id}.pdf'
        print('saving ROC to ' + fig_full_path)
        plt.savefig(fig_full_path, dpi = fig.dpi, bbox_inches='tight')
    else: plt.show()



#****************************************#
#           Runtime Params
#****************************************#


Parameters = namedtuple('Parameters', 'sample_id_qcd sample_id_sigs kfold_n read_n do_plots do_fpr_table')
params = Parameters(sample_id_qcd='qcdSigExt',
                    sample_id_sigs=['GtoWW35na', 'AtoHZ35', 'GtoWW15br'], 
                    kfold_n=10,
                    read_n=int(5e4),
                    do_plots=False,
                    do_fpr_table=True
                    )


# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*70+'\n'+'\t\t\t PLOTING RUN \n'+str(params)+'\n'+'*'*70)


#**********************************************************#
#    PLOT ALL SIGNALS, FIXED DIM=8, FIXED-N=600 (run 45)
#**********************************************************#

#****************************************#
#               READ DATA

run_n = 45
dim_z = 8
train_n = 600
study_title = 'Anomaly signature'
study_labels = ['Narrow 'r'G $\to$ WW 3.5 TeV', r'A $\to$ HZ $\to$ ZZZ 3.5 TeV', 'Broad 'r'G $\to$ WW 1.5 TeV']

# path setup
fig_dir = os.path.join(stco.reporting_fig_base_dir,'paper_submission_plots')
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)


input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

sample_qcd = jesa.JetSample.from_input_file(name=params.sample_id_qcd, path=input_data_dir+'/'+params.sample_id_qcd+'.h5').filter(slice(params.read_n))

dist_qcd = sample_qcd['classic_loss'].reshape(params.kfold_n,-1)
dist_q_qcd = sample_qcd['quantum_loss'].reshape(params.kfold_n,-1)

ll_dist_c_qcd = []; ll_dist_q_qcd = []
ll_dist_c_sigs = []; ll_dist_q_sigs = []
for sample_id_sig in params.sample_id_sigs:
    ll_dist_c_qcd.append(dist_qcd)
    ll_dist_q_qcd.append(dist_q_qcd)
    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))
    ll_dist_c_sigs.append(sample_sig['classic_loss'].reshape(params.kfold_n,-1))
    ll_dist_q_sigs.append(sample_sig['quantum_loss'].reshape(params.kfold_n,-1))

if params.do_plots:
    palette = ['forestgreen', '#EC4E20', 'darkorchid']
    plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n, save_dir=fig_dir, \
        pic_id='roc_qmeans_allSignals_r'+str(run_n)+'_z'+str(dim_z)+'_trainN_'+str(int(train_n)), palette=palette, legend_title=study_title, legend_loc='upper left')

if params.do_fpr_table:
    
    logger.info('printing fpr table for ' + study_title)
    compute_fpr_mean_and_std_for_fixpoint_tpr(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n)


#**********************************************************#
#    PLOT ALL LATENT DIMS, FIXED SIG=X, FIXED-N=600
#**********************************************************#

run_n_dict = OrderedDict([
    (32,47),
    (16,46),
    (8,45),
    (4,44),
])

train_n = 600

#****************************************#
#               A to HZ


sample_id_sig = 'AtoHZ35'
study_title = 'Latent dim.'
study_labels = ['32', '16', '8', '4'][-len(run_n_dict):]

ll_dist_c_qcd = []; ll_dist_q_qcd = []
ll_dist_c_sigs = []; ll_dist_q_sigs = []

for run_n in run_n_dict.values():

    input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

    sample_qcd = jesa.JetSample.from_input_file(name=params.sample_id_qcd, path=input_data_dir+'/'+params.sample_id_qcd+'.h5').filter(slice(params.read_n))

    dist_qcd = sample_qcd['classic_loss'].reshape(params.kfold_n,-1)
    dist_q_qcd = sample_qcd['quantum_loss'].reshape(params.kfold_n,-1)

    ll_dist_c_qcd.append(dist_qcd)
    ll_dist_q_qcd.append(dist_q_qcd)

    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))

    ll_dist_c_sigs.append(sample_sig['classic_loss'].reshape(params.kfold_n,-1))
    ll_dist_q_sigs.append(sample_sig['quantum_loss'].reshape(params.kfold_n,-1))


if params.do_plots:
    palette = ['black', '#3E96A1', '#EC4E20', '#FF9505']
    plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n, save_dir=fig_dir, pic_id='roc_qmeans_allZ_sig'+sample_id_sig+'_trainN_'+str(int(train_n)), legend_title=study_title, palette=palette)


if params.do_fpr_table:
    logger.info(f'printing fpr table for {study_title} of {sample_id_sig}')
    compute_fpr_mean_and_std_for_fixpoint_tpr(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n)

#****************************************#
#               G_RS 3.5TeV

sample_id_sig = 'GtoWW35na'

ll_dist_c_sigs = []; ll_dist_q_sigs = []

for run_n in run_n_dict.values():

    input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))

    ll_dist_c_sigs.append(sample_sig['classic_loss'].reshape(params.kfold_n,-1))
    ll_dist_q_sigs.append(sample_sig['quantum_loss'].reshape(params.kfold_n,-1))

if params.do_plots:

    plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n, save_dir=fig_dir, pic_id='roc_qmeans_allZ_sig'+sample_id_sig+'_trainN_'+str(int(train_n)), legend_title=study_title)

if params.do_fpr_table:
    logger.info(f'printing fpr table for {study_title} of {sample_id_sig}')
    compute_fpr_mean_and_std_for_fixpoint_tpr(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n)

#**********************************************************#
#    PLOT ALL Training sizes, FIXED SIG=X, FIXED-Z=8
#**********************************************************#

run_n_dict = OrderedDict([
    (6000,49),
    (600,45),
    (10,41),
])

dim_z = 8

#****************************************#
#               A to HZ


sample_id_sig = 'AtoHZ35'
study_title = 'Train size'
study_labels=list(run_n_dict.keys())

ll_dist_c_qcd = []; ll_dist_q_qcd = []
ll_dist_c_sigs = []; ll_dist_q_sigs = []

for run_n in run_n_dict.values():

    input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

    sample_qcd = jesa.JetSample.from_input_file(name=params.sample_id_qcd, path=input_data_dir+'/'+params.sample_id_qcd+'.h5').filter(slice(params.read_n))

    dist_qcd = sample_qcd['classic_loss'].reshape(params.kfold_n,-1)
    dist_q_qcd = sample_qcd['quantum_loss'].reshape(params.kfold_n,-1)

    ll_dist_c_qcd.append(dist_qcd)
    ll_dist_q_qcd.append(dist_q_qcd)

    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))

    ll_dist_c_sigs.append(sample_sig['classic_loss'].reshape(params.kfold_n,-1))
    ll_dist_q_sigs.append(sample_sig['quantum_loss'].reshape(params.kfold_n,-1))

if params.do_plots:

    plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n, save_dir=fig_dir, pic_id='roc_qmeans_allTrainN_sig'+sample_id_sig+'_z'+str(dim_z), legend_title=legend_title)

if params.do_fpr_table:
    logger.info(f'printing fpr table for {study_title} of {sample_id_sig}')
    compute_fpr_mean_and_std_for_fixpoint_tpr(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n)



#****************************************#
#               G_RS 3.5TeV

sample_id_sig = 'GtoWW35na'

ll_dist_c_sigs = []; ll_dist_q_sigs = []

for run_n in run_n_dict.values():

    input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))

    ll_dist_c_sigs.append(sample_sig['classic_loss'].reshape(params.kfold_n,-1))
    ll_dist_q_sigs.append(sample_sig['quantum_loss'].reshape(params.kfold_n,-1))

if params.do_plots:

    plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n, save_dir=fig_dir, pic_id='roc_qmeans_allLatDims_sig'+sample_id_sig+'_z'+str(dim_z), legend_title=legend_title)

if params.do_fpr_table:
    logger.info(f'printing fpr table for {study_title} of {sample_id_sig}')
    compute_fpr_mean_and_std_for_fixpoint_tpr(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, study_labels, params.kfold_n)

