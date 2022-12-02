import os
from collections import namedtuple
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import roc_curve, auc
import sklearn.metrics as skl
import mplhep as hep
from matplotlib.lines import Line2D

import laspaclu.util.logging as log
import laspaclu.util.string_constants as stco
import pofah.jet_sample as jesa


def get_fpr_and_tpr(neg_class_losses, pos_class_losses):


    true_vals = np.concatenate([np.zeros(len(neg_class_losses)), np.ones(len(pos_class_losses))])
    losses = np.concatenate([neg_class_losses, pos_class_losses])
    fpr, tpr, threshold = skl.roc_curve(true_vals, losses) # , drop_intermediate=False

    return np.nan_to_num(fpr), np.nan_to_num(tpr)



def get_roc_data(qcd, bsm, fix_tpr=False):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = skl.roc_curve(true_val, pred_val, drop_intermediate=False)
    if fix_tpr: return fpr_loss, tpr_loss, threshold_loss, true_val, pred_val
    return fpr_loss, tpr_loss

def get_FPR_for_fixed_TPR(tpr_window, fpr_loss, tpr_loss, true_data, pred_data, tolerance):
    position = np.where((tpr_loss>=tpr_window-tpr_window*tolerance) & (tpr_loss<=tpr_window+tpr_window*tolerance))[0]
    return np.mean(fpr_loss[position])

def get_mean_and_error(data):
    return [np.mean(data, axis=0), np.std(data, axis=0)]

def plot_ROC_kfold_mean(quantum_loss_qcd, quantum_loss_sig, classic_loss_qcd, classic_loss_sig, ids, n_folds, 
                       pic_id=None, xlabel='TPR', ylabel=r'1/FPR', legend_loc='best', legend_title='$ROC$', save_dir=None,
                       palette=['#3E96A1', '#EC4E20', '#FF9505', '#6C56B3']):

    lines_n = len(quantum_loss_qcd)
    palette = palette[:lines_n]
    
    styles = ['solid', 'dashed']
    plt.style.use(hep.style.CMS)   
    fig = plt.figure(figsize=(8, 8))
    anomaly_auc_legend = []; study_legend=[]

    # base fpr to interpolate for varying size fprs returned by scikit (omitting repeated values)
    base_tpr = np.linspace(0, 1, 10001)

    for i, id_name in enumerate(ids): # for each latent space or train size
        
        fpr_q=[]; fpr_c=[]
        auc_q=[]; auc_c=[]
        tpr_q=[]; tpr_c=[]
        # import ipdb; ipdb.set_trace()
        
        for j in range(n_folds):
        
            # quantum data
            fq, tq = get_roc_data(quantum_loss_qcd[i][j], quantum_loss_sig[i][j])
            # fq = np.interp(base_tpr, tq, fq)
        
            # classic data
            fc, tc = get_roc_data(classic_loss_qcd[i][j], classic_loss_sig[i][j])
            # fc = np.interp(base_tpr, tc, fc)

            auc_q.append(skl.auc(fq, tq)); auc_c.append(skl.auc(fc, tc))
            fpr_q.append(fq); fpr_c.append(fc)
            tpr_q.append(tq); tpr_c.append(tc)

            # auc_q.append(skl.auc(fq, base_tpr)); auc_c.append(skl.auc(fc, base_tpr))
            # fpr_q.append(fq); fpr_c.append(fc)
            # tpr_q.append(base_tpr); tpr_c.append(base_tpr)
        
        auc_data_q = get_mean_and_error(np.array(auc_q))
        auc_data_c = get_mean_and_error(np.array(auc_c))
        
        fpr_data_q = get_mean_and_error(1.0/np.array(fpr_q))
        fpr_data_c = get_mean_and_error(1.0/np.array(fpr_c))
        
        tpr_mean_q = np.mean(np.array(tpr_q), axis=0)
        tpr_mean_c = np.mean(np.array(tpr_c), axis=0)
        
        if ids[i]=='Narrow 'r'G $\to$ WW 3.5 TeV': # uncertainties are bigger for G_NA
            band_ind = np.where(tpr_mean_q > 0.6)[0] 
        else:
            band_ind = np.where(tpr_mean_q > 0.35)[0]
        
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
    legend3 = plt.legend([lines[i*2] for i in range(len(palette))], study_legend, markerscale=0.5, loc='center right', 
                         frameon=True, title=legend_title, fontsize=14, title_fontsize=15, bbox_to_anchor=(0.95,0.75))
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


Parameters = namedtuple('Parameters', 'sample_id_qcd sample_id_sigs kfold_n read_n')
params = Parameters(sample_id_qcd='qcdSigExt',
                    sample_id_sigs=['GtoWW35na', 'GtoWW15br', 'AtoHZ35'], 
                    kfold_n=5,
                    read_n=int(2e3))


# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*70+'\n'+'\t\t\t PLOTING RUN \n'+str(params)+'\n'+'*'*70)


#**********************************************************#
#    PLOT ALL SIGNALS, FIXED DIM=8, FIXED-N=600 (run 45)
#**********************************************************#

#****************************************#
#               READ DATA

run_n = 40
dim_z = 4
train_n = 10

# path setup
fig_dir = 'fig/paper_submission_plots'
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

# import ipdb; ipdb.set_trace()
palette = ['forestgreen', '#EC4E20', 'darkorchid']
legend_signal_names=['Narrow 'r'G $\to$ WW 3.5 TeV', 'Broad 'r'G $\to$ WW 1.5 TeV', r'A $\to$ HZ $\to$ ZZZ 3.5 TeV']
plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, legend_signal_names, params.kfold_n, save_dir=fig_dir, pic_id='qmeans_allSignals_run_'+str(run_n)+'_z'+str(dim_z)+'_trainN_'+str(int(train_n)))


#**********************************************************#
#    PLOT ALL LATENT DIMS, FIXED SIG=X, FIXED-N=600
#**********************************************************#

run_n_dict = {
    4: 40,
    8: 41,
    16: 42,
    32: 43
}

#****************************************#
#               A to HZ


sample_id_sig = 'AtoHZ35'

ll_dist_c_qcd = []; ll_dist_q_qcd = []
ll_dist_c_sigs = []; ll_dist_q_sigs = []

for z_dim, run_n in run_n_dict.items():

    input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

    sample_qcd = jesa.JetSample.from_input_file(name=params.sample_id_qcd, path=input_data_dir+'/'+params.sample_id_qcd+'.h5').filter(slice(params.read_n))

    dist_qcd = sample_qcd['classic_loss'].reshape(params.kfold_n,-1)
    dist_q_qcd = sample_qcd['quantum_loss'].reshape(params.kfold_n,-1)

    ll_dist_c_qcd.append(dist_qcd)
    ll_dist_q_qcd.append(dist_q_qcd)

    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))

    ll_dist_c_sigs.append(sample_sig['classic_loss'].reshape(params.kfold_n,-1))
    ll_dist_q_sigs.append(sample_sig['quantum_loss'].reshape(params.kfold_n,-1))


legend_signal_names=['lat4', 'lat8', 'lat16', 'lat32']
plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, legend_signal_names, params.kfold_n, save_dir=fig_dir, pic_id='qmeans_allLatDims_sig'+sample_id_sig+'_trainN_'+str(int(train_n)))


#****************************************#
#               G_RS 3.5TeV

sample_id_sig = 'GtoWW35na'

ll_dist_c_sigs = []; ll_dist_q_sigs = []

for z_dim, run_n in run_n_dict.items():

    input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))

    ll_dist_c_sigs.append(sample_sig['classic_loss'].reshape(params.kfold_n,-1))
    ll_dist_q_sigs.append(sample_sig['quantum_loss'].reshape(params.kfold_n,-1))


plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, legend_signal_names, params.kfold_n, save_dir=fig_dir, pic_id='qmeans_allLatDims_sig'+sample_id_sig+'_trainN_'+str(int(train_n)))


#**********************************************************#
#    PLOT ALL Training sizes, FIXED SIG=X, FIXED-Z=8
#**********************************************************#

run_n_dict = {
    10: 40,
    600: 44,
    6000: 48,
}

dim_z = 4

#****************************************#
#               A to HZ


sample_id_sig = 'AtoHZ35'

ll_dist_c_qcd = []; ll_dist_q_qcd = []
ll_dist_c_sigs = []; ll_dist_q_sigs = []

for z_dim, run_n in run_n_dict.items():

    input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

    sample_qcd = jesa.JetSample.from_input_file(name=params.sample_id_qcd, path=input_data_dir+'/'+params.sample_id_qcd+'.h5').filter(slice(params.read_n))

    dist_qcd = sample_qcd['classic_loss'].reshape(params.kfold_n,-1)
    dist_q_qcd = sample_qcd['quantum_loss'].reshape(params.kfold_n,-1)

    ll_dist_c_qcd.append(dist_qcd)
    ll_dist_q_qcd.append(dist_q_qcd)

    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))

    ll_dist_c_sigs.append(sample_sig['classic_loss'].reshape(params.kfold_n,-1))
    ll_dist_q_sigs.append(sample_sig['quantum_loss'].reshape(params.kfold_n,-1))


legend_signal_names=list(run_n_dict.keys())
plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, legend_signal_names, params.kfold_n, save_dir=fig_dir, pic_id='qmeans_allTrainN_sig'+sample_id_sig+'_z'+str(dim_z))


#****************************************#
#               G_RS 3.5TeV

sample_id_sig = 'GtoWW35na'

ll_dist_c_sigs = []; ll_dist_q_sigs = []

for z_dim, run_n in run_n_dict.items():

    input_data_dir = stco.cluster_out_data_dir+'/run_'+str(run_n)

    sample_sig = jesa.JetSample.from_input_file(name=sample_id_sig, path=input_data_dir+'/'+sample_id_sig+'.h5').filter(slice(params.read_n))

    ll_dist_c_sigs.append(sample_sig['classic_loss'].reshape(params.kfold_n,-1))
    ll_dist_q_sigs.append(sample_sig['quantum_loss'].reshape(params.kfold_n,-1))


plot_ROC_kfold_mean(ll_dist_q_qcd, ll_dist_q_sigs, ll_dist_c_qcd, ll_dist_c_sigs, legend_signal_names, params.kfold_n, save_dir=fig_dir, pic_id='qmeans_allLatDims_sig'+sample_id_sig+'_z'+str(dim_z))