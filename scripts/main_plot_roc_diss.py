import matplotlib as mpl
mpl.rc('font',**{'family':'serif','serif':['Times']})
mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preview'] = True
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
import sklearn.metrics as skl
import os
import mplhep as hep
import pathlib
from collections import namedtuple
from collections import OrderedDict

import laspaclu.src.analysis.roc as roc
import laspaclu.src.util.logging as log
import laspaclu.src.util.string_constants as stco
import pofah.jet_sample as jesa
import anpofah.model_analysis.roc_analysis as ra


def prepare_truths_and_scores(scores_bg, scores_sig):
    
    y_truths = np.concatenate([np.zeros(len(scores_bg)), np.ones(len(scores_sig))])
    y_scores = np.concatenate([scores_bg, scores_sig])
    
    return y_truths, y_scores


def plot_roc(ll_sc_bg_c, ll_sc_sig_c, ll_sc_bg_q, ll_sc_sig_q, main_legend_labels, main_legend_title, base_n=int(1e3), auc_legend_offset=0.07, plot_name='roc', fig_dir='fig', fig_format='.pdf'):
    
    plt.style.use(hep.style.CMS)
    line_styles = ['solid', 'dashed']
    fig = plt.figure(figsize=(8, 8))
    
    aucs = []
    for sc_bg_c, sc_bg_q, sc_sig_c, sc_sig_q, cc in zip(ll_sc_bg_c, ll_sc_bg_q, ll_sc_sig_c, ll_sc_sig_q, stco.multi_sig_palette):
        
        y_truths_q, y_scores_q = prepare_truths_and_scores(sc_bg_q, sc_sig_q)
        y_truths_c, y_scores_c = prepare_truths_and_scores(sc_bg_c, sc_sig_c)
        
        fpr_q, tpr_q, _ = skl.roc_curve(y_truths_q, y_scores_q)
        fpr_c, tpr_c, _ = skl.roc_curve(y_truths_c, y_scores_c)
            
        aucs.append(skl.roc_auc_score(y_truths_q, y_scores_q))
        aucs.append(skl.roc_auc_score(y_truths_c, y_scores_c))
        
        plt.loglog(tpr_q, 1./fpr_q, linestyle='solid', color=cc)
        plt.loglog(tpr_c, 1./fpr_c, linestyle='dashed', color=cc)
        
    # plot random decision line
    plt.loglog(np.linspace(0, 1, num=base_n), 1./np.linspace(0, 1, num=base_n), linewidth=1.2, linestyle='solid', color='silver')

    dummy_res_lines = [Line2D([0,1],[0,1],linestyle=s, color='gray') for s in line_styles[:2]]

    # add 2 legends (classic vs quantum and resonance types)
    lines = plt.gca().get_lines()
    
    legend1 = plt.legend(dummy_res_lines, [r'Quantum', r'Classic'], loc='lower left', frameon=False, title='algorithm', handlelength=1.5, fontsize=14, title_fontsize=17, bbox_to_anchor=(0,0.28))
    
    main_legend_labels = [r"{}".format(lbl) for lbl in main_legend_labels]
    legend2 = plt.legend([lines[i*len(line_styles)] for i in range(len(main_legend_labels))], main_legend_labels, loc='lower left', frameon=False, title=main_legend_title, fontsize=14, title_fontsize=17)
    
    auc_legend_labels = [r"$  {:.3f} \,\,|\,\, {:.3f}$".format(aucs[i*2],aucs[i*2+1]) for i in range(len(main_legend_labels))]
    auc_legend_title = r"auc q $\vert$ c$"
    legend3 = plt.legend([lines[i*len(line_styles)] for i in range(len(main_legend_labels))], auc_legend_labels, loc='lower center', frameon=False, title=auc_legend_title, fontsize=14, title_fontsize=17) 
    
    legend1._legend_box.align = "left"
    legend2._legend_box.align = "left"
    legend3._legend_box.align = "center"
    for leg in legend1.legendHandles:
        leg.set_linewidth(2.5)
        leg.set_color('gray')
    for leg in legend2.legendHandles:
        leg.set_linewidth(2.5)
    for leg in legend3.legendHandles:
        leg.set_visible(False)
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.gca().add_artist(legend3)
    
    plt.draw()
    
    # Get the bounding box of the original legend
    bb = legend3.get_bbox_to_anchor().inverse_transformed(plt.gca().transAxes)
    # Change to location of the legend. 
    bb.x0 += auc_legend_offset
    bb.x1 += auc_legend_offset
    legend3.set_bbox_to_anchor(bb, transform = plt.gca().transAxes)
    
    plt.grid()
    plt.xlabel('True positive rate',fontsize=17)
    plt.ylabel('1 / False positive rate',fontsize=17)
    plt.tight_layout()
        
    fig_path = os.path.join(fig_dir, plot_name + fig_format)
    print('writing figure to ' + fig_path)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


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
plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(1e4), plot_name='roc_allSig_r'+str(run_n), fig_dir=fig_dir)


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

plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(7e2), auc_legend_offset=-0.12, plot_name='roc_allTrainN_z'+str(dim_z)+'_'+str(sample_id_sig), fig_dir=fig_dir)


#****************************************#
#               G_RS 1.5TeV broad

sample_id_sig = 'GtoWW15br'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(1e4), auc_legend_offset=-0.12, plot_name='roc_allTrainN_z'+str(dim_z)+'_'+str(sample_id_sig), fig_dir=fig_dir)


#****************************************#
#               G_RS 3.5TeV na

sample_id_sig = 'GtoWW35na'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(1e2), auc_legend_offset=-0.12, plot_name='roc_allTrainN_z'+str(dim_z)+'_'+str(sample_id_sig), fig_dir=fig_dir)


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

plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(7e2), auc_legend_offset=-0.12, plot_name='roc_allZDim_trainN'+str(train_n)+'_'+str(sample_id_sig), fig_dir=fig_dir)


#****************************************#
#               G_RS 1.5TeV broad

sample_id_sig = 'GtoWW15br'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(5e3), auc_legend_offset=-0.12, plot_name='roc_allZDim_trainN'+str(train_n)+'_'+str(sample_id_sig), fig_dir=fig_dir)


#****************************************#
#               G_RS 3.5TeV na

sample_id_sig = 'GtoWW35na'

ll_scores_sig_c, ll_scores_sig_q = read_scores_from_multiple_runs(run_n_dict.values(), sample_id_sig, params.read_n) 

plot_roc(ll_scores_qcd_c, ll_scores_sig_c, ll_scores_qcd_q, ll_scores_sig_q, main_legend_labels=legend_labels, main_legend_title=legend_title, base_n=int(1e3), auc_legend_offset=-0.1, plot_name='roc_allZDim_trainN'+str(train_n)+'_'+str(sample_id_sig), fig_dir=fig_dir)

