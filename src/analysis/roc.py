import matplotlib as mpl
mpl.rc('font',**{'family':'serif','serif':['Times']})
mpl.rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
import sklearn.metrics as skl
import os
import pathlib

import laspaclu.src.util.string_constants as stco
import laspaclu.src.util.logging as log
import pofah.jet_sample as jesa


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
    auc_legend_title = r"auc q $\vert$ c"
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


def plot_roc_multiline(ll_sc_bg, ll_sc_sig, main_legend_labels, main_legend_title, base_n=int(1e3), auc_legend_offset=0.07, plot_name='roc', fig_dir='fig', fig_format='.pdf'):
    
    # ******
    #   ll_sc_bg : 3 x K x N
    #              3: quantum, hybrid, classic
    #              N: number of samples
    #              K: number of variations of parameter of interest (signal signature, latent dim, train_sz)
    #   ll_sc_sig : 3 x N x K
    # ******

    plt.style.use(hep.style.CMS)
    line_type_n = len(ll_sc_bg) # classic vs quantum (vs hybrid)
    line_styles = ['solid', 'dashed', 'dotted'][:line_type_n]
    fig = plt.figure(figsize=(8, 8))
    
    # transpose to get good plotting order study_objects, quantum/hybrid/classic, samples
    ll_sc_bg = np.asarray(ll_sc_bg).transpose(1,0,2)
    ll_sc_sig = np.asarray(ll_sc_sig).transpose(1,0,2)

    aucs = []
    # for each line set (classic, quantum, hybrid) of a signal signature/latent dim/train-sz variation
    for sc_bg_study_set, sc_sig_study_set, cc in zip(ll_sc_bg, ll_sc_sig, stco.multi_sig_palette):
        for sc_bg, sc_sig, ls in zip(sc_bg_study_set, sc_sig_study_set, line_styles):
        
            y_truths, y_scores = prepare_truths_and_scores(sc_bg, sc_sig)
            
            fpr, tpr, _ = skl.roc_curve(y_truths, y_scores)
                
            aucs.append(skl.roc_auc_score(y_truths, y_scores))
            
            plt.loglog(tpr, 1./fpr, linestyle=ls, color=cc)
        
    # plot random decision line
    plt.loglog(np.linspace(0, 1, num=base_n), 1./np.linspace(0, 1, num=base_n), linewidth=1.2, linestyle='solid', color='silver')

    dummy_res_lines = [Line2D([0,1],[0,1],linestyle=s, color='gray') for s in line_styles]

    # add 2 legends (classic vs quantum and resonance types)
    lines = plt.gca().get_lines()
    
    legend1 = plt.legend(dummy_res_lines, [r'Quantum', r'Hybrid', r'Classic'][:line_type_n], loc='lower left', frameon=False, title='algorithm', handlelength=1.5, fontsize=14, title_fontsize=17, bbox_to_anchor=(0,0.28))
    
    main_legend_labels = [r"{}".format(lbl) for lbl in main_legend_labels]
    legend2 = plt.legend([lines[i*len(line_styles)] for i in range(len(main_legend_labels))], main_legend_labels, loc='lower left', frameon=False, title=main_legend_title, fontsize=14, title_fontsize=17)
    
    auc_legend_labels = [r"$    {:.3f} \,\,|\,\, {:.3f} |\,\, {:.3f}$".format(aucs[i*line_type_n],aucs[i*line_type_n+1],aucs[i*line_type_n+2]) for i in range(len(main_legend_labels))]
    auc_legend_title = r"auc q \, $\vert$ \,\,h\,\, $\vert$ \, c"
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
