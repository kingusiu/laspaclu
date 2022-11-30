import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
import sklearn.metrics as skl
import os
import mplhep as hep
import pathlib
from collections import namedtuple

import laspaclu.src.analysis.roc as roc
import laspaclu.src.util.logging as log
import pofah.jet_sample as jesa
import anpofah.model_analysis.roc_analysis as roan




if __name__ == "__main__":


    #****************************************#
    #           Runtime Params
    #****************************************#


    Parameters = namedtuple('Parameters', 'sample_id_qcd sample_id_sigs read_n')
    params = Parameters(sample_id_qcd='qcdSigExt',
                        sample_id_sigs=['GtoWW35na', 'GtoWW15br', 'AtoHZ35'], 
                        read_n=int(2e3))


    # logging
    logger = log.get_logger(__name__)
    logger.info('\n'+'*'*70+'\n'+'\t\t\t PLOTING RUN \n'+str(params)+'\n'+'*'*70)


    #**********************************************************#
    #    PLOT ALL SIGNALS, FIXED DIM=8, FIXED-N=600 (run 45)
    #**********************************************************#

    #****************************************#
    #               READ DATA

    run_n = 45
    dim_z = 4
    train_n = 600

    # path setup
    fig_dir = os.path.join(stco.reporting_fig_base_dir,'qkmeans_run_'+str(params.run_n))
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

    # path setup
    fig_dir = os.path.join(stco.reporting_fig_base_dir,'qkmeans_multimodel/runs_'+'_'.join([str(r) for r in run_n_dict.values()]))
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

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






    var_sig = False
    var_train_sz = False
    var_latent_dim = True
    
    test_n = int(1e4)


    ### signal variation ROC
    if var_sig:
        # run_n, train_n = 22, 600
        # run_n, train_n = 24, 10
        run_n, train_n = 25, int(6e3)
        input_dir = '/eos/user/k/kiwoznia/data/laspaclu_results/run_'+str(run_n)
        fig_dir = 'fig/run_'+str(run_n)
        bg_id = 'qcdSigExt'
        sig_ids = ['GtoWW35na', 'AtoHZ35', 'GtoWW15br']
        read_n = int(1e4)

        # logging
        logger = log.get_logger(__name__)
        logger.info('\n'+'*'*70+'\n'+'\t\t\t plotting roc_analysis for run '+str(run_n)+'\n'+'*'*70)

        #****************************************#
        #               READ DATA
        #****************************************#

        sample_qcd = jesa.JetSample.from_input_file(name=bg_id, path=input_dir+'/'+bg_id+'.h5').filter(slice(read_n))
        samples_sig = [jesa.JetSample.from_input_file(name=sig_id, path=input_dir+'/'+sig_id+'.h5').filter(slice(read_n)) for sig_id in sig_ids]

        loss_qcd = [sample_qcd['quantum_loss'], sample_qcd['classic_loss']]
        losses_sig = [[sample['quantum_loss'], sample['classic_loss']] for sample in samples_sig]

        class_labels, losses = roc.prepare_labels_and_losses_signal_comparison(loss_qcd, losses_sig)
        legend_colors = [roc.sig_name_dict[sig_id] for sig_id in sig_ids]
        title = ' '.join(filter(None, [r"$N^{train}=$", '{:.0E}'.format(train_n).replace("E+0", "E"), r"$N^{test}=$", '{:.0E}'.format(test_n).replace("E+0", "E")]))

        roc.plot_roc(class_labels, losses, legend_colors=legend_colors, legend_colors_title='signals', title=title, plot_name='ROC_run_'+str(run_n)+'_allSig', fig_dir=fig_dir)


    ### training size variation ROC
    if var_train_sz:
        
        run_ns = [24, 22, 25]
        train_szs = [10, 600, 6000]

        bg_id = 'qcdSigExt'
        sig_id = 'GtoWW35na'
        read_n = int(1e4)
        fig_dir = 'fig/run_multi_'+'_'.join(str(n) for n in run_ns)
        pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

        losses_qcd = []
        losses_sig = []

        for run_n in run_ns:
            input_dir = '/eos/user/k/kiwoznia/data/laspaclu_results/run_'+str(run_n)

            sample_qcd = jesa.JetSample.from_input_file(name=bg_id, path=input_dir+'/'+bg_id+'.h5').filter(slice(read_n))
            sample_sig = jesa.JetSample.from_input_file(name=sig_id, path=input_dir+'/'+sig_id+'.h5').filter(slice(read_n))

            losses_qcd.append([sample_qcd['quantum_loss'], sample_qcd['classic_loss']])
            losses_sig.append([sample_sig['quantum_loss'], sample_sig['classic_loss']])

        class_labels, losses = roc.prepare_labels_and_losses_train_sz_comparison(losses_qcd, losses_sig)
        title = r'$N^{train}=$var, ' + r'$N^{test}=$' + '{:.0E}'.format(test_n).replace("E+0", "E")

        roc.plot_roc(class_labels, losses, legend_colors=[str(s) for s in train_szs], legend_colors_title='N train', \
            title=title, plot_name='ROC_run_multi_train_sz_compare_allSig', fig_dir=fig_dir)


   ### training size variation ROC
    if var_latent_dim:

        latent_dim_dict = {
            4 : r'$z \in \mathbb{R}^4$',
            8 : r'$z \in \mathbb{R}^8$',
            16 : r'$z \in \mathbb{R}^{16}$',
            24 : r'$z \in \mathbb{R}^{24}$',
            32 : r'$z \in \mathbb{R}^{32}$',
        }
    
        latent_dims = [4, 8, 16]
        run_ns = [29, 22, 32]

        default_sig_id = 'GtoWW35na'
        default_train_n = 600
        read_n = int(1e4)
        
        bg_id = 'qcdSigExt'

        fig_dir = 'fig/run_multi_'+'_'.join(str(n) for n in run_ns)
        pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

        losses_qcd = []
        losses_sig = []

        for run_n in run_ns:
            input_dir = '/eos/user/k/kiwoznia/data/laspaclu_results/run_'+str(run_n)

            sample_qcd = jesa.JetSample.from_input_file(name=bg_id, path=input_dir+'/'+bg_id+'.h5').filter(slice(read_n))
            sample_sig = jesa.JetSample.from_input_file(name=default_sig_id, path=input_dir+'/'+default_sig_id+'.h5').filter(slice(read_n))

            losses_qcd.append([sample_qcd['quantum_loss'], sample_qcd['classic_loss']])
            losses_sig.append([sample_sig['quantum_loss'], sample_sig['classic_loss']])

        class_labels, losses = roc.prepare_labels_and_losses_train_sz_comparison(losses_qcd, losses_sig)
        title = ' '.join(filter(None, [r"$N^{train}=$", '{:.0E}'.format(default_train_n).replace("E+0", "E"), r"$N^{test}=$", '{:.0E}'.format(test_n).replace("E+0", "E")]))

        aucs = roc.plot_roc(class_labels, losses, legend_colors=[str(latent_dim_dict[d]) for d in latent_dims], legend_colors_title='N train', \
        title=title, plot_name='ROC_multi_latent_dim_compare_allSig_'+'_'.join(str(d) for d in latent_dims), fig_dir=fig_dir, test_n=int(read_n/10))

        print(aucs)

