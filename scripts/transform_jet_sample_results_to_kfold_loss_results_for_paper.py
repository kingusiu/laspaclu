import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
import sklearn.metrics as skl
import os
import mplhep as hep
import pathlib
import h5py

import anpofah.model_analysis.roc_analysis as ra
import util.logging as log
import pofah.jet_sample as jesa


if __name__ == "__main__":

    kfold_n = 10
    run_n, train_n = 33, int(6e2)
    input_dir = '/eos/user/k/kiwoznia/data/laspaclu_results/run_'+str(run_n)
    output_dir = os.path.join(input_dir,'kfold_format')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    bg_id = 'qcdSigExt'
    sig_ids = ['GtoWW35na', 'AtoHZ35', 'GtoWW15br']
    read_n = int(2e4)

    # logging
    logger = log.get_logger(__name__)
    logger.info('\n'+'*'*70+'\n'+'\t\t\t postprocessing results to kfold format for run '+str(run_n)+'\n'+'*'*70)

    #****************************************#
    #               READ DATA
    #****************************************#

    sample_qcd = jesa.JetSample.from_input_file(name=bg_id, path=input_dir+'/'+bg_id+'.h5').filter(slice(read_n))
    samples_sig = [jesa.JetSample.from_input_file(name=sig_id, path=input_dir+'/'+sig_id+'.h5').filter(slice(read_n)) for sig_id in sig_ids]

    #****************************************#
    #               Transform Format
    #****************************************#

    loss_qcd_q, loss_qcd_c = sample_qcd['quantum_loss'].reshape(kfold_n,-1), sample_qcd['classic_loss'].reshape(kfold_n,-1) # each loss kfold_n x N/kfold_n
    losses_sig = [[sample['quantum_loss'].reshape(kfold_n,-1), sample['classic_loss'].reshape(kfold_n,-1)] for sample in samples_sig]

    #****************************************#
    #               WRITE DATA
    #****************************************#

    with h5py.File('loss_results', 'w') as ff:
        # write qcd
        ff.create_dataset('quantum_loss_qcd', data=loss_qcd_q)
        ff.create_dataset('classic_loss_qcd', data=loss_qcd_c)
        # write signals
        for sig_id, loss_sig in zip(sig_ids,losses_sig):
            ff.create_dataset('quantum_loss_'+sig_id, data=loss_sig[0])
            ff.create_dataset('classic_loss_'+sig_id, data=loss_sig[1])
