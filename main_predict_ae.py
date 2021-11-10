import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import namedtuple

import pofah.jet_sample as jesa
import pofah.path_constants.sample_dict_file_parts_input as sdi


"""
    pass datasample through autoencoder to obtain latent space representation and write to disk
    for further usage in clusering and pca
"""


#****************************************#
#           Runtime Params
#****************************************#

sample_ids = ['qcdSide', 'GtoWW35na']

Parameters = namedtuple('Parameters', ' run_n read_n')
params = Parameters(run_n=50, read_n=int(1e6))

paths = safa.SamplePathDirFactory(sdi.path_dict)
output_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.run)
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


#****************************************#
#           Load Autoencoder
#****************************************#

# load model
model_path_ae = pers.make_model_path(run_n=50, date='20211110', prefix='AE')

print('[main_predict_ae] >>> loading autoencoder ' + model_path_ae)
ae_model = tf.saved_model.load(model_path_ae)


for sample_id in sample_ids:

    #****************************************#
    #           Read Data
    #****************************************#

    sample = evsa.EventSample.from_input_dir(name=sample_id, path=paths.sample_dir_path(sample_id_qcd), read_n=params.read_n)
    p1_qcd, p2_qcd = sample.get_particles() 


    #****************************************#
    #           Load Autoencoder
    #****************************************#

    latent_coords_qcd = pred.map_to_latent_space(data_sample=np.vstack([p1_qcd, p2_qcd]), model=ae_model, read_n=params.read_n)


    #****************************************#
    #           Write results to list
    #****************************************#

    print('[main_predict_ae] >>> writing results for ' + sample_id + 'to ' + output_dir)

    # qcd results
    sample_out = jesa.JetSampleLatent.from_event_sample(sample)
    sample_out.add_latent_prepresentation(latent_coords_qcd)
    sample_out.dump(os.path.join(output_dir, sample_out.name+'.h5'))

