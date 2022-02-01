import numpy as np
from collections import namedtuple

import analysis.plotting as plot
import util.persistence as pers



#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'run_n_classic date_classic run_n_quantum date_quantum run_n_hybrid date_hybrid')
params = Parameters(
                    # classic model
                    run_n_classic=14, 
                    date_classic='20220201',
                    # quantum model
                    run_n_quantum=14,
                    date_quantum='20220201',
                    # hybrid model 
                    run_n_hybrid=14,
                    date_hybrid='20220201',
                    )
fig_dir = 'fig/run_'+str(params.run_n)
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
print('*'*50+'\n'+'model comparison '+str(params.run_n)+' on '+str(params.read_n)+' samples'+'\n'+'*'*50)


#****************************************#
#           Read in Models
#****************************************#

## classic model
model_path_km = pers.make_model_path(date=params.date_model, prefix='KM', run_n=params.run_n)
print('[main_predict_clustering] >>> loading classic model ' + model_path_km)

cluster_model = jli.load(model_path_km+'.joblib')    
cluster_centers = cluster_model.cluster_centers_
print('classic cluster centers: ')
print(cluster_centers)


## quantum model (distance calc and minimization = quantum) 

## hybrid model (distance calc = quantum, minimization = classic)

