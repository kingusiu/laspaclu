import numpy as np
from collections import namedtuple
import pathlib
import joblib as jli
import pandas as pd
import seaborn as sns
import mplhep as hep
import matplotlib.pyplot as plt

import analysis.plotting as plot
import util.persistence as pers
import util.logging as log


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
                    run_n_hybrid=20,
                    date_hybrid='20220207',
                    )
fig_dir = 'fig/run_'+str(params.run_n_classic)
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

# logging
logger = log.get_logger(__name__)
logger.info('*'*60+'\n'+'\t\t\t Model Comparison \n'+str(params)+'\n'+'*'*60)


#****************************************#
#           Read in Models
#****************************************#

## classic model
model_path_classic = pers.make_model_path(date=params.date_classic, prefix='KM', run_n=params.run_n_classic)
print('[model_compare] >>> loading classic model ' + model_path_classic)

cluster_model = jli.load(model_path_classic+'.joblib')    
cluster_centers_classic = cluster_model.cluster_centers_
print('classic cluster centers: ')
print(cluster_centers_classic)


## quantum model (distance calc and minimization = quantum) 

print('[model_compare] >>> loading qmeans')
model_path_quantum = pers.make_model_path(date=params.date_quantum, prefix='QM', run_n=params.run_n_quantum) + '.npy'
with open(model_path_quantum, 'rb') as f:
    cluster_centers_quantum = np.load(f)
print('quantum cluster centers: ')
print(cluster_centers_quantum)

## hybrid model (distance calc = quantum, minimization = classic)
print('[model_compare] >>> loading hybrid means')
model_path_hybrid = pers.make_model_path(date=params.date_hybrid, prefix='QM', run_n=params.run_n_hybrid) + '.npy'
with open(model_path_hybrid, 'rb') as f:
    cluster_centers_hybrid = np.load(f)
print('quantum cluster centers: ')
print(cluster_centers_hybrid)



### assemble to dataframe (8 latent dimensions)
columns = ["d"+str(i) for i in range(0,8)]
centroids = pd.DataFrame(cluster_centers_classic, columns=columns)
centroids_quantum = pd.DataFrame(cluster_centers_quantum, columns=columns)
centroids_hybrid = pd.DataFrame(cluster_centers_hybrid, columns=columns)
centroids = centroids.append(centroids_quantum, ignore_index=True).append(centroids_hybrid, ignore_index=True)
centroids['algorithm'] = ['classic']*2 +['quantum']*2 +['hybrid']*2

#****************************************#
#           Plot Scatter
#****************************************#

logger.info('plotting cluster center comparison to ' + fig_dir)

sns.set_style(hep.style.CMS)

plot = sns.pairplot(centroids, hue='algorithm', markers='*', plot_kws={'s': 550})
val = 0.9
plot.set(xlim=(-val,val))
plot.set(ylim=(-val,val))
sns.move_legend(plot, bbox_to_anchor=(0.5, -0.05), loc="lower center", ncol=2, labelspacing=0.8, fontsize=16, title='Algorithm')
plt.tight_layout()
plt.show()
plot.savefig(fig_dir+'/center_pair_scatter_classic_vs_quantum_vs_hybrid.png')


