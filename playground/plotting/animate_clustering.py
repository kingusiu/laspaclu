from collections import namedtuple
import numpy as np
import os
import pathlib
import pandas as pd
import seaborn as sns
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pofah.jet_sample as jesa
import laspaclu.util.persistence as pers
import laspaclu.util.logging as log







#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'run_n ae_run_n lat_dim read_n sample_id_train cluster_alg cluster_n, max_iter normalize quantum_min rtol mjj_center raw_format')
params = Parameters(run_n=38,
                    ae_run_n=50,
                    lat_dim=4,
                    read_n=int(100),
                    sample_id_train='qcdSig',
                    cluster_alg='kmeans',
                    cluster_n=2,
                    max_iter=10,
                    normalize=False,
                    quantum_min=True,
                    rtol=1e-2,
                    mjj_center=False,
                    raw_format=True)


# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*60+'\n'+'\t\t\t PLOT RUN \n'+str(params)+'\n'+'*'*60)

fig_dir = 'fig/test_plots'
gif_dir = 'gif/test_animations'
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(gif_dir).mkdir(parents=True, exist_ok=True)


Z=4 
def assign_clusters(latent_coords, cluster_centers,N):
    return np.random.choice(range(1,3),size=N).astype('int32')

old_centers = np.random.random(size=(2,Z))
def calc_new_centers(latent_coords, cluster_assigns):
    return old_centers+(np.random.random(size=(2,1))-0.5)*0.2


class ClusterTraining():

    def __init__(self,latent_coords,max_iter=6):
        self.latent_coords = latent_coords
        self.max_iter = max_iter

    def yield_next_step(self, cluster_centers_ini):
        self.cluster_centers = cluster_centers_ini
        N = len(self.latent_coords)
        i = 0
        while i < self.max_iter:
            cluster_assigns = assign_clusters(self.latent_coords, self.cluster_centers, N)
            self.cluster_centers = calc_new_centers(self.latent_coords, cluster_assigns)
            i += 1
            yield (cluster_assigns, self.cluster_centers, i-1)


class TrainingAnimator():

    def __init__(self, latent_coords, cluster_assigns, cluster_centers_ini):
        
        self.latent_coords = latent_coords
        self.N = len(latent_coords)
        self.feat_names = [r"$z_{"+str(z+1)+"}$" for z in range(latent_coords.shape[1])]
        df = pd.DataFrame(latent_coords, columns=self.feat_names).append(pd.DataFrame(cluster_centers_ini,columns=self.feat_names), ignore_index=True)
        df['assigned_cluster'] = np.append(cluster_assigns, [3, 4]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers

        self.palette = ['#21A9CE', '#5AD871', '#0052A3', '#008F5F']

        #import ipdb; ipdb.set_trace()
        
        self.gg = sns.PairGrid(df,hue='assigned_cluster')
        self.gg.map_offdiag(sns.scatterplot, palette=self.palette, alpha=0.75, size=[10]*self.N+[100]*2, markers=['.'], ec='face')
        self.gg.map_diag(sns.kdeplot, palette=self.palette, warn_singular=False)


    def clear_axes(self,gg):

        for ax in gg.axes.flat:
          ax.clear()
        for ax in gg.diag_axes:
            ax.clear()
            ax.set_ylim((0,0.9))
            ax.get_yaxis().set_visible(False)


    def animate(self, data):

        cluster_assignments, cluster_centers, i = data
        logger.info('iter '+str(i))
        logger.info('cluster_centers')
        logger.info(cluster_centers)

        df = pd.DataFrame(self.latent_coords,columns=self.feat_names).append(pd.DataFrame(cluster_centers,columns=self.feat_names), ignore_index=True)
        df['assigned_cluster'] = np.append(cluster_assignments, [3, 4]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers

        self.clear_axes(self.gg)
        self.gg.data = df
        off_dd = self.gg.map_offdiag(sns.scatterplot, palette=self.palette, alpha=0.75, size=[10]*self.N+[100]*2, markers='s', ec='face')
        dd = self.gg.map_diag(sns.kdeplot, palette=self.palette, warn_singular=False)
        self.gg.figure.suptitle('iteration '+str(i), ha='right') # y=1.02
        self.gg.figure.subplots_adjust(top=0.95)
        return off_dd.figure, dd.figure
  


def clustering_sim():

    N = 10
    # data
    latent_coords_qcd = np.random.random(size=(N,Z))
    cluster_assignments = np.random.choice(range(1,3),size=N).astype('int32')
    # init cluster centers randomly
    idx = np.random.choice(len(latent_coords_qcd), size=params.cluster_n, replace=False)
    cluster_centers_ini = latent_coords_qcd[idx]
    
    animator = TrainingAnimator(latent_coords_qcd, cluster_assignments, cluster_centers_ini)
    trainer = ClusterTraining(latent_coords_qcd, max_iter=6)
    animObj = animation.FuncAnimation(animator.gg.figure, animator.animate, frames=trainer.yield_next_step(cluster_centers_ini), repeat=False, interval=200, blit=True)

    ff = gif_dir+'/animated_training_generator.gif'
    logger.info('saving training gif to '+ff)
    writergif = animation.PillowWriter(fps=3) 
    animObj.save(ff, writer=writergif)

    logger.info('converged centers')
    logger.info(trainer.cluster_centers)


clustering_sim()



