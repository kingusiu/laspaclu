import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.animation as animation
from typing import List, Tuple, Generator
from collections.abc import Callable
import mplhep as hep

import laspaclu.src.util.logging as log
import laspaclu.src.quantum.dist_calc as dica
import laspaclu.src.quantum.minimization as mini
import laspaclu.src.analysis.plotting as plot


# logging config
logger = log.get_logger(__name__)

def calc_new_cluster_centers(data, cluster_assignments, n_clusters=2) -> np.ndarray:
    return np.array([data[cluster_assignments == i].mean(axis=0) for i in range(n_clusters)])


def quantum_distance_to_centers(sample, cluster_centers) -> List:

    """
        computes |a-b|**2
    """
    
    distances = []

    # calculate distance to each center
    for cluster_center in cluster_centers:

        qc = dica.overlap_circuit(sample, cluster_center)
        counts = dica.run_circuit(qc)
        distances.append(dica.calc_dist(counts, dica.calc_z(sample, cluster_center)))

    return distances


def assign_clusters(data, cluster_centers, quantum_min=True) -> Tuple[np.ndarray,np.ndarray]:

    cluster_assignments = []
    distances = []

    # for each sample
    for i, sample in enumerate(data):

        # import ipdb; ipdb.set_trace()
        dist = quantum_distance_to_centers(sample, cluster_centers)

        # find closest cluster index (duerr & hoyer minimization for quantum approach or numpy for classic approach)
        closest_cluster = mini.duerr_hoyer_minimization(dist) if quantum_min else np.argmin(dist)
        cluster_assignments.append(closest_cluster)
        distances.append(dist)

    return np.asarray(cluster_assignments), np.asarray(distances) 


class ClusterTrainer():

    def __init__(self, latent_coords, cluster_n, max_iter=6):
        self.latent_coords = latent_coords
        self.max_iter = max_iter
        self.cluster_n = cluster_n

    def yield_next_step(self, cluster_centers_ini, rtol):
        
        """
            train quantum k-means 
            :param latent_coords: input array of shape [N x Z] where N .. number of samples, Z .. dimension of latent space
            :return: np.ndarray of cluster centers
        """

        self.cluster_centers = cluster_centers_ini

        # loop until convergence or max_iter
        i = 0
        while True:

            if i > self.max_iter: 
                logger.info('>>> maximal number of iterations {} reached'.format(i))
                break

            cluster_assignments, _ = assign_clusters(self.latent_coords, self.cluster_centers, quantum_min=True)

            new_centers = calc_new_cluster_centers(self.latent_coords, cluster_assignments)
            logger.info('>>> iter {}: new centers {}'.format(i,new_centers))
            i = i+1

            if np.allclose(new_centers, self.cluster_centers, rtol=rtol):
                logger.info('>>> clustering converged')
                break

            self.cluster_centers = new_centers
            yield (cluster_assignments, self.cluster_centers,i-1)



class TrainingAnimator():

    def __init__(self, latent_coords, cluster_assigns, cluster_centers_ini):
        
        sns.set_style(hep.style.CMS)
        sns.set_style({'axes.linewidth': 0.5})

        self.latent_coords = latent_coords
        self.N = len(latent_coords)
        self.Z = latent_coords.shape[1]
        self.feat_names = [r"$z_{"+str(z+1)+"}$" for z in range(latent_coords.shape[1])]
        df = pd.DataFrame(latent_coords, columns=self.feat_names).append(pd.DataFrame(cluster_centers_ini,columns=self.feat_names), ignore_index=True)
        df['assigned_cluster'] = np.append(cluster_assigns, [2, 3]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers

        # if Z > 10, decrease df to O(10)
        if self.Z > 10:
            every_k = -(self.Z // -10) # ceil div
            drop_idx = list(range(1,self.Z,every_k))
            drop_cols = [j for i,j in enumerate(df.columns) if i in drop_idx]
            df = df.drop(drop_cols, axis=1)        

        self.palette = ['#21A9CE', '#5AD871', '#0052A3', '#008F5F']

        #import ipdb; ipdb.set_trace()
        
        self.gg = sns.PairGrid(df,hue='assigned_cluster')
        self.gg.map_offdiag(sns.scatterplot, palette=self.palette, alpha=0.7, size=[12]*self.N+[120]*2, markers=['.'], ec='face')
        self.gg.map_diag(sns.kdeplot, palette=self.palette, warn_singular=False)
        self.set_axes(self.gg,clear=False)


    def set_axes(self,gg,clear=True):

        for ax in gg.figure.axes:
            if clear:
                ax.clear()
            ax.set_xlim((-1,1))
            ax.set_ylim((-1,1))
            ax.tick_params(labelsize=15)
            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)
        for ax in gg.diag_axes:
            if clear:
                ax.clear()
            ax.set_ylim((0,1.7))
            ax.set_xlim((-1,1))   
            ax.get_yaxis().set_visible(False)
            ax.tick_params(labelsize=15)
            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)


    def animate(self, data):

        cluster_assignments, cluster_centers, i = data

        df = pd.DataFrame(self.latent_coords, columns=self.feat_names).append(pd.DataFrame(cluster_centers, columns=self.feat_names), ignore_index=True)
        df['assigned_cluster'] = np.append(cluster_assignments, [2, 3]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers

        if self.Z > 10:
            every_k = -(self.Z // -10) # ceil div
            drop_idx = list(range(1,self.Z,every_k))
            drop_cols = [j for i,j in enumerate(df.columns) if i in drop_idx]
            df = df.drop(drop_cols, axis=1)       

        self.set_axes(self.gg)
        self.gg.data = df
        off_dd = self.gg.map_offdiag(sns.scatterplot, palette=self.palette, alpha=0.7, size=[10]*self.N+[100]*2, markers='s', ec='face')
        dd = self.gg.map_diag(sns.kdeplot, palette=self.palette, warn_singular=False)
        self.gg.figure.suptitle('iteration '+str(i), x=0.96, ha='right', fontsize=18) # y=1.02
        self.gg.figure.subplots_adjust(top=0.94)
        return off_dd.figure, dd.figure


def train_qmeans_animated(data, cluster_centers_ini, cluster_n=2, quantum_min=True, rtol=1e-2, max_iter=30, gif_dir='gif'):
    
    cluster_assignments, _ = assign_clusters(data, cluster_centers_ini, quantum_min=quantum_min)

    animator = TrainingAnimator(data, cluster_assignments, cluster_centers_ini)
    trainer = ClusterTrainer(data, cluster_n=cluster_n, max_iter=max_iter)

    animObj = animation.FuncAnimation(animator.gg.figure, animator.animate, frames=trainer.yield_next_step(cluster_centers_ini, rtol), repeat=False, interval=200, blit=True)

    ff = gif_dir+'/animated_training.gif'
    logger.info('saving training gif to '+ff)
    writergif = animation.PillowWriter(fps=3) 
    animObj.save(ff, writer=writergif)

    return trainer.cluster_centers # return last cluster_centers



def train_qmeans(data, n_clusters=2, quantum_min=True, rtol=1e-2, max_iter=200):
    """
        train quantum k-means 
        :param data: input array of shape [N x Z] where N .. number of samples, Z .. dimension of latent space
        :return: np.ndarray of cluster centers
    """


    # init cluster centers randomly
    idx = np.random.choice(len(data), size=n_clusters, replace=False)
    cluster_centers = data[idx]

    # loop until convergence
    i = 0
    while True:

        if i > max_iter: break

        cluster_assignments, _ = assign_clusters(data, cluster_centers, quantum_min=quantum_min)

        new_centers = calc_new_cluster_centers(data, cluster_assignments)
        logger.info('>>> iter {}: new centers {}'.format(i,new_centers))
        i = i+1

        if np.allclose(new_centers, cluster_centers, rtol=rtol):
            break

        cluster_centers = new_centers

    logger.info('>>> cluster centers converged')
    return cluster_centers

