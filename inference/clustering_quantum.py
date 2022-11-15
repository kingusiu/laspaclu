import numpy as np
import util.logging as log
import quantum.dist_calc as dica
import quantum.minimization as mini
import laspaclu.analysis.plotting as plot
import pandas as pd
import seaborn as sns
import matplotlib.animation as animation



# logging config
logger = log.get_logger(__name__)

def calc_new_cluster_centers(data, cluster_assignments, n_clusters=2):
    return np.array([data[cluster_assignments == i].mean(axis=0) for i in range(n_clusters)])


def quantum_distance_to_centers(sample, cluster_centers):

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


def assign_clusters(data, cluster_centers, quantum_min=True):

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


def create_animated_figure(latent_coords, cluster_assignments, cluster_centers):
    palette = ['#21A9CE', '#00DE7E', '#008CB3', '#00C670']
    N = len(latent_coords)
    df = pd.DataFrame(latent_coords).append(pd.DataFrame(cluster_centers), ignore_index=True)
    df['assign'] = np.append(cluster_assignments, [2, 3]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers
    gg = sns.PairGrid(df,hue='assign')
    gg.map_offdiag(sns.scatterplot, palette=palette, alpha=0.6, size=[10]*N+[100]*2, markers='s', ec='face')
    gg.map_diag(sns.kdeplot, palette=palette, warn_singular=False)
    return gg


def create_animate(gg,N,latent_coords):

    palette = ['#21A9CE', '#00DE7E', '#008CB3', '#00C670']

    def animate(data):
    
        cluster_assignments, cluster_centers = data
        df = pd.DataFrame(latent_coords).append(pd.DataFrame(cluster_centers), ignore_index=True)
        df['assign'] = np.append(cluster_assignments, [2, 3]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers
    
        for ax in gg.axes.flat:
          ax.clear()
        for ax in gg.diag_axes:
            ax.clear()
        gg.data = df
        gg.map_offdiag(sns.scatterplot, palette=palette, alpha=0.6, size=[10]*N+[100]*2, markers='s', ec='face')
        gg.map_diag(sns.kdeplot, palette=palette, warn_singular=False)
    
    return animate


def yield_next_training_result(latent_coords, n_clusters, rtol, max_iter=30):
        
        """
            train quantum k-means 
            :param latent_coords: input array of shape [N x Z] where N .. number of samples, Z .. dimension of latent space
            :return: np.ndarray of cluster centers
        """


        # init cluster centers randomly
        idx = np.random.choice(len(latent_coords), size=n_clusters, replace=False)
        cluster_centers = latent_coords[idx]

        # loop until convergence or max_iter
        i = 0
        while True:

            if i > max_iter: 
                logger.info('>>> maximal number of iterations {} reached'.format(i))
                break

            cluster_assignments, _ = assign_clusters(latent_coords, cluster_centers, quantum_min=True)

            new_centers = calc_new_cluster_centers(latent_coords, cluster_assignments)
            logger.info('>>> iter {}: new centers {}'.format(i,new_centers))
            i = i+1

            if np.allclose(new_centers, cluster_centers, rtol=rtol):
                logger.info('>>> clustering converged')
                break

            cluster_centers = new_centers
            yield cluster_assignments, cluster_centers

        logger.info('>>> final cluster centers')
        logger.info(cluster_centers)

        yield cluster_centers


class TrainStepGenerator():

    def __init__(latent_coords, n_clusters, rtol, max_iter=30):
        self.gen = yield_next_training_result(latent_coords, n_clusters, rtol, max_iter)

    def __iter__():
        self.current_state = yield from self.gen
    


def train_qmeans_animated(data, n_clusters=2, quantum_min=True, rtol=1e-2, max_iter=30, gif_dir='gif'):
    
    # init cluster centers randomly
    idx = np.random.choice(len(data), size=n_clusters, replace=False)
    cluster_centers = data[idx]
    cluster_assignments, _ = assign_clusters(data, cluster_centers, quantum_min=quantum_min)

    N = len(data)

    gg = create_animated_figure(data, cluster_assignments, cluster_centers)
    frame_fun = TrainStepGenerator(data, n_clusters, rtol, max_iter=max_iter)
    animate_fun = create_animate(gg,N)

    animObj = animation.FuncAnimation(gg.figure, animate_fun, frames=frame_fun, repeat=True, interval=300)

    ff = gif_dir+'/animated_training.gif'
    logger.info('saving training gif to '+ff)
    writergif = animation.PillowWriter(fps=30) 
    animObj.save(ff, writer=writergif)

    return frame_fun.current_state # return last cluster_centers



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

