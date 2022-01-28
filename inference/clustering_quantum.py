import numpy as np
import logging
import quantum.dist_calc as dica
import quantum.minimization as mini


# logging config
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(funcName)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('clustering_quantum')


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


def train_qmeans(data, n_clusters=2, quantum_min=True):
    """
        train quantum k-means 
        :param data: input array of shape [N x Z] where N .. number of samples, Z .. dimension of latent space
        :return: np.ndarray of cluster centers
    """

    #import ipdb; ipdb.set_trace()

    # init cluster centers randomly
    idx = np.random.choice(len(data), size=n_clusters, replace=False)
    cluster_centers = data[idx]

    # loop until convergence
    i = 0
    while True:

        cluster_assignments, _ = assign_clusters(data, cluster_centers, quantum_min=quantum_min)

        new_centers = calc_new_cluster_centers(data, cluster_assignments)
        logger.info('>>> iter {}: new centers {}'.format(i,new_centers))
        i = i+1

        if np.allclose(new_centers, cluster_centers, rtol=1.e-2):
            break

        cluster_centers = new_centers

    logger.info('>>> cluster centers converged')
    return cluster_centers

