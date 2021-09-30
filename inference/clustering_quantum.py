import numpy as np
import quantum as qua

def train_qmeans(data, n_clusters=2): 
    """
        train quantum k-means 
        :param data: input array of shape [N x Z] where N .. number of samples, Z .. dimension of latent space
        :return: np.ndarray of cluster centers
    """

    # init cluster centers randomly
    cluster_centers = (np.random.random(size=(n_clusters, data.shape[1]))-0.5)*10

    distances = np.empty(shape=(len(data), n_clusters))

    # for each sample
    for i, sample in enumerate(data):

        distances = []

        # calculate distance to centers
        for cluster_center in cluster_centers:

            qc = qua.overlap_circuit(sample, cluster_center)
            counts = qua.run_circuit(qc)
            distances.append(qua.calc_dist(counts, qua.calc_z(sample, cluster_center)))

        # find closest cluster (duerr & hoyer minimization)

