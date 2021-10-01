import numpy as np
import quantum.dist_calc as dica
import quantum.minimization as mini

def calc_new_cluster_centers(data, cluster_assignments, n_clusters=2):
    return np.array([data[cluster_assignments == i].mean(axis=0) for i in range(n_clusters)])


def train_qmeans(data, n_clusters=2): 
    """
        train quantum k-means 
        :param data: input array of shape [N x Z] where N .. number of samples, Z .. dimension of latent space
        :return: np.ndarray of cluster centers
    """

    # init cluster centers randomly
    idx = np.random.choice(len(data), size=n_clusters, replace=False)
    cluster_centers = data[idx]

    # loop until convergence
    while True:

        cluster_assignments = []

        # for each sample
        for i, sample in enumerate(data):

            distances = []

            # calculate distance to each center
            for cluster_center in cluster_centers:

                qc = dica.overlap_circuit(sample, cluster_center)
                counts = dica.run_circuit(qc)
                distances.append(dica.calc_dist(counts, dica.calc_z(sample, cluster_center)))

            # find closest cluster (duerr & hoyer minimization)
            closest_cluster = mini.duerr_hoyer_minimization(distances)
            cluster_assignments.append(closest_cluster)

        new_centers = calc_new_cluster_centers(data, cluster_assignments)

        if np.allclose(new_centers, cluster_centers):
            break

        cluster_centers = new_centers

    return cluster_centers

