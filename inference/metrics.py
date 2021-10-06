import numpy as np


# euclidian distance to cluster center
dist_to_cluster_center = lambda a,b: np.sum((a-b)**2,axis=1)


def compute_metric_score(algo_str, coords, model):
    # compute euclidian distance to closest cluster center for kmeans
    if algo_str == 'kmeans':
        return np.sqrt(np.sum(model.transform(coords)**2, axis=1))
    # compute squared distance to separating hyperplane (shifting to all < 0) => Losing info of inlier vs outlier in this way (???)
    elif algo_str == 'one_class_svm':
        distances = model.decision_function(coords)
        distances = distances - np.max(distances)
        return distances**2


def compute_quantum_metric_score(sample_dist, cluster_assign, metric_type='sum_all_dist'):
    # compute squared sum of all distances
    if metric_type == 'sum_all_dist':
        return np.sqrt(np.sum(sample_dist**2, axis=1))
    # compute squared dist to closest cluster
    else: 
        return np.sqrt(sample_dist[range(len(sample_dist)), cluster_assign]**2)

        
