from sklearn import cluster
from sklearn import svm
import numpy as np


def train_kmeans(data, center_ini, n_clusters=2):
    model = cluster.KMeans(n_clusters=n_clusters, init=center_ini, verbose=0).fit(data) # TODO: tune initialization
    return model

def train_one_class_svm(data, outlier_frac=0.1):
    model = svm.OneClassSVM(nu=outlier_frac).fit(data)
    return model
