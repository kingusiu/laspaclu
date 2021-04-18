from sklearn import cluster
from sklearn import svm
import numpy as np


def train_kmeans(data):
    model = cluster.KMeans(n_clusters=2, verbose=0).fit(data) # TODO: tune initialization
    return model

def train_one_class_svm(data, outlier_frac=0.1):
    model = svm.OneClassSVM(nu=outlier_frac).fit(data)
    return model
