from sklearn.cluster import KMeans
import numpy as np

def train(data):
    model = KMeans(n_clusters=2, verbose=1).fit(data)
    return model

