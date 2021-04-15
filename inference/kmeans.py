from sklearn.cluster import KMeans
import numpy as np

def train(data):
    model = KMeans(n_clusters=2, verbose=0).fit(data) # TODO: tune initialization
    return model

