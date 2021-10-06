import inference.clustering_classic as cluster


#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'read_n sample_id_train cluster_alg')
params = Parameters(read_n=int(1e4), sample_id_train='qcdSide', cluster_alg='kmeans')


#****************************************#
#           Data Sample
#****************************************#

data_sample = dasa.DataSample(params.sample_id_train)



#****************************************#
#               CLUSTERING CLASSIC
#****************************************#

#****************************************#
#               KMEANS

if params.cluster_alg == 'kmeans':

    model_path = pers.make_model_path(prefix='KM')

    print('>>> training kmeans')
    cluster_model = cluster.train_kmeans(latent_coords_qcd)
    

#****************************************#
#           ONE CLASS SVM

else:

    model_path = pers.make_model_path(prefix='SVM')

    print('>>> training one class svm')
    cluster_model = cluster.train_one_class_svm(latent_coords_qcd)

# save
jli.dump(cluster_model, model_path+'.joblib') 



#****************************************#
#               QUANTUM CLUSTERING
#****************************************#

## train quantum kmeans

print('>>> training qmeans')
cluster_q_centers = cluster_q.train_qmeans(latent_coords_qcd)

model_path_qm = make_model_path(prefix='QM') + '.npy'
with open(model_path_qm, 'wb') as f:
    np.save(f, cluster_q_centers)
