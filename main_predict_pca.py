from collections import namedtuple



#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'run read_n sample_id_qcd sample_id_sig')
params = Parameters(run=12, read_n=int(5e4), sample_id_qcd='qcdSig', sample_id_sig='GtoWW35na')
fig_dir = 'fig/pca_run_'+str(params.run)
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
print('*'*50+'\n'+'pca projection run '+str(params.run)+' on '+str(params.read_n)+' samples'+'\n'+'*'*50)


#****************************************#
#           classic PCA
#****************************************#

model_path = pers.make_model_path(date='20211110', prefix='PCA', run_n=70)
pca = jli.load(model_path+'.joblib')

