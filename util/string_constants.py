import pathlib


###     categories    ###
# ----------------------#
# cluster - in - data
# cluster - out - data
# cluster - out - models



# *********************************************************** #
#                           IN/OUT DATA                       #
# *********************************************************** #


cluster_in_data_dir = '/eos/home-e/epuljak/private/epuljak/public/diJet/'

cluster_out_data_dir = '/eos/user/k/kiwoznia/data/laspaclu_results/events'
pathlib.Path(cluster_out_data_dir).mkdir(parents=True, exist_ok=True)



# *********************************************************** #
#                           OUT MODELS                        #
# *********************************************************** #
