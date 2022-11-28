import pathlib


###     categories    ###
# ----------------------#
# cluster - in - data
# cluster - out - data
# cluster - out - models
# cluster - out - results



# *********************************************************** #
#                           IN/OUT DATA                       #
# *********************************************************** #


cluster_in_data_dir = '/eos/home-e/epuljak/private/epuljak/public/diJet/'

cluster_out_data_dir = '/eos/user/k/kiwoznia/data/laspaclu_results/events'
pathlib.Path(cluster_out_data_dir).mkdir(parents=True, exist_ok=True)



# *********************************************************** #
#                           OUT MODELS                        #
# *********************************************************** #

cluster_out_model_dir = 'results/model'

# *********************************************************** #
#                           OUT RESULTS                       #
# *********************************************************** #

cluster_out_fig_base_dir = 'results/fig'
cluster_out_gif_base_dir = 'results/gif'


# *********************************************************** #
#                       LABELS & COLORS                       #
# *********************************************************** #

sample_name_dict = {

    'qcdSigExt': 'QCD signal-region',
    'GtoWW35na': r'$G(3.5 TeV)\to WW$ narrow',
    'GtoWW15br': r'$G(1.5 TeV)\to WW$ broad',
    'AtoHZ35': r'$A(3.5 TeV) \to HZ$'
}

# colors BG vs SIG

bg_blue = '#1b98e0' # background blue #1CAAE7
sig_red = '#F23602' #E4572E' # signal red/orange #DE1A1A
multi_sig_palette = ['#F4A615','#6C56B3','#F24236'] # yellow, violet, red


# colors classic vs quantum

classic_violett = '#6C56B3'
quantum_green = '#5AD871'


