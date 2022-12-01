import pathlib
from collections import OrderedDict

###     categories    ###
# ----------------------#
# cluster - in - data
# cluster - out - data
# cluster - out - models

# reporting - out - plots/animations



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
#                           OUT REPORTS                       #
# *********************************************************** #

reporting_fig_base_dir = 'results/fig'
reporting_gif_base_dir = 'results/gif'


# *********************************************************** #
#                       LABELS & COLORS                       #
# *********************************************************** #

sample_name_dict = OrderedDict(

    qcdSigExt = 'QCD signal-region',
    GtoWW35na = r'$G(3.5 TeV)\to WW$ narrow',
    GtoWW15br = r'$G(1.5 TeV)\to WW$ broad',
    AtoHZ35 = r'$A(3.5 TeV) \to HZ \to ZZZ$'
)

# colors BG vs SIG

bg_blue = '#1b98e0' # background blue #1CAAE7
sig_red = '#F23602' #E4572E' # signal red/orange #DE1A1A
multi_sig_palette = ['#F4A615','#6C56B3','#F24236','#ED701D'] # yellow, violet, red, pumpkin


# colors classic vs quantum

classic_violett = '#6C56B3'
quantum_green = '#5AD871'


