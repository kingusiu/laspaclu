#import setGPU
from collections import namedtuple
import numpy as np
import os
import pathlib
import pandas as pd
import seaborn as sns
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pofah.jet_sample as jesa
import laspaclu.util.persistence as pers
import laspaclu.util.logging as log







#****************************************#
#           Runtime Params
#****************************************#

Parameters = namedtuple('Parameters', 'run_n ae_run_n lat_dim read_n sample_id_train cluster_alg cluster_n, max_iter normalize quantum_min rtol mjj_center raw_format')
params = Parameters(run_n=38,
                    ae_run_n=50,
                    lat_dim=4,
                    read_n=int(100),
                    sample_id_train='qcdSig',
                    cluster_alg='kmeans',
                    cluster_n=2,
                    max_iter=10,
                    normalize=False,
                    quantum_min=True,
                    rtol=1e-2,
                    mjj_center=False,
                    raw_format=True)


# logging
logger = log.get_logger(__name__)
logger.info('\n'+'*'*60+'\n'+'\t\t\t PLOT RUN \n'+str(params)+'\n'+'*'*60)

fig_dir = 'fig/test_plots'
gif_dir = 'gif/test_animations'
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(gif_dir).mkdir(parents=True, exist_ok=True)

#****************************************#
#      load data latent representation
#****************************************#

# input_dir = "/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_"+str(params.ae_run_n)
input_dir = '/eos/home-e/epuljak/private/epuljak/public/diJet/'+str(int(params.lat_dim))
# latent_coords_qcd = pers.read_latent_rep_from_file(input_dir, sample_id=params.sample_id_train, read_n=params.read_n, raw_format=params.raw_format, shuffle=True)
# logger.info('read {} training samples ({} jets)'.format(len(latent_coords_qcd)/2, len(latent_coords_qcd))) # stacked j1 & j2

do_test_plot = False

if do_test_plot:

    N = 100
    Z = 4
    latent_coords_qcd = np.random.random(size=(N,Z))

    # init cluster centers randomly
    idx = np.random.choice(len(latent_coords_qcd), size=params.cluster_n, replace=False)
    cluster_centers = latent_coords_qcd[idx]

    sns.set_style(hep.style.CMS)
    palette = ['#21A9CE', '#5AD871', '#0052A3', '#008F5F']

    feat_names = [r"$z_{"+str(z+1)+"}$" for z in range(latent_coords_qcd.shape[1])]
    cluster_assignments = np.random.choice(range(1,3),size=N).astype('int32')
    df = pd.DataFrame(latent_coords_qcd, columns=feat_names).append(pd.DataFrame(cluster_centers,columns=feat_names), ignore_index=True)
    df['assigned_cluster'] = np.append(cluster_assignments, [3, 4]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers
    #df = df.assign(Cluster=df.assigned_cluster.map({1:'assigned 1', 2:'assigned 2', 3:'center 1', 4:'center 2'})).drop('assigned_cluster',axis='columns',inplace=True) # replace labels for legend
    # import ipdb; ipdb.set_trace()
    gg = sns.PairGrid(df,hue='assigned_cluster')
    gg.map_offdiag(sns.scatterplot, palette=palette, alpha=0.75, size=[10]*N+[100]*2, markers=['.'], ec='face')
    gg.map_diag(sns.kdeplot, palette=palette, warn_singular=False)
    gg.add_legend()
    # replace labels
    new_labels = ['assigned 1', 'assigned 2', 'center 1', 'center 2', None, None]
    h = gg.legend.legendHandles
    for t, l in zip(gg.legend.texts, new_labels):
        t.set_text(l)
    for hh in h[-2:]: hh.set_visible(False)
    sns.move_legend(gg, bbox_to_anchor=(0.5,-0.07), loc="lower center", ncol=4, labelspacing=0.8, fontsize=11, title='Cluster')
    plt.tight_layout() # rect=[0,0.1,1,1]
    fig = gg.figure
    fig.savefig(os.path.join(fig_dir,'test_pairplot.png'), bbox_inches="tight")



#****************************************#
#           Animation Routines
#****************************************#

do_test_gif = False

if do_test_gif:

    N=400
    Z=4
    latent_coords_qcd = np.random.random(size=(N,Z))

    # init cluster centers randomly
    idx = np.random.choice(len(latent_coords_qcd), size=params.cluster_n, replace=False)
    cluster_centers = latent_coords_qcd[idx]

    sns.set_style(hep.style.CMS)
    palette = ['#21A9CE', '#5AD871', '#0052A3', '#008F5F']

    feat_names = [r"$z_{"+str(z+1)+"}$" for z in range(latent_coords_qcd.shape[1])]
    cluster_assignments = np.random.choice(range(1,3),size=N).astype('int32')
    df = pd.DataFrame(latent_coords_qcd, columns=feat_names).append(pd.DataFrame(cluster_centers,columns=feat_names), ignore_index=True)
    df['assigned_cluster'] = np.append(cluster_assignments, [3, 4]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers

    gg = sns.PairGrid(df,hue='assigned_cluster')
    gg.map_offdiag(sns.scatterplot, palette=palette, alpha=0.75, size=[10]*N+[100]*2, markers=['.'], ec='face')
    gg.map_diag(sns.kdeplot, palette=palette, warn_singular=False)

    def clear_axes(gg):
        for ax in gg.axes.flat:
              ax.clear()
        for ax in gg.diag_axes:
            ax.clear()
            ax.set_ylim((0,0.9))
            ax.get_yaxis().set_visible(False)


    def animate(data):
        (df, i) = data
        clear_axes(gg)
        gg.data = df
        gg.map_offdiag(sns.scatterplot, palette=palette, alpha=0.75, size=[10]*N+[100]*2, markers='s', ec='face')
        gg.map_diag(sns.kdeplot, palette=palette, warn_singular=False)
        gg.figure.suptitle('iteration '+str(i), ha='right') # y=1.02
        gg.figure.subplots_adjust(top=0.95)

    def gen_func():
      for i in range(40):
        cluster_centers_i = cluster_centers+(np.random.random(size=(2,1))-0.5)*0.2
        df.iloc[-2:,:-1] = cluster_centers_i
        cluster_assigns_i = np.random.choice(range(1,3),size=N).astype('int32')
        df.iloc[:-2,-1] = cluster_assigns_i
        yield (df, i)


    animObj = animation.FuncAnimation(gg.figure, animate, frames=gen_func, interval=500)

    ff = gif_dir+'/animated_training.gif'
    logger.info('saving training gif to '+ff)
    writergif = animation.PillowWriter(fps=7) 
    animObj.save(ff, writer=writergif)


#****************************************#
#        Animation from Generator
#****************************************#

do_generator_gif = True

if do_generator_gif:

    N=400
    Z=4 
    def assign_clusters(latent_coords, cluster_centers):
        return np.random.choice(range(1,3),size=N).astype('int32')

    old_centers = np.random.random(size=(2,Z))
    def calc_new_centers(latent_coords, cluster_assigns):
        return old_centers+(np.random.random(size=(2,Z))-0.5)*0.2


    class ClusterAnimation():

        def __init__(self,latent_coords,gg,palette):
            self.latent_coords = latent_coords
            self.palette = palette
            self.gg = gg

        def yield_next_step(self,max_iter,cluster_centers):
            N = len(self.latent_coords)
            i = 0
            while i < max_iter:
                cluster_assigns = assign_clusters(self.latent_coords, cluster_centers)
                cluster_centers = calc_new_centers(self.latent_coords, cluster_assigns)
                i += 1
                yield (cluster_assigns, cluster_centers, i-1)
                # df.iloc[-2:,:-1] = cluster_centers_i
                # df.iloc[:-2,-1] = cluster_assigns_i
                # yield (df, i)

        def clear_axes(self,gg):
            for ax in gg.axes.flat:
              ax.clear()
            for ax in gg.diag_axes:
                ax.clear()
                ax.set_ylim((0,0.9))
                ax.get_yaxis().set_visible(False)


        def animate(self,data):
            cluster_assignments, cluster_centers, i = data
            df = pd.DataFrame(self.latent_coords).append(pd.DataFrame(cluster_centers), ignore_index=True)
            df['assign'] = np.append(cluster_assignments, [2, 3]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers

            self.clear_axes(self.gg)
            self.gg.map_offdiag(sns.scatterplot, palette=self.palette, alpha=0.75, size=[10]*N+[100]*2, markers='s', ec='face')
            self.gg.map_diag(sns.kdeplot, palette=self.palette, warn_singular=False)
            self.gg.figure.suptitle('iteration '+str(i), ha='right') # y=1.02
            self.gg.figure.subplots_adjust(top=0.95)



    def clear_axes(gg):
        for ax in gg.axes.flat:
          ax.clear()
        for ax in gg.diag_axes:
            ax.clear()
            ax.set_ylim((0,0.9))
            ax.get_yaxis().set_visible(False)

    def make_animate(gg, palette, latent_coords):

        def animate(data):
            cluster_assignments, cluster_centers, i = data
            df = pd.DataFrame(latent_coords).append(pd.DataFrame(cluster_centers), ignore_index=True)
            df['assign'] = np.append(cluster_assignments, [2, 3]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers

            #clear_axes(gg)
            gg.map_offdiag(sns.scatterplot, palette=palette, alpha=0.75, size=[10]*N+[100]*2, markers='s', ec='face')
            gg.map_diag(sns.kdeplot, palette=palette, warn_singular=False)
            gg.figure.suptitle('iteration '+str(i), ha='right') # y=1.02
            gg.figure.subplots_adjust(top=0.95)
        
        return animate

    def yield_next_step(latent_coords, cluster_centers, max_iter):

        N = len(latent_coords)
        i = 0
        while i < max_iter:
            cluster_assigns = assign_clusters(latent_coords, cluster_centers)
            cluster_centers = calc_new_centers(latent_coords, cluster_assigns)
            i += 1
            logger.info('iter '+str(i))
            logger.info('cluster_centers')
            logger.info(cluster_centers)
            yield (cluster_assigns, cluster_centers, i-1)
            # df.iloc[-2:,:-1] = cluster_centers_i
            # df.iloc[:-2,-1] = cluster_assigns_i
            # yield (df, i)        



    def clustering_sim():

        # data
        latent_coords_qcd = np.random.random(size=(N,Z))
        cluster_assignments = np.random.choice(range(1,3),size=N).astype('int32')
        # init cluster centers randomly
        idx = np.random.choice(len(latent_coords_qcd), size=params.cluster_n, replace=False)
        cluster_centers_ini = latent_coords_qcd[idx]
        feat_names = [r"$z_{"+str(z+1)+"}$" for z in range(latent_coords_qcd.shape[1])]
        df = pd.DataFrame(latent_coords_qcd, columns=feat_names).append(pd.DataFrame(cluster_centers_ini,columns=feat_names), ignore_index=True)
        df['assigned_cluster'] = np.append(cluster_assignments, [3, 4]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers

        palette = ['#21A9CE', '#5AD871', '#0052A3', '#008F5F']
        gg = sns.PairGrid(df,hue='assigned_cluster')
        gg.map_offdiag(sns.scatterplot, palette=palette, alpha=0.75, size=[10]*N+[100]*2, markers=['.'], ec='face')
        gg.map_diag(sns.kdeplot, palette=palette, warn_singular=False)

        animate_fun = make_animate(gg, palette, latent_coords_qcd)
        yield_next_step_gen = yield_next_step(latent_coords_qcd, cluster_centers_ini, max_iter=5)
        animator = ClusterAnimation(latent_coords_qcd, gg, palette)
        animObj = animation.FuncAnimation(gg.figure, animate_fun, frames=yield_next_step_gen, repeat=True, interval=300)

        ff = gif_dir+'/animated_training_generator.gif'
        logger.info('saving training gif to '+ff)
        writergif = animation.PillowWriter(fps=10) 
        animObj.save(ff, writer=writergif)


    clustering_sim()
