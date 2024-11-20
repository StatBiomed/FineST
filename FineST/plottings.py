import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import linear_model
import scipy.stats as stats
import seaborn as sns
#from utils import compute_pathway
from .utils import *
# from .sparseAEH import *
import holoviews as hv
from holoviews import opts, dim
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.io import export_svg, export_png
from bokeh.layouts import gridplot
from scipy.sparse import csc_matrix
from scipy import stats
from matplotlib import gridspec

hv.extension('bokeh')
hv.output(size=200)

import math
from matplotlib.cm import hsv
from sklearn.metrics.pairwise import cosine_similarity
from .evaluation import *
import logging
logging.getLogger().setLevel(logging.INFO)


from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set(style="white", font_scale=1.2)
plt.rcParams["figure.figsize"] = (5, 5)

import matplotlib.colors as clr
colors = ["#000003",  "#3b0f6f",  "#8c2980",   "#f66e5b", "#fd9f6c", "#fbfcbf"]
cnt_color = clr.LinearSegmentedColormap.from_list('magma', colors, N=256)





def compute_pathway(sample=None, all_interactions=None, interaction_ls=None, name=None, dic=None):
    """
    Compute enriched pathways for a list of pairs or a dic of SpatialDE results.
    :param sample: spatialdm obj
    :param ls: a list of LR interaction names for the enrichment analysis
    :param path_name: str. For later recall sample.path_summary[path_name]
    :param dic: a dic of SpatialDE results (See tutorial)
    """
    if interaction_ls is not None:
        dic = {name: interaction_ls}
    if sample is not None:
        ## the old one
        all_interactions = sample.uns['geneInter']
        ## Load the original object from a pickle file when needed
        # with open(sample.uns["geneInter"], "rb") as f:
        #     all_interactions = pickle.load(f)
                
    df = pd.DataFrame(all_interactions.groupby('pathway_name').interaction_name)
    df = df.set_index(0)
    total_feature_num = len(all_interactions)
    result = []
    for n,ls in dic.items():
        qset = set([x.upper() for x in ls]).intersection(all_interactions.index)
        query_set_size = len(qset)
        for modulename, members in df.iterrows():
            module_size = len(members.values[0])
            overlap_features = qset.intersection(members.values[0])
            overlap_size = len(overlap_features)

            negneg = total_feature_num + overlap_size - module_size - query_set_size
            # Fisher's exact test
            p_FET = stats.fisher_exact([[overlap_size, query_set_size - overlap_size],
                                        [module_size - overlap_size, negneg]], 'greater')[1]
            result.append((p_FET, modulename, module_size, overlap_size, overlap_features, n))
    result = pd.DataFrame(result).set_index(1)
    result.columns = ['fisher_p', 'pathway_size', 'selected', 'selected_inters', 'name']
    if sample is not None:
        sample.uns['pathway_summary'] = result
    return result


def dot(pathway_res, figsize, markersize, pdf):
    for i, name in enumerate(pathway_res.name.unique()):
        fig, legend_gs = make_grid_spec(figsize,
                                        nrows=2, ncols=1,
                                        height_ratios=(4, 1))
        dotplot = fig.add_subplot(legend_gs[0])
        result1 = pathway_res.loc[pathway_res.name == name]
        result1 = result1.sort_values('selected', ascending=False)
        cts = result1.selected
        perc = result1.selected / result1.pathway_size
        value = -np.log10(result1.loc[:, 'fisher_p'].values)
        size = value * markersize
        im = dotplot.scatter(result1.selected.values, result1.index, 
                             c=perc.loc[result1.index].values, s=size, cmap='Reds')
        dotplot.set_xlabel('Number of pairs')
        # dotplot.set_xticks(np.arange(0, max(result1.selected.values) + 2))

        # Set the x-axis tick positions and labels
        # Display a label every 2 ticks
        xticks_positions = np.arange(0, max(result1.selected.values) + 2, 4)  
        dotplot.set_xticks(xticks_positions)
        dotplot.set_xticklabels(xticks_positions)

        dotplot.tick_params(axis='y', labelsize=10)
        dotplot.set_title(name)
        plt.colorbar(im, location='bottom', label='percentage of pairs out of CellChatDB')
        #                 dotplot.tight_layout()
        plt.gcf().set_dpi(600)

        # plot size bar
        size_uniq = np.quantile(size, np.arange(1, 0, -0.1))
        value_uniq = np.quantile(value, np.arange(1, 0, -0.1))
        size_range = value_uniq
        size_legend_ax = fig.add_subplot(legend_gs[1])
        size_legend_ax.scatter(
            np.arange(len(size_uniq)) + 0.5,
            np.repeat(0, len(size_uniq)),
            s=size_uniq,
            color='gray',
            edgecolor='black',
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(value_uniq)) + 0.5)
        # labels = [
        #     "{}".format(np.round((x * 100), decimals=0).astype(int)) for x in size_range
        # ]
        size_legend_ax.set_xticklabels(np.round(np.exp(-value_uniq), 3),
                                       rotation=60, fontsize='small')

        # remove y ticks and labels
        size_legend_ax.tick_params(
            axis='y', left=False, labelleft=False, labelright=False
        )

        # remove surrounding lines
        size_legend_ax.spines['right'].set_visible(False)
        size_legend_ax.spines['top'].set_visible(False)
        size_legend_ax.spines['left'].set_visible(False)
        size_legend_ax.spines['bottom'].set_visible(False)
        size_legend_ax.grid(False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_title('fisher exact p-value (right tile)', y=ymax + 0.9, size='small')

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)
        if pdf != None:
            pdf.savefig()


def make_grid_spec(
    ax_or_figsize,
    nrows: int,
    ncols: int,
    wspace= None,
    hspace = None,
    width_ratios = None,
    height_ratios= None,
):
    kw = dict(
        wspace=wspace,
        hspace=hspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    if isinstance(ax_or_figsize, tuple):
        fig = plt.figure(figsize=ax_or_figsize)
        return fig, gridspec.GridSpec(nrows, ncols, **kw)
    else:
        ax = ax_or_figsize
        ax.axis('off')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.figure, ax.get_subplotspec().subgridspec(nrows, ncols, **kw)


def dot_path(adata, uns_key=None, dic=None, cut_off=1, groups=None, markersize=50,
             figsize=(6, 8), pdf=None,
             **kwargs):
    """
    Either input a dict containing lists of interactions, or specify a dict key in adata.uns
    :param sample: AnnData object.
    :param uns_key: a dict key in adata.uns
    :param dic: a dict containing 1 or more list(s) of interactions
    :param cut_off: Minimum number of spots to be plotted.
    :param groups: subgroups from all dict keys.
    :param markersize:
    :param figsize:
    :param pdf: export pdf under your current directory
    :param kwargs:
    :return:
    """
    # plt.figure(figsize=figsize)

    if uns_key is not None:
        dic = {uns_key: adata.uns[uns_key]}
    pathway_res = compute_pathway(adata, dic=dic)
    # print(pathway_res)
    pathway_res = pathway_res[pathway_res.selected >= cut_off]
    if groups is not None:
        pathway_res = pathway_res.loc[pathway_res.name.isin(groups)]
    # pathway_res = pathway_res.loc[pathway_res.name.isin(groups)]
    n_subplot = len(pathway_res.name.unique())
    if pdf != None:
        with PdfPages(pdf + '.pdf') as pdf:
            dot(pathway_res, figsize, markersize, pdf)
            plt.show()
            plt.close()
    else:
        dot(pathway_res, figsize, markersize, pdf)



#################################################
# 2024.11.12 add for pathway confusion matrix
#################################################
def plot_conf_mat(result_pattern_all, pattern_name='Pattern_0', pathway_name='WNT', save_path=None):
    result_pattern = result_pattern_all[result_pattern_all['name'] == pattern_name]

    confusion_matrix = np.array([[result_pattern.loc[pathway_name, 'overlap_size'], 
                                  result_pattern.loc[pathway_name, 'query_set_size'] - result_pattern.loc[pathway_name, 'overlap_size']],
                                 [result_pattern.loc[pathway_name, 'module_size'] - result_pattern.loc[pathway_name, 'overlap_size'], 
                                  result_pattern.loc[pathway_name, 'negneg']]])

    ## confusion matirix heatmap
    fig, ax = plt.subplots(figsize=(3.8,3.0))
    sns.heatmap(confusion_matrix, annot=True, cmap="coolwarm", fmt='d', 
                xticklabels=[str(pathway_name), "Others"], 
                yticklabels=[pattern_name, "Others"]
               )

    plt.title(str(pathway_name)+" pathway")
    plt.axis('equal')
    # plt.gcf().set_dpi(300)

    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


###################################
# 2024.11.12 adjust for spatialDM
###################################
def spatialDE_clusters(histology_results, patterns, spatialxy, w=None, s=10, save_path=None):
    plt.figure(figsize=(21,5))
    for i in range(w):
        plt.subplot(1, w, i + 1)
        if isinstance(patterns.columns, pd.RangeIndex):
            scatter = plt.scatter(spatialxy[:,0], spatialxy[:,1], marker = 's', 
                                  c=patterns[i], cmap="viridis", s=s)
        else:
            scatter = plt.scatter(spatialxy[:,0], spatialxy[:,1], marker = 's', 
                                  c=patterns[str(i)], cmap="viridis", s=s)
        plt.colorbar(scatter)  
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.title('Pattern {} - {} LR pairs'.format(i, 
                                                    histology_results.query('pattern == @i').shape[0]))
        plt.gcf().set_dpi(300)

    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


###################################
# 2024.11.12 adjust for sparseAEH
###################################
# def plot_clusters(gaussian_subspot:MixedGaussian, label='counts', w=None, s=None, save_path=None):
def sparseAEH_clusters(gaussian_subspot, label='counts', w=None, s=None, save_path=None):
    k = gaussian_subspot.K
    h = np.ceil(k / w).astype(int)  # Calculate the number of rows
    
    # Adjust figure size based on the number of columns
    if w == 3:
        plt.figure(figsize=(21,5))
    elif w == 2:
        plt.figure(figsize=(14,5))
    else:
        plt.figure(figsize=(7,5))

    for i in range(gaussian_subspot.K):
        plt.subplot(h, w, i + 1)
        scatter = plt.scatter(gaussian_subspot.kernel.spatial[:,0],
                              gaussian_subspot.kernel.spatial[:,1],marker='s', 
                              c=gaussian_subspot.mean[:,i], cmap="viridis", s=s)
        plt.colorbar(scatter)  
        plt.axis('equal')
        plt.gca().invert_yaxis()
        if label == 'counts':
            plt.title('{}'.format(np.sum(gaussian_subspot.labels==i)))
        else:
            plt.title('{}'.format(gaussian_subspot.pi[i]))

    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


def plot_selected_pair_dot(sample, pair, spots, selected_ind, figsize, cmap, cmap_l, cmap_r, 
                           marker, marker_size, **kwargs):
    i = pd.Series(selected_ind == pair).idxmax()
    L = sample.uns['ligand'].loc[pair].dropna().values
    R = sample.uns['receptor'].loc[pair].dropna().values
    l1, l2 = len(L), len(R)
    
    if isinstance(sample.obsm['spatial'], pd.DataFrame):
        spatial_loc = sample.obsm['spatial'].values
    else:
        spatial_loc = sample.obsm['spatial']
    
    plt.figure(figsize=figsize)
    plt.subplot(1, 5, 1)
    plt.scatter(spatial_loc[:,0], spatial_loc[:,1], c=spots.loc[pair], cmap=cmap,
                vmax=1, marker=marker, s=marker_size, **kwargs)
    plt_util_invert('Moran: ' + str(sample.uns['local_stat']['n_spots'].loc[pair]) + ' spots')
    
    for l in range(l1):
        plt.subplot(1, 5, 2 + l)
        plt.scatter(spatial_loc[:,0], spatial_loc[:,1], c=sample[:,L[l]].X.toarray(),
                    cmap=cmap_l, marker=marker, s=marker_size, **kwargs)
        plt_util_invert('Ligand: ' + L[l])
    for l in range(l2):
        plt.subplot(1, 5, 2 + l1 + l)
        plt.scatter(spatial_loc[:,0], spatial_loc[:,1], c=sample[:,R[l]].X.toarray(),
                    cmap=cmap_r, marker=marker, s=marker_size, **kwargs)
        plt_util_invert('Receptor: ' + R[l])


def plt_util_invert(title):
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.gca().invert_yaxis()   
    plt.colorbar()


def plot_pairs_dot(sample, pairs_to_plot, pdf=None, figsize=(56, 8),
               cmap='Greens', cmap_l='Purples', cmap_r='Purples', 
               marker='o', marker_size=5, **kwargs):
               # cmap='Greens', cmap_l='coolwarm', cmap_r='coolwarm', marker_size=5, **kwargs):
    if sample.uns['local_stat']['local_method'] == 'z-score':
        selected_ind = sample.uns['local_z_p'].index
        spots = 1 - sample.uns['local_z_p']
    if sample.uns['local_stat']['local_method'] == 'permutation':
        selected_ind = sample.uns['local_perm_p'].index
        spots = 1 - sample.uns['local_perm_p']
    if pdf != None:
        with PdfPages(pdf + '.pdf') as pdf:
            for pair in pairs_to_plot:
                plot_selected_pair_dot(sample, pair, spots, selected_ind, figsize, cmap=cmap,
                                   cmap_l=cmap_l, cmap_r=cmap_r, 
                                   marker=marker, marker_size=marker_size, **kwargs)
                pdf.savefig()
                plt.show()
                plt.close()

    else:
        for pair in pairs_to_plot:
            plot_selected_pair_dot(sample, pair, spots, selected_ind, figsize, cmap=cmap,
                               cmap_l=cmap_l, cmap_r=cmap_r, 
                               marker=marker, marker_size=marker_size, **kwargs)
            plt.show()
            plt.close()



###########################################
# 2024.11.11 For all spot gene expression
###########################################
def gene_expr_allspots(gene, spatial_loc_all, recon_ref_adata_image_f2, gene_hv, 
                       label, s=8, save_path=None):
    def plot_gene_data_dot(spatial_loc, genedata, title, ax, s):
        normalized_data = genedata
        scatter = ax.scatter(spatial_loc[:,0], spatial_loc[:,1], c=normalized_data, 
                             cmap=cnt_color, s=s)   
        ax.invert_yaxis()
        ax.set_title(title)
        return scatter

    fig, ax = plt.subplots(figsize=(9, 7))

    reconstruction_f2_reshape_pd_all = pd.DataFrame(recon_ref_adata_image_f2)
    reconstruction_f2_reshape_pd_all.columns = gene_hv
    genedata3 = reconstruction_f2_reshape_pd_all[[gene]].to_numpy()
    print(str(gene)+" gene expression dim: ", genedata3.shape)
    print(str(gene)+" gene expression: \n", genedata3)
    scatter3 = plot_gene_data_dot(spatial_loc_all, genedata3, 
                                  str(gene)+' expression: '+str(label), ax, s) 
    fig.colorbar(scatter3, ax=ax)

    # Save the figure if a save path is provided
    if save_path is not None:
        fig.savefig(save_path, save_path, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


###########################################
# 2024.11.11 Adjust comparision plot 
###########################################
def gene_expr_compare(adata, gene, data_impt_reshape, gene_hv, marker='o', s=2, 
                      cmap=cnt_color, save_path=None):
    def plot_gene_data_scale(spatial_loc, genedata, title, ax):
        normalized_data = (genedata - genedata.min()) / (genedata.max() - genedata.min())
        scatter = ax.scatter(spatial_loc[:,0], spatial_loc[:,1], c=normalized_data, 
                             marker=marker, s=s, cmap=cmap)   
        ax.invert_yaxis()
        ax.set_title(title)
        return scatter

    spatial_loc = adata.obsm['spatial']

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    ## Orignal test data
    # original_matrix = pd.DataFrame(adata.X.todense())

    ## Check if 'todense' attribute exists
    if hasattr(adata.X, 'todense'):
        original_matrix = pd.DataFrame(adata.X.todense())
    else:
        original_matrix = pd.DataFrame(adata.X)

    original_matrix.columns = gene_hv
    genedata1 = original_matrix[[gene]].to_numpy()
    scatter1 = plot_gene_data_scale(spatial_loc, genedata1, 
                                    str(gene)+" Expression: Orignal", axes[0])

    # Imputed test data
    imputed_matrix_test_exp = pd.DataFrame(data_impt_reshape)
    imputed_matrix_test_exp.columns = gene_hv
    genedata2 = imputed_matrix_test_exp[[gene]].to_numpy()
    scatter2 = plot_gene_data_scale(spatial_loc, genedata2, 
                                    str(gene)+" Expression: FineST", axes[1])

    fig.colorbar(scatter1, ax=axes.ravel().tolist())
    plt.show()

    # Save the figure if a save path is provided
    if save_path is not None:
        fig.savefig(save_path, save_path, format='pdf', dpi=300, bbox_inches='tight')


def gene_expr(adata, matrix_order_df, gene_selet, marker='h', s=22, 
              figsize=(9, 7), save_path=None):
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    scatter_plot = ax1.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], 
                               c=matrix_order_df[gene_selet], cmap=cnt_color, 
                               marker=marker, s=s) 
    ax1.invert_yaxis()
    ax1.set_title(str(gene_selet)+' Expression')
    fig.colorbar(scatter_plot, ax=ax1)
    plt.show()

    # Save the figure if a save path is provided
    if save_path is not None:
        fig.savefig(save_path, save_path, format='pdf', dpi=300, bbox_inches='tight')


def subspot_expr(C, value, marker='o', s=1800, save_path=None):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.scatter(C[:, 0], C[:, 1], c=value, marker=marker, cmap=cnt_color, s=s)
    ax.invert_yaxis()
    ax.set_title("First spot")
    plt.show()

    # Save the figure if a save path is provided
    if save_path is not None:
        fig.savefig(save_path, save_path, format='pdf', dpi=300, bbox_inches='tight')


###########################################
# 2024.11.08 Adjust
###########################################
def sele_gene_cor(adata, data_impt_reshape, gene_hv, gene, ylabel, title, size, save_path=None):

    if isinstance(adata.X, np.ndarray):
        original_matrix = pd.DataFrame(adata.X)
    else:
        original_matrix = pd.DataFrame(adata.X.todense())

    original_matrix.columns = gene_hv

    imputed_matrix_test_exp = pd.DataFrame(data_impt_reshape)
    imputed_matrix_test_exp.columns = gene_hv

    genedata1 = original_matrix[[gene]].to_numpy()
    genedata2 = imputed_matrix_test_exp[[gene]].to_numpy()  

    g = sns.JointGrid(x=genedata1[:, 0], y=genedata2[:, 0], space=0, height=size)
    g = g.plot_joint(sns.scatterplot, color="b")
    g = g.plot_marginals(sns.kdeplot, shade=True, color="b")

    pearson_corr, _ = pearsonr(genedata1[:, 0], genedata2[:, 0])
    cosine_sim = cosine_similarity(genedata1.reshape(1, -1), genedata2.reshape(1, -1))[0][0]

    lr = LinearRegression()
    lr.fit(genedata1, genedata2)
    x = np.array(g.ax_joint.get_xlim())
    y = lr.predict(x.reshape(-1, 1))
    g.ax_joint.plot(x, y[:, 0], color='red', linestyle='--')

    r2_value = r2_score(genedata2, lr.predict(genedata1))

    g.ax_joint.annotate(f'Pearson Correlation: {pearson_corr:.3f}\nCosine Similarity: {cosine_sim:.3f}\nR²: {r2_value:.3f}', 
                    xy=(0.4, 0.1), xycoords='axes fraction', fontsize=10)

    g.ax_joint.set_xlabel('Original Expression')
    g.ax_joint.set_ylabel(ylabel)
    g.fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()


#####################################################################
# 2024.11.16 add 'gene_only' for VisumHD 8um: only gene cor boxplot
#####################################################################
def mean_cor_box(adata, data_impt_reshape, gene_only=False, save_path=None):

    if isinstance(adata.X, np.ndarray):
        matrix_profile = np.array(adata.X)
    else:
        matrix_profile = np.array(adata.X.todense())

    if not gene_only:
        corr_spot = calculate_correlation(matrix_profile, data_impt_reshape, 
                                          method='pearson', sample="spot")
        mean_corr_spot = np.mean(corr_spot)
        print(mean_corr_spot)

    corr_gene = calculate_correlation(matrix_profile, data_impt_reshape, 
                                      method='pearson', sample="gene")
    ## avoid nan
    corr_gene = np.nan_to_num(corr_gene, nan=0.0)
    mean_corr_gene = np.mean(corr_gene)

    print(mean_corr_gene)

    data = pd.DataFrame({
        'Type': np.repeat('corr_gene', len(corr_gene)) if gene_only else np.concatenate([np.repeat('corr_spot', len(corr_spot)), np.repeat('corr_gene', len(corr_gene))]),
        'mean_corr': corr_gene if gene_only else np.concatenate([corr_spot, corr_gene])
    })

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 14

    plt.figure(figsize=(4, 4))
    sns.boxplot(x='Type', y='mean_corr', data=data, palette='Set2')

    plt.title('Pearson Correlation', fontsize=16)
    plt.xlabel('', fontsize=16)
    plt.ylabel('', fontsize=16)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    plt.gcf().set_dpi(100)

    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()







def generate_colormap(number_of_distinct_colors, number_of_shades = 7):
    '''
    Ref: https://stackoverflow.com/questions/42697933/colormap-with-maximum-distinguishable-colours
    :param number_of_distinct_colors:
    :param number_of_shades:
    :return: n distinct colors
    '''
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors \
            / number_of_shades) * number_of_shades)

    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / \
            number_of_distinct_colors_with_multiply_of_shades
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)
    arr_by_shade_columns = arr_by_shade_rows.T
    number_of_partitions = arr_by_shade_columns.shape[0]
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)
    initial_cm = hsv(nums_distributed_like_rising_saw)
    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier
    initial_cm = initial_cm[:,:3] * 255
    initial_cm = initial_cm.astype(int)
    initial_cm = np.array(['#%02x%02x%02x' % tuple(initial_cm[i]) for i in range(len(initial_cm))])
    return initial_cm

def ligand_ct(adata, pair):
    ct_L = (
        adata.uns['local_stat']['local_I'][:,adata.uns['selected_spots'].index==pair] * 
        adata.obsm['celltypes']
    )
    return ct_L

def receptor_ct(adata, pair):
    ct_R = (
        adata.uns['local_stat']['local_I_R'][:,adata.uns['selected_spots'].index==pair] *
        adata.obsm['celltypes']
    )
    return ct_R

def chord_celltype(adata, pairs, color_dic=None, title=None, min_quantile=0.5, ncol=1, save=None):
    """
    Plot aggregated cell type weights given a list of interaction pairs
    :param adata: Anndata object
    :param pairs: List of interactions. Must be consistent with adata.uns['selected_spots'].index
    :param color_dic: dict containing specified colors for each cell type
    :param title: default to names provided in pairs
    :param min_quantile: Minimum edge numbers (in quantile) to show in the plot, default to 0.5.
    :param ncol: number of columns if more than one pair will be plotted.
    :param save: 'svg' or 'png' or None
    :return: Chord diagram showing enriched cell types. Edge color indicates source cell types.
    """

    if color_dic is None:
        # adata.obsm['celltypes'] = adata.obs[adata.obs.columns]
        ct = adata.obsm['celltypes'].columns.sort_values()
        l = len(ct)
        l0 = max(l, 10)
        gen_col = generate_colormap(l0)[:l]
        color_dic = {ct[i]: gen_col[i] for i in range(len(ct))}
    ls = []

    if type(min_quantile) is float:
        min_quantile = np.repeat(min_quantile, len(pairs))
    for i, pair in enumerate(pairs):
        if title is None:
            t = pair
        type_interaction = adata.uns['geneInter'].loc[pair, 'annotation']
        if type_interaction == 'Secreted Signaling':
            w = adata.obsp['weight']
        else:
            w = adata.obsp['nearest_neighbors']

        ct_L = ligand_ct(adata, pair)
        ct_R = receptor_ct(adata, pair)

        sparse_ct_sum = [[(csc_matrix(w).multiply(ct_L[n1].values).T.multiply(ct_R[n2].values)).sum() \
                          for n1 in ct_L.columns] for n2 in ct_R.columns]
        sparse_ct_sum = np.array(sparse_ct_sum)

        Links = pd.DataFrame({'source': np.tile(ct_L.columns, ct_R.shape[1]),
                              'target': np.repeat(ct_R.columns, ct_L.shape[1]),
                              'value': sparse_ct_sum.reshape(1, -1)[0]})

        Nodes = pd.DataFrame({'name': ct_L.columns})
        Nodes.index = Nodes.name.values
        nodes = hv.Dataset(Nodes, 'index')

        chord = hv.Chord((Links.loc[Links.value > 0], nodes)).select(  # Links.value>min_link[i]
            value=(Links.value.quantile(min_quantile[i]), None))
        cmap_ct = pd.Series(color_dic)[chord.nodes.data['index'].values].values.tolist()
        adata.uns[pair + '_link'] = Links
        chord.opts(
            opts.Chord(  # cmap='Category20',
                edge_cmap=cmap_ct,
                edge_color=dim('source').str(),
                labels='name', node_color=dim('index').str(),
                node_cmap=cmap_ct,
                title=t))
        ls.append(chord)

    ar = np.array([hv.render(fig) for fig in ls])
    for n in ar:
        n.output_backend = "svg"
    plots = ar.reshape(-1, ncol).tolist()
    grid = gridplot(plots)
    if save is not None:
        file_format = save.split('.')[-1]
        if file_format == 'svg':
            export_svg(grid, filename=save)
        elif file_format == 'png':
            export_png(grid, filename=save)
    show(grid)
    return grid


def chord_LR(adata, senders, receivers, color_dic=None,
             title=None, min_quantile=0.5, ncol=1, save=None):
    """
        Plot aggregated interaction scores given a list of sender-receiver combinations.
        :param adata: Anndata object
        :param senders: (list) Sender cell types
        :param senders: (list) Receiver cell types. Must be of the same length with sender cell types.
        :param color_dic: dict containing specified colors for each sender-receiver combination.
        :param title: default to sender_receiver
        :param min_quantile: Minimum edge numbers (in quantile) to show in the plot, default to 0.5.
        :param ncol: number of columns if more than one combination will be plotted.
        :param save: 'svg' or 'png' or None
        :return: Chord diagram showing enriched interactions. Edge color indicates ligand.
    """
    if color_dic is None:
        subgeneInter = adata.uns['geneInter'].loc[adata.uns['selected_spots'].index]
        type_interaction = subgeneInter.annotation
        n_short_lri = (type_interaction!='Secreted Signaling').sum()
        ligand_all = subgeneInter.interaction_name_2.str.split('-').str[0]
        receptor_all = subgeneInter.interaction_name_2.str.split('-').str[1]
        genes_all = np.hstack((ligand_all, receptor_all))
        genes_all = pd.Series(genes_all).drop_duplicates().values
        l = len(genes_all)
        l0 = max(l, 10)
        gen_col = generate_colormap(l0)[:l]
        color_dic = {genes_all[i]: gen_col[i] for i in range(l)}

    ls = []
    if type(min_quantile) is float:
        min_quantile = np.repeat(min_quantile, len(senders))

    for i, (sender, receiver) in enumerate(zip(senders, receivers)):
        if title is None:
            t = ('_').join((sender, receiver))

        ct_L = adata.obs.loc[:,sender].values * adata.uns['local_stat']['local_I'].T
        ct_R = adata.obs.loc[:,receiver].values * adata.uns['local_stat']['local_I_R'].T

        sparse_ct_sum = np.hstack(([csc_matrix(adata.obsp['nearest_neighbors']).multiply(n1).T.multiply(n2).sum() \
                      for n1,n2 in zip(ct_L[:n_short_lri], ct_R[:n_short_lri])],
                                  [csc_matrix(adata.obsp['weight']).multiply(n1).T.multiply(n2).sum() \
                      for n1,n2 in zip(ct_L[n_short_lri:], ct_R[n_short_lri:])]))


        Links = pd.DataFrame({'source':ligand_all,
                    'target':receptor_all,
                  'value': sparse_ct_sum})
        adata.uns[t+'_link'] = Links

        Nodes = pd.DataFrame({'name': genes_all.astype(str)})
        Nodes.index = Nodes.name.values

        Nodes=Nodes.drop_duplicates()

        nodes = hv.Dataset(Nodes, 'index')

        chord = hv.Chord((Links.loc[Links.value>0], nodes)).select(
            value=(Links.value.quantile(min_quantile).drop_duplicates().values, None))

        cmap_ct = pd.Series(color_dic)[chord.nodes.data['index'].values].values.tolist()

        chord.opts(
            opts.Chord(#cmap='Category20',
                        edge_cmap=cmap_ct,
                       edge_color=dim('source').str(),
                       labels='name', node_color=dim('index').str(),
                       node_cmap=cmap_ct,
                       title = 'Undifferentiated_Colonocytes'))
        ls.append(chord)

    ar = np.array([hv.render(fig) for fig in ls])
    for n in ar:
        n.output_backend = "svg"
    plots = ar.reshape(-1, ncol).tolist()
    grid = gridplot(plots)
    if save is not None:
        file_format = save.split('.')[-1]
        if file_format == 'svg':
            export_svg(grid, filename=save)
        elif file_format == 'png':
            export_png(grid, filename=save)
    show(grid)
    return grid

def chord_celltype_allpairs(adata, color_dic=None,
                             min_quantile=0.9, ncol=3, save=None):
    """
       Plot aggregated cell type weights for all pairs in adata.uns['selected_spots']
       :param adata: Anndata object
       :param pairs: List of interactions. Must be consistent with adata.uns['selected_spots'].index
       :param color_dic: dict containing specified colors for each cell type
       :param title: default to names provided in pairs
       :param min_quantile: Minimum edge numbers (in quantile) to show in the plot, default to 0.5.
       :param ncol: number of columns if more than one pair will be plotted.
       :param save: 'svg' or 'png' or None
       :return: 3 chord diagrams showing enriched cell types, one for adjacent signaling, \
       one for secreted signaling, and the other for the aggregated.
       """

    if color_dic is None:
        ct = adata.obs.columns.sort_values()
        l = len(ct)
        l0 = max(l, 10)
        gen_col = generate_colormap(l0)[:l]
        color_dic = {ct[i]: gen_col[i] for i in range(len(ct))}

    long_pairs = adata.uns['geneInter'][adata.uns['geneInter'].annotation == \
                    'Secreted Signaling'].index.intersection(adata.uns['selected_spots'].index)
    short_pairs = adata.uns['geneInter'][adata.uns['geneInter'].annotation != \
                        'Secreted Signaling'].index.intersection(adata.uns['selected_spots'].index)
    ls=[]

    for by_range,pairs,w in zip(['long', 'short'],
                    [long_pairs, short_pairs],
                 [adata.obsp['weight'], adata.obsp['nearest_neighbors']]):
        sparse_ct_sum = [[[(csc_matrix(w).multiply(ligand_ct(adata, p)[n1].values).T.multiply(receptor_ct(adata, p)[n2].values)).sum() \
           for n1 in ct] for n2 in ct] for p in pairs]
        sparse_ct_sum = np.array(sparse_ct_sum).sum(0)

        Links = pd.DataFrame({'source':np.tile(ct, l),
                    'target':np.repeat(ct, l),
                  'value': sparse_ct_sum.reshape(1,-1)[0]})
        adata.uns[by_range]=Links

        Nodes = pd.DataFrame({'name': ct})
        Nodes.index = Nodes.name.values
        nodes = hv.Dataset(Nodes, 'index')

        chord = hv.Chord((Links.loc[Links.value>0], nodes)).select( #Links.value>min_link[i]
            value=(Links.value.quantile(min_quantile), None))
        cmap_ct = pd.Series(color_dic)[chord.nodes.data['index'].values].values.tolist()
        chord.opts(
            opts.Chord(#cmap='Category20',
                        edge_cmap=cmap_ct,
                       edge_color=dim('source').str(),
                       labels='name', node_color=dim('index').str(),
                       node_cmap=cmap_ct,
                       title = by_range))
        ls.append(chord)

    value = (len(long_pairs) * adata.uns['long'].value + len(short_pairs) * adata.uns['short'].value)/ \
            (len(long_pairs) + len(short_pairs))
    Links.value = value
    chord = hv.Chord((Links.loc[Links.value>0], nodes)).select( #Links.value>min_link[i]
            value=(Links.value.quantile(min_quantile), None))
    cmap_ct = pd.Series(color_dic)[chord.nodes.data['index'].values].values.tolist()
    chord.opts(
        opts.Chord(#cmap='Category20',
                    edge_cmap=cmap_ct,
                   edge_color=dim('source').str(),
                   labels='name', node_color=dim('index').str(),
                   node_cmap=cmap_ct,
                   title = 'Cell_type_interactions_between_all_identified_pairs'))
    ls.append(chord)

    ar = np.array([hv.render(fig) for fig in ls])
    for n in ar:
        n.output_backend = "svg"
    plots = ar.reshape(-1, ncol).tolist()
    grid = gridplot(plots)
    if save is not None:
        file_format = save.split('.')[-1]
        if file_format == 'svg':
            export_svg(grid, filename=save)
        elif file_format == 'png':
            export_png(grid, filename=save)
    show(grid)
    return grid





# def plot_selected_pair(sample, pair, spots, selected_ind, figsize, cmap, cmap_l, cmap_r, **kwargs):
#     i = pd.Series(selected_ind == pair).idxmax()
#     L = sample.uns['ligand'].loc[pair].dropna().values
#     R = sample.uns['receptor'].loc[pair].dropna().values
#     l1, l2 = len(L), len(R)
    
#     if isinstance(sample.obsm['spatial'], pd.DataFrame):
#         spatial_loc = sample.obsm['spatial'].values
#     else:
#         spatial_loc = sample.obsm['spatial']
    
#     plt.figure(figsize=figsize)
#     plt.subplot(1, 5, 1)
#     plt.scatter(spatial_loc[:,0], spatial_loc[:,1], c=spots.loc[pair], cmap=cmap,
#                 vmax=1, **kwargs)
#     plt_util('Moran: ' + str(sample.uns['local_stat']['n_spots'].loc[pair]) + ' spots')
    
#     for l in range(l1):
#         plt.subplot(1, 5, 2 + l)
#         plt.scatter(spatial_loc[:,0], spatial_loc[:,1], c=sample[:,L[l]].X.toarray(),
#                     cmap=cmap_l, **kwargs)
#         plt_util('Ligand: ' + L[l])
#     for l in range(l2):
#         plt.subplot(1, 5, 2 + l1 + l)
#         plt.scatter(spatial_loc[:,0], spatial_loc[:,1], c=sample[:,R[l]].X.toarray(),
#                     cmap=cmap_r, **kwargs)
#         plt_util('Receptor: ' + R[l])

# def plot_pairs(sample, pairs_to_plot, pdf=None, figsize=(35, 5),
#                cmap='Greens', cmap_l='coolwarm', cmap_r='coolwarm', **kwargs):
#     """
#     plot selected spots as well as LR expression.
#     :param sample: AnnData object.
#     :param pairs_to_plot: list or arrays. pair name(s), should be from spatialdm_local pairs .
#     :param pdf: str. pdf file prefix. save plots in a pdf file.
#     :param figsize: figsize for each pair. Default to (35, 5).
#     :param markersize: markersize for each spot. Default
#     :param cmap: cmap for selected local spots.
#     :param cmap_l: cmap for selected ligand. If None, no subplot for ligand expression.
#     :param cmap_r: cmap for selected receptor. If None, no subplot for receptor expression
#     :return: subplots of spatial scatter plots, 1 for local Moran p-values, others for the original expression values
#     """
#     if sample.uns['local_stat']['local_method'] == 'z-score':
#         selected_ind = sample.uns['local_z_p'].index
#         spots = 1 - sample.uns['local_z_p']
#     if sample.uns['local_stat']['local_method'] == 'permutation':
#         selected_ind = sample.uns['local_perm_p'].index
#         spots = 1 - sample.uns['local_perm_p']
#     if pdf != None:
#         with PdfPages(pdf + '.pdf') as pdf:
#             for pair in pairs_to_plot:
#                 plot_selected_pair(sample, pair, spots, selected_ind, figsize, cmap=cmap,
#                                    cmap_l=cmap_l, cmap_r=cmap_r, **kwargs)
#                 pdf.savefig()
#                 plt.show()
#                 plt.close()

#     else:
#         for pair in pairs_to_plot:
#             plot_selected_pair(sample, pair, spots, selected_ind, figsize, cmap=cmap,
#                                cmap_l=cmap_l, cmap_r=cmap_r, **kwargs)
#             plt.show()
#             plt.close()
