import logging
logging.getLogger().setLevel(logging.INFO)
from .utils import *
from .evaluation import *

import anndata
import matplotlib.pyplot as plt
import spatialdm as sdm
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.cm import hsv
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
import seaborn as sns
import holoviews as hv
from holoviews import opts, dim
from bokeh.io import show
from bokeh.io import export_svg, export_png
from bokeh.layouts import gridplot
from scipy.sparse import csc_matrix
from scipy import stats
from matplotlib import gridspec
from scipy.spatial.distance import jensenshannon
from matplotlib.ticker import MultipleLocator   # adjust axies

hv.extension('bokeh')
hv.output(size=200)

import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# sns.set(style="white", font_scale=1.2)
# plt.rcParams["figure.figsize"] = (5, 5)

from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib.colors as clr
colors = ["#000003",  "#3b0f6f",  "#8c2980",   "#f66e5b", "#fd9f6c", "#fbfcbf"]
cnt_color = clr.LinearSegmentedColormap.from_list('magma', colors, N=256)
## for L-R-TF-TG
from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.io as pio
import urllib, json
## for plot_half_violin
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import numpy as np
from scipy.stats import pearsonr


###################################################
# 2025.03.06 plot violin of 3 methods
###################################################
def plot_half_violin(method1_df, method2_df, method3_df, variable_name, value_property, property='PCC', 
                     fig_size=(2, 3.5), font_size=14, save_path=None):
    '''
    Plot half-box half-violin with p-value of wilcoxon_test or PCC
    Parameters: 
        method1_df, method2_df, method3_df: DataFrame with 1-col named 'Unnamed: 0' 2-col named 'others', 
        e.g.: method1_df:
            #######################
                    index      spot
            0       tumor  0.528174
            1           B  0.180316
            #######################
        variable_name: list of variable names to plot, 2-col name for three methods
        value_property: the column name of the values to plot, for example: Proportation, or JSE, RMSE
        property: 'PCC' for Pearson correlation coefficient or 'wilcoxon_test' for Wilcoxon test p-value
    '''
    # Merge DataFrames on their index
    df = pd.merge(method1_df, method2_df, left_index=True, right_index=True)
    df = pd.merge(df, method3_df, left_index=True, right_index=True)

    # Reshape the DataFrame for plotting
    df_melt = pd.melt(df.reset_index(), id_vars='index', value_vars=variable_name, 
                      var_name='Method', value_name=value_property)

    ## Configuration
    groups = variable_name
    palette = ['#B9DDD7', '#E8B0B4', '#B6A1D3']
    shift = 0.2
    pairs = list(itertools.combinations(groups, 2))

    fig, ax = plt.subplots(figsize=fig_size)

    ## Create half-violin plot
    violin_parts = sns.violinplot(x='Method', y=str(value_property), data=df_melt, inner=None, palette=palette)

    ## Modify violin patches to keep only the right half
    for vp in violin_parts.collections[::1]:
        for paths in vp.get_paths():
            vertices = paths.vertices
            vertices[:, 0] = np.clip(vertices[:, 0], np.median(vertices[:, 0]), np.inf)
            vertices[:, 0] += shift

    ## Create box plot
    sns.boxplot(x='Method', y=str(value_property), data=df_melt, width=0.3, fliersize=0, linewidth=1.0,
                boxprops={'edgecolor': 'gray'}, palette=palette, showcaps=True, whiskerprops={'linewidth': 1.0},
                capprops={'linewidth': 1.0}, medianprops={'color': 'gray'})

    ## Add mean values as scatter points
    mean_values = df_melt.groupby('Method')[str(value_property)].mean().values
    plt.scatter(groups, mean_values, color='white', edgecolor='gray', s=50, zorder=2)

    ## Add strip plot
    sns.stripplot(x='Method', y=str(value_property), data=df_melt, jitter=True, size=5, color=".3", linewidth=0)

    y, h, col = df_melt[str(value_property)].max() + 0.08, 0.08, 'k'
    for i, (method1, method2) in enumerate(pairs):
        method1_values = df_melt[df_melt['Method'] == method1][str(value_property)]
        method2_values = df_melt[df_melt['Method'] == method2][str(value_property)]

        if property == 'PCC': 
            ## Calculate Pearson correlation coefficients and add to the plot
            correlation, _ = pearsonr(method1_values, method2_values)
            print(f'correlation for {method1} vs {method2}: {correlation}')
            x1, x2 = groups.index(method1), groups.index(method2)
            plt.plot([x1, x1, x2, x2], [y + (h * i), y + h + (h * i), y + h + (h * i), y + (h * i)], lw=1.0, c=col)
            plt.text((x1 + x2) * .5, y + h + (h * i), f'r = {correlation:.2f}', ha='center', va='bottom', color=col)
        elif property == 'wilcoxon_test': 
            ## Calculate p-values and add to the plot
            _, p_value = wilcoxon(method1_values, method2_values)
            print(f'p-value for {method1} vs {method2}: {p_value}')
            if p_value < 0.05:
                x1, x2 = groups.index(method1), groups.index(method2)
                plt.plot([x1, x1, x2, x2], [y + (h * i), y + h + (h * i), y + h + (h * i), y + (h * i)], lw=1.0, c=col)
                plt.text((x1 + x2) * .5, y + h + (h * i), f'p = {p_value:.3e}', ha='center', va='bottom', color=col)
        y += h    

    ## Set axis labels and title
    ax.set_ylabel('Cell type proportions', fontsize=font_size)
    
    ## Adjust plot limits
    y_max = df_melt[str(value_property)].max()
    plt.xlim(-0.3, len(groups) - 0.25)
    plt.ylim(-0.3, y_max + 0.5)
    
    ## Adjust subplot and display plot
    plt.subplots_adjust()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.xticks(rotation=25)
    
    plt.gcf().set_dpi(150)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path is not None:
        plt.savefig(save_path, transparent=True, format='svg', dpi=300, bbox_inches='tight')

    plt.show()


###################################################
# 2025.03.06 plot cell type propotion of 3 methods
###################################################
def celltype_proportion(data, ctype_hex_map, 
                       figure_size=(2.5, 3.0), font_size=12, trans=False, format='svg', save_path=None):
    datasets = data.index
    categories = data.columns

    fig, ax = plt.subplots(figsize=figure_size)

    bottom = np.zeros(len(datasets))
    for category in categories:
        if category in ctype_hex_map:
            sizes = data[category]
            ax.bar(datasets, sizes, bottom=bottom, label=category, color=ctype_hex_map[category])
            bottom += sizes

    plt.xticks(rotation=25)

    ax2 = ax.twiny()
    ax2.set_xticks([0.2, 0.5, 0.8])
    ax2.set_xticklabels(['Reference', 'Visium', 'FineST'])
    plt.xticks(rotation=25)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

    ax.tick_params(axis='x', labelsize=font_size)  
    ax2.tick_params(axis='x', labelsize=font_size) 
    ax.tick_params(axis='y', labelsize=font_size)  

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')

    plt.show()    


###############################################
# 2024.02.12 Cluster-heatmap for colocalization
###############################################
def MoranR_colocalization(st_adata_cltp2, cmap, cell_type=None, linewidth=0.5, linecolor='whitesmoke', 
                          fig_size=(9, 9), font_size=12, trans=False, format='svg', 
                          xtick_rotation=90, ytick_rotation=0, save_path=None):
    """
    Plot clustermap.
    Parameters:
        st_adata_cltp2: AnnData, The annotated data matrix containing spatial transcriptomics data.
    """
    print("*** Calculate Moren_R using SpatialDM ***")
    ## calculate MorenR
    n_cell_types = st_adata_cltp2.obsm[str(cell_type)].shape[1]
    correlation_matrix_TransImp2 = np.zeros((n_cell_types, n_cell_types))

    for i in range(n_cell_types):
        for j in range(n_cell_types):
            correlation_matrix_TransImp2[i, j] = sdm.utils.Moran_R(
                st_adata_cltp2.obsm[str(cell_type)].values[:, i:i+1], 
                st_adata_cltp2.obsm[str(cell_type)].values[:, j:j+1], 
                st_adata_cltp2.obsp['weight']
            )[0]

    correlation_matrix2 = pd.DataFrame(correlation_matrix_TransImp2)
    correlation_matrix2.index = st_adata_cltp2.obsm[str(cell_type)].columns
    correlation_matrix2.columns = st_adata_cltp2.obsm[str(cell_type)].columns
    print("*** Calculate DONE! ***")

    ## Create clustermap
    g = sns.clustermap(correlation_matrix2, cmap=cmap, linewidth=linewidth, linecolor=linecolor, figsize=fig_size)

    ## Set x and y axis tick labels
    ax = g.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, ha='right', fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation, fontsize=font_size)

    ## Ensure all tick labels are visible
    plt.setp(ax.get_xticklabels(), visible=True)
    plt.setp(ax.get_yticklabels(), visible=True)

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')
    plt.show()

    return correlation_matrix2

#######################################
# 2024.02.11 Add LR global Moran R plot
#######################################
def LR_global_moranR(adata_impt_all_spot, pairs, fig_size=(6, 6), font_size=12,
                     trans=False, format='svg', save_path=None): 
    # Select P value
    if adata_impt_all_spot.uns['global_stat']['method'] == 'permutation':
        p = 'perm_pval'
    elif adata_impt_all_spot.uns['global_stat']['method'] == 'z-score':
        p = 'z_pval'
        
    ## Extract global_I values for the given pairs
    moran_r_values = []
    for pair in pairs:
        ligand_index = adata_impt_all_spot.uns['ligand'].index == pair
        moran_r_value = adata_impt_all_spot.uns['global_I'][ligand_index][0]
        # print(moran_r_value)
        moran_r_values.append(moran_r_value)
    
    ## Define colors for each pair
    colors = generate_colormap(max(10, len(pairs) + 2))[2:]
    pair_color_dict = {pair: colors[i % len(colors)] for i, pair in enumerate(pairs)}
    
    ## Sort pairs based on Moran_R values
    sorted_pairs = [pair for _, pair in sorted(zip(moran_r_values, pairs))]
    sorted_moran_r_values = sorted(moran_r_values)
    print(sorted_pairs)
    
    fig, axs = plt.subplots(figsize=fig_size)
    
    ## Original scatter plot with color and size variations
    for i, pair in enumerate(sorted_pairs):
        # ligand, receptor = pair.split('_')
        ligand, receptors = pair.split('_', maxsplit=1)   # for three
        
        ligand_index = adata_impt_all_spot.uns['ligand'].index == pair
        global_res_p = adata_impt_all_spot.uns['global_res'][p][ligand_index]
        
        size = np.exp(global_res_p) * 80
        color = pair_color_dict[pair]  # Use the original color for the pair
    
        scatter = axs.scatter(i, sorted_moran_r_values[i], c=[color], s=size, label=pair, edgecolor='k')
    
    axs.set_xlabel('Ligand', fontsize=font_size)
    axs.set_ylabel('Receptor', fontsize=font_size)
    axs.set_xticks(range(len(sorted_pairs)))
    axs.set_xticklabels([pair.split('_')[0] for pair in sorted_pairs])
    axs.set_yticks(sorted_moran_r_values)  # Ensure the number of ticks matches the labels
    axs.set_yticklabels([pair.split('_')[1] for pair in sorted_pairs])
    
    ## Add right side y-axis with Moran's I values
    ax2 = axs.twinx()
    ax2.set_ylim(axs.get_ylim())  
    ax2.set_yticks(range(len(sorted_moran_r_values)))  
    ax2.set_yticklabels([f'{val:.2f}' for val in sorted_moran_r_values])  
    ax2.set_ylabel("Global R", fontsize=font_size)
    axs.legend()
    ## Add colorbar
    # cbar = plt.colorbar(scatter, ax=ax2, pad=0.25)
    # cbar.set_label('Global R')
    fig.set_dpi(150)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')

    plt.show()

#######################################
# 2024.02.11 Add LR global Moran R plot
# using aver_expr as x, y coords
#######################################
def select_pairs(adata_impt_all_spot, pairs, fig_size=(6, 6)):
    ## Select P value
    if adata_impt_all_spot.uns['global_stat']['method'] == 'permutation':
        p = 'perm_pval'
    elif adata_impt_all_spot.uns['global_stat']['method'] == 'z-score':
        p = 'z_pval'

    coords = []
    for pair in pairs:
        ligand, receptor = pair.split('_')
        x_coord = adata_impt_all_spot.to_df()[ligand].mean()
        y_coord = adata_impt_all_spot.to_df()[receptor].mean()
        coords.append((x_coord, y_coord))

    fig, axs = plt.subplots(figsize=fig_size)

    ## Original scatter plot with color and size variations
    for i, pair in enumerate(pairs):
        ligand_index = adata_impt_all_spot.uns['ligand'].index == pair
        global_I = adata_impt_all_spot.uns['global_I'][ligand_index]
        global_res_p = adata_impt_all_spot.uns['global_res'][p][ligand_index]
        
        x_coord, y_coord = coords[i]
        size = np.exp(global_res_p)*80
        print(size)
        color = global_I

        scatter = axs.scatter(x_coord, y_coord, c=color, s=size, label=pair, cmap='viridis', edgecolor='k')

    axs.set_title('Original')
    axs.set_xlabel('Ligand mean')
    axs.set_ylabel('Receptor mean')
    axs.legend()
    cbar = plt.colorbar(scatter, ax=axs)
    cbar.set_label('Global I value')
    plt.tight_layout()
    plt.show()

#######################################
# 2024.02.11 Add LR interaction plot
#######################################
def LR_local_moranR(adata_LRpair, pair, fig_size=(12, 6), trans=False, format='svg', save_path=None):
    '''
    adata_LRpair: _adata_pattern_all_spot.h5ad from CCC 
    '''
    loczp = adata_LRpair.uns["local_z_p"]
    pair_data = 1 - loczp.loc[pair]
    non_zero_mean = pair_data[pair_data != 0].mean()
    formatted_mean = f"non_zero_mean: {non_zero_mean:.2f}"
    
    zero_proportion = (pair_data == 0).mean()
    non_zero_proportion = (pair_data != 0).mean()
    
    proportions = [zero_proportion, non_zero_proportion]
    labels = ['No', 'Yes']
    colors = ['whitesmoke', 'honeydew']
    explode = (0.1, 0)    # Highlight the first sector
    
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    ## pie
    wedges, texts, autotexts = axs[0].pie(proportions, explode=explode, colors=colors, 
                                          autopct='%1.1f%%', shadow=True, startangle=90)
    
    axs[0].legend(wedges, labels, title="Interaction", loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
    axs[0].set_title(f'{pair}: {formatted_mean} (LocalR mean)')
    
    ## violinplot
    sns.violinplot(data=pair_data[pair_data != 0], ax=axs[1], color='honeydew')
    axs[1].set_xlabel('Interaction')
    axs[1].set_ylabel('Strength')
    plt.tight_layout()
    fig.set_dpi(150)

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')
        
    plt.show()


#######################################
# 2024.02.05 Revised global_plot x-axis
#######################################
def global_plot(sample, pairs=None, figsize=(3,4), loc=2, max_step=0.1, min_step=0.1, **kwarg):
    """
    overview of global selected pairs for a SpatialDM obj
    parameters: 
        sample: AnnData object
        pairs: list of pairs to be highlighted, e.g. ['SPP1_CD44']
        figsize: default to (3,4)
        loc=2: left-up; 4: right-down; 5: right 
        reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    """
    if pairs is not None:
        color_codes = generate_colormap(max(10, len(pairs)+2))[2:]
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.75)
    ax.spines['bottom'].set_linewidth(0.75)

    ## Selecte P value
    if sample.uns['global_stat']['method'] == 'permutation':
        p = 'perm_pval'
    elif sample.uns['global_stat']['method'] == 'z-score':
        p = 'z_pval'
    ax.scatter(np.log1p(sample.uns['global_I']), -np.log1p(sample.uns['global_res'][p]),
                c=sample.uns['global_res'].selected, **kwarg)

    if pairs != None:
        for i, pair in enumerate(pairs):
            ax.scatter(np.log1p(sample.uns['global_I'])[sample.uns['ligand'].index==pair],
                        -np.log1p(sample.uns['global_res'][p])[sample.uns['ligand'].index==pair],
                        c=color_codes[i]) #TODO: perm pval only?

    ax.xaxis.set_major_locator(MultipleLocator(max_step))
    ax.xaxis.set_minor_locator(MultipleLocator(min_step))
    ax.set_xlabel('log1p (Global R)', fontsize=14)
    ax.set_ylabel('-log1p (P Value)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(np.hstack((['No-significant'], pairs)), loc=loc, fontsize=10)   # loc is location
    # ax.legend(np.hstack((['No-significant'], pairs)), bbox_to_anchor=(0,1), loc="upper left", fontsize=10)

    return ax


def generate_colormap(number_of_distinct_colors, number_of_shades=7):
    '''
    Ref: https://stackoverflow.com/questions/42697933/colormap-with-maximum-distinguishable-colours
    parameters: number_of_distinct_colors, number_of_shades:
    return: n distinct colors
    '''
    number_of_distinct_colors_with_multiply_of_shades = int(
        math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades
    )

    linearly_distributed_nums = np.arange(
        number_of_distinct_colors_with_multiply_of_shades
    ) / number_of_distinct_colors_with_multiply_of_shades

    arr_by_shade_rows = linearly_distributed_nums.reshape(
        number_of_shades, 
        number_of_distinct_colors_with_multiply_of_shades // number_of_shades
    )

    arr_by_shade_columns = arr_by_shade_rows.T
    number_of_partitions = arr_by_shade_columns.shape[0]
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half
    lower_half = lower_partitions_half * number_of_shades

    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8 / lower_half)

    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = (
                np.ones(number_of_shades) - 
                initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            )
            modifier = j * modifier / upper_partitions_half
            initial_cm[
                lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i
            ] += modifier

    initial_cm = initial_cm[:, :3] * 255
    initial_cm = initial_cm.astype(int)
    initial_cm = np.array([
        '#%02x%02x%02x' % tuple(initial_cm[i]) for i in range(len(initial_cm))
    ])

    return initial_cm


def cor_hist(adata, adata_df_infer, max_step=0.1, min_step=0.01,
             fig_size=(5, 4), trans=False, format='svg', save_path=None, label_fontsize=14, tick_fontsize=12):
    ## Check if input is AnnData or DataFrame and handle accordingly
    if isinstance(adata, pd.DataFrame):
        pearson_correlations = [stats.pearsonr(adata[col].values, adata_df_infer[col].values)[0] for col in adata.columns]
    else:
        pearson_correlations = [stats.pearsonr(adata.to_df()[col].values, adata_df_infer[col].values)[0] for col in adata.to_df().columns]
    print('Pearson correlations: ', np.mean(pearson_correlations))

    fig = plt.figure(figsize=fig_size)
    ax = sns.histplot(data=pearson_correlations, bins=30, kde=True)
    ax.xaxis.set_major_locator(MultipleLocator(max_step))
    ax.xaxis.set_minor_locator(MultipleLocator(min_step))
    plt.title("Histogram of Pearson Correlations", fontsize=label_fontsize)
    plt.xlabel("Pearson Correlation", fontsize=label_fontsize)
    plt.ylabel("Frequency", fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')
    plt.show()

#################################################
# 2025.01.28: plot the propotion of TransDeconv 
#################################################
def plot_stackedbar_p(df, labels, colors, title, subtitle, 
                      fig_size=(18, 4), trans=False, format='pdf', save_path=None):
    fields = labels

    sns.set(style="white", context="paper", font_scale=1.0) 
    fig, ax = plt.subplots(1, figsize=fig_size, dpi=150)

    left = len(df) * [0]
    for idx, name in enumerate(fields):
        plt.barh(df.index, df[name], left=left, color=colors[idx])
        left = left + df[name]

    plt.legend(labels, bbox_to_anchor=([1, 1]), ncol=2, frameon=False, fontsize='large')
    # plt.legend(labels, bbox_to_anchor=([.95, -0.14, 0, 0]), ncol=10, frameon=False)

    sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)

    xticks = np.arange(0, 1.1, 0.1)
    xlabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    plt.xticks(xticks, xlabels, fontsize=12)
    # plt.xlabel("Percentage", fontsize=14)
    plt.ylabel("", fontsize=14)
    # plt.title(title, fontsize=16)

    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    ax.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.6)

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, bbox_inches='tight', dpi=300)  

    plt.show()


#################################################
# 2025.01.24: plot the loss curve of trining 
#################################################
def loss_curve(train_losses, test_losses, best_epoch, best_loss, max_step=5, min_step=1,
               fig_size=(5, 4), trans=False, format='svg', save_path=None):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(train_losses, label='Train Loss')
    ax.plot(test_losses, label='Test Loss')

    ## Add a star for the best loss
    ax.scatter(best_epoch, best_loss, color='black', marker='*', label='Best Loss')
    ax.xaxis.set_major_locator(MultipleLocator(max_step))
    ax.xaxis.set_minor_locator(MultipleLocator(min_step))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()

    # fig.set_dpi(300)
    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')
    plt.show()

#################################################
# 2024.12.12 add PlotCell from SpatialScope: 
#################################################
def NucleiMap(adata, coords, annotation_list, size=0.8, alpha_img=0.3, lw=1, 
              subset=None, palette='tab20', 
              show_square=False, show_circle=False, legend=True, ax=None, **kwargs):
    """
    Plot cells with spatial coordinates.
    Parameters:
        adata : AnnData, Annotated data matrix.
        annotation_list : list, List of annotations to color the plot.
        size : float, optional, Size of the spots, by default 0.8.
        alpha_img : float, optional, Alpha value for the background image, by default 0.3.
        lw : int, optional, Line width for the square around spots, by default 1.
        subset : list, optional, Subset of annotations to plot, by default None.
        palette : str, optional, Color palette, by default 'tab20'.
        show_square : bool, optional, Whether to show squares around spots, by default True.
        legend : bool, optional, Whether to show the legend, by default True.
        ax : matplotlib.axes.Axes, optional, Axes object to draw the plot onto, by default None.
        kwargs : dict, Additional keyword arguments for sc.pl.spatial.
    """
    ## Prepare data
    merged_df = adata.uns['cell_locations'].copy()
    test = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
    test.obsm['spatial'] = merged_df[["x", "y"]].to_numpy()
    
    ## Adjust tissue image coordinates
    if coords[2][0] == 0: 
        test.obsm["spatial"] += np.array([coords[0][1], 0])
    else: 
        test.obsm["spatial"] += np.array([coords[0][1], coords[0][0]])

    test.uns = adata.uns

    if subset is not None:
        test.obs.loc[~test.obs[annotation_list].isin(subset), annotation_list] = None
        
    ## Plot spatial data
    sc.pl.spatial(
        test,
        color=annotation_list,
        size=size,
        frameon=False,
        alpha_img=alpha_img,
        show=False,
        palette=palette,
        na_in_legend=False,
        ax=ax,
        title='',
        sort_order=True,
        **kwargs
    )
    
    ## Add squares around spots if show_square is True
    if show_square:
        sf = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
        spot_radius = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['spot_diameter_fullres']/2
        for sloc in adata.obsm['spatial']:
            square = mpl.patches.Rectangle(
                (sloc[0] * sf - spot_radius * sf, sloc[1] * sf - spot_radius * sf),
                2 * spot_radius * sf,
                2 * spot_radius * sf,
                ec="grey",
                lw=lw,
                fill=False
            )
            ax.add_patch(square)

    ## Add circles around spots if show_circle is True
    if show_circle:
        sf = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
        spot_radius = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['spot_diameter_fullres']/2
        for sloc in adata.obsm['spatial']:
            rect = mpl.patches.Circle(
                (sloc[0] * sf, sloc[1] * sf),
                spot_radius * sf,
                ec="grey",
                lw=lw,
                fill=False
            )
            ax.add_patch(rect)

    ## Hide axis labels
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)
    ## Remove legend if not needed
    if not legend:
        ax.get_legend().remove()
    ## Make frame visible
    for _, spine in ax.spines.items(): 
        spine.set_visible(True)
        
    ax.set_aspect('equal', adjustable='box') 
    ax.grid(False) 
    ax.set_xticks([]) 
    ax.set_yticks([])
    ax.set_title('Nuclei detection') 
    ax.set_xlabel('X Coordinate') 
    ax.set_ylabel('Y Coordinate')
        
    plt.tight_layout()
    plt.show()


#################################################
# 2024.12.06 add PCC calculate: 
#################################################
def PCC(shared_visium_df, shared_xenium_df):
    ## Calculate Pearson correlation coefficient and p-value for each column
    columns_corr = []
    columns_p_value = []
    for column in shared_visium_df.columns:
        corr, p_value = pearsonr(shared_visium_df[column], shared_xenium_df[column])
        columns_corr.append(corr)
        columns_p_value.append(p_value)

    ## Calculate Pearson correlation coefficient and p-value for each row
    rows_corr = []
    rows_p_value = []
    for idx, row in shared_visium_df.iterrows():
        corr, p_value = pearsonr(row, shared_xenium_df.loc[idx])
        rows_corr.append(corr)
        rows_p_value.append(p_value)

    ## Save results to dataframes
    columns_result_df = pd.DataFrame({'Gene': shared_visium_df.columns, 
                                      'correlation_coefficient': columns_corr, 
                                      'p_value': columns_p_value})
    rows_result_df = pd.DataFrame({'Sample': shared_visium_df.index, 
                                   'correlation_coefficient': rows_corr, 
                                   'p_value': rows_p_value})

    return columns_result_df, rows_result_df

#################################################
# 2024.12.06 add PCC plot: 
#################################################
def compute_jsd_between_matrices(matrix1, matrix2, axis=0):
    probabilities1 = matrix1 / np.sum(matrix1, axis=axis, keepdims=True)
    probabilities2 = matrix2 / np.sum(matrix2, axis=axis, keepdims=True)
    jsd = np.zeros(matrix1.shape[1 - axis])
    for i in range(matrix1.shape[1 - axis]):
        p, q = (probabilities1[:, i], probabilities2[:, i]) \
            if axis == 0 else (probabilities1[i, :], probabilities2[i, :])
        jsd[i] = jensenshannon(p, q)
    return jsd

def rmse(y_pred, y_mean_pred):
    mse = ((y_mean_pred - y_pred)**2).mean()
    return mse**0.5


def plot_PCC_revised(df1, df2, column_name, x_label, y_label, gene_set=None, title=None, 
                             max_step=0.2, min_step=0.1, fig_size=(6, 5), 
                             trans=False, format='pdf', save_path=None):
    
    merged_df = pd.merge(df1, df2, on=column_name)
    merged_df = merged_df[np.isfinite(merged_df[x_label]) & np.isfinite(merged_df[y_label])]
    print("merged_df:", merged_df.shape)

    ## Create a main plot and two subplots for the histograms
    fig = plt.figure(figsize=fig_size)  
    grid = plt.GridSpec(6, 6, hspace=0.1, wspace=0.1)  
    x_hist = fig.add_subplot(grid[0, :-2])  
    main_ax = fig.add_subplot(grid[1:, :-2], sharex=x_hist)  
    y_hist = fig.add_subplot(grid[1:, -2], sharey=main_ax)  
    ## Add a subplot for the colorbar
    cax = fig.add_subplot(grid[1:, -1])  

    ## Create a scatter plot in the main plot
    xy = np.vstack([merged_df[x_label], merged_df[y_label]])
    density = gaussian_kde(xy)(xy)
    norm = clr.Normalize(vmin=density.min(), vmax=density.max())
    colors = cm.viridis(norm(density))
    scatter = main_ax.scatter(merged_df[x_label], merged_df[y_label], 
                              c=colors, alpha=0.7, label='Data points')

    ## Created automatically.
    main_ax.xaxis.set_major_locator(MultipleLocator(max_step))
    main_ax.xaxis.set_major_formatter('{x:.1f}')
    main_ax.xaxis.set_minor_locator(MultipleLocator(min_step))
    main_ax.yaxis.set_major_locator(MultipleLocator(max_step))
    main_ax.yaxis.set_major_formatter('{x:.1f}')
    main_ax.yaxis.set_minor_locator(MultipleLocator(min_step))

    ## Adjust the font size of the tick labels
    main_ax.tick_params(axis='both', which='major', labelsize=12)
    ## Add a colorbar
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax, shrink=0.5)
    # cb.set_label('Density')

    ## Add histograms for x and y data
    sns.histplot(x=merged_df[x_label], ax=x_hist, kde=True, color='gray', 
                 legend=False, bins=30, alpha=0.6)
    sns.histplot(y=merged_df[y_label], ax=y_hist, kde=True, color='gray', 
                 legend=False, bins=30, alpha=0.6, orientation='horizontal')
    # Remove x-axis tick labels for the x_hist
    plt.setp(x_hist.get_xticklabels(), visible=False)
    plt.setp(y_hist.get_yticklabels(), visible=False)
    ## Remove x-axis tick labels
    x_hist.set_xlabel("")
    y_hist.set_ylabel("")

    # Adjust the font size of the tick labels
    x_hist.tick_params(axis='both', which='major', labelsize=12)
    # x_hist.tick_params(axis='both', which='minor', labelsize=10)
    y_hist.tick_params(axis='both', which='major', labelsize=12)
    # y_hist.tick_params(axis='both', which='minor', labelsize=10)

    ## mark genes
    # genes_to_annotate = ['TGFB1', 'BMP2']
    genes_to_annotate = gene_set if gene_set is not None else []
    for gene in genes_to_annotate:
        gene_data = merged_df.loc[merged_df[column_name] == gene, [x_label, y_label]]
        if not gene_data.empty:
            main_ax.scatter(*gene_data.values[0], c='red', alpha=0.7)
            main_ax.annotate(gene, gene_data.values[0], textcoords="offset points", 
                             xytext=(0, 5), ha='center', fontsize=14, color='red')

    ## setting
    main_ax.set_xlabel(x_label, fontsize=14)
    main_ax.set_ylabel(y_label, fontsize=14)

    ## add Diagonal line
    min_value = min(merged_df[x_label].min(), merged_df[y_label].min())
    max_value = max(merged_df[x_label].max(), merged_df[y_label].max())
    main_ax.axline((min_value, min_value), (max_value, max_value), linestyle='--', 
                   color='gray', label='Diagonal line')
    
    sns.despine()
    plt.gcf().set_dpi(180)
    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')
    plt.show


def plot_SSIM_revised(df1, df2, column_name, x_label, y_label, gene_set=None, title=None,
                      max_step=0.1, min_step=0.1, fig_size=(6, 5), 
                      trans=False, format='pdf', save_path=None):
    
    merged_df = pd.merge(df1, df2, on=column_name)
    merged_df = merged_df[np.isfinite(merged_df[x_label]) & np.isfinite(merged_df[y_label])]
    print("merged_df:", merged_df.shape)

    ## Create a main plot and two subplots for the histograms
    fig = plt.figure(figsize=fig_size)  # 
    grid = plt.GridSpec(6, 6, hspace=0.1, wspace=0.1)  
    x_hist = fig.add_subplot(grid[0, :-2])  
    main_ax = fig.add_subplot(grid[1:, :-2], sharex=x_hist)  
    y_hist = fig.add_subplot(grid[1:, -2], sharey=main_ax)  
    # Add a subplot for the colorbar
    cax = fig.add_subplot(grid[1:, -1])  

    ## Create a scatter plot in the main plot
    xy = np.vstack([merged_df[x_label], merged_df[y_label]])
    density = gaussian_kde(xy)(xy)
    norm = clr.Normalize(vmin=density.min(), vmax=density.max())
    colors = cnt_color(norm(density))

    scatter = main_ax.scatter(merged_df[x_label], merged_df[y_label], 
                              c=colors, alpha=0.7, label='Data points')

    # Set major and minor ticks
    main_ax.xaxis.set_major_locator(MultipleLocator(max_step))
    main_ax.xaxis.set_major_formatter('{x:.1f}')
    main_ax.xaxis.set_minor_locator(MultipleLocator(min_step))
    main_ax.yaxis.set_major_locator(MultipleLocator(max_step))
    main_ax.yaxis.set_major_formatter('{x:.1f}')
    main_ax.yaxis.set_minor_locator(MultipleLocator(min_step))

    # Adjust the font size of the tick labels
    main_ax.tick_params(axis='both', which='major', labelsize=12)
    # main_ax.tick_params(axis='both', which='minor', labelsize=10)
    
    ## Add a colorbar  
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cnt_color), cax=cax, shrink=0.5) # 'viridis'
    # cb.set_label('Density')

    ## Add histograms for x and y data
    sns.histplot(x=merged_df[x_label], ax=x_hist, kde=True, color='gray', 
                 legend=False, bins=30, alpha=0.6)
    sns.histplot(y=merged_df[y_label], ax=y_hist, kde=True, color='gray', 
                 legend=False, bins=30, alpha=0.6, orientation='horizontal')

    ## Remove x-axis tick labels for the x_hist
    plt.setp(x_hist.get_xticklabels(), visible=False)
    plt.setp(y_hist.get_yticklabels(), visible=False)
    ## Remove x-axis tick labels
    x_hist.set_xlabel("")
    y_hist.set_ylabel("")

    ## Adjust the font size of the tick labels
    x_hist.tick_params(axis='both', which='major', labelsize=12)
    y_hist.tick_params(axis='both', which='major', labelsize=12)
    
    ## mark genes
    genes_to_annotate = gene_set if gene_set is not None else []
    for gene in genes_to_annotate:
        gene_data = merged_df.loc[merged_df[column_name] == gene, [x_label, y_label]]
        if not gene_data.empty:
            main_ax.scatter(*gene_data.values[0], c='red', alpha=0.7)
            main_ax.annotate(gene, gene_data.values[0], textcoords="offset points", 
                             xytext=(0, 5), ha='center', fontsize=14, color='red')

    ## setting
    main_ax.set_xlabel(x_label, fontsize=14)
    main_ax.set_ylabel(y_label, fontsize=14)

    ## add Diagonal line
    min_value = min(merged_df[x_label].min(), merged_df[y_label].min())
    max_value = max(merged_df[x_label].max(), merged_df[y_label].max())
    main_ax.axline((min_value, min_value), (max_value, max_value), linestyle='--', 
                   color='gray', label='Diagonal line')

    sns.despine()
    plt.gcf().set_dpi(180)
    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')
    plt.show

#################################################
# 2024.11.28 add Sankey plot: Ligand-Receptor-TF 
#################################################
def sankey_LR2TF2TG(subdf, width=600, height=400, title='Pattern 0', alpha_color=0.6,
                    save_path=None, fig_format='svg'):
    """
    Create Sankey diagram from ligand-receptor-TF data.
    Parameters:
        subdf: a DataFrame with 'Ligand_symbol', 'Receptor_symbol', 'TF' and 'value' columns
        save_path: the path to save the SVG file
    """
    ## Create lists of unique node labels and their indices
    node_label = list(set(subdf['Ligand_symbol'].tolist() + 
                          subdf['Receptor_symbol'].tolist() + 
                          subdf['TF'].tolist()+ 
                          subdf['Target'].tolist()))
    
    source = ([node_label.index(i) for i in subdf['Ligand_symbol'].tolist()] + 
              [node_label.index(i) for i in subdf['Receptor_symbol'].tolist()] + 
              [node_label.index(i) for i in subdf['TF'].tolist()])
    
    target = ([node_label.index(i) for i in subdf['Receptor_symbol'].tolist()] + 
              [node_label.index(i) for i in subdf['TF'].tolist()] + 
              [node_label.index(i) for i in subdf['Target'].tolist()])
    
    value = subdf['value'].tolist() * 3  # adjust the multiplication factor
    

    ## Load color data from online JSON file
    url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())

    #########################
    # 70%  color
    #########################
    def adjust_alpha(color, alpha=alpha_color):
        # Check if color is in rgba format
        if color.startswith('rgba'):
            color = color.rstrip(')').split(',')
            color[-1] = str(alpha)
            color = ','.join(color) + ')'
        # Check if color is in rgb format
        elif color.startswith('rgb'):
            color = color.rstrip(')').split(',')
            color.append(str(alpha))
            color = ','.join(color) + ')'
        return color
    
    mycol_vector_list = [adjust_alpha(color) for color in data['data'][0]['node']['color']]

    ## Create Sankey diagram
    data_trace = go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = node_label,
            color = mycol_vector_list
        ),
        link = dict(
            source = source,
            target = target,
            value = value,
            label = node_label,
            color = mycol_vector_list 
        )
    )

    fig = go.Figure(data=data_trace)

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        title=title, 
        annotations=[
            go.layout.Annotation(
                text="Ligand",
                align='center',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.0,
                y=-0.15,
                font=dict(size=15)
            ),
            go.layout.Annotation(
                text="Receptor",
                align='center',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.25,
                y=-0.15,
                font=dict(size=15)
            ),
            go.layout.Annotation(
                text="TF",
                align='center',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.68,
                y=-0.15,
                font=dict(size=15)
            ),
            go.layout.Annotation(
                text="Target",
                align='center',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.0,
                y=-0.15,
                font=dict(size=15)
            )
        ]
    )

    ## Save SVG figure
    if save_path is not None and fig_format != 'html':
        pio.write_image(fig, save_path, format=fig_format)
    else:
        fig_obj = go.Figure(fig)
        fig_obj.write_html(str(save_path) + '_sankey_diagram.html')
        with open(str(save_path) + '_sankey_diagram.html', 'r') as f:
            html_string = f.read()

        display(HTML(html_string))
        

def sankey_LR2TF(subdf, width=600, height=400, title='Pattern 0', save_path=None, fig_format='svg'):
    """
    Create a Sankey diagram from ligand-receptor-TF data.
    Parameters:
        subdf: a DataFrame with 'Ligand_symbol', 'Receptor_symbol', 'TF' and 'value' columns
        save_path: the path to save the SVG file
    """
    ## Create lists of unique node labels and their indices
    node_label = list(set(subdf['Ligand_symbol'].tolist() + subdf['Receptor_symbol'].tolist() + subdf['TF'].tolist()))
    source = [node_label.index(i) for i in subdf['Ligand_symbol'].tolist()] + [node_label.index(i) for i in subdf['Receptor_symbol'].tolist()]
    target = [node_label.index(i) for i in subdf['Receptor_symbol'].tolist()] + [node_label.index(i) for i in subdf['TF'].tolist()]
    value = subdf['value'].tolist() * 2  # assuming the value for both edges is the same

    ## Load color data from online JSON file
    url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    ## override gray link colors with 'source' colors
    mycol_vector_list = ['rgba(255,0,255, 0.8)' if color == "magenta" else color for color in data['data'][0]['node']['color']]

    ## Create Sankey diagram
    data_trace = go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = node_label,
            color = mycol_vector_list
        ),
        link = dict(
            source = source,
            target = target,
            value = value,
            label = node_label,
            color = mycol_vector_list 
        )
    )

    fig = go.Figure(data=data_trace)

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        title=title, 
        annotations=[
            go.layout.Annotation(
                text="Ligand",
                align='center',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.0,
                y=-0.15,
                font=dict(size=15)
            ),
            go.layout.Annotation(
                text="Receptor",
                align='center',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.15,
                font=dict(size=15)
            ),
            go.layout.Annotation(
                text="TF",
                align='center',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.0,
                y=-0.15,
                font=dict(size=15)
            )
        ]
    )

    # Save SVG figure
    if save_path is not None and fig_format != 'html':
        pio.write_image(fig, save_path, format=fig_format)
    else:
        fig_obj = go.Figure(fig)
        fig_obj.write_html(str(save_path) + 'sankey_diagram.html')

        from IPython.display import display, HTML

        with open(str(save_path) + 'sankey_diagram.html', 'r') as f:
            html_string = f.read()

        display(HTML(html_string))


def plot_time_bars(time, bar_height=0.25, fig_size=(5, 4),
                   inter_value_l=40, inter_value_r=90, end=180,
                   trans=False, format='pdf', save_path=None):

    ## Set position of bar on Y axis
    r = [np.arange(len(time)) + i*bar_height for i in range(len(time.columns[1:]))]

    fig, ax = plt.subplots(figsize=fig_size)
    colors = ['#80CA80', '#BFB0D5', '#D7DF04', '#F8BE82', '#ADD8E6', '#6F6F6F']

    labels_added = []
    methods = time.columns[1:]

    for i, method in enumerate(methods):
        ax.barh(r[i], time[method], color=colors[i], height=bar_height, edgecolor='grey', label=method)

    ## Add yticks on the middle of the group bars
    ax.set_yticks([r[i][0] + 2.0*bar_height for i in range(len(time))])
    ax.set_yticklabels(time['Task'], fontsize=12)
    ax.invert_yaxis()

    ## Create legend & Show graphic
    ax.legend(loc=4, fontsize=12)

    ## Add grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # #Add labels and title
    ax.set_xlabel("Time", fontsize=12)
    # ax.set_ylabel("Task", fontsize=12)
    ax.set_title("Time Bar Plot", fontsize=12)

    ## Set xticks
    ax.set_xticks(np.arange(0, end+1, 20))
    ax.set_xticklabels(np.arange(0, end+1, 20), fontsize=12)

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')
        
    plt.rcParams['svg.fonttype'] = 'none'
    plt.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True)
    plt.show()


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


def dot(pathway_res, figsize, markersize, pdf, step=4):
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
        xticks_positions = np.arange(0, max(result1.selected.values) + 2, step)  
        dotplot.set_xticks(xticks_positions)
        dotplot.set_xticklabels(xticks_positions)

        dotplot.tick_params(axis='y', labelsize=10)
        dotplot.set_title(name)
        plt.colorbar(im, location='bottom', label='Percentage of pairs out of CellChatDB')
        #                 dotplot.tight_layout()
        plt.gcf().set_dpi(150)

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
        size_legend_ax.set_title('Fisher exact p-value (right tile)', y=ymax + 0.9, size='small')

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


####################################
# 2025.02.07 add the p_value cutoff
####################################
def dot_path(adata, uns_key=None, dic=None, num_cutoff=1, p_cutoff=None, 
             groups=None, markersize=50, step=4, figsize=(6, 8), pdf=None, **kwargs):
    """
    Either input a dict containing lists of interactions, or specify a dict key in adata.uns
    :param adata: AnnData object.
    :param uns_key: a dict key in adata.uns
    :param dic: a dict containing 1 or more list(s) of interactions
    :param num_cutoff: Minimum number of spots to be plotted.
    :param p_cutoff: p-value cutoff for filtering interactions.
    :param groups: subgroups from all dict keys.
    :param markersize: Size of the markers in the plot.
    :param step: Step size for plotting.
    :param figsize: Size of the figure.
    :param pdf: Export PDF under your current directory.
    :param kwargs: Additional keyword arguments.
    :return: None
    """
    if uns_key is not None:
        dic = {uns_key: adata.uns[uns_key]}
    pathway_res = compute_pathway(adata, dic=dic)

    ## Apply the num_cutoff filter
    pathway_res = pathway_res[pathway_res.selected >= num_cutoff]
    ## Apply the p_cutoff filter if provided
    if p_cutoff is not None:
        pathway_res = pathway_res[pathway_res.fisher_p <= p_cutoff]

    if groups is not None:
        pathway_res = pathway_res.loc[pathway_res.name.isin(groups)]
    # pathway_res = pathway_res.loc[pathway_res.name.isin(groups)]
    n_subplot = len(pathway_res.name.unique())
    if pdf != None:
        with PdfPages(pdf + '.pdf') as pdf:
            dot(pathway_res, figsize, markersize, pdf, step)
            plt.show()
            plt.close()
    else:
        dot(pathway_res, figsize, markersize, pdf, step)


#################################################
# 2024.11.12 add for pathway confusion matrix
#################################################
def plot_conf_mat(result_pattern_all, pattern_name='Pattern_0', pathway_name='WNT', 
                  font=14, save_path=None):
    result_pattern = result_pattern_all[result_pattern_all['name'] == pattern_name]

    confusion_matrix = np.array([
        [
            result_pattern.loc[pathway_name, 'overlap_size'], 
            result_pattern.loc[pathway_name, 'query_set_size'] - 
            result_pattern.loc[pathway_name, 'overlap_size']
        ],
        [
            result_pattern.loc[pathway_name, 'module_size'] - 
            result_pattern.loc[pathway_name, 'overlap_size'], 
            result_pattern.loc[pathway_name, 'negneg']
        ]
    ])

    ## confusion matirix heatmap
    fig, ax = plt.subplots(figsize=(3.8,3.0))
    cax = sns.heatmap(confusion_matrix, annot=True, cmap="coolwarm", fmt='d', 
                    xticklabels=[str(pathway_name), "Others"], 
                    yticklabels=[pattern_name, "Others"],
                    annot_kws={"size": 16}  # Adjust size here
                    )

    ## Adjust tick size
    ax.tick_params(axis='both', which='major', labelsize=font)

    ## Adjust colorbar size
    cbar = cax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font)

    plt.title(str(pathway_name)+" pathway")
    plt.axis('equal')
    plt.gcf().set_dpi(150)

    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


###################################
# 2024.11.12 adjust for spatialDM
###################################
def spatialDE_clusters(histology_results, patterns, spatialxy, w=None, marker='s', s=10,
                       figsize=(21,5), trans=False, format='pdf', save_path=None):
    plt.figure(figsize=figsize)
    for i in range(w):
        plt.subplot(1, w, i + 1)
        if isinstance(patterns.columns, pd.RangeIndex):
            scatter = plt.scatter(spatialxy[:,0], spatialxy[:,1], marker = marker, 
                                  c=patterns[i], cmap="viridis", s=s)
        else:
            scatter = plt.scatter(spatialxy[:,0], spatialxy[:,1], marker = marker, 
                                  c=patterns[str(i)], cmap="viridis", s=s)
        colorbar = plt.colorbar(scatter)
        colorbar.ax.tick_params(labelsize=14)  # Adjust colorbar tick label size here
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.title('Pattern {} - {} LR pairs'.format(i, 
                                                    histology_results.query('pattern == @i').shape[0]))
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.gcf().set_dpi(300)

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')

    plt.show()


###################################
# 2024.11.12 adjust for sparseAEH
###################################
def sparseAEH_clusters(gaussian_subspot, label='counts', w=None, s=5, marker='s', resolution=None,
                       trans=False, format='pdf', save_path=None):
    k = gaussian_subspot.K
    h = np.ceil(k / w).astype(int)  # Calculate the number of rows
    
    if resolution is not None:
        if w == 3 and h == 2:
            plt.figure(figsize=(21,11))
        elif w == 3:
            plt.figure(figsize=(19,5))
        elif w == 2:
            plt.figure(figsize=(14,5))
        else:
            plt.figure(figsize=(7,5))
    else:
        ## Adjust figure size based on the number of columns
        if w == 3 and h == 2:
            plt.figure(figsize=(21,11))
        elif w == 3:
            plt.figure(figsize=(21,5))
        elif w == 2:
            plt.figure(figsize=(14,5))
        else:
            plt.figure(figsize=(7,5))

    for i in range(gaussian_subspot.K):
        plt.subplot(h, w, i + 1)
        scatter = plt.scatter(gaussian_subspot.kernel.spatial[:,0],
                            gaussian_subspot.kernel.spatial[:,1], 
                            c=gaussian_subspot.mean[:,i], cmap="viridis",
                            marker=marker, s=s)
        colorbar = plt.colorbar(scatter)
        colorbar.ax.tick_params(labelsize=14)  # Adjust colorbar tick label size here
        plt.axis('equal')
        plt.gca().invert_yaxis()
        if label == 'counts':
            plt.title('Pattern {} - {} LR pairs'.format(i, np.sum(gaussian_subspot.labels==i)))
        else:
            plt.title('Pattern {} - {} LR pairs'.format(i, gaussian_subspot.pi[i]))
        plt.gcf().set_dpi(300)
        plt.tick_params(axis='both', which='major', labelsize=14)

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format, dpi=300, bbox_inches='tight')

    plt.show()


#############################
# 2025.02.07 update font size
#############################
def plot_selected_pair_dot(sample, pair, spots, selected_ind, figsize, cmap, cmap_l, cmap_r, 
                           marker, marker_size, title_font_size=16, tick_font_size=16, **kwargs):
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
    scatter = plt.scatter(spatial_loc[:,0], spatial_loc[:,1], c=spots.loc[pair], cmap=cmap,
                vmax=1, marker=marker, s=marker_size, **kwargs)
    colorbar = plt.colorbar(scatter)
    colorbar.ax.tick_params(labelsize=tick_font_size)
    plt_util_invert('Moran: ' + str(sample.uns['local_stat']['n_spots'].loc[pair]) + ' spots', 
                    title_font_size=title_font_size, tick_font_size=tick_font_size)
    
    for l in range(l1):
        plt.subplot(1, 5, 2 + l)
        scatter = plt.scatter(spatial_loc[:,0], spatial_loc[:,1], c=sample[:,L[l]].X.toarray(),
                    cmap=cmap_l, marker=marker, s=marker_size, **kwargs)
        colorbar = plt.colorbar(scatter)
        colorbar.ax.tick_params(labelsize=tick_font_size)
        plt_util_invert('Ligand: ' + L[l],  title_font_size=title_font_size, tick_font_size=tick_font_size)
    for l in range(l2):
        plt.subplot(1, 5, 2 + l1 + l)
        scatter = plt.scatter(spatial_loc[:,0], spatial_loc[:,1], c=sample[:,R[l]].X.toarray(),
                    cmap=cmap_r, marker=marker, s=marker_size, **kwargs)
        colorbar = plt.colorbar(scatter)
        colorbar.ax.tick_params(labelsize=tick_font_size)
        plt_util_invert('Receptor: ' + R[l],  title_font_size=title_font_size, tick_font_size=tick_font_size)


def plt_util_invert(title, title_font_size=14, tick_font_size=14):
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=title_font_size)
    plt.tick_params(axis='both', labelsize=tick_font_size)
    plt.gca().invert_yaxis()


def plot_pairs_dot(sample, pairs_to_plot, pdf=None, trans=False, figsize=(56, 8),
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
                pdf.savefig(transparent=trans)
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
def gene_expr_allspots(gene, spatial_loc_all, recon_ref_adata_image_f2, 
                       gene_hv, label, marker='h', s=8, figsize=(9, 7), save_path=None):
    def plot_gene_data_dot(spatial_loc, genedata, title, ax, s):
        scatter = ax.scatter(spatial_loc[:,0], spatial_loc[:,1], c=genedata, 
                             cmap=cnt_color, marker=marker, s=s)   
        ax.invert_yaxis()
        ax.set_title(title)
        return scatter

    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(recon_ref_adata_image_f2, anndata.AnnData):
        genedata3 = recon_ref_adata_image_f2.to_df()[[gene]].to_numpy()
    else:
        genedata3 = pd.DataFrame(recon_ref_adata_image_f2, columns=gene_hv)[[gene]].to_numpy()

    print(f"{gene} gene expression dim: {genedata3.shape}")
    print(f"{gene} gene expression: \n {genedata3}")
    scatter3 = plot_gene_data_dot(spatial_loc_all, genedata3, f'{gene} expression: {label}', ax, s) 
    fig.colorbar(scatter3, ax=ax)

    # Save the figure if a save path is provided
    if save_path is not None:
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

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

    if save_path is not None:
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')


###########################################
# 2025.01.29 Add the infer and impt FineST
###########################################
def gene_expr(adata, matrix_order_df, gene_selet, marker='h', s=22, 
              figsize=(9, 7), cnt_color=None, 
              trans=False, format='pdf', save_path=None):
    if isinstance(matrix_order_df, pd.DataFrame):
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        scatter_plot = ax1.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], 
                                   c=matrix_order_df[gene_selet], cmap=cnt_color, 
                                   marker=marker, s=s) 
    
    elif isinstance(matrix_order_df, torch.Tensor):
        matrix_order_df = pd.DataFrame(matrix_order_df.numpy(), columns=adata.var_names)

        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        scatter_plot = ax1.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], 
                                   c=matrix_order_df[[gene_selet]].to_numpy(), cmap=cnt_color, 
                                   marker=marker, s=s) 
    ax1.invert_yaxis()
    ax1.set_title(str(gene_selet)+' Expression')
    cbar = fig.colorbar(scatter_plot, ax=ax1)

    if max(figsize) >=9:
        ## Set tick parameters
        ax1.tick_params(axis='both', which='major', labelsize=18)
        cbar.ax.tick_params(labelsize=18) 

    if save_path is not None:
        plt.savefig(save_path, transparent=trans, format=format,
                    dpi=300, bbox_inches='tight')
    plt.show()

def subspot_expr(C, value, patch_size=56, dataset_class=None, 
                 marker='o', s=1800, rotation=None,
                 fig_size=(2.5, 2.5), trans=False, format='pdf', save_path=None):
    fig, ax = plt.subplots(figsize=fig_size)
    scatter = ax.scatter(C[:, 0], C[:, 1], c=value, marker=marker, s=s)
    ax.set_title("First spot")

    if rotation is None:
        rotation_value = 0
    else:
        rotation_value = rotation

    ## Set `split_num` according to `dataset_class`
    if dataset_class == 'Visium16':
        split_num = 16
    elif dataset_class == 'Visium64':
        split_num = 64
    elif dataset_class == 'VisiumSC':
        split_num = 1
    elif dataset_class == 'VisiumHD':
        split_num = 4
    else:
        raise ValueError('Invalid dataset_class. Only "Visium16", '
                 '"Visium64", "VisiumSC" and "VisiumHD" are supported.')

    x_ticks = np.unique(C[:, 0])
    y_ticks = np.unique(C[:, 1])
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticks, rotation=rotation_value) 
    ax.set_yticklabels(y_ticks)

    ## Ensure that the points are evenly distributed
    ax.set_xlim([C[:, 0].min() - patch_size/(2*np.sqrt(split_num)), C[:, 0].max() + patch_size/(2*np.sqrt(split_num))])
    ax.set_ylim([C[:, 1].min() - patch_size/(2*np.sqrt(split_num)), C[:, 1].max() + patch_size/(2*np.sqrt(split_num))])
    ax.set_aspect('equal', 'box') 

    if save_path is not None:
        fig.set_dpi(300)
        plt.savefig(save_path, transparent=trans, format=format, bbox_inches='tight')

    plt.show()


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

    print('mean correlation of genes: ', mean_corr_gene)

    data = pd.DataFrame({
        'Type': (
            np.repeat('corr_gene', len(corr_gene))
            if gene_only
            else np.concatenate([
                np.repeat('corr_spot', len(corr_spot)),
                np.repeat('corr_gene', len(corr_gene))
            ])
        ),
        'mean_corr': (
            corr_gene
            if gene_only
            else np.concatenate([corr_spot, corr_gene])
        )
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
