from dataclasses import dataclass
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mpl_ticker

import seaborn as sns

# Set plotting parameters for uniform style (loaded from file)
def set_plotting_environment():
    sns.set_theme(style="ticks", palette="colorblind")

    # Load custom styleparameters from file `mscThesisPlotting.mplstyle`
    import ast
    with open('/home/tobiac/polybox/studium/mscThesis/personal_repo/data_visualization/src/mscPlotting.mplstyle', "r") as data:
        custom_rcParams = ast.literal_eval(data.read())
    
    mpl.rcParams.update(custom_rcParams)
    
    return custom_rcParams


# Line properties generator (for varying line properies, i.e. markers, linetypes)
def line_properties_generator(plot_kwargs):
    # By default uses 8 colors
    colormap = ListedColormap(sns.color_palette("colorblind").as_hex())
    markers = ['v', '^', '<', 's', 'D', 'x', 's', '1']
    index = 0
    
    contains_markers = False
    if 'markers' in plot_kwargs.keys():
        contains_markers = plot_kwargs['markers']
        
        if plot_kwargs['markers'] == False:
            plot_kwargs.pop('markers')
    
    while True:
        plot_kwargs['color'] = colormap.colors[index]
        
        if contains_markers:
            plot_kwargs['marker'] = markers[index]

        yield plot_kwargs
        index = (index + 1) % len(colormap.colors)

# Define fontsizes for plots
@dataclass
class SizeParams:
    ticksize: float = 14
    label_fontsize: float = 16
    label_linesize: float = 1.5
    label_pad: float = 15
    title_fontsize: float = 18
    title_pad: float = 15
    spine_linewidth: float = 1
        
# Plot lines of a given slope (for convergence plots)
def order_lines(axes, sizes, intercepts, slopes):
    if not isinstance(axes, np.ndarray):
        ax_arr = np.array([axes,])
    else:
        ax_arr = axes
    
    for ax in ax_arr:
        for intercept, slope in zip(intercepts, slopes):
            order = np.zeros(len(sizes))
            factor = intercept*(sizes[0]**slope)
            for i, elem in enumerate(sizes):
                order[i] = factor/(elem**slope)

            ax.plot(sizes, order, '--', color='gray', label=f'Order {slope}')

# Plot spectral convergence lines of a given base
def order_exp(axes, sizes, intercepts, base):
    if not isinstance(axes, np.ndarray):
        ax_arr = np.array([axes,])
    else:
        ax_arr = axes
    
    # Generate points on [min, max] of x-axis    
    sizes = np.linspace(min(sizes), max(sizes), 100)
    
    for ax in ax_arr:
        for intercept, slope in zip(intercepts, slopes):
            order = np.zeros(len(sizes))
            factor = intercept*(base**sizes[0])
            for i, elem in enumerate(sizes):
                order[i] = factor/(base**elem)

            ax.plot(sizes, order, '--', color='gray', label=f'Order $\textnormal{N^{slope}}$')

# Set axis labels / ticks and legend for convergence plots
def prepare_convergence_plots(axes, x_ticks, labels=None, x_label=r'Mesh Points', y_label=r'$\eta$'):
    if not isinstance(axes, np.ndarray):
        ax_arr = np.array([axes,])
    else:
        ax_arr = axes
    
    size_params = SizeParams()

    for ax in ax_arr:
        ax.set_xlabel(x_label, fontsize=size_params.label_fontsize, labelpad=size_params.label_pad)
        ax.set_ylabel(y_label, fontsize=size_params.label_fontsize, labelpad=size_params.label_pad)

        ax.tick_params(axis='both', labelsize=size_params.ticksize, color='gray')

        ax.tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
        ax.set_xticks(x_ticks)
        ax.get_xaxis().set_major_formatter(mpl_ticker.ScalarFormatter())
        
        # Turn off minor ticks
        ax.minorticks_off()
        
        # Change color of border of axis
        for spine in ax.spines:
            ax.spines[spine].set_color('gray')
            ax.spines[spine].set_linewidth(1)
        
        # Override legend labels with latex font
        if labels is None:
            _, labels = ax.get_legend_handles_labels()
        leg = ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=3, fancybox=True, shadow=True, fontsize=SizeParams().ticksize)
        
        # Set larger lines inside the legend box (to make them more visible)
        leg_lines = leg.get_lines()
        # bulk-set the properties of all lines and texts
        plt.setp(leg_lines, linewidth=size_params.label_linesize)
        
# Set axis labels / ticks and legend for asymptotic fall-off plots
def prepare_asymptotic_plots(axes, x_ticks, labels=None, x_label=r'Mesh Points', y_label=r'$\eta$'):
    if not isinstance(axes, np.ndarray):
        ax_arr = np.array([axes,])
    else:
        ax_arr = axes
    
    size_params = SizeParams()

    for ax in ax_arr:
        ax.set_xlabel(x_label, fontsize=size_params.label_fontsize, labelpad=size_params.label_pad)
        ax.set_ylabel(y_label, fontsize=size_params.label_fontsize, labelpad=size_params.label_pad)

        ax.tick_params(axis='both', labelsize=size_params.ticksize, color='gray')
        
        # Turn off minor ticks
        ax.minorticks_off()

        # ax.tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
        # ax.set_xticks(x_ticks)
        # ax.get_xaxis().set_major_formatter(mpl_ticker.ScalarFormatter())
        
        # Change color of border of axis
        for spine in ax.spines:
            ax.spines[spine].set_color('gray')
            ax.spines[spine].set_linewidth(1)
        
        # Override legend labels with latex font
        if labels is None:
            _ , labels = ax.get_legend_handles_labels()
        leg = ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=3, fancybox=True, shadow=True, fontsize=size_params.ticksize)
             
        # Set larger lines inside the legend box (to make them more visible)
        leg_lines = leg.get_lines()
        # bulk-set the properties of all lines and texts
        plt.setp(leg_lines, linewidth=size_params.label_linesize)


# Set axis labels / ticks and legend for convergence plots
def prepare_emittance_plots(axes, x_label, y_label, plot_limit=True, legend_loc='right', bbox_to_anchor=None, ncol=1):
    if not isinstance(axes, np.ndarray):
        ax_arr = np.array([axes,])
    else:
        ax_arr = axes
    
    size_params = SizeParams()

    for ax in ax_arr:
        
        if plot_limit is True:
            ax.axhline(0.491, linestyle='dashed', color='grey')

        ax.set_xlabel(x_label, fontsize=size_params.label_fontsize, labelpad=size_params.label_pad)
        ax.set_ylabel(y_label, fontsize=size_params.label_fontsize, labelpad=size_params.label_pad)

        ax.tick_params(axis='both', labelsize=size_params.ticksize, color='gray')
        
        # Change color of border of axis
        for spine in ax.spines:
            ax.spines[spine].set_color('gray')
            ax.spines[spine].set_linewidth(1)

        
        if legend_loc == 'right':
            if bbox_to_anchor is None:
                bbox_to_anchor = (1.1, 0.5)
            leg = ax.legend(loc='center left', bbox_to_anchor=bbox_to_anchor, \
                            ncols=ncol, fancybox=True, shadow=True, \
                            fontsize=size_params.ticksize)
        elif legend_loc == 'top':
            if bbox_to_anchor is None:
                bbox_to_anchor = (0.5, 1.5)
            leg = ax.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, \
                            ncols=ncol, fancybox=True, shadow=True, \
                            fontsize=size_params.ticksize)
        else:
            raise ValueError('Passed argument for "legend_loc" not supported.')
                
        # Set larger lines inside the legend box (to make them more visible)
        leg_lines = leg.get_lines()
        # bulk-set the properties of all lines and texts
        plt.setp(leg_lines, linewidth=size_params.label_linesize)
        
        
# Set axis labels / ticks and legend for imshow plots
def prepare_imshow_plots(axes, x_label, y_label, plot_limit=True, legend_loc='right', bbox_to_anchor=None, ncol=1):
    if not isinstance(axes, np.ndarray):
        ax_arr = np.array([axes,])
    else:
        ax_arr = axes
    
    size_params = SizeParams()

    for ax in ax_arr:
        
        if plot_limit is True:
            ax.axhline(0.491, linestyle='dashed', color='gray')

        ax.set_xlabel(x_label, fontsize=size_params.label_fontsize, labelpad=size_params.label_pad)
        ax.set_ylabel(y_label, fontsize=size_params.label_fontsize, labelpad=size_params.label_pad)

        ax.tick_params(axis='both', labelsize=size_params.ticksize, color='gray')
        
        # Change color of border of axis
        for spine in ax.spines:
            ax.spines[spine].set_color('gray')
            ax.spines[spine].set_linewidth(1)
        