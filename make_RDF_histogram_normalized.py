import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm
import scipy.stats as stats
import matplotlib.mlab as mlab
from fractions import Fraction

def make_plot( plot_option ):

    """ This section relevant only to geometry analysis. Variables overwritten for diameter analysis """
    x_left = 0
    x_right = 350
    y_max = 2.0
    y_min = 0.0

    """ Make the Y-axis fractions """
#    y_range = np.arange(0,59,1)
#    y_range = y_range[0:6]
#    y_fracs = y_range * (1/59.)
#    y_tick = np.array([])
#    for index, item in enumerate(y_range):
#        y_tick = np.append(y_tick,Fraction(numerator=index, denominator=59))


    # Setting up custom bin edges. Want range of 0.8-1.2 and 80 (ish) bins
    my_binedges = np.linspace(0,351, num=352)
#    my_binedges = my_binedges - 0.0025

    # Plot size for geometry analysis
    plt.figure(figsize=(6,2))

    if plot_option == 'RDFs_test':
        my_data = np.loadtxt('one_particle_evaluated.txt')
        my_title = 'Radial Distribution Function'
        #my_color = 'blue'
        my_color = 'grey'
        my_alpha = 1.0
        inline_title = 'Theoretical'

    if plot_option == 'RDFs_norm':
        my_data = np.loadtxt('B19_5fa_RDFs.txt')
        my_title = 'Radial Distribution Function'
        #my_color = 'blue'
        my_color = 'grey'
        my_alpha = 1.0
        inline_title = 'Normalized'

    my_hist, bin_edges = np.histogram(my_data)
    print( my_hist )

    norm_hist, norm_bin_edges = np.histogram(np.loadtxt('one_particle_evaluated.txt'),my_binedges)

    # For gaussian curve
#    mu, std = norm.fit(my_data)
#    percentile_5 = np.percentile(my_data, 5)
#    percentile_10 = np.percentile(my_data, 10)
#    percentile_90 = np.percentile(my_data, 90)
#    percentile_95 = np.percentile(my_data, 95)
#    median = np.median(my_data)

    N_points = len(my_data)
    N_unique = np.unique(my_data).shape
    #n_bins = 100

    """ Modified from stackoverflow.com/questions/38650550 """
    print(my_binedges)
#    print(median)

    # Histogram:
    # Bin it
    #n, bin_edges = np.histogram(my_data, n_bins)
    n, bin_edges = np.histogram(my_data, my_binedges)
    n_bins = len(my_binedges) - 1
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = n/float(n.sum())
    norm_bin_probability = norm_hist/float(norm_hist.sum())


    # Get the mid points of every bin
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
    # Compute the bin-width
    bin_width = bin_edges[1]-bin_edges[0]
    # Plot the histogram as a bar plot
#    plt.bar(bin_middles, bin_probability, width=bin_width, color=my_color, alpha=my_alpha)
    plt.bar(bin_middles, bin_probability/norm_bin_probability, width=bin_width, color=my_color, alpha=my_alpha)

    # Fit to normal distribution
#    (mu, sigma) = stats.norm.fit(my_data)
    # The pdf should not normed anymore but scaled the same way as the data
#    y = mlab.normpdf(bin_middles, mu, sigma)*bin_width
#    l = plt.plot(bin_middles, y, 'r', linewidth=2, color='black')

    plt.grid(True, alpha=0.2)
    """ End stackoverflow code """

    # Plot it
    xmin, xmax = (x_left, x_right)
    plt.xlim(x_left, x_right)
    plt.ylim(y_min, y_max)
#    plt.yticks(np.unique(y_fracs),y_tick)

    ### Geometry-specific
    #plt.title( my_title )

    plt.xlabel('Ångstroms', size='16')
    plt.ylabel('Observed/Expected', size='14')
    plt.axhline(y=1, linewidth=0.5, color='black')

#    annotation = " mean: %.3f\n std: %.3f\n bin width: %.3f\n datapoints: %.0f\n\n  5th percentile: %.3f\n 95th percentile: %.3f" % (np.mean(my_data), np.std(my_data), bin_width, N_points, percentile_5, percentile_95)


#    if geometry != 'diameters':
#        plt.annotate(annotation, xy=(0.65,0.25), xycoords='axes fraction')
#        plt.annotate(inline_title, xy=(0.05,0.7), xycoords='axes fraction', size='20', bbox=dict(boxstyle="round", fc="w"))
#    elif geometry == 'diameters':
#        plt.annotate(annotation, xy=(0.65,0.45), xycoords='axes fraction')
#        #plt.annotate(inline_title, xy=(0.05,0.85), xycoords='axes fraction', size='20', bbox=dict(boxstyle="round", fc="w"))

    plt.annotate(inline_title, xy=(0.05,0.75), xycoords='axes fraction', size='16', bbox=dict(boxstyle="round", fc="w"))
    #plt.annotate(inline_title, xy=(0.65,0.75), xycoords='axes fraction', size='16', bbox=dict(boxstyle="round", fc="w"))


    print(my_data, N_points, N_unique[0])
#    print(percentile_5, percentile_10, percentile_90, percentile_95)
    print(np.mean(my_data))

    #plt.xaxis.set_major_locator(ticker.MultipleLocator(0.1))


    #plt.show()

    my_imagename = ''.join([plot_option, '_histogram_3digits.png'])

    plt.savefig(my_imagename)

def main(args):

    make_plot( args.plot_option )

    sys.exit()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--plot_option", required=True, choices=['RDFs_norm'])

    sys.exit(main(parser.parse_args()))

