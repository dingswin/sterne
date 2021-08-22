#!/usr/bin/env python
"""
Used to make plots of the publication quality;
Written in python3 by Hao Ding; 
Created on 22 August 2021.
"""
import sys, corner
import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

def cornerplot(saved_posteriors='outdir/posterior_samples.dat',\
        output_figure='outdir/bayesian_corner_plot.pdf', **kwargs):
    """
    Input parameters
    ----------------
    kwargs : dict
        1. cornerplot_variables_dict : dict
            {parameter : cornerplot_label_name}.
            For example, {'ra_0':r"$\Delta\alpha$\,(mas)", 'px_0':r"$\varpi$\,(mas)",
                'dec_0':r"$\Delta\delta$\,(mas)", 
                'mu_a_0':r"$\mu_{\alpha}\,(\mathrm{mas~{yr}^{-1}})$",
                'mu_d_0':r"$\mu_{\delta}\,(\mathrm{mas~{yr}^{-1}})$"}.
            If not provded, parameter names will be adopted as axis label names.
    """
    cornerplot_samples = Table.read(saved_posteriors, format='ascii')
    cornerplot_samples.remove_columns(['log_likelihood','log_prior'])
    parameters = cornerplot_samples.colnames
    cornerplot_variables = []
    try:
        cornerplot_variables_dict = kwargs['cornerplot_variables_dict']
        plt.rc('text', usetex=True)
        for parameter in parameters:
            cornerplot_variables.append(cornerplot_variables_dict[parameter])
    except KeyError:
        for parameter in parameters:
            cornerplot_variables.append(parameter)
    print(cornerplot_variables)
    ## >>> convert ra/dec to offset from median ra/dec to ease illustration
    for parameter in parameters:
        if 'dec' in parameter:
            median_dec_rad = np.median(cornerplot_samples[parameter])
            cornerplot_samples[parameter] -= median_dec_rad
            cornerplot_samples[parameter] *= (u.rad).to(u.mas)
        if 'ra' in parameter:
            median_ra_rad = np.median(cornerplot_samples[parameter])
            cornerplot_samples[parameter] -= median_ra_rad
            cornerplot_samples[parameter] *= (u.rad).to(u.mas)
            cornerplot_samples[parameter] *= np.cos(median_dec_rad)
    ## <<<
    cornerplot_2darray = transfer_astropy_Table_to_2darray_accepted_by_corner(cornerplot_samples)
    
    fig = corner.corner(cornerplot_2darray, labels=cornerplot_variables,\
        label_kwargs={"fontsize": 22})
    fig.tight_layout()
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=14, pad=0)
    fig.savefig(output_figure)
    print("%s is made." % output_figure)
    
def transfer_astropy_Table_to_2darray_accepted_by_corner(table):
    parameters = table.colnames    
    array = np.array([[]]*len(table))
    for parameter in parameters:
        array = np.column_stack((array, table[parameter]))
    return array
