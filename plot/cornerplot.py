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
    Examples
    --------
    1. 8P full corner plot:    
        cornerplot.cornerplot(cornerplot_variables_dict={'ra_0':r"$\Delta \alpha$\,(mas)", 'px_0':r"$\varpi$\,(mas)", 'dec_0':r"$\Delta \delta$\,(mas)", 'mu_a_0':r"$\mu_{\delta}\,(\mathrm{mas~{yr}^{-1}})$", 'mu_d_0':r"$\mu_{\alpha}\,(\mathrm{mas~{yr}^{-1}})$", 'efac_0':r"$\eta_\mathrm{EFAC}$", 'incl_0':r"$i$\,(deg)", 'om_asc_0':r"$\Omega_\mathrm{asc}$\,(deg)"})
    
    2. Only the two orbital parameters:
        cornerplot.cornerplot(cornerplot_variables_dict={'ra_0':r"$\Delta \alpha$\,(mas)", 'px_0':r"$\varpi$\,(mas)", 'dec_0':r"$\Delta \delta$\,(mas)", 'mu_a_0: ':r"$\mu_{\delta}\,(\mathrm{mas~{yr}^{-1}})$", 'mu_d_0':r"$\mu_{\alpha}\,(\mathrm{mas~{yr}^{-1}})$", 'efac_0':r"$\eta_\mathrm{EFAC}$", 'incl_0':r"$i$\,(deg)", 'om_asc_0':r"$\Omega_\mathrm{asc}$\,(deg)"}, exclude_parameters=['ra_0','dec_0','efac_0','px_0','mu_a_0','mu_d_0'], bins=30) 
        

    Input parameters
    ----------------
    kwargs : dict
        1. cornerplot_variables_dict : dict
            {parameter : cornerplot_label_name}.
            For example, {'ra_0':r"$\Delta\alpha$\,(mas)", 'px_0':r"$\varpi$\,(mas)",
                'dec_0':r"$\Delta\delta$\,(mas)", 
                'mu_a_0':r"$\mu_{\alpha}\,(\mathrm{mas~{yr}^{-1}})$",
                'mu_d_0':r"$\mu_{\delta}\,(\mathrm{mas~{yr}^{-1}})$"}.
            If not provided, parameter names will be adopted as axis label names.
        2. truths : list
            the 'truths' parameter for corner.corner(), in alphabetical order for
            parameters. For example, if the truth parallax is 0.95 (mas), then the
            truths can be [None,None,None,0.95,None].
        3. bins : int
            number of bins for corner.corner().
        4. ranges : list
            the 'range' for corner.corner()
        5. exclude_parameters : list
            a list of parameters to not plot.
    """
    cornerplot_samples = Table.read(saved_posteriors, format='ascii')
    try:
        exclude_parameters = kwargs['exclude_parameters']
    except KeyError:
        exclude_parameters = []
    exclude_parameters += ['log_likelihood','log_prior']
    cornerplot_samples.remove_columns(exclude_parameters)
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
    try:
        truths = kwargs['truths']
    except KeyError:
        truths = [None] * len(parameters)
    try:
        bins = kwargs['bins']
    except KeyError:
        bins = 20
    try:
        ranges = kwargs['ranges']
    except KeyError:
        ranges = [1.] * len(parameters)
    
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
        if 'incl' in parameter:
            cornerplot_samples[parameter] *= 180./np.pi ## rad to deg
    ## <<<
    cornerplot_2darray = transfer_astropy_Table_to_2darray_accepted_by_corner(cornerplot_samples)
    
    fig = corner.corner(cornerplot_2darray, labels=cornerplot_variables,\
        label_kwargs={"fontsize": 22}, truths=truths, plot_contour=True, bins=bins, range=ranges)
    #corner.overplot_lines(fig, mean_values, color='g')
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
