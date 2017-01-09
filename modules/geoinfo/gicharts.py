#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This geoinfo module contains plot functions using matplotlib
1. Class GICharts:

- scatter_plot(): Create a regression plot.
- line_plot()
- rspec_plot()

Usage:
1. Import:
from geoinfo.gicharts import GICharts

2. Call the method/function:
GICharts.scatter_plot()
GICharts.line_plot()
GICharts.rspec_plot()

2. Further functions:

bar_plot(): Plotting bars
barh_plot(): Plotting horizontal bars
plot_lines(): Create line plots. More sophisticated than GICHarts.line_plot(),
              using GICharts.line_plot().
plot_regression(): Create a regression plot. More sophisticated than GICHarts.scatter_plot(),
                   using GICharts.scatter_plot()
box_plot(): Create a "Notched Box & Whisker" plot
create_config_plot(): Create any plot based on a config file
timeseries_plot(): Create a timeseries plot.
histogram_plot(): Create a histogram plot.

Usage:
1. Import:
import geoinfo.gicharts as gicharts

2. Call the method/function:
gicharts.bar_plot()

"""

import configparser

# Already loaded by other imports
# noinspection PyUnresolvedReferences
import math as math
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.dates import MONDAY
import matplotlib.patches as patches

# Already loaded by other imports
# noinspection PyUnresolvedReferences
import numpy as np

import pandas as pd
from pylab import *
from scipy import stats
from scipy import spatial
import scipy.odr as odr

import geoinfo.gidatatools as gidatatools
import geoinfo.gistats as gistats
import geoinfo.giutils as giutils


# Todo: Standardize arguments and I/O variables of all chart functions
class GICharts:
    """ Plot functions defined within this class:

    scatter_plot(...)
    line_plot(...)
    rspec_plot(...)

    Usage: see above
    """

    @staticmethod
    def scatter_plot(axes, graphs,
                     xoutlier=None, youtlier=None,
                     data_labels=None,
                     x_min=None, y_min=None,
                     x_max=None, y_max=None,
                     separate_values=True,
                     identity=True,
                     identity_color='0.8',
                     identity_line_width=0.5,
                     confidence=10,
                     trend=True,
                     orthogonal_distance=False,
                     reg_line_color='#888888',
                     reg_line_width=0.5,
                     reg_line_style='-',
                     density_plot=False,
                     # Relative class width
                     hist_ratio=0.02,
                     invx=False,
                     logscale=False, logregr=True,
                     marker_size=20,
                     marker_face_color=None,
                     # marker_edge_color=None,
                     legend=False,
                     show_empty_datasets_in_legend=False,
                     legend_alpha=None,
                     legend_location='best',  # for details please refer to
                     # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
                     show_grid=False, grid_color=None,
                     show_a=False, show_b=False, show_func=False,
                     show_r=False, show_r2=False,
                     show_p=False, show_stderr=False,
                     show_stderr_estimate=False,
                     show_median_ratio=False,
                     show_bias=False,
                     show_mean_abs_err=False,
                     show_rmse_abs=False,
                     show_rmse_rel=False,
                     show_rmse_bias=False,
                     show_res_abs=False,
                     show_res_rel=False,
                     show_n=False,
                     dec_places=3,
                     txtpos=None,
                     legend_delta_y=0.4,
                     font_family='times new roman',
                     tick_font_size=14,
                     text_font_size=14,
                     text_background_color='#000000',
                     text_background_alpha=0.2,
                     axis_font_size=16,
                     legend_font_size=12):

        """ Create a scatter plot:

        :param show_rmse_bias:
        :param show_empty_datasets_in_legend:
        :param orthogonal_distance:
        :param reg_line_color:
        :param axes: the axes to be used for plotting the scatter plot
        :param graphs: the data, ...
        :param xoutlier: x values of outliers
        :param youtlier: y values of outliers
        :param data_labels: Labeld for data points
        :param x_min: minimum limit of horizontal axis
        :param y_min: minimum limit of vertical axis
        :param x_max: maximum limit of horizontal axis
        :param y_max: maximum limit of vertical axis
        :param separate_values: calculate a seperate regression for each dataset, if True
        :param identity: plot line of identity if True
        :param identity_color: line of identity color
        :param identity_line_width: line of identity width
        :param confidence: confidence interval in percent
        :param trend: plot the regression line, if True
        :param reg_line_width: regression line width
        :param reg_line_style: regression line style
        :param density_plot: plot data as density plot
        :param hist_ratio: define the number of cells in the density plot
        :param invx: invert x axis scale
        :param logscale: plot logscale
        :param logregr: calculate regression for logarithmis values
        :param marker_size: marker size
        :param marker_face_color:
        :param legend: show a legend, if True
        :param legend_alpha: transparency (0-1, where 0 is transparent)
        :param legend_location: argument to place the legend
        :param show_grid: grid for major ticks is plotted, if True
        :param grid_color: color of the grid
        :param show_a: plot the slope, if True
        :param show_b: plot the intercept, if True
        :param show_func: dito regression function
        :param show_r: dito correlation coefficient
        :param show_r2: dito coefficient of determination
        :param show_p: dito error probability
        :param show_stderr: dito standard error
        :param show_stderr_estimate: dito standard error of the estimate (RMSE model)
        :param show_median_ratio: dito median of the ratios between y and x
        :param show_bias: dito bias
        :param show_mean_abs_err: dito mean absolute error
        :param show_rmse_abs: dito absolute RMSE
        :param show_rmse_rel: dito relative RMSE
        :param show_res_abs: dito absolute residuals
        :param show_res_rel: dito rel residuals
        :param show_n: dito number of samples
        :param dec_places: number of decimal places to be shown
        :param txtpos: position of the statistical output (lower left corner)
        :param legend_delta_y: distance between multiple legends (height + padding)
        :param font_family: font family
        :param tick_font_size: tick font size
        :param text_font_size:text font size
        :param text_background_color: background color
        :param text_background_alpha: transparency of the background (0-1, where 0 is transparent)
        :param axis_font_size: axi label font size
        :param legend_font_size: legend font size

        :return: x_min, x_max
        """

        if txtpos is None:
            txtpos = [0.03, 0.80]

        if x_min is not None and x_max is None:
            pass
        if y_min is not None and y_max is None:
            pass
        # 1. Define fonts
        tick_font = font_manager.FontProperties(family=font_family, style='normal',
                                                weight='normal', size=tick_font_size)
        # axis_font = {'family': font_family, 'size': axis_font_size, 'style': 'normal', 'weight': 'normal'}
        text_font = {'family': font_family, 'size': text_font_size, 'style': 'normal', 'weight': 'normal'}
        legend_font = {'family': font_family, 'size': legend_font_size, 'style': 'normal'}

        for ticklabel in axes.get_xticklabels():
            ticklabel.set_fontproperties(tick_font)

        for ticklabel in axes.get_yticklabels():
            ticklabel.set_fontproperties(tick_font)

        # 2. Create appropriate axis ranges
        x_mins = []
        x_maxs = []
        y_mins = []
        y_maxs = []

        # Get minima and maxima from values
        for graph in graphs:

            if graph[0] is not None:
                if len(graph[0]) > 0:
                    if x_min is None:
                        x_mins.append(np.nanmin(graph[0]))
                    if x_max is None:
                        x_maxs.append(np.nanmax(graph[0]))
                else:
                    print('No x values to be scattered.')
            else:
                print('No x values to be scattered.')

            if len(graph[1]) > 0:
                if y_min is None:
                    y_mins.append(np.nanmin(graph[1]))
                if y_max is None:
                    y_maxs.append(np.nanmax(graph[1]))
            else:
                # graph[0] = []
                print('No y values to be scattered.')

        # 2.1 Create appropriate range for lin-scale
        if logscale is False:
            if x_min is None and len(x_mins) > 0:
                x_min = giutils.lin_rangemin(x_mins)
            if x_max is None and len(x_maxs) > 0:
                # print('x_maxs:', x_maxs)
                x_max = giutils.lin_rangemax(x_maxs)
            if y_min is None and len(y_mins) > 0:
                # print('y_mins:', y_mins)
                y_min = giutils.lin_rangemin(y_mins)
            if y_max is None and len(y_maxs) > 0:
                # print('y_maxs:', y_maxs)
                y_max = giutils.lin_rangemax(y_maxs)

        # 2.2 Create appropriate range for log-scale
        # Todo: Shall we distinguish between log-scale for x and y? E. g. xlogscale, ylogscale?
        if logscale is True:

            axes.set_yscale('log', basex=10, basey=10, nonposy='clip')
            axes.set_xscale('log', basex=10, basey=10, nonposx='clip')

            if x_min is None:
                # print('x_mins:', x_mins)
                x_min = giutils.log_rangemin(x_mins)
            if x_max is None:
                # print('x_maxs:', x_maxs)
                x_max = giutils.log_rangemax(x_maxs)
            if y_min is None:
                # print('y_mins:', y_mins)
                y_min = giutils.log_rangemin(y_mins)
            if y_max is None:
                # print('y_maxs:', y_maxs)
                y_max = giutils.log_rangemax(y_maxs)

        # 3. Adjust ranges in the identity case
        if identity:

            # Set x and y range to equal values
            if x_min is not None and y_min is not None:
                if x_min > 0 and y_min > 0 and logscale is False:
                    x_min = y_min = 0
                else:
                    x_min = y_min = min(x_min, y_min)
            if x_max is not None and y_max is not None:
                x_max = y_max = max(x_max, y_max)

        # Decimal data type is not supported
        if x_min is not None:
            x_min = float(x_min)
        if x_max is not None:
            x_max = float(x_max)
        if y_min is not None:
            y_min = float(y_min)
        if y_max is not None:
            y_max = float(y_max)

        axes.set_xlim([x_min, x_max])
        axes.set_ylim([y_min, y_max])

        # 4. Plot line of equality and the +/- confidence range
        # Todo: Insert concave confidence limits for data_sh or regression line
        # See: https://en.wikipedia.org/wiki/File:Regression_confidence_band.svg

        if identity or confidence:
            if x_min is not None and x_max is not None:
                # linspace creates a linear array for the 1:1 line
                print('xmin:', x_min, 'xmax:', x_max)
                line = np.linspace(x_min - x_max, x_max * 2, 100)

                if identity is True:
                    axes.plot(line, line, color=identity_color, linewidth=identity_line_width)

                # linspace creates a linear array for the confidence area
                # Todo: Insert concave confidence limits for data_sh or regression line
                # See: https://en.wikipedia.org/wiki/File:Regression_confidence_band.svg
                if confidence is not None:
                    if confidence > 0:
                        if logscale:
                            print('Please do not use confidence=True with logscale=True.')
                        else:
                            axes.fill_between(line,
                                              (1.0 - confidence / 100.0) * line,
                                              (1.0 + confidence / 100.0) * line,
                                              color='k', alpha=0.2, label='Confidence')

        # 5. Invert x axis (e. g. for Secchi depth case)
        if invx:
            axes.invert_xaxis()

        # 6. Set aspect ratio
        if not identity and density_plot is False:
            axes.set_aspect('auto')
        else:
            axes.set_aspect('equal')

        # 7. Extract the lists x, y, err and dict prop from graphs, plot the values, errors and outliers
        idx_graph = 0

        for graph in graphs:

            x, y, err, prop = graph

            # Todo: Check number of data before commands, which raise an error if x or y are empty lists!
            # if x is not None and y is not None:
            #     if len(x) > 0 and len(y) > 0:
            #         data_available = True
            #     else:
            #         data_available = False

            if density_plot is False:

                # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
                #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
                if show_empty_datasets_in_legend is True or len(x) > 0:
                    # if marker_face_color is not None:
                    #     color = marker_face_color
                    # else:
                    #     color = prop['c']

                    axes.scatter(x, y, s=marker_size, **prop)

                    if data_labels is not None:
                        for idx_value in range(len(x)):
                            axes.annotate(data_labels[idx_graph][idx_value],
                                          xy=(x[idx_value], y[idx_value]))
            else:
                # Prepare the data for density plot
                num_values = len(x)
                num_bins = num_values * hist_ratio

                if num_bins < 50:
                    print('Number of values bins will be set to 50.')
                    num_bins = 50

                hist_data, y_limits, x_limits = np.histogram2d(y, x, bins=(num_bins, num_bins))
                hist_data_masked = np.ma.masked_where(hist_data < 1, hist_data)

                x_mesh, y_mesh = np.meshgrid(x_limits, y_limits)
                axes.pcolormesh(x_mesh, y_mesh, hist_data_masked)

            if err is not None and density_plot is False:
                if show_empty_datasets_in_legend is True or len(x) > 0:
                    if marker_face_color is not None:
                        color = marker_face_color
                    else:
                        color = prop['c']

                    axes.errorbar(x, y, yerr=err,
                                  linestyle='none',
                                  label=prop['label'],
                                  c=color,
                                  marker=prop['marker'])
            idx_graph += 1

        if xoutlier is not None and len(xoutlier) > 0:
            axes.scatter(xoutlier, youtlier, marker='*', s=marker_size)

        # 8. Regression results
        if trend is True:

            delta_y = 0

            for i_graph in range(len(graphs)):

                x, y, err, props = graphs[i_graph]

                # 1. Shall datasets be combined (not to be seperated)?
                if separate_values is False and i_graph == 0:
                    # Collect values from other graphs
                    merged_x_values = [] + list(x)
                    merged_y_values = [] + list(y)

                    for i_graph2 in range(1, len(graphs)):
                        x2, y2, err, props = graphs[i_graph2]
                        merged_x_values = merged_x_values + list(x2)
                        merged_y_values = merged_y_values + list(y2)

                    x = merged_x_values
                    y = merged_y_values

                if (separate_values is False and i_graph == 0) or separate_values is True:
                    # 8.1 Calculate logarithmic values (log10(x) and log10(y))
                    std_err_estimate = None
                    median_ratio = None
                    mean_abs_err = None

                    rmse_rel = 0
                    res_rel = 0

                    if logscale and logregr:

                        x_log_values = []
                        y_log_values = []

                        i_log_values_omitted = 0

                        for i_value in range(len(x)):
                            if x[i_value] > 0 and y[i_value] > 0:
                                x_log_values.append(math.log10(x[i_value]))
                                y_log_values.append(math.log10(y[i_value]))
                            else:
                                i_log_values_omitted += 1

                        if i_log_values_omitted > 0:
                            print(str(i_log_values_omitted) + ' values are <= 0 and omitted for logarithmic display.')

                        (a, b, r, tt, std_err) = stats.linregress(x_log_values, y_log_values)

                        if len(x_log_values) > 0:
                            bias = gistats.get_bias(x_log_values, y_log_values)
                            rmse_abs = gistats.get_rmse(x_log_values, y_log_values)
                            rmse_rel = gistats.get_rmse_rel(x_log_values, y_log_values)
                            res_abs = gistats.get_mean_residual(x_log_values, y_log_values)
                            res_rel = gistats.get_mean_residual_rel(x_log_values, y_log_values)
                        else:
                            bias = np.nan
                            rmse_abs = np.nan
                            rmse_rel = np.nan
                            res_abs = np.nan
                            res_rel = np.nan

                        xlinspace = np.linspace(x_min, x_max, 1000)

                    else:

                        if show_median_ratio is True:
                            median_ratio = gistats.get_median_ratio(x, y)
                        bias = gistats.get_bias(x, y)
                        mean_abs_err = gistats.get_mae(x, y)

                        rmse_abs = gistats.get_rmse(x, y)
                        if show_rmse_rel is True:
                            rmse_rel = gistats.get_rmse_rel(x, y)

                        res_abs = gistats.get_mean_residual(x, y)
                        if show_res_rel is True:
                            res_rel = gistats.get_mean_residual_rel(x, y)

                        # Calculate regression and correlation results based on ordinary least square regression
                        (a, b, r, tt, std_err) = stats.linregress(x, y)
                        y_model = a * np.array(x) + b
                        std_err_estimate = gistats.get_std_err_estimate(y, y_model)

                        if show_stderr_estimate is True or show_stderr_estimate is True:
                            print('\nSE by stats.linregress():', std_err)
                            print('SE by gistats:', std_err_estimate)

                        # (Re-) Calculate slope and intercept based on orthogonal distance regression
                        if orthogonal_distance is True:
                            def ff(vector, x):
                                """ Linear function y = m * x + b
                                - B is a vector of the parameters.
                                - x is an array of the current x values.
                                  x is in the same format as the x passed to Data or RealData.

                                Return an array in the same format as y passed to Data or RealData.

                                :param vector:
                                :param x:
                                :return: vector[0] * x + vector[1]
                                """

                                return vector[0] * x + vector[1]

                            # 1. Create a Model.:
                            linear = odr.Model(ff)

                            # 2. Create a Data or RealData instance.:
                            sx = gistats.get_std_dev(x)
                            sy = gistats.get_std_dev(y)

                            mydata = odr.Data(x, y, wd=1./np.power(sx, 2), we=1./np.power(sy, 2))
                            #    or, when the actual covariances are known:
                            # mydata = odr.RealData(x, y, sx=sx, sy=sy)

                            # 3. Instantiate ODR with your data, model and initial parameter estimate.:
                            my_odr = odr.ODR(mydata, linear, beta0=[1., 2.])

                            # 4. Run the fit.:
                            odr_result = my_odr.run()

                            # 5. Examine output.:

                            odr_result.pprint()

                            a = odr_result.beta[0]
                            b = odr_result.beta[1]
                            y_model = a * np.array(x) + b
                            std_err_estimate = gistats.get_std_err_estimate(y, y_model)

                        if x_min is not None and x_max is not None:
                            xlinspace = np.linspace(x_min, x_max, 10000)
                        else:
                            xlinspace = None

                    if reg_line_color is not None:
                        color = reg_line_color
                    else:
                        color = props['c']
                        color = color if color else 'black'

                    if separate_values is False:
                        # color = 'k'
                        ticklabel = 'Trend'
                    else:
                        ticklabel = props['label']
                        ticklabel = 'Trend of ' + ticklabel if ticklabel else 'Trend'

                    # 8.2 Plot regression line
                    if logscale and logregr:

                        x_logspace = []
                        y_logspace = []

                        if xlinspace is not None:
                            for i_xlinspace in range(len(xlinspace)):
                                if xlinspace[i_xlinspace] > 0:
                                    x_logspace.append(xlinspace[i_xlinspace])
                                    y_log = a * math.log10(xlinspace[i_xlinspace]) + b
                                    y_logspace.append(math.pow(10, y_log))

                            axes.plot(x_logspace, y_logspace,
                                      color,
                                      linewidth=reg_line_width,
                                      linestyle=reg_line_style,
                                      label=ticklabel)

                    else:
                        if xlinspace is not None:
                            axes.plot(xlinspace, a * xlinspace + b, color,
                                      linewidth=reg_line_width,
                                      linestyle=reg_line_style,
                                      label=ticklabel)

                    reg_text = ''

                    # 8.3 Plot statistical results
                    # Todo: Include arguments for RMSE and Standard deviation (show_rmse, show_sd) and
                    # implement the functionality

                    # decimal_threshold = 10**(-dec_places)

                    def format_result(value, decimal_places):
                        """ Format number by decimal places:

                        :param value:
                        :param decimal_places:
                        :return:
                        """
                        min_threshold = 10**(-decimal_places+1)
                        max_threshold = 10**decimal_places
                        if abs(value) > max_threshold or abs(value) < min_threshold:
                            number_format = '{:.' + str(decimal_places - 1) + 'E}'
                            result = number_format.format(value)
                        else:
                            if abs(value) < 0.1:
                                result = str(round(value, dec_places + 1))
                            else:
                                result = str(round(value, dec_places))
                        return result

                    if show_stderr is True:
                        reg_text = r'$stderr=' + format_result(std_err, dec_places) + '$'

                    if show_stderr_estimate is True:
                        reg_text = r'$stderr_m_o_d_e_l=' + format_result(std_err_estimate, dec_places) + '$'

                    if show_p is True:
                        p2 = tt
                        p1 = p2 / 2
                        if reg_text != '':
                            reg_text += '$\n'
                        if p1 < 0.01:
                            reg_text += r'$p <$ $0.01$'
                        else:
                            reg_text = reg_text + r'$p =' + format_result(p1, dec_places) + '$'

                    if show_a is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$a=' + format_result(a, dec_places) + '$'

                    if show_b is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$b=' + format_result(b, dec_places) + '$'
                    if show_func is True:
                        if reg_text != '':
                            reg_text += '\n'
                        if logscale:
                            reg_text = reg_text + r'$log(y)=' + format_result(a, dec_places) + '*log(x)'
                            # +' + str(round(b, 4)) + '$'
                        else:
                            reg_text = reg_text + r'$y=' + format_result(a, dec_places) + '*x'
                            # +' + str(round(b, 4)) + '$'

                        if b >= 0 or format_result(b, dec_places) == '0.0':
                            reg_text = reg_text + r' + ' + format_result(b, dec_places) + '$'
                        else:
                            reg_text = reg_text + r' ' + format_result(b, dec_places) + '$'

                    if show_r is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$R=' + format_result(r, dec_places) + '$'

                    if show_r2 is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$R^2=' + format_result(r * r, dec_places) + '$'

                    if show_median_ratio is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$Median.ratio=' + format_result(median_ratio, dec_places) + '$'

                    if show_bias is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$Bias=' + format_result(bias, dec_places) + '$'

                    if show_mean_abs_err is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$Mean abs. error=' + format_result(mean_abs_err, dec_places) + '$'

                    if show_rmse_abs is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$RMSE.abs=' + format_result(rmse_abs, dec_places) + '$'

                    if show_rmse_rel is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$RMSE.rel=' + format_result(rmse_rel, dec_places) + '$'

                    if show_rmse_bias is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$RMSE.bias=' + format_result(rmse_abs/bias, dec_places) + '$'

                    if show_res_abs is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$Mean Res.abs=' + format_result(res_abs, dec_places) + '$'

                    if show_res_rel is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + r'$Mean Res.rel=' + format_result(res_rel, dec_places) + '$'
                        pass
                    if show_n is True:
                        if reg_text != '':
                            reg_text += '\n'
                        reg_text = reg_text + '$n=' + str(len(x)) + '$'

                    if reg_text != '':
                        if txtpos is None:
                            if r > 0:
                                txtpos = [0.65, 0.1]
                            else:
                                txtpos = [0.1, 0.1]

                        axes.text(txtpos[0], txtpos[1] + delta_y, reg_text, transform=axes.transAxes,
                                  fontdict=text_font, bbox=dict(color=text_background_color,
                                                                alpha=text_background_alpha))

                    delta_y -= legend_delta_y

                # If separate_values is False, then datasets have already been combined and regression has been
                # calculated. Therefore, skip the loop for i_graph > 0.
                if separate_values is False:
                    break

        axes.xaxis.label.set_size(axis_font_size)
        axes.yaxis.label.set_size(axis_font_size)

        # 9. Add the legend
        if legend:

            if legend_location is None:
                bbox_to_anchor = [1.05, 0.5]
                legend_location = 'center left'
                # Move axes to the left and reduce size to find space for the outside barh
                pos = [0.125, 0.1, 0.5, 0.8]
                axes.set_position(pos, which='both')
            elif legend_location == 'outside':
                # Put legend right from the plot
                bbox_to_anchor = [1.05, 0.5]
                legend_location = None
            else:    # if legend_location == 'best':
                bbox_to_anchor = None

            # Todo: Shall trend lines (0 - n-1) or data_sh (n - 2 * 2 - 1) appear in the figure?
            handles, labels = axes.get_legend_handles_labels()

            # Datasets should be sorted before calling plot function, otherwise use:
            # labels = sorted(labels)
            num_handles = len(graphs)
            if trend:
                if separate_values is False:
                    num_handles += 1

            axes.legend([handle for i, handle in enumerate(handles) if i in range(num_handles)],
                        [label for i, label in enumerate(labels) if i in range(num_handles)],
                        scatterpoints=1, loc=legend_location, bbox_to_anchor=bbox_to_anchor,
                        prop=legend_font, framealpha=legend_alpha)

        if show_grid:
            if grid_color is None:
                grid_color = 'k'
            axes.grid(c=grid_color)

        return x_min, x_max

    @staticmethod
    def line_plot(axes, graphs,
                  x_min=None, x_max=None,
                  y_min=None, y_max=None,
                  logscale=False,
                  show_grid=False,
                  identity=False,
                  marker_size=20,
                  # text_font_size=15,
                  axis_font_size=14,
                  legend=False, legend_alpha=0.5,
                  distinct_labels_only=False,
                  font_family='sans-serif',
                  font_style='normal'
                  # line_plot_text='',
                  # y_axis_index=1
                  ):

        """ Create a line plot:

        :param show_grid:
        :param axes: the axes to be used for plotting the scatter plot
        :param graphs: the data, ...
        :param x_min: minimum limit of horizontal axis
        :param y_min: minimum limit of vertical axis
        :param x_max: maximum limit of horizontal axis
        :param y_max: maximum limit of vertical axis
        :param identity: plot line of identity if True
        :param logscale: plot logscale
        :param marker_size: Marker size
        :param axis_font_size: axis label font size
        :param legend: plot a legend, if True
        :param legend_alpha: transparency of the legend background (0-1, where 0 is transparent)
        :param distinct_labels_only: remove redundant labels if true
        :param font_family: font family
        :param font_style: font style

        :return: x_min, x_max
        """

        # if y_axis_index == 2:
        #     axes_twin = axes.twinx()

        # Scale x axis
        if x_min is None:
            # Set appropriate axis ranges
            x_min = []
            x_max = []

            for graph in graphs:
                x_values = graph[0]
                if len(x_values) > 0:
                    x_min.append(np.nanmin(x_values))
                    x_max.append(np.nanmax(x_values))

            x_min = np.nanmin(x_min)
            x_max = np.nanmax(x_max)

            # if x_max - x_min > abs(x_min):
            #     if x_min >= 0:
            #         x_min = 0.8 * x_min
            #     else:
            #         x_min = 1.1 * x_min
            #
            #     if x_max > 0:
            #         x_max = 1.2 * x_max
            #     else:
            #         x_max = 0.8 * x_max
            # else:
            x_min = x_min - (x_max - x_min) / 10
            x_max = x_max + (x_max - x_min) / 10

            # if y_axis_indices is None:
            if x_max > 0:
                x_max = (math.ceil(10 ** (-int(math.floor(math.log10(x_max))) + 1) *
                                   x_max)) / 10 ** (-int(math.floor(math.log10(x_max))) + 1)

            if identity:
                x_min = 0
                x_max = max(x_max)

            # Here it is assumed that all plots have a 0/0 origin - may be edited if neg. values are considered
            # if y_axis_indices is None:
            if x_min > 0:
                x_min = (math.floor(10 ** (-int(math.floor(math.log10(x_min))) + 1) * x_min)) / 10 ** (
                    -int(math.floor(math.log10(x_min))) + 1)

        axes.set_xlim([x_min, x_max])

        # Scale y axis
        if y_min is None:

            # Set appropriate axis ranges
            y_min = []
            y_max = []

            for graph in graphs:

                for idx_y in range(len(graph[1])):
                    if type(graph[1][idx_y]) is str:
                        graph[1][idx_y] = float(graph[1][idx_y])
                y_values = graph[0]
                if len(y_values) > 0:
                    y_min.append(np.nanmin(y_values))
                    y_max.append(np.nanmax(y_values))

            y_min = 0.8 * np.nanmin(y_min)
            y_max = 1.2 * np.nanmax(y_max)

            # print('recent value:', y_min, y_max, graph[1])

            if 0 > y_max > -10 ** -5:
                y_max = 0
            elif y_max > 0:
                # print('Y_Max:', y_max)
                if y_max == inf:
                    pass
                else:
                    y_max = (math.ceil(10 ** (-int(math.floor(math.log10(y_max))) + 1) *
                             y_max)) / 10 ** (-int(math.floor(math.log10(y_max))) + 1)

            if identity:
                y_min = 0
                y_max = max(x_max, y_max)

                x_max = y_max

            # Here it is assumed that all plots have a 0/0 origin - may be edited if neg. values are considered
            if y_min != 0:
                if y_min > 0:
                    y_min = (math.floor(10 ** (-int(math.floor(math.log10(y_min))) + 1) * y_min)) / 10 ** (
                            -int(math.floor(math.log10(y_min))) + 1)

        # if y_axis_index == 2:
        #     axes_twin.set_ylim([y_min, y_max])
        # else:
        axes.set_ylim([y_min, y_max])

        # logarithmic scale option
        if logscale:
            # if y_axis_index == 2:
            #     axes_twin.set_yscale('log')
            # else:
            axes.set_yscale('log')

        for idx_graph in range(len(graphs)):

            graph = graphs[idx_graph]
            # the graph tuples are divided in the lists x, y, err and the dict prop; and then plotted
            x, y, err, prop = graph

            # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
            #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
            # if y_axis_indices is not None and y_axis_indices != '':
            #     y_axis_index = y_axis_indices[idx_graph]
            # else:
            #     y_axis_index = None

            # if y_axis_index is not None:
            #    if y_axis_index == 2:
            #        axes_twin.plot(x, y, markersize=marker_size, linewidth=1.5, **prop)  # , s=marker_size,
            #    elif y_axis_index == 1:
            #        axes.plot(x, y, markersize=marker_size, linewidth=1.5, **prop)  # , s=marker_size,
            #    else:
            #        print('line_plot: Only 1 and 2 are accepted values for y_axis_index.')
            #        exit(-1)
            # else:
            axes.plot(x, y, markersize=marker_size, linewidth=1.5, **prop)  # , s=marker_size,

            tick_font = font_manager.FontProperties(family=font_family,
                                                    style=font_style,
                                                    size=axis_font_size)
            if err is not None:
                # if len(err) != 0:
                #     if y_axis_index == 2:
                #         axes_twin.errorbar(x, y, yerr=err, ecolor=prop['c'], fmt=None)
                #     else:
                axes.errorbar(x, y, yerr=err, ecolor=prop['c'], fmt='none')
                # Todo: Apply tick_font
                # Todo: Insert text

        if legend is True:

            handles, labels = axes.get_legend_handles_labels()
            # if y_axis_index == 2:
            #     if axes_twin.legend() is not None:
            #         handles2, labels2 = axes_twin.get_legend_handles_labels()
            #         handles = handles + handles2
            #         labels = labels + labels2

            if distinct_labels_only is True:

                distinct_handles = []
                distinct_labels = []

                for idx_label in range(len(labels)):
                    label = labels[idx_label]
                    if label not in distinct_labels:
                        distinct_labels.append(label)
                        distinct_handles.append(handles[idx_label])
                handles = distinct_handles
                labels = distinct_labels

            axes.legend([handle for i, handle in enumerate(handles) if i in range(len(graphs))],
                        [label for i, label in enumerate(labels) if i in range(len(graphs))],
                        numpoints=1, handlelength=3,
                        loc='best', framealpha=legend_alpha)
        axes.grid(show_grid)

        axes.xaxis.label.set_size(axis_font_size)
        # if y_axis_index == 2:
        #     axes_twin.yaxis.label.set_size(axis_font_size)
        # else:
        axes.yaxis.label.set_size(axis_font_size)

        return x_min, x_max


def rspec_plot(axes, graphs,
               logscale=False, legend=True,
               text='',
               y_max=0, x_min=400,
               x_max=900,
               txtpos=None, txty=0, text_size=14,
               # font_style='normal',
               tick_font_size=14,
               font_family='times new roman',
               # text_font_size=14,
               axis_font_size=16, legend_font_size=10,
               err_bars=False, fill=False, invert_y=False):

    """ Create a reflectance plot:

    :param axes: axes to be used for plotting
    :param graphs: data, ...
    :param logscale: use log scale, if True
    :param legend: plot a legend, if True
    :param text: text to be plotted
    :param y_max: mximum limit of vertical axis
    :param x_min: minimum limit of horizontal axis
    :param x_max: maximum limit of horizontal axis
    :param txtpos: position of the text
    :param txty: offset for maximum y limit
    :param text_size: text font size
    :param tick_font_size: tick font size
    :param font_family: font family
    :param axis_font_size: axis label font size
    :param legend_font_size: legend font size
    :param err_bars:
    :param fill:
    :param invert_y:

    :return: x_min, x_max
    """

    # mpl.rcParams['text.usetex'] = True

    # set font properties
    # Todo: Fonts setting with mathrm-strings is complicated. Install Latex if further options are needed.
    # See: http://matplotlib.org/users/usetex.html
    tick_font = font_manager.FontProperties(family=font_family, style='normal',
                                            weight='normal', size=tick_font_size)

    # axis_font = {'family': font_family, 'size': axis_font_size, 'style': 'normal', 'weight': 'normal'}
    # text_font = {'family': font_family, 'size': text_font_size, 'style': 'normal', 'weight': 'normal'}
    # legend_font = {'family': font_family, 'size': legend_font_size, 'style': 'normal'}

    if txtpos is None:
        txtpos = [0.75, 0.03]

    axes.xaxis.label.set_size(axis_font_size)
    axes.yaxis.label.set_size(axis_font_size)

    for ticklabel in axes.get_xticklabels():
        ticklabel.set_fontproperties(tick_font)
    for ticklabel in axes.get_yticklabels():
        ticklabel.set_fontproperties(tick_font)

    # Set appropriate y-axis range if not in kwarg
    y_min = 0
    if y_max == 0:
        y_max = []
        for graph in graphs:
            y_max.append(np.nanmax(graph[1]))
        y_max = txty * giutils.lin_rangemax(y_max)

    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])

    # logarithmic scale option
    if logscale:
        axes.set_yscale('log')

    # invert y-axis option
    if invert_y:
        axes.invert_yaxis()

    for n_graph, graph in enumerate(graphs):

        # the graph tuples are divided in the lists x, y, err and the dict prop; and then plotted
        x, y, err, prop = graph

        # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
        #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
        axes.plot(x, y, linewidth=1.5, **prop)

        if err_bars is True and len(err) != 0:
            axes.errorbar(x, y, yerr=err, ecolor=prop['c'], fmt=None)

        # fill the space below graphs in fill-list if provided
        if fill is not False:
            if n_graph in fill:
                axes.fill_between(x, 0, y, facecolor='black', alpha=.15)

    if text != '':
        axes.text(txtpos[0], txtpos[1], text, transform=axes.transAxes,
                  fontdict={'family': font_family, 'fontsize': text_size},
                  va='top', ha='left', bbox=dict(color='k', alpha=0.2)
                  )

    if legend:
        handles, labels = axes.get_legend_handles_labels()
        axes.legend([handle for i, handle in enumerate(handles) if i in range(len(graphs))],
                    [label for i, label in enumerate(labels) if i in range(len(graphs))],
                    numpoints=1, handlelength=3, loc='best',
                    prop={'size': legend_font_size, 'family': font_family})

    axes.grid(True)

    return x_min, x_max


def regression_plot(xij, yi,
                    marker_style='o',
                    # calculate_intercept=True,
                    regression_type='linear',
                    orthogonal_regression=False,
                    y_error_is_zero=False,
                    line_style='--r',
                    reg_plot_text='',
                    txtpos=None,
                    close_figure=True):

    """ This function calculates and plots a linear or non-linear regression considering ordinary or
    orthogonal distance regression:
        - linear: y = a * xi + b
        - multilinear: y = a1 xi_1 + a2 * xi_2 + ... + ai * xi_j + b (x_ij.shape = [j, i])
        - exponential: y = a * e^xi + b
        - logarithmic: y = a * log(xi) + b
        - quadratic: y = a * xi^2 + b * xi + c
        - cubic = a * xi^3 + b * xi^2 + c * xi + d

    :param xij: matrix of i independent values for each of the j coefficients to be estimated
                 (list of j lists of i values)
    :param yi: List of i "dependent" values
    :param marker_style:
    :param regression_type: Choose either exponential, linear, quadratic, cubic.
                            If type = 'linear' and x_ij.shape[0] > 1, a multilinear regression
                            will be calculated.
    :param orthogonal_regression:
    :param y_error_is_zero:
    :param line_style:
    :param reg_plot_text:
    :param txtpos:
    :param close_figure:

    :return: True
    """

    def get_regression_results(xij, yi,
                               regression_type=regression_type,
                               orthogonal_regression=False,
                               calculate_intercept=True
                               ):
        """ Calculate regression results

        :param xij:
        :param yi:
        :param regression_type:
        :param orthogonal_regression:
        :param calculate_intercept:
        :return:
        """
        def f_linear(vector, x):
            """ Linear function y = m*x + b

            Return an array in the same format as y passed to Data or RealData.

            :param vector: vector of the parameters.
            :param x: an array of the current x values.
                      x is in the same format as the x passed to Data or RealData.
            :return:
            """
            return vector[0] * x + vector[1]

        # 2. Create a Model:
        model = None
        if regression_type == 'linear':
            model = odr.Model(f_linear)
        else:
            print('Regression type', regression_type, 'not implemented.')
            exit(-1)

        # 3. Create a Data or RealData instance.:
        if orthogonal_regression is True:
            sy = gistats.get_std_dev(yi)
            if len(xij) == 1:
                sx = gistats.get_std_dev(xij[0])
                mydata = odr.Data(xij, yi, wd=1./np.power(sx, 2), we=1./np.power(sy, 2))
            else:
                mydata = odr.Data(xij, yi, we=1./np.power(sy, 2))

            #    or, when the actual covariances are known:
            # mydata = odr.RealData(x, y, sx=sx, sy=sy)

            # 4. Instantiate ODR with your data, model and initial parameter estimate.:
            myodr = odr.ODR(mydata, model, beta0=[1., 2.])
            # 5. Run the fit.:
            regression_result = myodr.run()
            # 6. Examine output.:
            regression_result.pprint()
        else:
            # yi = c1 * np.exp(-xi) + c2 * xi
            # zi = yi + 0.05 * np.max(yi) * np.random.randn(len(yi))
            if calculate_intercept is True:
                if y_error_is_zero is False:
                    a = np.c_[xij[0][:, np.newaxis], np.ones(len(xij[0]))[:, np.newaxis]]
                else:
                    a = np.c_[xij[0][:, np.newaxis], np.ones(len(xij[0]))[:, np.newaxis]]
            else:
                a = np.c_[xij[0][:, np.newaxis]]
            # c, resid, rank, sigma = linalg.lstsq(A, yi)
            regression_result = linalg.lstsq(a, yi)

        return regression_result

    def create_regression_plot(x_values, y_values,
                               regression_path,
                               line_style='--r',
                               text='',
                               txtpos=None
                               ):
        """ Create a regression plot:

        :param x_values:
        :param y_values:
        :param regression_path:
        :param line_style:
        :param text:
        :param txtpos:
        :return:
        """
        if txtpos is None:
            txtpos = [0.1, 0.9]
        plt.plot(regression_path[0], regression_path[1], line_style)
        plt.plot(x_values, y_values, marker_style)
        axes = plt.gca()
        axes.text(txtpos[0], txtpos[1], text, transform=axes.transAxes)
        # fontdict=text_font, bbox=dict(color=text_background_color, alpha=text_background_alpha))

        if close_figure is True:
            plt.show()

        return True

    if txtpos is None:
        txtpos = [0.1, 0.9]

    regression_result = None
    if y_error_is_zero is False:
        regression_result = get_regression_results(xij, yi,
                                                   regression_type=regression_type,
                                                   orthogonal_regression=orthogonal_regression
                                                   )
    elif regression_type == 'linear':
        regression_result = get_regression_results([yi], xij[0],
                                                   regression_type=regression_type,
                                                   orthogonal_regression=orthogonal_regression
                                                   )
    else:
        print('Option y_error_is_zero not implemented for', regression_type + '.')
        exit(-1)

    if orthogonal_regression is True:
        regression_coeff = regression_result.beta
        # text = 'Orthogonal regression (green):'
    else:
        regression_coeff = regression_result[0]
        # text = 'Ordinary linear regression (blue):'

    if y_error_is_zero is False:
        a = regression_coeff[0]
        b = regression_coeff[1]
    else:
        a = 1 / regression_coeff[0]
        b = -regression_coeff[1] / regression_coeff[0]

    y0 = a * np.array(xij[0]) + b
    x_y_model = xij + [yi] + [y0]

    # Range of independent data
    x_min = np.min(xij[0])
    x_max = np.max(xij[0])

    # Regression curve
    xx = np.linspace(x_min, x_max, num=1000)
    yy = a * xx + b

    regression_path = [xx, yy]
    dec_places = 3

    reg_plot_text = reg_plot_text + '\ny = ' + str(round(a, 3)) + ' * x + ' + str(round(b, dec_places))

    create_regression_plot(x_y_model[0], x_y_model[1],
                           regression_path,
                           line_style=line_style,
                           text=reg_plot_text,
                           txtpos=txtpos
                           )

    return True


def bar_plot(x_values, thresholds=None,
             x_axis_label='x axis', y_axis_label='y axis',
             x_tick_labels=None,
             simple_tex=False,
             y_min=None, y_max=None,
             fig_size=None,
             fig_size_px=None,  # e. g. [1500, 700] for 1500 * 700 px
             # logscale=False,
             errors=None,
             title='',
             colors=None,
             font_style='normal',
             font_family='serif',
             legend_labels=None,
             title_font_size=24,
             text_font_size=15, axis_font_size=18,
             legend_font_size=12, tick_font_size=14,
             text=None, txt_pos=None,
             show_legend=False,
             legend_location='best',
             # for details please refer to
             # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
             img_file_name=''):

    """ Create a (vertical) bar plot:

    :param x_values: datasets containing x values as a list of lists of values
    :param thresholds: thresholds to be marked as dotted lines
    :param x_axis_label:
    :param y_axis_label:
    :param x_tick_labels:
    :param simple_tex: If True, "_" and "^" will be interpreted as sub-script and super-script.
                       Use curly brackets for multiple chars, e. g. ^{-1} or _{min}
    :param y_min: minimum value of vertical axis range
    :param y_max: maximum value of vertical axis range
    :param fig_size: fig size in inch
    :param fig_size_px: Output size in pixel
    :param errors: A list of lists of error values
    :param title: Plot title, to be printed above the plot.
    :param colors: Colors of the datasets. A list of colors, e. g. "k", "b", "#ff0000"
    :param font_style: normal, italic, oblique
    :param font_family: 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
    :param legend_labels: Dataset names for legend
    :param title_font_size: Plot title size
    :param text_font_size: General text size (not fully implemented, see text_pos
    :param axis_font_size: Axis label size
    :param legend_font_size: Legend label size
    :param tick_font_size: Tick mark size
    :param text:
    :param txt_pos: Position for extra text
    :param show_legend: plot legend, if True
    :param legend_location: for details please refer to
           http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
    :param img_file_name: Leave blank, if the figure should not be saved as an image file

    :return: True
    """

    if thresholds is None:
        thresholds = []

    if txt_pos is None:
        txt_pos = [0.2, 0.8]

    num_datasets = len(x_values)
    num_groups = len(x_values[0])
    width = 1 / (num_datasets + 1)
    delta = width / 2

    if fig_size_px is not None:
        dpi = 100
        width = fig_size_px[0] / dpi
        height = fig_size_px[1] / dpi
        fig_size = [width, height]

    fig = plt.figure(figsize=fig_size)
    rect = fig.patch
    rect.set_facecolor('#ffffff')
    ax = fig.add_subplot(111)

    if simple_tex:
        x_axis_label = giutils.convert_2_tex(x_axis_label, font_family=font_family, font_style=font_style)
        y_axis_label = giutils.convert_2_tex(y_axis_label, font_family=font_family, font_style=font_style)
        title = giutils.convert_2_tex(title, font_family=font_family, font_style=font_style)

    ax.set_xlabel(x_axis_label, fontdict={'family': font_family, 'style': font_style, 'fontsize': axis_font_size})
    ax.set_ylabel(y_axis_label, fontdict={'family': font_family, 'style': font_style, 'fontsize': axis_font_size})
    ax.set_title(title, fontdict={'family': font_family, 'style': font_style, 'fontsize': title_font_size})

    positions = np.arange(num_groups)
    ax.set_xticks(positions + 0.5)

    if x_tick_labels is None:
        x_tick_labels = np.arange(num_groups)

    ax.set_xticklabels(x_tick_labels)

    if y_max is not None:
        if y_min is None:
            y_min = 0
        ax.set_ylim([y_min, y_max])

    # ax.legend( (bars[0]), ('Men', 'Women') )
    legend_bars = []

    for i_dataset in range(len(x_values)):

        # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
        #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
        bars = ax.bar(positions + delta + i_dataset * width,
                      x_values[i_dataset],
                      width,
                      color=colors[i_dataset],
                      yerr=errors[i_dataset],
                      ecolor=[0, 0, 0])
        legend_bars.append(bars[0])

    if thresholds is not None and len(thresholds) > 0:
        for threshold in thresholds:
            ax.plot([0, threshold], [len(x_values[0]), threshold],
                    color='#888888', linestyle='--')

    if show_legend is True:
        if legend_labels is not None:
            ax.legend(legend_bars, legend_labels, fontsize=legend_font_size, loc=legend_location)
        else:
            print('No legend labels.')

    tick_font = font_manager.FontProperties(family=font_family,
                                            style=font_style,
                                            size=tick_font_size)

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontproperties(tick_font)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontproperties(tick_font)

    text_font = {'family': font_family, 'size': text_font_size, 'style': 'normal', 'weight': 'normal'}

    if text is not None:
        ax.text(txt_pos[0], txt_pos[1], text,
                transform=ax.transAxes,
                fontdict=text_font,
                bbox=dict(color='w', alpha=0.2))

    if img_file_name == '':
        plt.show()
    else:
        fig.savefig(img_file_name)
        print('Plot was saved as: ', img_file_name)
    plt.close()

    return True


def barh_plot(x_values,
              thresholds=None,
              x_axis_label='x axis',
              y_axis_label='y axis',
              x_tick_labels=None,
              simple_tex=False,
              y_min=None, y_max=None,
              fig_size=None,
              fig_size_px=None,
              # logscale=False,
              errors=None,
              title='',
              colors=None,
              font_style='normal',
              font_family='serif',
              legend_labels=None,
              legend_font_size=20,
              title_font_size=24,
              text_font_size=15,
              axis_font_size=18,
              tick_font_size=14,
              text=None, txt_pos=None,
              legend=False, legend_location='best',
              # for details please refer to
              # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
              img_file_name=''):

    """Plot a horizontal bar plot:

    :param fig_size:
    :param x_values: list of list of values
    :param thresholds: thresholds to be marked as dotted lines
    :param x_axis_label: label for (here!) vertical axis
    :param y_axis_label: label for (here!) vertical axis
    :param x_tick_labels: data group labels
    :param simple_tex: option to activate a simple tex syntax
    :param y_min: minimum for horizontal axis
    :param y_max: maximum for horizontal axis
    :param fig_size_px:
    :param errors: error values for error bars
    :param title:
    :param colors:
    :param font_style: normal, italic
    :param font_family: serif, sansserif, Helvetica
    :param legend_labels:
    :param legend_font_size:
    :param title_font_size:
    :param text_font_size:
    :param axis_font_size:
    :param tick_font_size:
    :param text: supplemental text to be plotted at position txt_pos
    :param txt_pos: position for supplemental text, e. g. [3.4, 5.6]
    :param legend: True or False
    :param legend_location: for details please refer to
                            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
    :param img_file_name:

    :return: True
    """

    # Todo: Create a stacked horizontal bar.
    # import numpy as np
    # from matplotlib import pyplot as plt
    #
    # dataset = [{'A': 19, 'B': 39, 'C': 61, 'D': 70},
    #            {'A': 34, 'B': 68, 'C': 32, 'D': 38},
    #            {'A': 35, 'B': 45, 'C': 66, 'D': 50},
    #            {'A': 23, 'B': 23, 'C': 21, 'D': 16},
    #            {'A': 35, 'B': 45, 'C': 66, 'D': 50}]
    # categories = ['A', 'B', 'C', 'D']
    # labels = ['1', '2', '3', '4', '5']
    # colors = ["red", "yellow", "green", "blue", "purple", "magenta"]
    # ticks = np.arange(len(dataset))
    #
    # for i, data in enumerate(dataset):
    #     values = np.array([data[key] for key in categories])
    #     starts = np.zeros(len(values))
    #     pos = np.array([i for key in categories])
    #     for k in range(1, len(starts)):
    #         starts[k] = starts[k - 1] + values[k - 1]
    #     plt.barh(pos, left=starts, width=values, color=colors, align='center', height=0.5)
    #
    # plt.yticks(ticks, labels)
    # plt.show()

    if txt_pos is None:
        txt_pos = [0.02, 0.8]

    if thresholds is None:
        thresholds = []

    num_datasets = len(x_values)
    num_groups = len(x_values[0])
    width = 1 / (num_datasets + 1)
    delta = width / 2

    if fig_size_px is not None:
        dpi = 100
        width = fig_size_px[0] / dpi
        height = fig_size_px[1] / dpi
        fig_size = [width, height]

    fig = plt.figure(figsize=fig_size)

    rect = fig.patch
    rect.set_facecolor('#ffffff')
    ax = fig.add_subplot(111)

    if simple_tex:
        x_axis_label = giutils.convert_2_tex(x_axis_label, font_family=font_family, font_style=font_style)
        y_axis_label = giutils.convert_2_tex(y_axis_label, font_family=font_family, font_style=font_style)
        title = giutils.convert_2_tex(title, font_family=font_family, font_style=font_style)

    ax.set_xlabel(x_axis_label, fontdict={'family': font_family, 'style': font_style, 'fontsize': axis_font_size})
    ax.set_ylabel(y_axis_label, fontdict={'family': font_family, 'style': font_style, 'fontsize': axis_font_size})
    ax.set_title(title, fontdict={'family': font_family, 'style': font_style, 'fontsize': title_font_size})

    ax = fig.add_subplot(111)

    # Move axes to the right and reduce size to find space for the y-axis labels
    # (x, y, width, height)
    pos = [0.3, 0.125, 0.6, 0.75]
    ax.set_position(pos, which='both')

    # Redundant (see below):
    # ax.set_xlabel(x_axis_label, fontdict={'family':font_family, 'style':font_style, 'fontsize':legend_font_size})
    # ax.set_ylabel(y_axis_label, fontdict={'family':font_family, 'style':font_style, 'fontsize':legend_font_size})

    ax.set_title(title, fontdict={'family': font_family, 'style': font_style, 'fontsize': title_font_size})

    positions = np.arange(num_groups)
    ax.set_yticks(positions + 0.5)

    if x_tick_labels is None:
        x_tick_labels = np.arange(num_groups)

    # Todo: 1
    ax.set_yticklabels(x_tick_labels)

    if y_max is not None:
        if y_min is None:
            y_min = 0
        ax.set_xlim([y_min, y_max])

    if colors is None:
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'r']

    ecolor = [0, 0, 0]
    legend_bars = []

    for i_dataset in range(len(x_values)):

        # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
        #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
        bars = ax.barh(positions + delta + i_dataset * width,
                       x_values[i_dataset],
                       width,
                       color=colors[i_dataset],
                       xerr=errors[i_dataset],
                       ecolor=ecolor)
        legend_bars.append(bars[0])

    if legend:
        if legend_labels is not None:
            ax.legend(legend_bars,
                      legend_labels,
                      fontsize=legend_font_size,
                      loc=legend_location)
        else:
            print('No legend labels.')

    if thresholds is not None and len(thresholds) > 0:
        for threshold in thresholds:
            ax.plot([threshold, threshold], [0, len(x_values[0])],
                    color='#888888', linestyle='--')

    ax.xaxis.label.set_size(axis_font_size)
    ax.yaxis.label.set_size(axis_font_size)

    tick_font = font_manager.FontProperties(family=font_family,
                                            style=font_style,
                                            size=tick_font_size)

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontproperties(tick_font)

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontproperties(tick_font)

    text_font = {'family': font_family, 'size': text_font_size, 'style': 'normal', 'weight': 'normal'}

    if text is not None:
        ax.text(txt_pos[0], txt_pos[1], text,
                transform=ax.transAxes,
                fontdict=text_font,
                bbox=dict(color='w', alpha=0.2))
    if img_file_name == '':
        plt.show()
    else:
        fig.savefig(img_file_name)

    plt.close()

    return True


def add_data_to_line_plot(x_values, y_values,
                          dataset_labels=None,
                          # x_axis_label='x axis',
                          # y_axis_label='y axis',
                          x_min=None, x_max=None,
                          y_min=None, y_max=None,
                          errors=None,
                          text='',
                          text_pos=None,
                          extra_lines=None,
                          extra_points=None,
                          title='',
                          title_offset=None,
                          # axis_label_offset=0,
                          colors=None,
                          markers=None, marker_size=20,
                          markeredgecolors=None,
                          line_styles=None,
                          font_style='normal',
                          font_family='sans-serif',
                          # label_font_size=16,
                          title_font_size=14,
                          text_font_size=15, axis_font_size=14,
                          legend=False,
                          show_empty_datasets_in_legend=False,
                          # legend_alpha=0.5,
                          distinct_labels_only=False
                          # img_file_name='',
                          # close_plot=True
                          ):

    """ Add data to a line plot:

    :param x_values:
    :param y_values:
    :param dataset_labels:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param errors:
    :param text:
    :param text_pos:
    :param extra_lines:
    :param extra_points:
    :param title:
    :param title_offset:
    :param colors:
    :param markers:
    :param marker_size:
    :param markeredgecolors:
    :param line_styles:
    :param font_style:
    :param font_family:
    :param title_font_size:
    :param text_font_size:
    :param axis_font_size:
    :param legend:
    :param show_empty_datasets_in_legend:
    :param distinct_labels_only:

    :return: True
    """
    if title_offset is None:
        title_offset = [0, 0]

    # graph = x (array), y (array), errorValues (array or None), prop (dict)
    graphs = []

    if dataset_labels is None:
        dataset_labels = []
        for idx in range(0, len(x_values)):
            if len(x_values[idx]) > 0 or show_empty_datasets_in_legend is True:
                dataset_labels.append('Dataset ' + str(idx + 1))

    if colors is None:
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'r']

    if markers is None:
        markers = ['o', 'v', 's', '*', '+', 'x', 'D', '^', '8', 'h', 'p', '.', ',', 'H', '<', 'd', '|', '>']

    for idx in range(0, len(x_values)):

        if len(x_values[idx]) == 0 and show_empty_datasets_in_legend is False:
            continue

        if len(colors) == 1:
            color = colors[0]
        else:
            color = colors[idx]

        if line_styles is not None:
            if len(line_styles) > 1:
                line_style = line_styles[idx]
            else:
                line_style = line_styles[0]
        else:
            line_style = 'None'

        # markeredgewidth = 1.5

        if markeredgecolors is not None:
            if len(markeredgecolors) > 1:
                markeredgecolor = markeredgecolors[idx]
            else:
                markeredgecolor = markeredgecolors[0]

            # if markeredgecolor == color:
            #     markeredgewidth = 0.0
        else:
            markeredgecolor = None

        if len(markers) > idx:
            marker = markers[idx]
        else:
            div, mod_value = divmod(idx, len(markers))
            marker = markers[mod_value]

        if errors is None:
            error = None
        else:
            error = errors[idx]

        the_properties = dict(c=color, marker=marker, label=dataset_labels[idx],
                              markeredgecolor=markeredgecolor, linestyle=line_style)
        # , markeredgewidth=markeredgewidth)

        graphs.append([x_values[idx], y_values[idx],
                      error, the_properties])

    # x1 und x2 enthalten die Grenzen der Y-Achse
    # - identity: line of equality darstellen?
    # - confidence: linearer Vertrausensbereich um die "Line of equality"
    # - trend: Regressionsgerade anzeigen?
    axes = plt.gca()

    x1, x2 = GICharts.line_plot(axes, graphs, logscale=False,
                                x_min=x_min, x_max=x_max,
                                y_min=y_min, y_max=y_max,
                                marker_size=marker_size,
                                # text_font_size=text_font_size,
                                axis_font_size=axis_font_size,
                                font_style='normal',
                                legend=legend,
                                distinct_labels_only=distinct_labels_only
                                # line_plot_text=''
                                )

    print('x range:', x1, x2)

    if extra_lines is not None:
        for idx_line in range(len(extra_lines)):
            x1 = extra_lines[idx_line][0][0]
            x2 = extra_lines[idx_line][1][0]
            y1 = extra_lines[idx_line][0][1]
            y2 = extra_lines[idx_line][1][1]

            axes.plot([x1, x2], [y1, y2], color='#ffA050', linestyle='solid')

    if extra_points is not None:
        for idx_point in range(len(extra_points)):
            x = extra_points[idx_point][0]
            y = extra_points[idx_point][1]
            axes.plot([x], [y], 'bo')

    # x_label = axes.set_xlabel(x_axis_label,
    #                           fontdict={'family': font_family,
    #                                     'style': font_style,
    #                                     'fontsize': axis_font_size})
    # x_label.set_position((x_label.get_position()[0],
    # x_label.get_position()[1] - axis_label_offset))

    # fontdict = {'family': font_family, 'style': font_style, 'fontsize': axis_font_size}
    # y_label = axes.set_ylabel(y_axis_label,
    #                           fontdict=fontdict)
    # y_label.set_position((y_label.get_position()[0] + axis_label_offset, y_label.get_position()[1]))

    title = axes.set_title(title, fontdict={'family': font_family, 'style': font_style, 'fontsize': title_font_size})
    title.set_position((title.get_position()[0] + title_offset[0],
                        title.get_position()[1] + title_offset[1]))

    if x_min is not None:
        axes.set_xlim(x_min, x_max)

    if y_min is not None:
        axes.set_ylim(y_min, y_max)

    font_dict = {'family': font_family, 'size': text_font_size, 'style': 'normal', 'weight': 'normal'}

    if text_pos is not None:
        axes.text(text_pos[0], text_pos[1], text,
                  fontdict=font_dict,
                  bbox=dict(color='k', alpha=0.2))

    # # Todo: Consider errors (see scatterplot function)
    # if legend:
    # handles, labels = axes.get_legend_handles_labels()
    # axes.legend([handle for i,handle in enumerate(handles) if i in range(len(graphs))],
    # [label for i,label in enumerate(labels) if i in range(len(graphs))],
    # scatterpoints=1, loc='best')

    return True


def plot_lines(x_values, y_values,
               dataset_labels=None,
               # x_axis_label='x axis', y_axis_label='y axis',
               x_min=None, x_max=None,
               y_min=None, y_max=None,
               errors=None,
               fig_size=None,
               fig_size_px=None,
               text='',
               text_pos=None,
               extra_lines=None,
               extra_points=None,
               title='',
               title_offset=None,
               # axis_label_offset=0,
               colors=None,
               markers=None, marker_size=20,
               markeredgecolors=None,
               line_styles=None,
               font_style='normal',
               font_family='sans-serif',
               # label_font_size=16,
               title_font_size=14, text_font_size=15, axis_font_size=14,
               legend=False,
               show_empty_datasets_in_legend=False,
               # legend_alpha=0.5,
               # distinct_labels_only=False,
               img_file_name='',
               close_plot=True):

    """ Plot lines:

    :param close_plot:
    :param show_empty_datasets_in_legend:
    :param x_values: dataset for x values as a list of lists of values
    :param y_values: dataset for y values as a list of lists of values
    :param dataset_labels: labels for legend
    :param x_min: minimum x axis limit
    :param x_max: maximum x axis limit
    :param y_min: minimum y axis limit
    :param y_max: maximum y axis limit
    :param errors: dataset for error values as a list of lists of values
    :param fig_size: fig size in inch
    :param fig_size_px: figure size in pixel
    :param text: text to be plotted
    :param text_pos: position of the text
    :param extra_lines: supplemental lines to be plotted
    :param extra_points: supplemental points to be plotted
    :param title: plot title to be plotted at the top of the figure
    :param title_offset: distance from plot
    :param colors: colors to be used for markers
    :param markers: marketr types
    :param marker_size: marker size
    :param markeredgecolors: marker edge color
    :param line_styles: list of line types, such as '-', '--', '.-' or ':'
    :param font_style: normal, italic
    :param font_family: serif, sansserif, Helvetica
    :param title_font_size: plot title font size
    :param text_font_size: text font size
    :param axis_font_size: axis label font size
    :param legend: True or False plot a legend, if True
    :param img_file_name: leave blank, if figure should not be saved as image file

    :return: True
    """

    if title_offset is None:
        title_offset = [0, 0]

    if fig_size_px is not None:
        dpi = 100
        width = fig_size_px[0] / dpi
        height = fig_size_px[1] / dpi
        fig_size = [width, height]

    # fig = plt.gcf()
    fig = plt.gcf()    # figure(figsize=fig_size)
    if len(fig.axes) == 0:
        fig.set_size_inches(fig_size)
        # Add white background for whole figure
        rect = fig.patch
        rect.set_facecolor('#ffffff')
        fig.add_subplot(111)

    add_data_to_line_plot(x_values, y_values,
                          dataset_labels=dataset_labels,
                          # x_axis_label=x_axis_label,
                          # y_axis_label=y_axis_label,
                          x_min=x_min, x_max=x_max,
                          y_min=y_min, y_max=y_max,
                          errors=errors,
                          text=text,
                          text_pos=text_pos,
                          extra_lines=extra_lines,
                          extra_points=extra_points,
                          title=title,
                          title_offset=title_offset,
                          colors=colors,
                          markers=markers,
                          marker_size=marker_size,
                          markeredgecolors=markeredgecolors,
                          line_styles=line_styles,
                          font_style=font_style,
                          font_family=font_family,
                          # label_font_size=None,
                          title_font_size=title_font_size,
                          text_font_size=text_font_size,
                          axis_font_size=axis_font_size,
                          legend=legend,
                          show_empty_datasets_in_legend=show_empty_datasets_in_legend,
                          # legend_alpha=legend_alpha
                          # distinct_labels_only=None,
                          # img_file_name=img_file_name,
                          # close_plot=close_plot
                          )

    if img_file_name == '' or img_file_name is None:
        if close_plot is True:
            plt.show()
    else:
        fig.savefig(img_file_name)

    if close_plot is True:
        plt.close()

    return True


def plot_regression_from_df(xy_dataframe, x_col='CHL', y_col='CHL',
                            group_col='Station',
                            change_group_name=None,
                            x_axis_label='x axis',
                            y_axis_label='y axis',
                            simple_tex=True,
                            merge_data=True,
                            fig_size=None,  # e. g. (12, 6)
                            fig_size_px=None,  # e. g. (1200, 600)
                            conf=0, trend=False, identity=False, logscale=False,
                            errors=None,
                            plot_title='Regression plot',
                            markers=None,
                            marker_size=20,
                            colors=None,
                            font_style='normal', font_family='sans-serif',
                            # labels=None,
                            # label_font_size=16,
                            title_font_size=14,
                            text_font_size=15, axis_font_size=14,
                            show_a=False, show_b=False, show_r=False, show_r2=False,
                            show_p=False, show_stderr=False, show_n=False, show_func=False,
                            txtpos=None,
                            legend=True, legend_location='outside',
                            # for details please refer to
                            # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
                            legend_delta_y=0.12,
                            img_file_name=''):

    """ Call plot_regression with data extracted from a dataframe
    For further details details see plot_regression().

    :param xy_dataframe: dataframe containing the data
    :param x_col: name of the dataframe column containing the x data
    :param y_col:name of the dataframe column containing the y data
    :param group_col: name of the columns for grouping
    :param change_group_name: change group names, if True
    :param x_axis_label:
    :param y_axis_label:
    :param simple_tex:
    :param merge_data:
    :param fig_size:
    :param fig_size_px:
    :param conf:
    :param trend:
    :param identity:
    :param logscale:
    :param errors:
    :param plot_title:
    :param markers:
    :param marker_size:
    :param colors:
    :param font_style:
    :param font_family:
    :param title_font_size:
    :param text_font_size:
    :param axis_font_size:
    :param show_a:
    :param show_b:
    :param show_r:
    :param show_r2:
    :param show_p:
    :param show_stderr:
    :param show_n:
    :param show_func:
    :param txtpos:
    :param legend:
    :param legend_location:
    :param legend_delta_y:
    :param img_file_name:

    :return: True
    """

    if txtpos is None:
        txtpos = [0.2, 0.9]
    # 1 Group the dataframe
    # Todo: Repair this statement
    print(group_col)
    station_dict = gidatatools.get_group_dict(xy_dataframe)  # , group_col)

    x_values = []
    y_values = []
    plot_labels = []

    # 2. Iterate over groups to extract x and y data_sh
    for idx_station, station_data in enumerate(station_dict.items()):

        # 3.1 The first element is the group name, second element is the corresponding dataframe
        if change_group_name is None:
            station_name = station_data[0]
        else:
            station_name = change_group_name(station_data[0])

        station_values = station_data[1]

        # 3.2 Add IS and EO data_sh for the station as a new XY dataset
        if x_col == y_col:
            x_col += '_x'
            y_col += '_y'

        x_values.append(station_values[x_col].values)
        y_values.append(station_values[y_col].values)

        # 3.3 Create label for the dataset
        plot_labels.append(station_name)

    if merge_data is True:
        separate_values = False
    else:
        separate_values = True

    # 4. Call regression plot function
    plot_regression(x_values, y_values,
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    simple_tex=simple_tex,
                    separate_values=separate_values,
                    fig_size=fig_size,  # e. g. (12, 6)
                    fig_size_px=fig_size_px,
                    conf=conf, trend=trend,
                    identity=identity, logscale=logscale,
                    errors=errors,
                    title=plot_title,
                    markers=markers,
                    marker_size=marker_size,
                    colors=colors,
                    font_style=font_style, font_family=font_family,
                    labels=plot_labels,
                    # label_font_size=label_font_size,
                    title_font_size=title_font_size,
                    text_font_size=text_font_size, axis_font_size=axis_font_size,
                    show_a=show_a, show_b=show_b, show_r=show_r, show_r2=show_r2,
                    show_p=show_p, show_stderr=show_stderr, show_n=show_n, show_func=show_func,
                    txtpos=txtpos,
                    legend=legend,
                    legend_location=legend_location,  # for details please refer to
                    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
                    legend_delta_y=legend_delta_y,
                    img_file_name=img_file_name)

    return True


def plot_regression(x_values, y_values, x_axis_label='x axis', y_axis_label='y axis',
                    data_labels=None,
                    simple_tex=False,
                    x_min=None, x_max=None, y_min=None, y_max=None,
                    separate_values=True,
                    fig_size=None,
                    fig_size_px=None,
                    conf=0, trend=False, identity=False,
                    density_plot=False, hist_ratio=0.02,
                    logscale=False, logregr=True,
                    errors=None,
                    title='',
                    markers=None, marker_size=20, markeredgecolors=None,
                    colors=None,
                    font_style='normal',
                    font_family='serif', labels=None,
                    # label_font_size=20,
                    title_font_size=24,
                    text_font_size=15, axis_font_size=14, legend_font_size=12, tick_font_size=14,
                    show_grid=False,
                    show_a=False, show_b=False, show_func=False,
                    show_r=False, show_r2=False,
                    show_p=False, show_stderr=False, show_n=False,
                    dec_places=4,
                    txtpos=None,
                    legend=False,
                    show_empty_datasets_in_legend=False,
                    legend_alpha=0.5,
                    legend_location='best',  # for details please refer to
                    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
                    legend_delta_y=1.6,
                    convex_hull=False,
                    img_file_name=''):

    """ Create a regression plot:

    :param legend_alpha:
    :param show_empty_datasets_in_legend:
    :param convex_hull:
    :param data_labels:
    :param x_values: independent data for regression
    :param y_values: dependent data for regression
    :param x_axis_label: label for horizontal axis
    :param y_axis_label: label for vertical axis
    :param simple_tex: True or False (activate basic tex functions)
    :param x_min: minimum limit of horizontal axis
    :param x_max: maximum limit of horizontal axis
    :param y_min: minimum limit of vertical axis
    :param y_max: maximum limit of vertical axis
    :param separate_values: True, if a regression should be calculated for each dataset
    :param fig_size: size of the figure
    :param fig_size_px: figure size in pixels
    :param conf: confidence lines in percent
    :param trend: True or False to plot the regression line
    :param identity: True or False (line of identity, where y = x)
    :param density_plot: True or False (density plot instead of single values)
    :param hist_ratio: number of bins as percentage of the number of values,
           e. g. 0.01 for 10000 values results in 100 x 100 bins
    :param logscale: True or False (shall both axes use a logarithmic scale?)
    :param logregr: True or False (shall regression be based on log values?)
    :param errors: list of error values ( n = number of values)
    :param title:
    :param markers:
    :param marker_size:
    :param markeredgecolors:
    :param colors:
    :param font_style: normal, italic, ...
    :param font_family: serif, sansserif, Helvetica, Tahoma, ...
    :param labels: list of labels for each dataset to be used in the legend (legend=True)
    :param title_font_size:
    :param text_font_size:
    :param axis_font_size:
    :param legend_font_size:
    :param tick_font_size:
    :param show_grid: True or False (Grid lines at major tick marks)
    :param show_a: True or False ()
    :param show_b: True or False ()
    :param show_func: True or False ()
    :param show_r: True or False ()
    :param show_r2: True or False ()
    :param show_p: True or False ()
    :param show_stderr: True or False ()
    :param show_n: True or False ()
    :param dec_places: True or False ()
    :param txtpos:
    :param legend: True or False (show legend)
    :param legend_location:
    :param legend_delta_y: distance between legends for datasets
    :param img_file_name: leave blank if figure should not be saved as uimage file

    :return: True

    """

    if txtpos is None:
        txtpos = [0.02, 0.8]

    if fig_size_px is not None:
        dpi = 100

        width = fig_size_px[0] / dpi
        height = fig_size_px[1] / dpi
        fig_size = [width, height]

    fig = plt.figure(figsize=fig_size)
    rect = fig.patch
    rect.set_facecolor('#ffffff')

    axes = fig.add_subplot(111)

    graphs = []

    if colors is None:
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'r']

    if markers is None:
        markers = ['o', 'v', 's', '*', '+', 'x', 'D', '^', '8', 'h', 'p', '.', ',', 'H', '<', 'd', '|', '>']

    for idx in range(0, len(x_values)):

        if len(x_values[idx]) == 0:
            continue

        dataset_color = colors[idx]

        if markeredgecolors is not None:
            markeredgecolor = markeredgecolors[idx]
            # if markeredgecolor == dataset_color:
            #     markeredgewidth = 0.0
        else:
            markeredgecolor = None

        marker = markers[idx]

        if labels is None:
            label = 'Dataset ' + str(idx)
        else:
            label = labels[idx]

        if errors is None:
            error = None
        else:
            error = errors[idx]

        the_properties = dict(c=dataset_color, marker=marker, label=label, edgecolor=markeredgecolor)
        graphs.append([x_values[idx], y_values[idx], error, the_properties])

    # x1 und x2 enthalten die Grenzen der Y-Achse
    # - identity: line of equality darstellen?
    # - confidence: linearer Vertrausensbereich um die "Line of equality"
    # - trend: Regressionsgerade anzeigen?
    x1, x2 = GICharts.scatter_plot(axes, graphs, xoutlier=None, youtlier=None,
                                   data_labels=data_labels,
                                   separate_values=separate_values,
                                   x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                   identity=identity, confidence=conf,
                                   trend=trend, density_plot=density_plot, hist_ratio=hist_ratio,
                                   logscale=logscale, logregr=logregr,
                                   marker_size=marker_size,
                                   legend=legend,
                                   show_empty_datasets_in_legend=show_empty_datasets_in_legend,
                                   legend_alpha=legend_alpha,
                                   legend_location=legend_location,
                                   # for details please refer to
                                   # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
                                   show_grid=show_grid,
                                   show_a=show_a, show_b=show_b, show_func=show_func,
                                   show_r=show_r, show_r2=show_r2,
                                   show_p=show_p, show_stderr=show_stderr, show_n=show_n,
                                   dec_places=dec_places,
                                   txtpos=txtpos, legend_delta_y=legend_delta_y,
                                   font_family=font_family,
                                   tick_font_size=tick_font_size, text_font_size=text_font_size,
                                   axis_font_size=axis_font_size, legend_font_size=legend_font_size
                                   )
    if simple_tex:
        x_axis_label = giutils.convert_2_tex(x_axis_label, font_family=font_family, font_style=font_style)
        y_axis_label = giutils.convert_2_tex(y_axis_label, font_family=font_family, font_style=font_style)
        title = giutils.convert_2_tex(title, font_family=font_family, font_style=font_style)

    axes.set_xlabel(x_axis_label, fontdict={'family': font_family, 'style': font_style, 'fontsize': axis_font_size})
    axes.set_ylabel(y_axis_label, fontdict={'family': font_family, 'style': font_style, 'fontsize': axis_font_size})
    axes.set_title(title, fontdict={'family': font_family, 'style': font_style, 'fontsize': title_font_size})

    if x1 > 0 and x_min is None:
        axes.set_xbound(lower=0)

    y1, y2 = axes.set_ylim()

    if y1 > 0 and y_min is None:
        axes.set_ybound(lower=0)

    if convex_hull is True and 1 == 0:
        list_of_points = np.array([x_values[0], y_values[0]]).transpose()
        hull = spatial.Delaunay(list_of_points).convex_hull
        polygon = matplotlib.patches.Polygon(hull, closed=True)
        axes.add_patch(polygon)

    if img_file_name == '':
        plt.show()
    else:
        fig.savefig(img_file_name)

    plt.close()

    return True


# Todo: dataframe with missing values should be taken as input
def plot_correlation_matrix(datasets,  # list of list of n values each
                            x_axis_label='x axis', y_axis_label='y axis',
                            data_labels=None,
                            simple_tex=False,
                            x_min=0, x_max=1, y_min=0, y_max=1,
                            sub_plot_size=200,
                            fig_size=None,
                            fig_size_px=None,
                            conf=0, trend=False, identity=False,
                            density_plot=False, hist_ratio=0.02,
                            logscale=False, logregr=True,
                            title='',
                            markers=None, marker_size=20, colors=None,
                            font_style='normal',
                            font_family='serif', labels=None,
                            label_font_size=20, title_font_size=24, sub_title_font_size=20,
                            text_font_size=15, axis_font_size=14, legend_font_size=12, tick_font_size=14,
                            show_grid=False, grid_color=None,
                            show_a=False, show_b=False, show_func=False,
                            show_r=False, show_r2=False,
                            show_p=False, show_stderr=False, show_n=False,
                            dec_places=4,
                            txtpos=None,
                            legend=False,
                            legend_location='best',  # for details please refer to
                            # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
                            legend_delta_y=1.6,
                            img_file_name=''):

    """ Calculate a correlation matrix:

    :param data_labels:
    :param datasets: independent data for regression
    :param x_axis_label: label for horizontal axis
    :param y_axis_label: label for vertical axis
    :param simple_tex: True or False (activate basic tex functions)
    :param x_min: minimum limit of horizontal axis
    :param x_max: maximum limit of horizontal axis
    :param y_min: minimum limit of vertical axis
    :param y_max: maximum limit of vertical axis
    :param sub_plot_size: sub plot size
    :param fig_size: figure size in inch
    :param fig_size_px: figure size in pixels
    :param conf: confidence lines in percent
    :param trend: True or False to plot the regression line
    :param identity: True or False (line of identity, where y = x)
    :param density_plot: True or False (density plot instead of single values)
    :param hist_ratio: number of bins as percentage of the number of values,
           e. g. 0.01 for 10000 values results in 100 x 100 bins
    :param logscale: True or False (shall both axes use a logarithmic scale?)
    :param logregr: True or False (shall regression be based on log values?)
    :param title: plot title
    :param markers: marker type
    :param marker_size: marker size
    :param colors: marker and line colors
    :param font_style: normal, italic, ...
    :param font_family: serif, sansserif, Helvetica, Tahoma, ...
    :param labels: list of labels for each dataset to be used in the legend (legend=True)
    :param label_font_size: label font size
    :param title_font_size: plot title font size
    :param sub_title_font_size: sub title font size
    :param text_font_size: text font size
    :param axis_font_size:axis label font size
    :param legend_font_size: legend font size
    :param tick_font_size:tick mark font size
    :param show_grid: True or False (Grid lines at major tick marks)
    :param grid_color: grid color
    :param show_a: True or False ()
    :param show_b: True or False ()
    :param show_func: True or False ()
    :param show_r: True or False ()
    :param show_r2: True or False ()
    :param show_p: True or False ()
    :param show_stderr: True or False ()
    :param show_n: True or False ()
    :param dec_places: True or False ()
    :param txtpos: position of the text to be plotted
    :param legend: True or False (show legend)
    :param legend_location: location of the legend
    :param legend_delta_y: distance between legends for datasets
    :param img_file_name: leave blank if figure should not be saved as uimage file
    :return: True

    """

    if txtpos is None:
        txtpos = [0.02, 0.8]

    n_datasets = len(datasets)

    dpi = 100

    margin_left = sub_plot_size / 2
    margin_bottom = sub_plot_size / 2
    margin_top = sub_plot_size / 2
    margin_right = sub_plot_size / 4

    if fig_size is None:

        if fig_size_px is not None:
            fig_width_px = fig_size_px[0]
            fig_height_px = fig_size_px[1]

            fig_width = fig_width_px * dpi / 100
            fig_height = fig_height_px * dpi / 100
            fig_size = [fig_width, fig_height]
        else:
            fig_width_px = n_datasets * sub_plot_size + margin_left + margin_right
            fig_height_px = n_datasets * sub_plot_size + margin_top + margin_bottom

            fig_width = fig_width_px * dpi / 100 / 100
            fig_height = fig_height_px * dpi / 100 / 100
            fig_size = [fig_width, fig_height]
    else:
        fig_width_px = fig_size[0] * dpi
        fig_height_px = fig_size[1] * dpi

    fig, axes = plt.subplots(nrows=n_datasets,
                             ncols=n_datasets,
                             figsize=fig_size,
                             sharey=True, sharex=True
                             )

    fig.set_facecolor('#aaaaaa')  # ffffff')

    rel_margin_left = margin_left / fig_width_px
    rel_margin_right = margin_right / fig_width_px
    rel_margin_bottom = margin_bottom / fig_height_px
    rel_margin_top = margin_top / fig_height_px

    fig.subplots_adjust(left=rel_margin_left,
                        bottom=rel_margin_bottom,  # fig_margin/fig_height,
                        right=1 - rel_margin_right,
                        top=1 - rel_margin_top,
                        hspace=0.1, wspace=0.1
                        )

    sup_title = fig.suptitle(title,
                             x=0.5, y=0.98,
                             fontsize=title_font_size,
                             horizontalalignment='center',
                             verticalalignment='top'
                             )

    # Adapt title position
    title_offset = (0, 0)
    sup_title.set_position((sup_title.get_position()[0] + title_offset[0],
                            sup_title.get_position()[1] + title_offset[1]))
    enumerator = axes.flat

    for idx_ax, ax in enumerate(enumerator, start=0):  # for ax in axes.flat:

        idx_row = int(idx_ax / n_datasets)
        idx_col = idx_ax % n_datasets

        x_values = datasets[idx_col]
        y_values = datasets[idx_row]

        if labels is not None:
            sub_plot_title = labels[idx_col]
            if idx_row == idx_col:
                ax_title = ax.set_title(str.strip(sub_plot_title),
                                        fontdict={'family': font_family,
                                                  'style': font_style,
                                                  'fontsize': sub_title_font_size}
                                        )

                ax_title.set_position((0.5, 0.8))

        axis_label_font_dict = {'family': font_family, 'style': font_style, 'fontsize': label_font_size}

        if simple_tex:
            x_axis_label = giutils.convert_2_tex(x_axis_label, font_family=font_family, font_style=font_style)
            y_axis_label = giutils.convert_2_tex(y_axis_label, font_family=font_family, font_style=font_style)
            # title_tex = giutils.convert_2_tex(title, font_family=font_family, font_style=font_style)

        if idx_row == n_datasets - 1:
            ax.set_xlabel(x_axis_label, fontdict=axis_label_font_dict, color='k')

        if idx_col % n_datasets == 0:
            ax.set_ylabel(y_axis_label, fontdict=axis_label_font_dict, color='k')

        # if labels is not None:
        # font_dict = {'family':font_family, 'style':font_style, 'fontsize':sub_title_font_size}
        # ax.set_title(labels[idx_row], fontdict=font_dict, color='k')

        graphs = []

        if colors is None:
            colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'r']

        if markers is None:
            markers = ['o', 'v', 's', '*', '+', 'x', 'D', '^', '8', 'h', 'p', '.', ',', 'H', '<', 'd', '|', '>']

        color = colors[0]
        marker = markers[0]

        if labels is None:
            label = 'Dataset ' + str(idx_row)
        else:
            label = labels[idx_row]

        errors = None

        the_properties = dict(c=color, marker=marker, label=label)
        graphs.append([x_values, y_values, errors, the_properties])

        # x1 und x2 enthalten die Grenzen der Y-Achse
        # - identity: line of equality darstellen?
        # - confidence: linearer Vertrausensbereich um die "Line of equality"
        # - trend: Regressionsgerade anzeigen?

        if idx_row == idx_col and legend:
            show_legend = True
        else:
            show_legend = False

        x1, x2 = GICharts.scatter_plot(ax, graphs, xoutlier=None, youtlier=None,
                                       data_labels=data_labels,
                                       x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                       identity=identity, confidence=conf,
                                       trend=trend, density_plot=density_plot, hist_ratio=hist_ratio,
                                       logscale=logscale, logregr=logregr,
                                       marker_size=marker_size,
                                       legend=show_legend,
                                       legend_location=legend_location,
                                       # for details please refer to
                                       # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
                                       show_grid=show_grid, grid_color=grid_color,
                                       show_a=show_a, show_b=show_b, show_func=show_func,
                                       show_r=show_r, show_r2=show_r2,
                                       show_p=show_p, show_stderr=show_stderr, show_n=show_n,
                                       dec_places=dec_places,
                                       txtpos=txtpos, legend_delta_y=legend_delta_y,
                                       font_family=font_family,
                                       tick_font_size=tick_font_size, text_font_size=text_font_size,
                                       axis_font_size=axis_font_size, legend_font_size=legend_font_size
                                       )

        # ax.set_xlabel(x_axis_label, fontdict={'family':font_family, 'style':font_style, 'fontsize':label_font_size})
        # ax.set_ylabel(y_axis_label,fontdict={'family':font_family, 'style':font_style, 'fontsize':label_font_size})
        # ax.set_title(title, fontdict={'family':font_family, 'style':font_style, 'fontsize':title_font_size})

        if x1 > 0 and x_min is None:
            ax.set_xbound(lower=0)

        y1, y2 = ax.set_ylim()
        if y1 > 0 and y_min is None:
            ax.set_ybound(lower=0)

    if img_file_name == '':
        plt.show()
    else:
        fig.savefig(img_file_name)

    plt.close()

    return True


def box_plot(datasets, dataset_names, dataset_colors,
             x_axis_title, y_axis_title,
             y_limits=None,
             plot_title='',
             fig_size=None,
             fig_size_px=None,
             bg_color='#ffffff',
             iqrs=1.5,
             font_size=15,
             font_style='normal',
             font_family='sans-serif',
             img_file_name=''):

    """ Create a box plot:

    :param datasets: list of lists of values
    :param dataset_names: dataset lables for x axis
    :param dataset_colors: dataset colors
    :param x_axis_title:
    :param y_axis_title:
    :param y_limits: y axis range (list of two values)
    :param plot_title:
    :param fig_size:
    :param fig_size_px:
    :param bg_color: defaults to white
    :param iqrs: number of interquartile ranges to define min and max and thgus outliers
    :param font_size:
    :param font_style: normal, italic, ..
    :param font_family: serif, sansserif, Helvetica, Tahoma, ...
    :param img_file_name: leave blank, if figure should not be saved as image file

    :return: True

    """

    def set_box_colors(boxes, colors):

        """ Set colors for box plot:

        :param boxes: box plot to be colored
        :param colors: colors

        :return: True
        """
        for idx_dataset in range(len(colors)):
            idx1 = idx_dataset * 2
            idx2 = idx1 + 1

            color = colors[idx_dataset]

            # Box (P25 to P75)
            setp(boxes['boxes'][idx_dataset], color=color)

            # upper limit
            setp(boxes['caps'][idx1], color=color)
            # lower limit
            setp(boxes['caps'][idx2], color=color)

            # vertical lines (P0 to P25 and P75 to P100)
            setp(boxes['whiskers'][idx1], color=color)
            setp(boxes['whiskers'][idx2], color=color)

            # horizontal line indicating the median; optionally 'means'

            setp(boxes['medians'][idx_dataset], color=color)

        for idx_fliers in range(len(boxes['fliers'])):
            setp(boxes['fliers'][idx_fliers], color='#000000')

        return True

    # 1. Check configuration
    # 1.1 Check number of datasets
    if dataset_names is not None:
        if len(datasets) != len(dataset_names):
            print('Error in configuration. Number of my_datasets does not correspond to number of ',
                  'dataset names.')
            exit()

    if dataset_colors is not None:
        if len(datasets) != len(dataset_colors):
            print('Error in configuration. Number of my_datasets does not correspond to number of ',
                  'dataset colors.')
            exit()

    if bg_color == '' or bg_color is None:
        bg_color = '#ffffff'

    if fig_size_px is not None:
        dpi = 100
        width = fig_size_px[0] / dpi
        height = fig_size_px[1] / dpi
        fig_size = [width, height]

    fig = plt.figure(figsize=fig_size)

    rect = fig.patch
    rect.set_facecolor('#ffffff')

    ax = fig.add_subplot(111, axisbg=bg_color)

    # 1.2 Check y scale
    if y_limits is not None:
        if isinstance(y_limits, list) and len(y_limits) == 2:
            y_min = y_limits[0]
            y_max = y_limits[1]

            y_min_value = giutils.min_of_nested_list(datasets)
            if y_min_value < y_min:
                print('Warning: Minimum value exceeds limits of y axis.')

            y_max_value = giutils.max_of_nested_list(datasets)
            if y_max_value > y_max:
                print('Warning: Maximum value exceeds limits of y axis.')
            ax.set_ylim(y_min, y_max)

    datasets = giutils.get_numbers_from_2d_list(datasets)

    # pos = np.array(range(len(datasets)))+1
    pos = range(1, len(datasets) + 1)

    # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
    #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
    bp = ax.boxplot(datasets,
                    sym='k+',
                    positions=pos,
                    notch=None,
                    bootstrap=None,
                    usermedians=None,
                    conf_intervals=None,
                    whis=iqrs)

    font_dict = {'family': font_family, 'style': font_style, 'fontsize': font_size}

    ax.set_xlabel(x_axis_title, fontdict=font_dict)
    ax.set_ylabel(y_axis_title, fontdict=font_dict)

    ax.set_xticklabels(dataset_names, family=font_family, style=font_style, fontsize=font_size - 2)

    tick_font = font_manager.FontProperties(family=font_family, style='normal', weight='normal', size=font_size - 2)

    for ticklabel in ax.get_xticklabels():
        ticklabel.set_fontproperties(tick_font)

    for ticklabel in ax.get_yticklabels():
        ticklabel.set_fontproperties(tick_font)

    # xtickrange = []
    # for i in range(1, len(datasets) + 1):
    # xtickrange.append(i)
    # ax.set_xticks(xtickrange)

    ax.set_title(plot_title + '\n', fontdict={'family': font_family,
                                              'style': font_style,
                                              'fontsize': font_size + 2})

    setp(bp['whiskers'], color='k', linestyle='-')
    setp(bp['fliers'], markersize=10.0)

    if dataset_colors is None:
        pass
    else:
        if len(dataset_colors) == 0:
            pass
        else:
            set_box_colors(bp, dataset_colors)

    if img_file_name == '' or img_file_name is None:

        plt.show()
    else:
        fig.savefig(img_file_name, bbox_inches='tight')

    plt.close()

    return True


def create_config_plot(config_file,
                       input_path='',
                       input_file='',
                       x_col='',
                       y_col='',
                       output_path='',
                       output_file='',
                       iqrs=None):

    """ Based on the config file a plot is created. The config file includes all input file
    specific details. However, it can be re-used, when the file specific details are given as
    optional args. They overwrite the respective key values in the config file.

    :param config_file: path and filename of configuration file (*.ini)
    :param input_path: path of input file
    :param input_file: filename of input file
    :param x_col: x column name
    :param y_col: y column name
    :param output_path: path of output file
    :param output_file: filename of output file
    :param iqrs: coefficient of interquartile range for outlier detection

    :return: image file name
    """

    # 1. Read configuration file
    if not giutils.does_file_exist(config_file):
        print('Config file for statistical calculations not found.')
        return False

    config = configparser.ConfigParser()
    config.read(config_file)
    config.sections()
    plot_type = config['plot']['plot_type']

    if plot_type != 'regression':
        print('Plot type', plot_type, 'not implemented yet.')
        return False

    # 1.1 Input properties
    # Data import
    # Todo: If filename is a directory, consider to read all text files in the respective directory
    # Should a file pattern be used?

    if input_path == '':
        input_path = config['input']['input_path']
    else:
        input_path = input_path

    if input_file == '':
        file_name = config['input']['file_name']
    else:
        file_name = input_file

    input_file = input_path + file_name

    delimiter = config['input']['delimiter']
    comments = config['input']['comments']

    # Column names exist in row 1?
    names = giutils.convert_string2bool(config['input']['names'])

    # Shall leading and trailing whitespaces be omitted (stripped) automatically?
    autostrip = giutils.convert_string2bool(config['input']['autostrip'])

    i_dataset = 1
    if x_col == '':
        x_cols = []
        y_cols = []
        x_col = ''

        while x_col is not None:
            try:
                x_col = config['input']['x' + str(i_dataset) + '_col']
                y_col = config['input']['y' + str(i_dataset) + '_col']
                x_cols.append(x_col)
                y_cols.append(y_col)
            except ValueError:
                # End of loop
                x_col = None
            else:
                x_col = None

            i_dataset += i_dataset
    else:
        # Todo: Shall we define x and y column names for more than 1 dataset per file?
        x_cols = [x_col]
        y_cols = [y_col]

    # 1.2 Output properties
    # No string if map shall not be saved
    if output_path == '':
        output_path = config['output']['output_path']

    if output_file == '':
        output_file = config['output']['img_file_name']

    img_file_name = output_path + output_file

    # 1.3 Plot properties
    # No string if no title shall be plotted
    font_style = config['plot']['font_style']
    font_family = config['plot']['font_family']
    plot_title = config['plot']['plot_title']
    title_size = int(config['plot']['title_size'])
    fig_height_cm = int(config['plot']['fig_height_cm'])
    fig_width_cm = int(config['plot']['fig_width_cm'])

    # 1.4 Graph properties
    # Properties for dataset1
    i_dataset = 1

    dataset_colors = []
    dataset_markers = []
    dataset_labels = []
    dataset_color = ''

    while dataset_color is not None:
        try:
            dataset_color = config['graphs']['color' + str(i_dataset)]
            dataset_marker = config['graphs']['marker' + str(i_dataset)]
            dataset_label = config['graphs']['label' + str(i_dataset)]
            dataset_colors.append(dataset_color)
            dataset_markers.append(dataset_marker)
            dataset_labels.append(dataset_label)
        except ValueError:
            # End of loop
            dataset_color = None
        else:
            dataset_color = None

        i_dataset += i_dataset

    # Graph layout
    marker_size = int(config['graphs']['marker_size'])
    confidence_range = int(config['graphs']['confidence_range'])
    show_regression_line = giutils.convert_string2bool(config['graphs']['show_regression_line'])

    # Output text for regression results
    text_size = config['graphs']['text_size']
    txtpos_x = float(config['graphs']['txtpos_x'])
    # where (0, 0) corresponds to bottom left corner of the plot
    txtpos_y = float(config['graphs']['txtpos_y'])
    txtpos = [txtpos_x, txtpos_y]

    show_a = giutils.convert_string2bool(config['graphs']['show_a'])
    show_b = giutils.convert_string2bool(config['graphs']['show_b'])
    show_r = giutils.convert_string2bool(config['graphs']['show_r'])
    show_p = giutils.convert_string2bool(config['graphs']['show_p'])
    show_stderr = giutils.convert_string2bool(config['graphs']['show_stderr'])
    show_n = giutils.convert_string2bool(config['graphs']['show_n'])
    show_line_of_equality = giutils.convert_string2bool(config['graphs']['show_line_of_equality'])

    # Legend for datasets
    show_legend = giutils.convert_string2bool(config['graphs']['show_legend'])
    legend_delta_y = float(config['graphs']['legend_delta_y'])
    # choose higher value for legend_delta_y , if text is too long and blocks overlap

    # 1.5 Axes properties
    # Definition of axes
    x_axis_label = config['axes']['x_axis_label']
    y_axis_label = config['axes']['y_axis_label']

    # Set minimum values to None if min should be set automatically or
    # choose a distinct value
    x_min = giutils.convert_string2float(config['axes']['x_min'],
                                         nan_string='None', nan_replace_value=None)
    x_max = giutils.convert_string2float(config['axes']['x_max'],
                                         nan_string='None', nan_replace_value=None)
    y_min = giutils.convert_string2float(config['axes']['y_min'],
                                         nan_string='None', nan_replace_value=None)
    y_max = giutils.convert_string2float(config['axes']['y_max'],
                                         nan_string='None', nan_replace_value=None)

    # Shall the x axes be inverted
    invert_x_axis = giutils.convert_string2bool(config['axes']['invert_x_axis'])

    # Shall a logarithmic scale be used?
    use_log_scale = giutils.convert_string2bool(config['axes']['use_log_scale'])
    axis_tick_size = int(config['axes']['axis_tick_size'])
    axis_label_size = int(config['axes']['axis_label_size'])

    # 2. Read data_sh file
    if delimiter == '\\t' or delimiter == '':
        my_data = np.recfromtxt(input_file,
                                delimiter=b'\t',  # Tabulator muss explizit als b'\t' beschrieben werden
                                missing_values='NA',  # Consecutive delimiters will not be considered as missing value
                                filling_values=None,
                                names=names,  # True indicates a header row with column names
                                comments=comments,  # Comment marker
                                autostrip=autostrip)  # Leading and trailing spaces will be omitted
    else:
        my_data = np.recfromtxt(input_file,
                                delimiter=delimiter,
                                missing='',  # Consecutive delimiters will not be considered as missing value
                                names=names,  # True indicates a header row with column names
                                comments=comments,  # Comment marker
                                autostrip=autostrip)  # Leading and trailing spaces will be omitted

    x_datasets = []
    y_datasets = []
    x_omitted_dataset = []
    y_omitted_dataset = []

    errors = []

    for i_dataset in range(len(x_cols)):

        x_dataset = my_data[x_cols[i_dataset]]
        y_dataset = my_data[y_cols[i_dataset]]

        x_values = []
        y_values = []

        # Omit NAN values pairwise
        for i in range(len(x_dataset)):
            x_value = x_dataset[i]
            y_value = y_dataset[i]

            if math.isnan(x_value) is False and math.isnan(y_value) is False:
                x_values.append(x_value)
                y_values.append(y_value)

        if iqrs is not None and iqrs > 0:
            xy_outlier_datasets = gistats.omit_xy_outliers_by_iqr(x_values, y_values, iqrs=iqrs)
            x_dataset = xy_outlier_datasets[0]
            y_dataset = xy_outlier_datasets[1]
            x_omitted_dataset += xy_outlier_datasets[2]
            y_omitted_dataset += xy_outlier_datasets[3]
        else:
            x_dataset = x_values
            y_dataset = y_values

        x_datasets.append(x_dataset)
        y_datasets.append(y_dataset)
        # Todo: Consider definition of errors (to be defined in config file?)
        errors.append(None)

    # 3. Create plot
    # 3.1 Create new figure
    # fig = figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    fig = plt.figure(figsize=(fig_width_cm, fig_height_cm))
    rect = fig.patch
    rect.set_facecolor('#ffffff')

    # 3.2 Add plot axes at a relative position
    # e. g. 111 = row 1, col 1, plot 1
    def create_axes(row_position=1, col_position=1, num_plots=1):

        """ Define the index of the axes:

        :param row_position: number of rows in the subplot matrix
        :param col_position: column of columns in the subplot matrix
        :param num_plots: number of the subplot, starting with 1

        :return: axes
        """

        sub_plot_arg = 100 * row_position + 10 * col_position + num_plots
        axes = fig.add_subplot(sub_plot_arg)

        return axes

    # Todo: Modify if more than 1 plot should be plotted in one figure
    axes = create_axes()

    # 3.3 Define a so-called "graph"ical representation for each dataset
    def create_graphs(x_datasets, y_datasets,
                      errors,
                      dataset_colors,
                      dataset_markers,
                      dataset_labels):
        """ Create graph dictionnaries:

        :param x_datasets:
        :param y_datasets:
        :param errors:
        :param dataset_colors:
        :param dataset_markers:
        :param dataset_labels:

        :return: graphs
        """

        graphs = []
        for i_dataset in range(len(x_datasets)):
            graphs.append((x_datasets[i_dataset], y_datasets[i_dataset], errors[i_dataset],
                           dict(c=dataset_colors[i_dataset], marker=dataset_markers[i_dataset],
                                label=dataset_labels[i_dataset])))
        return graphs

    graphs = create_graphs(x_datasets, y_datasets, errors,
                           dataset_colors, dataset_markers, dataset_labels)

    # 3.4 Create the plot
    x1, x2 = GICharts.scatter_plot(axes, graphs,
                                   xoutlier=x_omitted_dataset,
                                   youtlier=y_omitted_dataset,
                                   identity=show_line_of_equality,
                                   confidence=confidence_range,
                                   trend=show_regression_line,
                                   invx=invert_x_axis,
                                   logscale=use_log_scale,
                                   marker_size=marker_size,
                                   legend=show_legend,
                                   show_grid=False,
                                   show_a=show_a, show_b=show_b,
                                   show_r=show_r, show_p=show_p,
                                   show_stderr=show_stderr,
                                   show_n=show_n,
                                   txtpos=txtpos,
                                   legend_delta_y=legend_delta_y,
                                   text_font_size=text_size,
                                   axis_font_size=axis_tick_size,
                                   font_family=font_family)

    print('x range:', x1, x2)
    # 3.5 Axes layout
    axes.set_xbound(lower=x_min, upper=x_max)
    axes.set_ybound(lower=y_min, upper=y_max)

    axes.set_xlabel(' \n' + x_axis_label,
                    fontdict={'family': font_family,
                              'style': font_style,
                              'fontsize': axis_label_size})
    axes.set_ylabel(y_axis_label + '\n',
                    fontdict={'family': font_family,
                              'style': font_style,
                              'fontsize': axis_label_size})
    axes.set_title(plot_title + '\n',
                   fontdict={'family': font_family,
                             'style': font_style,
                             'fontsize': title_size})

    # 3.6 Save plot as image file
    if img_file_name == '':
        plt.show()
    else:
        fig.savefig(img_file_name)
        print('Plot was saved as', img_file_name)

    plt.close()

    return img_file_name


def timeseries_plot_from_dataframe(my_dataframe, ts_plot_props):
    """ Create a timeseries plot for data of a dataframe:

    :param my_dataframe: dataframe containing the data
    :param ts_plot_props: timeseries plot properties

    :return: True
    """

    # Split the data by:
    # 1. value coliumns or
    # 2. group

    # X value column
    date_col = ts_plot_props['date_col']

    # Y value columns
    value_cols = ts_plot_props['value_cols']
    if type(value_cols) is str:
        value_cols = [value_cols]

    # Errors
    error_cols = ts_plot_props['error_cols']
    if type(error_cols) is str:
        error_cols = [error_cols]

    group_col = ts_plot_props['group_col']

    x_values = []
    y_values = []

    if error_cols is not None:
        errors = []
    else:
        errors = None

    dataset_labels = []

    if group_col is None:

        for idx_value_col in range(len(value_cols)):
            value_col = value_cols[idx_value_col]
            x_values.append(my_dataframe[date_col].values)
            y_values.append(my_dataframe[value_col].values)

            if errors is not None:
                errors.append(error_cols[idx_value_col])
    else:
        groups = my_dataframe[group_col].unique()
        bool_array = np.ones((1, my_dataframe.shape[0]), dtype=bool)
        for group in groups:
            bool_index = pd.Series(bool_array[0], index=my_dataframe.index)
            bool_index &= my_dataframe[group_col] == group
            result = my_dataframe[bool_index]
            x_values.append(result[date_col].values)
            y_values.append(result[value_cols[0]].values)

            if errors is not None:
                errors.append(result[error_cols[0]].values)

            dataset_labels.append(group)

    date_format = ts_plot_props['date_format']
    # x axis scaling is defined
    x_min = ts_plot_props['x_min']
    x_max = ts_plot_props['x_max']

    # Use start date from Time range selection
    start_date = None

    if x_min is not None:
        start_date = datetime.datetime.strptime(x_min, date_format)
    else:
        start = ts_plot_props['start']
        if start is not None:
            start_date = datetime.datetime.strptime(start, date_format)
    ts_plot_props['x_min'] = start_date

    if x_max is not None:
        end_date = datetime.datetime.strptime(x_max, date_format)
    else:
        end = ts_plot_props['end']
        if end is not None:
            end_date = datetime.datetime.strptime(end, date_format)
        else:
            end_date = None

    ts_plot_props['x_max'] = end_date

    ts_plot_props['y1_labels'] = dataset_labels
    ts_plot_props['y2_labels'] = None

    timeseries_plot_2(
                      x_values, y_values,
                      None, None,
                      errors1=errors, errors2=None,
                      y1_labels=dataset_labels, y2_labels=None,
                      props=ts_plot_props
                      )

    return True


def timeseries_plot_2(dates1, y1_values,
                      dates2, y2_values,
                      y1_labels=None, y2_labels=None,
                      errors1=None, errors2=None,
                      props=None):
    """ Time series plot 2: Call timeseries_plot() using a dictionary from a config file section.

    :param dates1: x values for left vertical axis
    :param y1_values: y values for left vertical axis:
    :param dates2: x values for right vertical axis
    :param y2_values: y values for right vertical axis:
    :param y1_labels: left vertical axis label
    :param y2_labels: right vertical axis label
    :param errors1: errors values for left vertical axis
    :param errors2: errors values for right vertical axis
    :param props: plot properties

    :return: True
    """

    if props is None:
        props = {
            'x_min': None,
            'x_max': None,
            'y_min': None,
            'y_max': None,
            'fig_size': None,
            'fig_size_px': None,
            'axes_pos': [0.05, 0.07, 0.9, 0.87],
            'symbol_defs': None,
            'markers': None,
            'marker_face_colors': None,
            'marker_edge_colors': None,
            'marker_sizes': None,
            'line_styles': None,
            'line_colors': None,
            'line_widths': None,
            'twelve_months_scale': False,
            'x_axis_label': 'Time',
            'interval': '',
            'y1_axis_label': None,
            'y2_axis_label': None,
            'error_label': None,
            'extra_lines': None, 'extra_line_styles': None,
            'extra_line_colors': None, 'extra_line_width': 1,
            'extra_points': None,
            'plot_title': None,
            'plot_title_offset': (0, 0),
            'title_size': 18,
            'img_filename': '',
            'thresholds': None,
            'threshold_line_styles': None,
            'text_font_size': 16, 'label_font_size': 16,
            'tick_font_size': 12, 'rotation': 30,
            'font_style': 'normal', 'font_family': 'serif',
            'simple_tex': False,
            'show_legend': False,
            'legend_loc': 'best',
            'show_grid': False
        }

    x_min = props['x_min']
    x_max = props['x_max']
    y_min = props['y_min']
    y_max = props['y_max']
    fig_size = props['fig_size']
    fig_size_px = props['fig_size_px']
    axes_pos = props['axes_pos']
    symbol_defs = props['symbol_defs']
    markers = props['markers']
    marker_face_colors = props['marker_face_colors']
    marker_edge_colors = props['marker_edge_colors']
    marker_sizes = props['marker_sizes']
    line_styles = props['line_styles']
    line_colors = props['line_colors']
    line_widths = props['line_widths']
    twelve_months_scale = props['twelve_months_scale']
    x_axis_label = props['x_axis_label']
    interval = props['interval']
    y1_axis_label = props['y1_axis_label']
    y2_axis_label = props['y2_axis_label']
    error_label = props['error_label']
    extra_lines = props['extra_lines']
    extra_line_styles = props['extra_line_styles']
    extra_line_colors = props['extra_line_colors']
    extra_line_width = props['extra_line_width']
    extra_points = props['extra_points']
    plot_title = props['plot_title']
    plot_title_offset = props['plot_title_offset']
    title_size = props['title_size']
    img_filename = props['img_filename']
    thresholds = props['thresholds']
    threshold_line_styles = props['threshold_line_styles']
    text_font_size = props['text_font_size']
    label_font_size = props['label_font_size']
    tick_font_size = props['tick_font_size']
    rotation = props['rotation']
    font_style = props['font_style']
    font_family = props['font_family']
    simple_tex = props['simple_tex']
    show_legend = props['show_legend']
    legend_loc = props['legend_loc']
    show_grid = props['show_grid']

    timeseries_plot(dates1, y1_values, dates2, y2_values,
                    y1_labels=y1_labels, y2_labels=y2_labels,
                    errors1=errors1, errors2=errors2,
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                    fig_size=fig_size,
                    fig_size_px=fig_size_px,
                    axes_pos=axes_pos,
                    symbol_defs=symbol_defs,
                    markers=markers,
                    marker_face_colors=marker_face_colors,
                    marker_edge_colors=marker_edge_colors,
                    marker_sizes=marker_sizes,
                    line_styles=line_styles, line_colors=line_colors, line_widths=line_widths,
                    twelve_months_scale=twelve_months_scale,
                    x_axis_label=x_axis_label,
                    interval=interval,
                    y1_axis_label=y1_axis_label,
                    y2_axis_label=y2_axis_label,
                    error_label=error_label,
                    extra_lines=extra_lines, extra_line_styles=extra_line_styles,
                    extra_line_colors=extra_line_colors, extra_line_width=extra_line_width,
                    extra_points=extra_points,
                    plot_title=plot_title,
                    plot_title_offset=plot_title_offset,
                    title_size=title_size,
                    img_filename=img_filename,
                    thresholds=thresholds,
                    threshold_line_styles=threshold_line_styles,
                    text_font_size=text_font_size, label_font_size=label_font_size,
                    tick_font_size=tick_font_size, rotation=rotation,
                    font_style=font_style, font_family=font_family,
                    simple_tex=simple_tex,
                    show_legend=show_legend,
                    legend_loc=legend_loc,
                    show_grid=show_grid)

    return True


def timeseries_plot(dates1, y1_values,
                    dates2, y2_values,
                    y1_labels=None, y2_labels=None,
                    errors1=None, errors2=None,
                    x_min=None, x_max=None, y_min=None, y_max=None,
                    fig_size=None,
                    fig_size_px=None,
                    axes_pos=None,
                    symbol_defs=None,
                    markers=None,
                    marker_face_colors=None,
                    marker_edge_colors=None,
                    marker_sizes=None,
                    line_styles='solid', line_colors=None, line_widths=None,
                    twelve_months_scale=False,
                    x_axis_label='Time',
                    interval='',
                    y1_axis_label=None,
                    y2_axis_label=None,
                    error_label=None,
                    extra_lines=None, extra_line_styles=None, extra_line_colors=None, extra_line_width=1,
                    extra_points=None,
                    plot_title=None,
                    plot_title_offset=(0, 0),
                    title_size=18,
                    img_filename='',
                    thresholds=None,
                    threshold_line_styles=None,
                    text_font_size=16, label_font_size=16,
                    tick_font_size=12,
                    rotation=30,
                    font_style='normal', font_family='serif',
                    simple_tex=False,
                    show_legend=False,
                    legend_loc='best',
                    show_grid=False):

    """ Create a  time series plot

    - a set of datasets can be plotted on the left y axis:
      - dates1:  a list of list of dates
      - y1_values: a list of list of values
      - errors1 (optional): a list of list of error values
    - a second set of datasets can be plotted on the right y axis
      - dates2:  a list of list of dates
      - y2_values: a list of list of values
      - errors2 (optional): a list of list of error values

    Mandatory parameter:
    :param rotation:
    :param simple_tex:
    :param plot_title_offset:
    :param axes_pos:
    :param marker_edge_colors:
    :param y_min:
    :param x_min:
    :param dates1: list of lists of dates
    :param y1_values: list of lists of values
    :param dates2: list of lists of dates
    :param y2_values: list of lists of values

    Optional parameter:
    :param y1_labels: list of labels for y1 datasets
    :param y2_labels: list of labels for y2 datasets
    :param x_min and x_max: range for x axis
    :param y_min and y_max: range for left y axis
    :param errors1: list of lists of error values for y1 datasets
    :param errors2: list of lists of error values for y2 datasets
    :param fig_size: figure size in pyplot units
    :param fig_size_px. figure size in pixel
    :param symbol_defs: e. g. 'o-b' to keep it short. Optionally, you can use the following parameters.
                        Only implemented for datasets of left y-axis.
    :param markers:
    :param marker_face_colors:
    :param marker_sizes:
    :param line_styles:
    :param line_colors:
    :param line_widths:
    :param twelve_months_scale: if set to True, a generic axis from Jan to Dec
                                will be created and years will be neglected.
    :param x_axis_label: x axis label
    :param interval: interval between x axis ticks, e. g. 'year', 'month' or 'day'
    :param y1_axis_label: left y axis label
    :param y2_axis_label: right y axis label
    :param error_label: not implemented
    :param extra_lines: a list of lists of two tupels representing line end points
    :param extra_line_styles:
    :param extra_line_colors:
    :param extra_line_width:
    :param extra_points: a list of supplemental points
    :param plot_title:
    :param title_size:
    :param img_filename: if empty string or None, the plot will not be saved.
    :param thresholds: list of values for supplemental horizontal lines
    :param threshold_line_styles:
    :param text_font_size:
    :param label_font_size:
    :param tick_font_size:
    :param font_style: normal, italic, ...
    :param font_family: serif, sansserif, ...
    :param show_legend: if set to True legend will be shown
    :param legend_loc: 'best' or 'right'
    :param show_grid: if set to True a grid will be plotted for major ticks
    """

    if axes_pos is None:
        axes_pos = [0.05, 0.1, 0.9, 0.87]

    # Convert all dates to data type datetime.datetime (e. g. Series indices, datetime64, ...
    if dates1 is not None:
        if type(dates1[0][0]) == np.datetime64:
            dates1_as_datetime = []
            for idx_dataset in range(len(dates1)):
                next_dates_as_datetime = []
                for idx_date in range(len(dates1[idx_dataset])):
                    next_date = dates1[idx_dataset][idx_date]
                    dt = pd.to_datetime(str(next_date))
                    next_dates_as_datetime.append(dt.to_datetime())
                dates1_as_datetime.append(next_dates_as_datetime)

            # if type(next_date) is not datetime.datetime and type(next_date) is not datetime.date:
            # new_dates = dates1.to_pydatetime()
            # dates1[idx_dataset] = new_dates

            dates1 = dates1_as_datetime

    if dates2 is not None:
        # for idx_dates in range(len(dates2)):
        #     dates = dates2[idx_dates]
        #     if type(dates[0]) is not datetime.datetime and type(dates[0]) is not datetime.date:
        #         new_dates = dates.to_pydatetime()
        #         dates2[idx_dates] = new_dates

        if type(dates2[0][0]) == np.datetime64:
            dates2_as_datetime = []
            for idx_dataset in range(len(dates2)):
                next_dates_as_datetime = []
                for idx_date in range(len(dates1[idx_dataset])):
                    next_date = dates1[idx_dataset][idx_date]
                    dt = pd.to_datetime(str(next_date))
                    next_dates_as_datetime.append(dt.to_datetime())
                dates2_as_datetime.append(next_dates_as_datetime)

            # if type(next_date) is not datetime.datetime and type(next_date) is not datetime.date:
            # new_dates = dates1.to_pydatetime()
            # dates1[idx_dataset] = new_dates

            dates2 = dates2_as_datetime

    if twelve_months_scale is True:
        if dates1 is not None:
            for idx_dates in range(len(dates1)):

                dates = dates1[idx_dates]
                if type(dates[0]) is not datetime.datetime and type(dates[0]) is not datetime.date:
                    dates = [date.to_pydatetime() for date in dates]

                values = y1_values[idx_dates]

                new_dates = []
                new_values = []

                last_date = None

                for idx_date in range(len(dates)):
                    # Change year to 1900
                    old_date = dates[idx_date]

                    # The following lines insert "None" dates where a timeseries has a month without data
                    # Data points that are surrounded by two such "None" dates will not be plotted
                    # in pdfs by pyplot.errorbar
                    # https://github.com/matplotlib/matplotlib/issues/7293
                    # ToDo: see if this issue is resolved with the next 2.0.1 bugfix
                    new_date = None
                    if old_date is not None:
                        if idx_date > 0 and (old_date.month < last_date.month or old_date.month - 1 > last_date.month):
                            new_values.append(None)
                            new_dates.append(None)

                        if old_date.month == 2 and old_date.day == 29:
                            new_date = datetime.date(2000, old_date.month, 28)
                        else:
                            new_date = datetime.date(2000, old_date.month, old_date.day)

                        new_dates.append(new_date)
                        new_values.append(values[idx_date])

                    last_date = new_date

                dates1[idx_dates] = new_dates
                y1_values[idx_dates] = new_values

        if dates2 is not None:
            for idx_dates in range(len(dates2)):
                dates = dates2[idx_dates]
                if type(dates[0]) is not datetime.datetime and type(dates[0]) is not datetime.date:
                    dates = dates.to_pydatetime()

                for idx_date in range(len(dates)):
                    # Change year to 1900
                    old_date = dates[idx_date]

                    if old_date.month == 2 and old_date.day == 29:
                        new_date = datetime.date(2000, old_date.month, 28)
                    else:
                        new_date = datetime.date(2000, old_date.month, old_date.day)
                    dates[idx_date] = new_date

    num_datasets = len(dates1)

    if symbol_defs is None:

        if line_colors is None:
            all_line_colors = ['#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                               '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                               '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                               '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                               '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                               '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff']
            line_colors = all_line_colors[(len(all_line_colors) - num_datasets):]
        if line_styles is None:
            all_line_styles = ['None', 'None', 'None', 'None', 'None',
                               'None', 'None', 'None', 'None', 'None',
                               'None', 'None', 'None', 'None', 'None',
                               'None', 'None', 'None', 'None', 'None',
                               'None', 'None', 'None', 'None', 'None',
                               'None', 'None', 'None', 'None', 'None']
            line_styles = all_line_styles[(len(all_line_styles) - num_datasets):]
        elif line_styles == 'solid':
            all_line_styles = ['solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid']
            line_styles = all_line_styles[(len(all_line_styles) - num_datasets):]
        if line_widths is None:
            all_line_widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            line_widths = all_line_widths[(len(all_line_widths) - num_datasets):]
        if markers is None:
            all_markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o',
                           's', 's', 's', 's', 's', 's', 's', 's', 's', 's',
                           'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']
            markers = all_markers[(len(all_markers) - num_datasets - 1):]
        if marker_face_colors is None:
            all_marker_face_colors = ['#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                                      '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                                      '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                                      '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                                      '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                                      '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff']
            marker_face_colors = all_marker_face_colors[(len(all_marker_face_colors) - num_datasets):]
        if marker_edge_colors is None:
            all_marker_edge_colors = ['#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                                      '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                                      '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                                      '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                                      '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                                      '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff']
            marker_edge_colors = all_marker_edge_colors[(len(all_marker_edge_colors) - num_datasets):]

        if marker_sizes is None:
            all_marker_sizes = [8, 8, 8, 8, 8, 8, 7, 8, 8, 8,
                                8, 8, 8, 8, 8, 8, 7, 8, 8, 8,
                                8, 8, 8, 8, 8, 8, 7, 8, 8, 8]
            marker_sizes = all_marker_sizes[(len(all_marker_sizes) - num_datasets):]

    # if marker_edge_colors is not None:
    #     if marker_edge_colors == colors:
    #         marker_edge_width = 0.0

    # Todo: Major and minor tick arguments should be considered.
    if y1_labels is None:
        y1_labels = []
        for idx in range(len(y1_values)):
            y1_labels.append('')

    if y2_labels is None and y2_values is not None:
        y2_labels = []
        for idx in range(len(y2_values)):
            y2_labels.append('')

    if fig_size_px is not None:
        dpi = 100
        width = fig_size_px[0] / dpi
        height = fig_size_px[1] / dpi
        fig_size = [width, height]

    if fig_size is None:
        fig_size = (15, 7)

    fig = plt.figure(figsize=fig_size)
    rect = fig.patch
    rect.set_facecolor('#ffffff')

    if axes_pos is not None:
        ax1 = fig.add_axes(axes_pos)
    else:
        ax1 = fig.add_subplot(111)

    font_dict = {'family': font_family, 'style': font_style, 'fontsize': label_font_size}

    if simple_tex:
        x_axis_label = giutils.convert_2_tex(x_axis_label, font_family=font_family, font_style=font_style)
        y1_axis_label = giutils.convert_2_tex(y1_axis_label, font_family=font_family, font_style=font_style)
        plot_title = giutils.convert_2_tex(plot_title, font_family=font_family, font_style=font_style)

    ax1.set_xlabel(x_axis_label, fontdict=font_dict, color='k')
    ax1.set_ylabel(y1_axis_label, fontdict=font_dict, color='k')

    if plot_title is not None:
        font_dict = {'family': font_family, 'style': font_style, 'fontsize': title_size}
        title_element = ax1.set_title(plot_title, fontdict=font_dict, color='k')
        if plot_title_offset is not None:
            title_element.set_position((title_element.get_position()[0] + plot_title_offset[0],
                                        title_element.get_position()[1] + plot_title_offset[1]))

    if twelve_months_scale is True:
        start_date = datetime.date(2000, 1, 1)
        end_date = datetime.date(2000, 12, 31)
    else:
        start_date = dates1[0][0]
        end_date = dates1[0][-1]

    # Call syntax: plot_date(x, y, fmt='bo', tz=None, xdate=True, ydate=False, **kwargs)
    for idx_dataset in range(num_datasets):

        if dates1[idx_dataset][0] < start_date:
            start_date = dates1[idx_dataset][0]

        if dates1[idx_dataset][-1] > end_date:
            end_date = dates1[idx_dataset][-1]

        if symbol_defs is not None:
            ax1.plot(dates1[idx_dataset], y1_values[idx_dataset], symbol_defs[idx_dataset],
                     linewidth=1, label=y1_labels[idx_dataset])
        else:
            # Schwarze Linie für weiße Symbole
            if line_colors[idx_dataset] == '#ffffff':
                line_color = '#000000'
            else:
                line_color = line_colors[idx_dataset]

            if y1_labels is None:
                labels = None
            else:
                labels = y1_labels[idx_dataset]

            if errors1 is None:
                errors = None
            else:
                errors = errors1[idx_dataset]

            ax1.errorbar(dates1[idx_dataset], y1_values[idx_dataset],
                         label=labels,
                         marker=markers[idx_dataset],
                         markerfacecolor=marker_face_colors[idx_dataset],
                         markeredgecolor=marker_edge_colors[idx_dataset],
                         markersize=marker_sizes[idx_dataset],
                         linestyle=line_styles[idx_dataset],
                         color=line_color,
                         linewidth=line_widths[idx_dataset],
                         yerr=errors)

            # Plot marker and error bars using distinct colors
            # if errors1 is not None:
            # ax1.errorbar(dates1[idx_dataset], y1_values[idx_dataset], yerr=errors1[idx_dataset], label=error_label)
            # #, fmt=marker_style, ecolor=err_color, mew=0)

    if extra_lines is not None:
        for idx_line in range(len(extra_lines)):
            x1 = extra_lines[idx_line][0][0]
            x2 = extra_lines[idx_line][1][0]
            y1 = extra_lines[idx_line][0][1]
            y2 = extra_lines[idx_line][1][1]

            if extra_line_styles is not None:
                extra_line_style = extra_line_styles[idx_line]
            else:
                extra_line_style = 'solid'

            if extra_line_colors is not None:
                extra_line_color = extra_line_colors[idx_line]
            else:
                extra_line_color = '#ffA050'

            ax1.plot([x1, x2], [y1, y2],
                     color=extra_line_color,
                     linestyle=extra_line_style,
                     linewidth=extra_line_width)

    if extra_points is not None:
        for idx_point in range(len(extra_points)):
            x = extra_points[idx_point][0]
            y = extra_points[idx_point][1]
            ax1.plot([x], [y], 'bo')

    if thresholds is not None:
        for idx_threshold in range(len(thresholds)):

            if threshold_line_styles is None:
                linestyle = 'solid'
            else:
                linestyle = threshold_line_styles[idx_threshold]

            ax1.plot([start_date, end_date], [thresholds[idx_threshold], thresholds[idx_threshold]],
                     color='#ffA050', linestyle=linestyle)

    if x_min is not None:
        ax1.set_xlim(x_min, x_max)
    if y_min is not None:
        ax1.set_ylim([y_min, y_max])

    duration_days = (end_date - start_date).days

    my_major_locator = None
    my_minor_locator = None
    my_formatter = None

    def get_date_formatter(interval, date_format_2):
        """ Define date formatter:

        :param interval:
        :param date_format_2:

        :return: date_formatter
        """

        if interval == '':
            my_formatter = DateFormatter(date_format_2)
        else:
            if interval == 'year':
                my_date_format = '%Y'
            elif interval == 'month':
                my_date_format = '%b %Y'
            elif interval == 'day':
                my_date_format = '%d %b %Y'
            else:
                my_date_format = date_format_2

            my_formatter = DateFormatter(my_date_format)

        return my_formatter

    # For details about DateFormatter see:
    # - http://strftime.org/
    # - https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior
    if twelve_months_scale is True:
        my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=1)
        my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=13)
        my_formatter = get_date_formatter(interval, "%b")
        plt.xlim(datetime.date(2000, 1, 1), datetime.date(2000, 12, 31))
    elif interval == 'year':
        # Jahr markieren
        if duration_days > 365:
            my_major_locator = YearLocator(base=2, month=1, day=1, tz=None)
        else:
            my_major_locator = YearLocator(base=1, month=1, day=1, tz=None)
        my_minor_locator = YearLocator(base=1, month=1, day=1, tz=None)
        my_formatter = get_date_formatter(interval, "%Y")
    elif duration_days > (365 * 50):
        # Jahr markieren
        my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=12)
        my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=6)
        my_formatter = get_date_formatter(interval, "%b %Y")
    elif duration_days > (365 * 30):
        # Jahr markieren
        my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=12)
        my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=6)
        my_formatter = get_date_formatter(interval, "%b %Y")
    elif duration_days > (365 * 15):
        # Jahr markieren
        my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=12)
        my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=3)
        my_formatter = get_date_formatter(interval, "%b %Y")
    elif duration_days > (365 * 10):
        # Jahr markieren
        my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=6)
        my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=3)
        my_formatter = get_date_formatter(interval, "%b %Y")
    elif duration_days > (365 * 5):
        # Jahr markieren
        my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=6)
        my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=2)
        my_formatter = get_date_formatter(interval, "%b %Y")

    elif duration_days > 730:
        # Quartal markieren
        my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=3)
        my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=1)
        my_formatter = get_date_formatter(interval, "%b %Y")

    elif duration_days > 365:
        # 2 Monate markieren
        my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=2)
        my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=1)
        my_formatter = get_date_formatter(interval, "%b %Y")

    elif duration_days > 153 or interval == 'month':
        # Monat markieren
        my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=1)
        my_minor_locator = WeekdayLocator(MONDAY, interval=1)
        my_formatter = get_date_formatter(interval, "%b %Y")

    elif duration_days > 72:
        # 2 Wochen markieren
        my_major_locator = WeekdayLocator(MONDAY, interval=2)
        my_minor_locator = WeekdayLocator(MONDAY, interval=1)
        my_formatter = get_date_formatter(interval, "%d %b %Y")

    elif duration_days > 35 or interval == 'week':
        # Woche markieren
        # every monday
        my_major_locator = WeekdayLocator(MONDAY, interval=1)
        my_minor_locator = None
        my_formatter = get_date_formatter(interval, "%d %b %Y %H:%M:%S")
    elif duration_days > 5:
        # Tag markieren
        my_major_locator = DayLocator(interval=1)
        my_minor_locator = HourLocator(interval=6)
        my_formatter = get_date_formatter(interval, "%d %b %y %H:%M:%S")
    elif duration_days > 2:
        # 6 Stunden markieren
        my_major_locator = HourLocator(interval=6)
        my_minor_locator = HourLocator(interval=1)
        my_formatter = get_date_formatter(interval, "%d %b %Y %H:%M:%S")
    # Do not change locator if start = end, e. g. only one value is available
    elif duration_days > 0:

        # 2 Stunden markieren
        my_major_locator = HourLocator(interval=2)
        my_minor_locator = HourLocator(interval=1)
        my_formatter = get_date_formatter(interval, "%d %b %Y %H:%M:%S")

    if my_major_locator:
        ax1.xaxis.set_major_locator(my_major_locator)
    # plt.xticks(rotation=5)
    if my_formatter:
        ax1.xaxis.set_major_formatter(my_formatter)
    if my_minor_locator:
        ax1.xaxis.set_minor_locator(my_minor_locator)

    ax1.autoscale_view()

    tick_font = font_manager.FontProperties(family=font_family,
                                            style=font_style,
                                            size=tick_font_size)

    # Activate TeX ()
    # fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'], 'weight' : 'normal', 'size' : sizeOfFont}
    # rc('text', usetex=True)
    # rc('font',**fontProperties)

    for label in ax1.get_xticklabels():
        label.set_fontproperties(tick_font)

    for label in ax1.get_yticklabels():
        label.set_fontproperties(tick_font)

    ax1.xaxis.grid(show_grid, 'major')
    ax1.xaxis.grid(False, 'minor')
    ax1.yaxis.grid(show_grid, 'major')
    ax1.yaxis.grid(False, 'minor')

    if y1_axis_label is not None:
        ax1.set_ylabel(y1_axis_label, color='k')

    for tl in ax1.get_yticklabels():
        tl.set_color('k')

    if y2_values == []:
        y2_values = None

    if y2_values is not None:

        ax2 = ax1.twinx()

        if y2_axis_label is not None:
            ax2.set_ylabel(y2_axis_label, color='k')

        # ax2.plot(dates2, y2_values, 'ro-', label=y2_label[0])

        for idx_dataset in range(len(dates2)):
            # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
            #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
            ax2.plot(dates2[idx_dataset], y2_values[idx_dataset],
                     'bo-', label=y1_labels)

            # Plot marker and error bars using distinct colors
            if errors2 is not None:
                # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
                #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
                the_dates = dates2[idx_dataset]
                the_y2_values = y2_values[idx_dataset]
                ax2.errorbar(the_dates,
                             the_y2_values,
                             yerr=errors2[idx_dataset],
                             label=error_label)

        for t2 in ax2.get_yticklabels():
            t2.set_color('k')

    fig.autofmt_xdate(bottom=0.2, rotation=rotation, ha='right')

    if show_legend:
        plt.legend(loc=legend_loc, prop={'size': text_font_size, 'family': font_family}, numpoints=1)

    # plt.legend(loc='best')

    if img_filename == '' or img_filename is None:
        plt.show()
    else:
        fig.savefig(img_filename)
        print('Plot was saved as: ', img_filename)

    plt.close()

    return True


def timeseries_multiplot(dates1_collection, y1_values_collection,
                         dates2_collection, y2_values_collection,
                         y1_labels=None, y2_labels=None,
                         x_min=None, x_max=None, y_min=0, y_max=None,
                         errors1=None, errors2=None,
                         fig_size=None,
                         fig_size_px=None,
                         symbol_defs=None,
                         markers=None, marker_face_colors=None, marker_sizes=None,
                         line_styles=None, line_colors=None, line_widths=None,
                         twelve_months_scale=False,
                         x_axis_label='Time',
                         interval='',
                         y1_axis_label=None,
                         y2_axis_label=None,
                         error_label=None,
                         plot_cols=1, plot_rows=1,
                         row_labels=None,
                         col_labels=None,
                         extra_lines=None, extra_line_styles=None,
                         extra_line_colors=None, extra_line_width=1,
                         extra_points=None,
                         plot_title=None,
                         title_size=18,
                         sub_plot_titles=None,
                         sub_title_size=12,
                         img_filename='',
                         thresholds=None,
                         threshold_line_styles=None,
                         text_font_size=16, label_font_size=16,
                         tick_font_size=12, font_style='normal',
                         font_family='serif',
                         simple_tex=False,
                         show_legend=False,
                         legend_loc='best',
                         show_grid=False):
    """ Time series plot multiplot:

    - a set of n x m sets of datasets can be plotted as an n x m plot matrix.
      Each dataset is plotted on the left y axis:
      - dates1:  a list of list of dates
      - y1_values: a list of list of values
      - errors1 (optional): a list of list of error values
    - a second set of datasets can be plotted on the right y axis
      - dates2:  a list of list of dates
      - y2_values: a list of list of values
      - errors2 (optional): a list of list of error values

    Mandatory parameter:
    :param row_labels:
    :param simple_tex:
    :param col_labels:
    :param y_min:
    :param x_min:
    :param y2_values_collection:
    :param dates2_collection:
    :param y1_values_collection:
    :param dates1_collection:

    Optional parameter:
    :param y1_labels: list of (list of labels for y1 datasets)
    :param y2_labels: list of (list of labels for y2 datasets)
    :param x_min and x_max: range for x axis
    :param y_min and y_max: range for left y axis
    :param errors1: list of (list of lists of error values for y1 datasets)
    :param errors2: list of (list of lists of error values for y2 datasets)
    :param fig_size: figure size in pyplot units
    :param fig_size_px. figure size in pixel
    :param symbol_defs: e. g. 'o-b' to keep it short. Optionally, you can use the following parameters.
                        Only implemented for datasets of left y-axis.
    :param markers:
    :param marker_face_colors:
    :param marker_sizes:
    :param line_styles:
    :param line_colors:
    :param line_widths:
    :param twelve_months_scale: if set to True, a generic axis from Jan to Dec will
                                be created and years will be neglected
    :param x_axis_label: x axis label
    :param interval: interval between x axis ticks, e. g. 'year', 'month' or 'day'
    :param y1_axis_label: left y axis label
    :param y2_axis_label: right y axis label
    :param error_label: not implemented
    :param plot_rows: number of rows in the plot matrix
    :param plot_cols: number of columns in the plot matrix
    :param extra_lines: a list of lists of two tupels representing line end points
    :param extra_line_styles:
    :param extra_line_colors:
    :param extra_line_width:
    :param extra_points: a list of supplemental points
    :param plot_title:
    :param title_size:
    :param sub_plot_titles: List of sub titles
    :param sub_title_size: Sub title font size
    :param img_filename: if empty string or None, the plot will not be saved.
    :param thresholds: list of values for supplemental horizontal lines
    :param threshold_line_styles:
    :param text_font_size:
    :param label_font_size:
    :param tick_font_size:
    :param font_style: normal, italic, ...
    :param font_family: serif, sansserif, ...
    :param show_legend: if set to True legend will be shown
    :param legend_loc: 'best' or 'right'
    :param show_grid: if set to True a grid will be plotted for major ticks
    """

    # Convert all dates to data type datetime.datetime (e. g. Series indices, datetime64, ...
    if dates1_collection is not None:
        for idx_col in range(len(dates1_collection)):
            for idx_row in range(len(dates1_collection)):
                for idx_dates in range(len(dates1_collection[idx_col][idx_row])):
                    dates = dates1_collection[idx_col][idx_row][idx_dates]
                    if type(dates[0]) is not datetime.datetime and type(dates[0]) is not datetime.date:
                        new_dates = dates.to_pydatetime()
                        dates1_collection[idx_col][idx_row][idx_dates] = new_dates

    # if dates2_list is not None:
    # for idx_dates in range(len(dates2_list)):
    # dates = dates2_list[idx_dates]
    # if type(dates[0]) is not datetime.datetime and type(dates[0]) is not datetime.date:
    # new_dates = dates.to_pydatetime()
    # dates2_list[idx_dates] = new_dates

    if dates2_collection is not None:
        for idx_col in range(len(dates2_collection)):
            for idx_row in range(len(dates2_collection)):
                for idx_dates in range(len(dates2_collection[idx_col][idx_row])):
                    dates = dates2_collection[idx_col][idx_row][idx_dates]
                    if type(dates[0]) is not datetime.datetime and type(dates[0]) is not datetime.date:
                        new_dates = dates.to_pydatetime()
                        dates2_collection[idx_col][idx_row][idx_dates] = new_dates

    if twelve_months_scale is True:
        if dates1_collection is not None:
            for idx_list in range(len(dates1_collection)):
                for idx_dates in range(len(dates1_collection[idx_list])):
                    dates = dates1_collection[idx_list][idx_dates]
                    if type(dates[0]) is not datetime.datetime and type(dates[0]) is not datetime.datetime:
                        dates = dates.to_pydatetime()

                    values = y1_values_collection[idx_dates]

                    new_dates = []
                    new_values = []

                    last_date = None

                    for idx_date in range(len(dates)):
                        # Change year to 1900
                        old_date = dates[idx_date]

                        new_date = None
                        if old_date is not None:
                            if idx_date > 0 and (
                                            old_date.month < last_date.month or old_date.month - 1 > last_date.month):
                                new_values.append(None)
                                new_dates.append(None)

                            if old_date.month == 2 and old_date.day == 29:
                                new_date = datetime.date(2000, old_date.month, 28)
                            else:
                                new_date = datetime.date(2000, old_date.month, old_date.day)

                            new_dates.append(new_date)
                            new_values.append(values[idx_date])

                        last_date = new_date

                    dates1_collection[idx_dates] = new_dates
                    y1_values_collection[idx_dates] = new_values

        if dates2_collection is not None:
            for idx_list in range(len(dates2_collection)):
                for idx_dates in range(len(dates2_collection[idx_list])):
                    dates = dates2_collection[idx_list][idx_dates]
                    if type(dates[0]) is not datetime.datetime and type(dates[0]) is not datetime.date:
                        dates = dates.to_pydatetime()

                    for idx_date in range(len(dates)):
                        # Change year to 1900
                        old_date = dates[idx_date]

                        if old_date.month == 2 and old_date.day == 29:
                            new_date = datetime.datetime(2000, old_date.month, 28)
                        else:
                            new_date = datetime.datetime(2000, old_date.month, old_date.day)
                        dates[idx_date] = new_date

    # Todo: Nur den gewünschten wert berechnen, falls genau eine Angabe (x_min exoder x_max) gemacht wurde
    start_dates = []
    end_dates = []

    if x_min is None or x_max is None:

        for idx_col in range(plot_cols):

            start_date = datetime.datetime(2099, 12, 31)
            end_date = datetime.datetime(1900, 1, 1)

            for idx_row in range(plot_rows):
                for idx_dataset in range(len(dates1_collection[idx_col][idx_row])):
                    date_list = dates1_collection[idx_col][idx_row][idx_dataset]
                    min_date = date_list[0]
                    max_date = date_list[-1]
                    if min_date < start_date:
                        start_date = min_date
                    if max_date > end_date:
                        end_date = max_date
                    x_min = start_date
                    x_max = end_date

            start_dates.append(x_min)
            end_dates.append(x_max)

    num_datasets = len(dates1_collection[0])

    if num_datasets != plot_rows:
        print('Number of datasets does not fit to matrix size.')
        return False

    if symbol_defs is None:

        if line_colors is None:
            all_line_colors = ['#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                               '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                               '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                               '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                               '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                               '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff']
            line_colors = all_line_colors[(len(all_line_colors) - num_datasets):]
        if line_styles is None:
            all_line_styles = ['solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid',
                               'solid', 'solid', 'solid', 'solid', 'solid']
            line_styles = all_line_styles[(len(all_line_styles) - num_datasets):]
        if line_widths is None:
            all_line_widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            line_widths = all_line_widths[(len(all_line_widths) - num_datasets):]
        if markers is None:
            all_markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o',
                           's', 's', 's', 's', 's', 's', 's', 's', 's', 's',
                           'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', ]
            markers = all_markers[(len(all_markers) - num_datasets - 1):]
        if marker_face_colors is None:
            all_marker_face_colors = ['#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                                      '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                                      '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                                      '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff',
                                      '#ffffff', '#ffff66', '#333333', '#cccccc', '#aa77cc',
                                      '#ff9944', '#cc5555', '#93a274', '#6699dd', '#77ddff']
            marker_face_colors = all_marker_face_colors[(len(all_marker_face_colors) - num_datasets):]
        if marker_sizes is None:
            all_marker_sizes = [8, 8, 8, 8, 8, 8, 7, 8, 8, 8,
                                8, 8, 8, 8, 8, 8, 7, 8, 8, 8,
                                8, 8, 8, 8, 8, 8, 7, 8, 8, 8]
            marker_sizes = all_marker_sizes[(len(all_marker_sizes) - num_datasets):]

    # Todo: Major and minor tick arguments should be considered.
    if y1_labels is None:
        y1_labels = []
        for idx in range(len(y1_values_collection[0])):
            y1_labels.append('')

    if y2_labels is None and y2_values_collection is not None:
        y2_labels = []
        for idx in range(len(y2_values_collection)):
            y2_labels.append('')

    if fig_size_px is not None:

        dpi = 100
        width = fig_size_px[0] / dpi
        height = fig_size_px[1] / dpi
        fig_size = [width, height]

    plot_width = 400
    plot_height = 300
    fig_margin = 100
    fig_left = 0
    fig_top = 100
    plot_padding = 0

    if fig_size is None:
        fig_height = fig_top + fig_margin * 2 + plot_rows * plot_height + (plot_rows - 1) * plot_padding
        fig_width = fig_left + fig_margin * 2 + plot_cols * plot_width + (plot_cols - 1) * plot_padding
    else:
        fig_width = fig_size[0]
        fig_height = fig_size[1]

    # fig = plt.figure(figsize=fig_size)
    # rect = fig.patch
    # rect.set_facecolor('#ffffff')
    # # #
    fig, axes = plt.subplots(nrows=plot_rows,
                             ncols=plot_cols,
                             figsize=(int(fig_width / 100),
                                      int(fig_height / 100)),
                             sharey=True, sharex=False
                             )

    fig.set_facecolor('#aaaaaa')  # ffffff')

    fig.subplots_adjust(left=(fig_left + fig_margin) / fig_width,
                        bottom=0,  # fig_margin/fig_height,
                        right=1 - (fig_margin / fig_width),
                        top=1 - ((fig_top + fig_margin) / fig_height),
                        hspace=0.05, wspace=0.05
                        )

    # sup_title = giutils.convert_2_tex(abs_plot_title)    # , font_family=font_family, font_style=font_style)
    # sup_title = fig.suptitle(plot_title,
    #                          x=0.5, y=0.98,
    #                          fontsize=title_size,
    #                          horizontalalignment='center',
    #                          verticalalignment='top'
    #                          )

    # Adapt title position
    # title_offset = (0, 0)
    # sup_title.set_position((sup_title.get_position()[0] + title_offset[0],
    # sup_title.get_position()[1] + title_offset[1]))

    if plot_cols == 1 and plot_rows == 1:
        enumerator = [axes]
    else:
        enumerator = axes.flat

    for idx_ax, ax in enumerate(enumerator, start=0):  # for ax in axes.flat:

        idx_row = int(idx_ax / 3)
        idx_col = idx_ax % 3

        dates1 = dates1_collection[idx_col][idx_row]
        num_datasets = len(dates1)

        y1_values = y1_values_collection[idx_col][idx_row]

        # # #
        print()
        print('Plot:', idx_ax + 1)

        if sub_plot_titles is not None:
            ax.set_title(str.strip(sub_plot_titles[idx_row]),
                         fontdict={'family': font_family,
                                   'style': font_style,
                                   'fontsize': sub_title_size}
                         )

        axis_label_font_dict = {'family': font_family, 'style': font_style, 'fontsize': label_font_size}

        if simple_tex:
            x_axis_label = giutils.convert_2_tex(x_axis_label, font_family=font_family, font_style=font_style)
            y1_axis_label = giutils.convert_2_tex(y1_axis_label, font_family=font_family, font_style=font_style)
            plot_title = giutils.convert_2_tex(plot_title, font_family=font_family, font_style=font_style)

        ax.set_xlabel(x_axis_label, fontdict=axis_label_font_dict, color='k')
        if idx_col % plot_cols == 0:
            ax.set_ylabel(y1_axis_label, fontdict=axis_label_font_dict, color='k')

        if sub_plot_titles is not None:
            font_dict = {'family': font_family, 'style': font_style, 'fontsize': title_size}
            ax.set_title(sub_plot_titles[idx_row], fontdict=font_dict, color='k')

        # Todo: Min und Max für Datum bestimmen
        start_date = dates1[0][0]
        end_date = dates1[0][-1]

        # Call syntax: plot_date(x, y, fmt='bo', tz=None, xdate=True, ydate=False, **kwargs)
        for idx_dataset in range(num_datasets):

            # if dates1[idx_dataset][0] < start_date:
            # start_date = dates1[idx_dataset][0]
            #
            # if dates1[idx_dataset][-1] > end_date:
            # end_date = dates1[idx_dataset][-1]

            if symbol_defs is not None:
                # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
                #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
                ax.plot(dates1[idx_dataset], y1_values[idx_dataset], symbol_defs[idx_dataset],
                        linewidth=1, label=y1_labels[idx_dataset])
            else:
                # Schwarze Linie für weiße Symbole
                if line_colors[idx_dataset] == '#ffffff':
                    line_color = '#000000'
                else:
                    line_color = line_colors[idx_dataset]

                if y1_labels is None:
                    label = None
                else:
                    label = y1_labels[idx_dataset]

                # if errors1 is None:
                #     errors = None
                # else:
                #     errors = errors1[idx_dataset]

                # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
                #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
                ax.errorbar(dates1[idx_dataset], y1_values[idx_dataset],
                            label=label,
                            marker=markers[idx_dataset],
                            markerfacecolor=marker_face_colors[idx_dataset],
                            markersize=marker_sizes[idx_dataset],
                            linestyle=line_styles[idx_dataset],
                            color=line_color,
                            linewidth=line_widths[idx_dataset],
                            yerr=errors1)

                if x_min is not None:
                    # axes.set_xlim([x_min, x_max])
                    ax.set_xlim([start_dates[idx_col], end_dates[idx_col]])

                if y_max is not None:
                    # axes.set_ylim([y_min, y_max])
                    ax.set_ylim([y_min, y_max])

                    # Plot marker and error bars using distinct colors
                    # if errors1 is not None:
                    # ax1.errorbar(dates1[idx_dataset], y1_values[idx_dataset],
                    #              yerr=errors1[idx_dataset], label=error_label)
                    # #, fmt=marker_style, ecolor=err_color, mew=0)

        if extra_lines is not None:
            for idx_line in range(len(extra_lines)):

                x1 = extra_lines[idx_line][0][0]
                x2 = extra_lines[idx_line][1][0]
                y1 = extra_lines[idx_line][0][1]
                y2 = extra_lines[idx_line][1][1]

                if extra_line_styles is not None:
                    extra_line_style = extra_line_styles[idx_line]
                else:
                    extra_line_style = 'solid'

                if extra_line_colors is not None:
                    extra_line_color = extra_line_colors[idx_line]
                else:
                    extra_line_color = '#ffA050'

                ax.plot([x1, x2], [y1, y2], color=extra_line_color, linestyle=extra_line_style,
                        linewidth=extra_line_width)

        if extra_points is not None:
            for idx_point in range(len(extra_points)):
                x = extra_points[idx_point][0]
                y = extra_points[idx_point][1]
                ax.plot([x], [y], 'bo')

        if thresholds is not None:
            for idx_threshold in range(len(thresholds)):

                if threshold_line_styles is None:
                    linestyle = 'solid'
                else:
                    linestyle = threshold_line_styles[idx_threshold]

                # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
                #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
                ax.plot([start_date[idx_col], end_date[idx_col]],
                        [thresholds[idx_threshold], thresholds[idx_threshold]],
                        color='#ffA050', linestyle=linestyle)

        duration_days = (end_date - start_date).days

        # for idx_plot in range(plot_rows * plot_cols):
        # ax = plt.subplot(plot_rows, plot_cols, idx_axes + 1)
        # idx_col = idx_axes % plot_cols

        my_major_locator = None
        my_minor_locator = None
        my_formatter = None

        def get_date_formatter(interval, date_format_2):
            """ Define the date formatter:

            :param interval: Interval as string (empty string, if date_format shall be used)
            :param date_format_2: Date format as string, e. g. "%d %b %Y", where b corresponds to month

            :return: date formatter
            """

            if interval == '':
                my_formatter = DateFormatter(date_format_2)
            else:
                if interval == 'year':
                    my_date_format = '%Y'
                elif interval == 'month':
                    my_date_format = '%b %Y'
                elif interval == 'day':
                    my_date_format = '%d %b %Y'
                else:
                    my_date_format = date_format_2

                my_formatter = DateFormatter(my_date_format)

            return my_formatter

        # For details about DateFormatter see:
        # - http://strftime.org/
        # - https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior
        if twelve_months_scale is True:
            my_major_locator = YearLocator(base=1, month=1, day=1, tz=None)
            my_minor_locator = YearLocator(base=1, month=1, day=1, tz=None)
            my_formatter = get_date_formatter(interval, "%b")
        elif interval == 'year':
            # Jahr markieren
            if duration_days > 365:
                my_major_locator = YearLocator(base=2, month=1, day=1, tz=None)
            else:
                my_major_locator = YearLocator(base=1, month=1, day=1, tz=None)
            my_minor_locator = YearLocator(base=1, month=1, day=1, tz=None)
            my_formatter = get_date_formatter(interval, "%Y")
        elif duration_days > (365 * 50):
            # Jahr markieren
            my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=12)
            my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=6)
            my_formatter = get_date_formatter(interval, "%b %Y")
        elif duration_days > (365 * 30):
            # Jahr markieren
            my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=12)
            my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=6)
            my_formatter = get_date_formatter(interval, "%b %Y")
        elif duration_days > (365 * 15):
            # Jahr markieren
            my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=12)
            my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=3)
            my_formatter = get_date_formatter(interval, "%b %Y")
        elif duration_days > (365 * 10):
            # Jahr markieren
            my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=6)
            my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=3)
            my_formatter = get_date_formatter(interval, "%b %Y")
        elif duration_days > (365 * 5):
            # Jahr markieren
            my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=6)
            my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=2)
            my_formatter = get_date_formatter(interval, "%b %Y")
        elif duration_days > 730:
            # Quartal markieren
            my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=3)
            my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=1)
            my_formatter = get_date_formatter(interval, "%b %Y")
        elif duration_days > 365:
            # 2 Monate markieren
            my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=2)
            my_minor_locator = MonthLocator(range(1, 13), bymonthday=1, interval=1)
            my_formatter = get_date_formatter(interval, "%b %Y")
        elif duration_days > 153 or interval == 'month':
            # Monat markieren
            my_major_locator = MonthLocator(range(1, 13), bymonthday=1, interval=1)
            my_minor_locator = WeekdayLocator(MONDAY, interval=1)
            my_formatter = get_date_formatter(interval, "%b %Y")
        elif duration_days > 72:
            # 2 Wochen markieren
            my_major_locator = WeekdayLocator(MONDAY, interval=2)
            my_minor_locator = WeekdayLocator(MONDAY, interval=1)
            my_formatter = get_date_formatter(interval, "%d %b %Y")
        elif duration_days > 35 or interval == 'week':
            # Woche markieren
            # every monday
            my_major_locator = WeekdayLocator(MONDAY, interval=1)
            my_minor_locator = None
            my_formatter = get_date_formatter(interval, "%d %b %Y %H:%M:%S")
        elif duration_days > 5:
            # Tag markieren
            my_major_locator = DayLocator(interval=1)
            my_minor_locator = HourLocator(interval=6)
            my_formatter = get_date_formatter(interval, "%d %b %y %H:%M:%S")
        elif duration_days > 2:
            # 6 Stunden markieren
            my_major_locator = HourLocator(interval=6)
            my_minor_locator = HourLocator(interval=1)
            my_formatter = get_date_formatter(interval, "%d %b %Y %H:%M:%S")
        # Do not change locator if start = end, e. g. only one value is available
        elif duration_days > 0:
            # 2 Stunden markieren
            my_major_locator = HourLocator(interval=2)
            my_minor_locator = HourLocator(interval=1)
            my_formatter = get_date_formatter(interval, "%d %b %Y %H:%M:%S")

        if my_major_locator:
            ax.xaxis.set_major_locator(my_major_locator)
        # plt.xticks(rotation=5)
        if my_formatter:
            ax.xaxis.set_major_formatter(my_formatter)
        if my_minor_locator:
            ax.xaxis.set_minor_locator(my_minor_locator)

        ax.autoscale_view()

        tick_font = font_manager.FontProperties(family=font_family,
                                                style=font_style,
                                                size=tick_font_size)

        # Activate TeX ()
        # fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'], 'weight' : 'normal', 'size' : sizeOfFont}
        # rc('text', usetex=True)
        # rc('font',**fontProperties)

        for label in ax.get_xticklabels():
            label.set_fontproperties(tick_font)

        for label in ax.get_yticklabels():
            label.set_fontproperties(tick_font)

        ax.xaxis.grid(show_grid, 'major')
        ax.xaxis.grid(False, 'minor')
        ax.yaxis.grid(show_grid, 'major')
        ax.yaxis.grid(False, 'minor')

        # if y1_axis_label is not None:
        # ax.set_ylabel(y1_axis_label, color='k')

        for tl in ax.get_yticklabels():
            tl.set_color('k')

        if len(y2_values_collection) == 0:
            y2_values_collection = None

        if y2_values_collection is not None:

            ax2 = ax.twinx()

            if y2_axis_label is not None:
                ax2.set_ylabel(y2_axis_label, color='k')

            # ax2.plot(dates2, y2_values, 'ro-', label=y2_label[0])

            for idx_dataset in range(len(dates2_collection)):

                # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
                #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
                ax2.plot(dates2_collection[idx_dataset], y2_values_collection[idx_dataset], 'bo-', label=y1_labels)
                # Plot marker and error bars using distinct colors
                if errors2 is not None:
                    # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
                    #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)
                    ax2.errorbar(dates2_collection[idx_dataset], y2_values_collection[idx_dataset],
                                 yerr=errors2[idx_dataset], label=error_label)

            for t2 in ax2.get_yticklabels():
                t2.set_color('k')

        # x_min = start_dates[idx_col]
        # x_max = end_dates[idx_col]

        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')

        if show_legend:
            if idx_row == plot_rows - 1 and idx_col == int((plot_cols - 1) / 2):
                plt.legend(loc=legend_loc, prop={'size': text_font_size, 'family': font_family})

                legend_location = None

                if legend_location == 'best':
                    bbox_to_anchor = None
                else:
                    bbox_to_anchor = (0.88, -0.6)

                # Todo: Shall trend lines (0 - n-1) or data_sh (n - 2 * 2 - 1) appear in the figure?
                handles, labels = ax.get_legend_handles_labels()

                # Datasets should be sorted before calling plot function, otherwise use:
                # labels = sorted(labels)
                num_handles = num_datasets

                ax.legend([handle for i, handle in enumerate(handles) if i in range(num_handles)],
                          [label for i, label in enumerate(labels) if i in range(num_handles)],
                          scatterpoints=1, loc=legend_location, bbox_to_anchor=bbox_to_anchor
                          # prop=legend_font
                          )

                # plt.legend(loc='best')

    font_properties = mpl.font_manager.FontProperties(family=font_family,
                                                      style=font_style,
                                                      size=label_font_size + 2)
    if row_labels is not None:

        x_pos = (fig_margin + plot_width * 0.05) / fig_width

        for idx_row in range(plot_rows):
            real_plot_height = 246
            y_pos = 1 - (fig_margin + 0.5 * real_plot_height + idx_row * real_plot_height + 6) / fig_height
            plt.figtext(x_pos, y_pos, row_labels[idx_row],
                        font_properties=font_properties,
                        horizontalalignment=u'left')

    if col_labels is not None:

        y_pos = 1 - (fig_top + fig_margin) / fig_height * 0.9

        for idx_col in range(plot_cols):
            real_left_margin = fig_margin
            real_plot_width = 408
            real_fig_width = 1400
            x_pos = (real_left_margin + 0.5 * real_plot_width + idx_col * real_plot_width) / real_fig_width
            plt.figtext(x_pos, y_pos, col_labels[idx_col],
                        font_properties=font_properties,
                        horizontalalignment=u'center')

    for idx_row in range(len(axes)):
        for idx_ax in range(len(axes[0])):
            print(axes[idx_row][idx_ax].bbox)
    if img_filename == '' or img_filename is None:
        plt.show()
    else:
        fig.savefig(img_filename, bbox='tight')
        print('Plot was saved as: ', img_filename)

    plt.close()

    return True


def histogram_plot(data, thresholds=None,
                   x_min=None, x_max=None,
                   y_min=0, y_max=None,

                   class_width=None, classes=10,
                   plot_title='', x_axis_label='', y_axis_label='',
                   simple_tex=False,
                   fig_size=None,
                   fig_size_px=None,
                   legend=False,
                   labels=None,
                   text_pos=None, text='',
                   font_family='sans serif', font_style='normal',
                   title_font_size=20, axis_label_size=16, tick_font_size=12,
                   bar_color=None,
                   img_file_name=''):

    """ Plot a histogram based on a list of values:

    :param data: list of values for the histogram
    :param thresholds: list of values to be marked with a horizontal line
    :param x_min: minimum x axis limit
    :param x_max: maximum x axis limit
    :param y_min: minimum y axis limit
    :param y_max: maximum y axis limit
    :param class_width: width of classes
    :param classes: number of classes
    :param plot_title: plot title
    :param x_axis_label: x axis label
    :param y_axis_label: a axis label
    :param simple_tex: True or False, to activate basic Tex functions
    :param fig_size: figure size
    :param fig_size_px: figure size in pixels
    :param legend: True or False
    :param labels: label for the dataset to be used in legend
    :param text_pos: position of supplemental text
    :param text: supplemental text
    :param font_family: serif, sansserif, Helvetica, Tahoma
    :param font_style: normal, italic, ...
    :param title_font_size: title font size
    :param axis_label_size: axis label font size
    :param tick_font_size: tick font size
    :param bar_color: bar color
    :param img_file_name: leave blank, if figure should not be saved as image file

    :return: True
    """

    # Todo: Consider x_0 and x_n for class definition.
    # 1. Define the range
    if x_min is None:
        range_min = min(data)
    else:
        range_min = x_min

    if x_max is None:
        range_max = max(data)
    else:
        range_max = x_max

    hist_range = (range_min, range_max)

    # 2. Define class limits or number of classes
    if classes is None and class_width is not None:
        limit = range_min
        class_limits = [limit]
        while limit < range_max:
            limit = limit + class_width
            class_limits.append(limit)
        classes = class_limits

    frequencies, class_limits = np.histogram(data, bins=classes, range=hist_range)  # hist_range)

    if y_min is None:
        y_min = min(frequencies)
    if y_max is None:
        y_max = max(frequencies)

    width = (class_limits[1] - class_limits[0])
    class_means = (class_limits[:-1] + class_limits[1:]) / 2

    if fig_size_px is not None:
        dpi = 100
        width = fig_size_px[0] / dpi
        height = fig_size_px[1] / dpi
        fig_size = [width, height]

    fig = plt.figure(figsize=fig_size)

    # Add white background for whole figure
    rect = fig.patch
    rect.set_facecolor('#ffffff')

    if bar_color is None or bar_color == '':
        bar_color = '#aaaaaa'

    if labels is not None:
        label = labels[0]
    else:
        label = None

    # Todo: Choose 2nd y axis if needed (e. g. for 4 datasets:
    #           y_axis_id = [1, 1, 2, 2], y_min_2 = 0, y_max_2 = 1)

    plt.bar(class_means, frequencies, align='center', width=width, facecolor=bar_color, label=label)

    if thresholds is not None:
        for threshold in thresholds:
            plt.plot([threshold, threshold], [y_min, y_max],
                     color='#ffA050', linestyle='solid',
                     label='250 MW/Jahr', linewidth=2)

    if legend:
        plt.legend()

    plt.xlim(range_min, range_max)
    plt.ylim(y_min, y_max)

    locs, labels = plt.xticks()
    plt.xticks(locs, locs, fontsize=tick_font_size, family=font_family, style=font_style)

    locs, labels = plt.yticks()
    plt.yticks(locs, locs, fontsize=tick_font_size, family=font_family, style=font_style)

    if simple_tex:
        x_axis_label = giutils.convert_2_tex(x_axis_label, font_family=font_family, font_style=font_style)
        y_axis_label = giutils.convert_2_tex(y_axis_label, font_family=font_family, font_style=font_style)
        plot_title = giutils.convert_2_tex(plot_title, font_family=font_family, font_style=font_style)

    font_dict = {'family': font_family,
                 'style': font_style,
                 'fontsize': axis_label_size}

    plt.xlabel(x_axis_label, font_dict)
    plt.ylabel(y_axis_label, font_dict)

    font_dict['fontsize'] = title_font_size
    plt.title(plot_title, fontdict=font_dict)

    if text_pos is not None:
        x_pos = x_min + text_pos[0] * (x_max - x_min)
        y_pos = y_min + text_pos[1] * (y_max - y_min)

        plt.text(x_pos, y_pos, text,
                 fontdict=font_dict,
                 bbox=dict(color=bar_color, alpha=0.2))

    if img_file_name == '':
        plt.show()
    else:
        fig.savefig(img_file_name)

    plt.close()

    return True


def create_confusion_matrix(matrix_data, properties=None):

    """ Create a confusion matrix as an image file. Text output on console.

    :param matrix_data: a list of list of data, where each list corresponds to a column, i. e.:
                        - matrix_data[0][0] is the value in the upper left corner
                        - matrix_data[0][n-1] is the value in the lower left corner
    :param properties: properties as a dictionnary, e. g.:
                       - properties['text_size'], e. g. 12
                       - properties['title_size'], e. g. 12
                       - properties['matrix_title']
                       - properties['font_family']
                       - properties['font_style']
                       - properties['col_title'], e. g. 'In-Situ Database'
                       - properties['row_title'], e. g. 'IdePix'
                       - properties['col_labels'], e. g. ['Water', 'Cloud'}
                       - properties['row_labels'], e. g. ['WATER', 'CLOUD']
                       - properties['num_rows'], e. g. 2
                       - properties['num_cols'], e. g. 2
                       - properties['output_path'], e. g. ''
                       - properties['output_file'], e. g. 'output.txt'
                       - properties['output_image'], e. g. 'output.png'

                       In case that we have two classes, i. e. 2 rows and 2 columns:
                       - properties['qry_row_1'], e. g. 'NOT(ccf_F_COASTLINE = 1 OR ccf_F_LAND = 1) AND
                                                         NOT(ccf_F_CLOUD_SHIMEZ = 1 OR ccf_F_CLOUD_OTSU = 1 OR
                                                         ccf_F_CLOUD_CLOST = 1)'
                       - properties['qry_row_2'], e. g. 'NOT(ccf_F_COASTLINE = 1 OR ccf_F_LAND = 1) AND
                                                         (ccf_F_CLOUD_SHIMEZ = 1 OR ccf_F_CLOUD_OTSU = 1 OR
                                                         ccf_F_CLOUD_CLOST = 1)'
                       - properties['qry_col_1'], e. g. 'PIXEL_SURFACE_TYPE_ID = 2'
                       - properties['qry_col_2'], e. g. 'PIXEL_SURFACE_TYPE_ID = 0'

    :return: True
    """

    if properties is None:
        print('create_confusion_matrix_2: Properties for confusion matrix are missing.')
        exit(-1)

    text_size = properties['text_size']
    if text_size is None:
        text_size = 12

    title_size = properties['title_size']
    if title_size is None:
        title_size = 16

    col_title = properties['col_title']
    row_title = properties['row_title']
    col_labels = properties['col_labels']
    if col_labels is None:
        print('create_confusion_matrix: Column labels are missing.')
        exit(-1)
    row_labels = properties['row_labels']

    output_path = properties['output_path']
    if output_path is None:
        output_path = ''

    output_filename = properties['output_filename']
    if output_filename is None:
        print('Please provide filename in config file.')
        exit(-1)

    output_img_file = output_filename + '.png'
    output_txt_file = output_filename + '.txt'

    output_file = open(output_path + output_txt_file, "a")

    font_family = properties['font_family']
    font_style = properties['font_style']

    text_delta = 0

    num_classes = len(matrix_data)

    # Create the figure
    fig_scaling = properties['fig_scaling']
    fig = plt.figure(facecolor='#ffffff', figsize=(fig_scaling * 8,
                                                   fig_scaling * 6))
    ax = fig.add_subplot(111)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_aspect(0.4)
    x_scaling = properties['fig_scaling']
    if x_scaling is None:
        x_scaling = 1.2

    # result_array = np.zeros((num_classes + 5, num_classes + 8))
    ax.set_xlim([0, x_scaling * (num_classes + 4)])
    ax.set_ylim([0, num_classes + 4])
    ax.invert_yaxis()

    matrix_title = properties['matrix_title']
    title_font_properties = mpl.font_manager.FontProperties(family=font_family,
                                                            style=font_style,
                                                            size=text_size + 2)
    if matrix_title is not None:
        ax.annotate(matrix_title,
                    size=title_size, xy=(0.5, 0.8),
                    horizontalalignment='center',
                    verticalalignment='center',
                    textcoords='figure fraction',
                    fontproperties=title_font_properties)

    face_color = properties['bg_color_labels']
    line_color = properties['line_color']
    rect = patches.Rectangle((0, 0), x_scaling * (num_classes + 2),
                             num_classes + 2,
                             fc=face_color, ec=line_color)
    ax.add_patch(rect)
    rect = patches.Rectangle((1 * x_scaling, 1),
                             x_scaling * (num_classes + 1),
                             num_classes + 1,
                             fc='#ffffff', ec=line_color)
    ax.add_patch(rect)

    for i in range(num_classes):
        i_rect = i + 1
        bg_color_identity = properties['bg_color_identity']
        rect = patches.Rectangle((x_scaling * i_rect, i_rect), x_scaling * 1, 1,
                                 fc=bg_color_identity, ec=bg_color_identity)
        ax.add_patch(rect)

    rect = patches.Rectangle((1 * x_scaling, 1), x_scaling * num_classes, num_classes,
                             fill=False, ec=line_color, lw=1.5)
    ax.add_patch(rect)

    # plt.grid(b=True, which='both', color='0', linestyle='-')

    ax.annotate('Class', size=text_size, xy=(x_scaling * 0.5, 0.5),
                horizontalalignment='center',
                verticalalignment='center')

    if col_title is None:
        col_title = 'Dataset 1'
    ax.annotate(col_title, size=text_size, xy=(0.5, 0.75),
                xycoords='figure fraction',
                horizontalalignment='center',
                verticalalignment='center')

    if row_title is None:
        row_title = 'Dataset 2'
    annotation = row_title
    ax.annotate(annotation, size=text_size, xy=(0.1, 0.5),
                xycoords='figure fraction',
                rotation='vertical',
                horizontalalignment='center',
                verticalalignment='center')

    for i_class in range(num_classes):

        annotation = col_labels[i_class]
        ax.annotate(annotation, size=text_size,
                    xy=(x_scaling * (i_class + 1.5), 0.5),
                    horizontalalignment='center',
                    verticalalignment='center')

        if row_labels is None:
            row_labels = col_labels

        if row_labels is None:
            annotation = str(i_class + 1)
        else:
            annotation = row_labels[i_class]
        ax.annotate(annotation, size=text_size,
                    xy=(x_scaling * 0.5, i_class + 1.5),
                    horizontalalignment='center',
                    verticalalignment='center')

    ax.annotate('Sum', xy=(x_scaling * (num_classes + 1.5), 0.5),
                size=text_size,
                horizontalalignment='center',
                verticalalignment='center')
    ax.annotate('U A', xy=(x_scaling * (num_classes + 2.5), 0.5),
                size=text_size,
                horizontalalignment='center',
                verticalalignment='center')
    ax.annotate('E', xy=(x_scaling * (num_classes + 3.5), 0.5),
                size=text_size,
                horizontalalignment='center',
                verticalalignment='center')
    ax.annotate('Sum', xy=(x_scaling * 0.5, num_classes + 1.5),
                size=text_size,
                horizontalalignment='center',
                verticalalignment='center')
    ax.annotate('P A', xy=(x_scaling * 0.5, num_classes + 2.5),
                size=text_size,
                horizontalalignment='center',
                verticalalignment='center')
    ax.annotate('E', xy=(x_scaling * 0.5, num_classes + 3.5),
                size=text_size,
                horizontalalignment='center',
                verticalalignment='center')

    # Matrix values
    width = num_classes
    height = num_classes

    for x in range(width):
        for y in range(height):
            ax.annotate(str(matrix_data[x][y]),
                        size=text_size,
                        xy=(x_scaling * (x + 1.5), y+1.5),
                        horizontalalignment='center',
                        verticalalignment='center')
    # Column and row sums
    col_sums = []
    row_sums = []
    for column in matrix_data:
        col_sums.append(sum(column))

    for i_row in range(num_classes):
        row_sum = 0
        for column in matrix_data:
            row_sum = row_sum + column[i_row]
        row_sums.append(row_sum)

    for i_class in range(num_classes):
        ax.annotate(col_sums[i_class], size=text_size + text_delta,
                    xy=(x_scaling * (i_class + 1.5), num_classes + 1.5),
                    horizontalalignment='center',
                    verticalalignment='center')
        ax.annotate(row_sums[i_class], size=text_size+text_delta,
                    xy=(x_scaling * (num_classes + 1.5), i_class + 1.5),
                    horizontalalignment='center',
                    verticalalignment='center')

    if sum(col_sums) != sum(row_sums):
                    print('Warnung! Spalten- und Zeilensumme nicht identisch')
                    exit()

    ax.annotate(str(sum(col_sums)), size=text_size+text_delta,
                xy=(x_scaling * (num_classes + 1.5),
                    num_classes+1.5),
                horizontalalignment='center',
                verticalalignment='center')

    # plt.xticks(range(width + 2),'')
    # plt.yticks(range(height + 2),'')

    # Output
    for x in range(width):
        output_row = ''
        for y in range(height):
            output_row += str(matrix_data[y][x])
            if y != height - 1:
                output_row += '\t'

        output_file.write(output_row)
        print(output_row)

    # Calculation of statistical estimates
    users_accuracies = []
    producers_accuracies = []
    diagonal_sum = 0

    for i in range(num_classes):
        if row_sums[i] == 0:
            users_accuracies.append(0)
        else:
            users_accuracies.append(round(matrix_data[i][i] / row_sums[i] * 100, 1))
        if col_sums[i] == 0:
            producers_accuracies.append(0)
        else:
            producers_accuracies.append(round(matrix_data[i][i] / col_sums[i] * 100, 1))
        diagonal_sum = diagonal_sum + matrix_data[i][i]

    for i_class in range(num_classes):
        ax.annotate(producers_accuracies[i_class],
                    size=text_size+text_delta,
                    xy=(x_scaling * (i_class + 1.5), num_classes + 2.5),
                    horizontalalignment='center',
                    verticalalignment='center')
        ax.annotate(users_accuracies[i_class],
                    size=text_size+text_delta,
                    xy=(x_scaling * (num_classes + 2.5), i_class + 1.5),
                    horizontalalignment='center',
                    verticalalignment='center')
        ax.annotate(round(100 - producers_accuracies[i_class], 1),
                    size=text_size+text_delta,
                    xy=(x_scaling * (i_class + 1.5), num_classes + 3.5),
                    horizontalalignment='center',
                    verticalalignment='center')
        ax.annotate(round(100 - users_accuracies[i_class], 1),
                    size=text_size+text_delta,
                    xy=(x_scaling * (num_classes + 3.5), i_class + 1.5),
                    horizontalalignment='center',
                    verticalalignment='center')

    oaa = round(diagonal_sum / sum(col_sums)*100, 2)

    # OAA
    ax.annotate('OAA:', size=text_size+text_delta,
                xy=(x_scaling * (num_classes + 2.5), num_classes + 2.5),
                horizontalalignment='center',
                verticalalignment='center')

    ax.annotate(oaa, size=text_size+text_delta,
                xy=(x_scaling * (num_classes + 3.5), num_classes + 2.5),
                horizontalalignment='center',
                verticalalignment='center')

    print('PA:', producers_accuracies)
    print('UA:', users_accuracies)
    print('OAA:', oaa)
    pa_str = ('\nPA: ' + str(producers_accuracies[0]) + ' ' +
              str(producers_accuracies[1]) + '\n')
    ua_str = ('UA: ' + str(users_accuracies[0]) + ' ' +
              str(users_accuracies[1]) + '\n')
    oaa_str = 'OAA: ' + str(oaa) + '\n'

    output_file.write(pa_str)
    output_file.write(ua_str)
    output_file.write(oaa_str)

    users_accuracy = np.zeros(num_classes, float)
    producers_accuracy = np.zeros(num_classes, float)
    overall_accuracy = 0

    for i in range(num_classes):
        overall_accuracy = overall_accuracy + matrix_data[i][i]
        users_accuracy[i] = matrix_data[i][i]
        producers_accuracy[i] = matrix_data[i][i]

    matrix_sum = sum(col_sums)

    overall_accuracy = overall_accuracy / matrix_sum

    producers_accuracy = producers_accuracy / row_sums    # matrix_data.sum(axis=0)
    users_accuracy = users_accuracy / col_sums    # matrix_data.sum(axis=1)

    matrix_data_np = np.array(matrix_data)

    norm_conf_matrix = matrix_data_np / matrix_sum
    koinzidenzmatrix = matrix_data_np + matrix_data_np.transpose()

    percent_agreement = 1
    scott_interim_value_1 = 0
    scott_interim_value_2 = 0
    cohen_interim_value_1 = 0
    cohen_interim_value_2 = 0
    krippendorf_interim_value_1 = 0
    krippendorf_interim_value_2 = 0

    for i in range(num_classes):

        scott_interim_value_1 = scott_interim_value_1 + norm_conf_matrix[i][i]
        cohen_interim_value_1 = cohen_interim_value_1 + norm_conf_matrix[i][i]
        scott_interim_value_2 = (scott_interim_value_2 +
                                 math.pow((norm_conf_matrix.sum(axis=0)[i] +
                                           norm_conf_matrix.sum(axis=1)[i]) / 2., 2))

        cohen_interim_value_2 = cohen_interim_value_2 + (
            norm_conf_matrix.sum(axis=0)[i] * norm_conf_matrix.sum(axis=1)[i])
        for j in range(num_classes):  # column
            if i != j:
                percent_agreement = percent_agreement - norm_conf_matrix[i][j]
            if i != j:
                krippendorf_interim_value_1 = (krippendorf_interim_value_1 +
                                               koinzidenzmatrix[i][j])
                krippendorf_interim_value_2 = (krippendorf_interim_value_2 +
                                               koinzidenzmatrix.sum(axis=0)[i] *
                                               koinzidenzmatrix.sum(axis=1)[j])

    krippendorf_interim_value_1 = (krippendorf_interim_value_1 /
                                   koinzidenzmatrix.sum())
    krippendorf_interim_value_2 = (krippendorf_interim_value_2 /
                                   (koinzidenzmatrix.sum() * (koinzidenzmatrix.sum()-1.)))

    scotts_pi = int((scott_interim_value_1 - scott_interim_value_2) /
                    (1 - scott_interim_value_2) * 1000) / 1000
    cohens_kappa = int((cohen_interim_value_1 - cohen_interim_value_2) /
                       (1 - cohen_interim_value_2) * 1000) / 1000
    krippendorf_alpha = int((1. - (krippendorf_interim_value_1 /
                                   krippendorf_interim_value_2)) * 1000) / 1000

    print('overall_accuracy', overall_accuracy)
    print('users_accuracy', users_accuracy)
    print('producers_accuracy', producers_accuracy)
    print('percent_agreement', percent_agreement)
    print('scotts_pi', scotts_pi)
    print('cohens_kappa', cohens_kappa)
    print('krippendorf_alpha', krippendorf_alpha)
    overall_accuracy_str = 'overall_accuracy ' + str(overall_accuracy) + '\n'

    users_accuracy_str = 'users_accuracy '
    for ua in users_accuracy:
        users_accuracy_str = users_accuracy_str + ' ' + str(ua)
    users_accuracy_str += '\n'

    producers_accuracy_str = 'producers_accuracy '
    for pa in producers_accuracy:
        producers_accuracy_str = producers_accuracy_str + ' ' + str(pa)
    producers_accuracy_str += '\n'

    scotts_pi_str = str(scotts_pi) + '\n'
    cohens_kappa_str = str(cohens_kappa) + '\n'
    krippendorf_alpha_str = str(krippendorf_alpha) + '\n'
    percent_agreement_str = str(percent_agreement) + '\n'

    output_file.write(overall_accuracy_str)
    output_file.write(users_accuracy_str)
    output_file.write(producers_accuracy_str)
    output_file.write(percent_agreement_str)
    output_file.write(scotts_pi_str)
    output_file.write(cohens_kappa_str)
    output_file.write(krippendorf_alpha_str)

    ax.annotate('Scott''s Pi: ' + str(scotts_pi),
                size=text_size+text_delta,
                xy=(0.5, 0.22),
                xycoords='figure fraction',
                horizontalalignment='center',
                verticalalignment='center')

    ax.annotate('Krippendorf''s alpha: ' + str(krippendorf_alpha),
                size=text_size+text_delta,
                xy=(0.5, 0.15),
                xycoords='figure fraction',
                horizontalalignment='center',
                verticalalignment='center')

    ax.annotate('Cohen''s kappa: ' + str(cohens_kappa),
                size=text_size+text_delta,
                xy=(0.5, 0.08),
                xycoords='figure fraction',
                horizontalalignment='center',
                verticalalignment='center')

    plt.savefig(output_path + output_img_file)

    output = ''

    col_label_string = ''
    for idx_class in range(num_classes):
        col_label_string = col_label_string + '\t' + col_labels[idx_class]

    output = (output + '\n' + '\tClass' + col_label_string +
              '\tSum\tUser accuracies\tError')

    row_string = row_title

    for idx_row in range(len(matrix_data[0])):
        if idx_row > 0:
            row_string = ''
        row_string = row_string + '\t' + row_labels[idx_row]
        for idx_col in range(num_classes):
            row_string = row_string + '\t' + str(matrix_data[idx_col][idx_row])
        row_string = row_string + '\t' + str(row_sums[idx_row])
        row_string = row_string + '\t' + str(users_accuracies[idx_row])
        row_string = row_string + '\t' + str(100 - users_accuracies[idx_row])
        output = output + '\n' + row_string

    sum_row = '\tSum'
    for idx_col in range(num_classes):
        sum_row = sum_row + '\t' + str(col_sums[idx_col])
    sum_row = sum_row + '\t' + str(matrix_sum)
    output = output + '\n' + sum_row

    pa_row = '\tProducer accuracies'
    for idx_col in range(num_classes):
        pa_row = pa_row + '\t' + str(producers_accuracies[idx_col])
    pa_row = pa_row + '\t\tOverall accuracy\t' + str(overall_accuracy)
    output = output + '\n' + pa_row

    e_row = '\tError'
    for idx_col in range(num_classes):
        e_row = e_row + '\t' + str(100 - producers_accuracies[idx_col])
    output = output + '\n' + e_row

    output = output + "\nScott's Pi:\t" + str(scotts_pi)
    output = output + "\nKrippendorf's Alpha:\t" + str(krippendorf_alpha)
    output = output + "\nCohen's Kappa:\t" + str(cohens_kappa)

    print(output)

    output_file.write(output)
    output_file.close()

    print()
    print('KA:', gistats.get_krippendorfs_alpha(matrix_data))
    print('SP:', gistats.get_scotts_pi(matrix_data))
    print('CK:', gistats.get_cohens_kappa(matrix_data))

    return True
