__author__ = 'Daniel'

import math as math
import numpy as np
import os as os
import fnmatch as fnmatch


def calculate_haversine(lon1, lat1, lon2, lat2):

    # calculate the great circle distance between two points on Earth (specified in decimal degrees)
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    km = 6367 * c
    return km


def get_tick_positions(lower, upper, n_ticks):
    coord_range = upper - lower
    exponent = round(np.log(coord_range))
    lower_floored = np.floor(lower * pow(10, - exponent)) * pow(10, exponent)
    upper_ceiled = np.ceil(upper * pow(10, - exponent)) * pow(10, exponent)
    range_section = (upper_ceiled - lower_floored) / coord_range
    grid_step = (upper_ceiled - lower_floored) / (n_ticks * range_section)
    decimal = 1
    while grid_step < 10:
        grid_step = grid_step * 10
        decimal = decimal * 10
    if grid_step < 20:
        grid_step_round = 10 / decimal
    elif grid_step < 50:
        grid_step_round = 20 / decimal
    elif grid_step < 100:
        grid_step_round = 50 / decimal
    tick_list = [lower_floored]
    current = lower_floored + grid_step_round
    while tick_list[-1] < upper_ceiled:
        tick_list.append(current)
        current = current + grid_step_round
    return tick_list


def get_range_specs(param_percentile):
    range_intervals = [5000, 2000, 1000,
                       500, 200, 100,
                       50, 20, 10,
                       5, 2, 1,
                       0.5, 0.2, 0.1,
                       0.05, 0.02, 0.01,
                       0.005, 0.002, 0.001,
                       0.0005, 0.0002, 0.0001]
    if param_percentile <= 0.0001:
        param_range = [0, 0.0001]
    else:
        for n_interval in range(2, len(range_intervals)):
            if param_percentile > range_intervals[n_interval]:
                param_range = [0, range_intervals[n_interval - 2]]
                break
    if param_range[1] >= 10:
        legend_decimals = '%.0f'
    elif param_range[1] >= 1:
        legend_decimals = '%.1f'
    elif param_range[1] >= 0.1:
        legend_decimals = '%.2f'
    elif param_range[1] >= 0.01:
        legend_decimals = '%.3f'
    elif param_range[1] >= 0.001:
        legend_decimals = '%.4f'
    elif param_range[1] >= 0.0001:
        legend_decimals = '%.5f'
    elif param_range[1] >= 0.00001:
        legend_decimals = '%.6f'
    elif param_range[1] >= 0.000001:
        legend_decimals = '%.7f'
    return(param_range, legend_decimals)


def get_legend_str(layer_str):
    if layer_str == 'num_obs':
        legend_str = '$\mathbf{number\/of\/observations}$'
    elif layer_str == 'chl_mph_mean':
        legend_str = '$\mathbf{mean\/MPH\/CHL-\mathit{a}\/[mg/m^{3}]}$'
    elif layer_str == 'chl_mph_sigma':
        legend_str = '$\mathbf{sdev\/MPH\/CHL-\mathit{a}\/[mg/m^{3}]}$'
    elif layer_str == 'chl_fub_mean':
        legend_str = '$\mathbf{mean\/FUB\/CHL-\mathit{a}\/[mg/m^{3}]}$'
    elif layer_str == 'chl_fub_sigma':
        legend_str = '$\mathbf{sdev\/FUB\/CHL-\mathit{a}\/[mg/m^{3}]}$'
    elif layer_str == 'cdom_fub_mean':
        legend_str = '$\mathbf{mean\/FUB\/CDOM\/[m^{-1}]}$'
    elif layer_str == 'cdom_fub_sigma':
        legend_str = '$\mathbf{sdev\/FUB\/CDOM\/[m^{-1}]}$'
    elif layer_str == 'tsm_cc_mean':
        legend_str = '$\mathbf{mean\/CoastColour\/TSM\/[g/m^{3}]}$'
    elif layer_str == 'tsm_cc_sigma':
        legend_str = '$\mathbf{sdev\/CoastColour\/TSM\/[g/m^{3}]}$'
    elif layer_str == 'turbidity_cc_mean':
        legend_str = '$\mathbf{mean\/CoastColour\/turbidity\/[FNU]}$'
    elif layer_str == 'turbidity_cc_sigma':
        legend_str = '$\mathbf{sdev\/CoastColour\/turbidity\/[FNU]}$'
    elif layer_str == 'immersed_cyanobacteria_mean':
        legend_str = '$\mathbf{mean\/immersed\/cyanobacteria\/probability}$'
    elif layer_str == 'floating_cyanobacteria_mean':
        legend_str = '$\mathbf{mean\/floating\/cyanobacteria\/probability}$'
    elif layer_str == 'floating_vegetation_mean':
        legend_str = '$\mathbf{mean\/floating\/vegetation\/probability}$'
    elif layer_str == 'owt_cc_dominant_class_mode':
        legend_str = '$\mathbf{dominant\/optical\/water\/type}$'
    elif layer_str == 'lswt_d_mean':
        legend_str = '$\mathbf{ARCLake\/mean\/daytime\/LSWT\/[deg\/C]}$'
    elif layer_str == 'lswt_n_mean':
        legend_str = '$\mathbf{ARCLake\/mean\/nighttime\/LSWT\/[deg\/C]}$'
    elif layer_str in ['extent', 'shallow']:
        legend_str = '$\mathbf{Maximum\/lake\/extent\/(2=shallow)}$'
    return legend_str


def get_single_label(product):
    if '2003-01-01_2011-12-31' in os.path.splitext(product)[0]:
        arranged_labels = [['2003-2011']]
    elif ('-01-01_' in os.path.splitext(product)[0]) and ('-12-31' in os.path.splitext(product)[0]):
        arranged_labels = [[os.path.splitext(product)[0][str.find(os.path.splitext(product)[0],
                           '_') + 1 : str.find(os.path.splitext(product)[0], '_') + 5]]]
    else:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
        product_basename = os.path.splitext(product)[0]
        product_year = str.split(product_basename, '_')[-1][0:4]
        product_month = months[int(str.split(product_basename, '_')[-1][5:7]) - 1]
        arranged_labels = [[product_month + ' ' + product_year]]
    return arranged_labels


def get_dby_strings(in_folder):
    arranged_labels= [['2003', '2004', '2005'],
                      ['2006', '2007', '2008'],
                      ['2009', '2010', '2011']]
    arrangement = [['2003-01-01_2003-12-31', '2004-01-01_2004-12-31', '2005-01-01_2005-12-31'],
                   ['2006-01-01_2006-12-31', '2007-01-01_2007-12-31', '2008-01-01_2008-12-31'],
                   ['2009-01-01_2009-12-31', '2010-01-01_2010-12-31', '2011-01-01_2011-12-31']]
    filepaths = []
    arranged_filepaths = []
    for file in os.listdir(in_folder):
        if file.endswith(".tif"):
            filepaths.append(file)
    for row in arrangement:
        row_filenames = []
        for element in row:
            row_filenames.append(in_folder + '/' + fnmatch.filter(filepaths, '*'+element+'*')[0])
        arranged_filepaths.append(row_filenames)
    return (arranged_filepaths, arranged_labels)


def get_ybm_strings(in_folder, year):
    filenames = []
    arranged_filepaths = []
    for file in os.listdir(in_folder):
        if file.endswith(".tif"):
            filenames.append(file)
    months_found = 0
    for filename in filenames:
        if year in filename:
            months_found = months_found + 1
    if months_found != 12:
        print('   year ' + str(year) + ' not complete, year-by-month plot skipped')
        arranged_labels = ''
    else:
        if year in ['2004', '2008', '2012']:
            feb = '29'
        else:
            feb = '28'
        arrangement = [[year + '-01-01_' + year + '-01-31', year + '-02-01_' + year + '-02-' + feb,
                        year + '-03-01_' + year + '-03-31', year + '-04-01_' + year + '-04-30'],
                       [year + '-05-01_' + year + '-05-31', year + '-06-01_' + year + '-06-30',
                        year + '-07-01_' + year + '-07-31', year + '-08-01_' + year + '-08-31'],
                       [year + '-09-01_' + year + '-09-30', year + '-10-01_' + year + '-10-31',
                        year + '-11-01_' + year + '-11-30', year + '-12-01_' + year + '-12-31']]
        arranged_labels = [['Jan ' + year, 'Feb ' + year, 'Mar ' + year, 'Apr ' + year],
                           ['May ' + year, 'Jun ' + year, 'Jul ' + year, 'Aug ' + year],
                           ['Sep ' + year, 'Oct ' + year, 'Nov ' + year, 'Dec ' + year]]
        for row in arrangement:
            row_filepaths = []
            for element in row:
                row_filepaths.append(in_folder + '/' + fnmatch.filter(filenames, '*'+element+'*')[0])
            arranged_filepaths.append(row_filepaths)
    return (arranged_filepaths, arranged_labels)