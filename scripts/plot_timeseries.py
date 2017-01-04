#!python
__author__ = 'Daniel'

import configparser as configparser
import os as os
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.font_manager as font_manager
import modules.auxiliary_functions as divaux


def plot_timeseries(x_dates, y_values, y_label, x_min, x_max, y_min, y_max, errors, img_filename, plot_title,
                    y_axis_label):

    marker_size = 10
    tick_font_size = 12
    font_style = 'normal'
    font_family = 'Times New Roman'

    fig = plt.figure(figsize=(15, 7))
    rect = fig.patch
    rect.set_facecolor('#ffffff')
    axes = fig.add_subplot(111)

    font_dict = {'family':font_family, 'style':font_style, 'fontsize':16}
    axes.set_xlabel('$\mathbf{Date}$', fontdict=font_dict, color='k')
    axes.set_ylabel(y_axis_label, fontdict=font_dict, color='k')

    font_dict = {'family':font_family, 'style':font_style, 'fontsize':18}
    axes.set_title('\n' + plot_title, fontdict=font_dict, color='k')

    start_date = x_min
    end_date = x_max

    axes.errorbar(x_dates, y_values, label=y_label, marker='+', markerfacecolor='#666666',
                 markersize=marker_size, linestyle='-', color='#666666', linewidth=1, yerr=errors)

    if x_min is not None:
        axes.set_xlim(x_min, x_max)
    if y_min is not None:
        axes.set_ylim([y_min, y_max])

    duration_days = (end_date - start_date).days
    if duration_days > (365 * 5) and len(y_values) < 10:
        my_major_locator = dates.MonthLocator(range(1,13), bymonthday=1, interval=12)
        my_minor_locator = dates.MonthLocator(range(1,13), bymonthday=1, interval=3)
        my_formatter = dates.DateFormatter("%Y")
    elif duration_days > (365 * 5):
        my_major_locator = dates.MonthLocator(range(1,13), bymonthday=1, interval=6)
        my_minor_locator = dates.MonthLocator(range(1,13), bymonthday=1, interval=3)
        my_formatter = dates.DateFormatter("%b %Y")
    else:
        my_major_locator = dates.MonthLocator(range(1,13), bymonthday=1, interval=1)
        my_formatter = dates.DateFormatter("%b %Y")
        my_minor_locator = False
    if my_major_locator:
        axes.xaxis.set_major_locator(my_major_locator)
    if my_formatter:
        axes.xaxis.set_major_formatter(my_formatter)
    if my_minor_locator:
        axes.xaxis.set_minor_locator(my_minor_locator)
    axes.autoscale_view()
    tick_font = font_manager.FontProperties(family=font_family,
                                            style=font_style,
                                            size=tick_font_size)

    for label in axes.get_xticklabels():
        label.set_fontproperties(tick_font)
    for label in axes.get_yticklabels():
        label.set_fontproperties(tick_font)
    axes.set_ylabel(y_axis_label, color='k')
    for tl in axes.get_yticklabels():
        tl.set_color('k')

    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend(loc='upper left', prop={'family':'times new roman', 'size':12, 'style':'normal'})

    fig.savefig(img_filename)
    print('Plot was saved as: ', img_filename)

    plt.close()


def plot_dbm_timeseries(param_range, param_str, d2products_folder, lake, stats_str, blacklist):

    if d2products_folder == '':
        input_file_path = 'Lake-' + lake + '/parameter-stats-monthly/Lake-' + lake + '_' + param_str + '.txt'
    else:
        input_file_path = d2products_folder + '/Lake-' + lake + '/parameter-stats-monthly/Lake-' + lake + '_' + param_str + '.txt'

    if not os.path.exists(input_file_path):
        print('Missing ' + input_file_path)
        print('Please run extract_stats.py first')
        return
    else:
        print('Plotting decade_by_months time series for file Lake-' + lake + '_' + param_str + '.txt')

    if d2products_folder == '':
        dbm_plot_folder = 'Lake-' + lake + '/timeseries-monthly'
    else:
        dbm_plot_folder = d2products_folder + '/Lake-' + lake + '/timeseries-monthly'

    dbm_plot_path = dbm_plot_folder + '/Lake-' + lake + '_' + param_str + '_' + stats_str + '_decade.png'
    if not os.path.exists(dbm_plot_folder):
        os.makedirs(dbm_plot_folder)

    plot_title = lake.upper() + ', spatial aggregate'
    x_min = datetime.date(2002, 1, 1)
    x_max = datetime.date(2012, 7, 1)
    if param_range:
        y_min = param_range[0]
        y_max = param_range[1]
    y_axis_label = divaux.get_legend_str(param_str)
    y_label = stats_str

    # Read the parameter-stats-monthly table
    input_table = open(input_file_path)
    line_number = 0
    meas_dates = []
    meas_values = []
    if stats_str == 'average':
        errors = []
    else:
        errors = None

    for line in input_table:
        line = line.lstrip()
        line = line.replace('\n', '')
        table_row_list = line.split('\t')
        if line_number == 0:
            stats_column = table_row_list.index(stats_str)
            error_column = table_row_list.index('sigma')
        else:
            if blacklist != '' and table_row_list[0] in blacklist:
                meas_values.append(np.nan)
                if stats_str == 'average':
                    errors.append(0)
            elif round(float(table_row_list[stats_column])) == -999:
                meas_values.append(np.nan)
                if stats_str == 'average':
                    errors.append(0)
            else:
                meas_values.append(float(table_row_list[stats_column]))
                if stats_str == 'average':
                    errors.append(float(table_row_list[error_column]))
            meas_dates.append(datetime.datetime.strptime(table_row_list[0], '%Y-%m-%d'))

        line_number += 1

    # Convert LSWT to centigrade
    if param_str in ['lswt_d_mean', 'lswt_n_mean']:
        meas_values = [meas_value - 273.15 for meas_value in meas_values]

    if not param_range:
        y_min = divaux.get_range_specs(max(meas_values)*0.5)[0][0]
        y_max = divaux.get_range_specs(max(meas_values)*0.5)[0][1]

    plot_timeseries(meas_dates, meas_values, y_label=y_label, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                    errors=errors, img_filename=dbm_plot_path, plot_title=plot_title, y_axis_label=y_axis_label)


def plot_ybm_timeseries(param_range, param_str, d2products_folder, lake, stats_str, blacklist):

    if d2products_folder == '':
        input_file_path = 'Lake-' + lake + '/parameter-stats-monthly/Lake-' + lake + '_' + param_str + '.txt'
    else:
        input_file_path = d2products_folder + '/Lake-' + lake + '/parameter-stats-monthly/Lake-' + lake + '_' + param_str + '.txt'

    if not os.path.exists(input_file_path):
        print('Missing ' + input_file_path)
        print('Please run extract_stats.py first')
        return
    else:
        print('Plotting year_by_months time series for file Lake-' + lake + '_' + param_str + '.txt')

    if d2products_folder == '':
        dbm_plot_folder = 'Lake-' + lake + '/timeseries-monthly'
    else:
        dbm_plot_folder = d2products_folder + '/Lake-' + lake + '/timeseries-monthly'

    if not os.path.exists(dbm_plot_folder):
        os.makedirs(dbm_plot_folder)

    plot_title = lake.upper() + ', spatial aggregate'
    if param_range:
        y_min = param_range[0]
        y_max = param_range[1]
    y_axis_label = divaux.get_legend_str(param_str)
    y_label = stats_str

    # Read the parameter-stats-monthly table
    input_table = open(input_file_path)
    line_number = 0
    meas_dates = []
    meas_values = []
    if stats_str == 'average':
        errors = []
    else:
        errors = None

    for line in input_table:
        line = line.lstrip()
        line = line.replace('\n', '')
        table_row_list = line.split('\t')
        if line_number == 0:
            stats_column = table_row_list.index(stats_str)
            error_column = table_row_list.index('sigma')
        else:
            if blacklist != '' and table_row_list[0] in blacklist:
                meas_values.append(np.nan)
                if stats_str == 'average':
                    errors.append(0)
            elif round(float(table_row_list[stats_column])) == -999:
                meas_values.append(np.nan)
                if stats_str == 'average':
                    errors.append(0)
            else:
                meas_values.append(float(table_row_list[stats_column]))
                if stats_str == 'average':
                    errors.append(float(table_row_list[error_column]))
            meas_dates.append(datetime.datetime.strptime(table_row_list[0], '%Y-%m-%d'))

        line_number += 1

    # Convert LSWT to centigrade
    if param_str in ['lswt_d_mean', 'lswt_n_mean']:
        meas_values = [meas_value - 273.15 for meas_value in meas_values]

    if not param_range:
        y_min = divaux.get_range_specs(max(meas_values)*0.5)[0][0]
        y_max = divaux.get_range_specs(max(meas_values)*0.5)[0][1]

    for year in range(2003, 2012):
        x_min = datetime.date(year - 1, 12, 15)
        x_max = datetime.date(year, 12, 15)
        ybm_plot_path = dbm_plot_folder + '/Lake-' + lake + '_' + param_str + '_' + stats_str + '_' + str(year) + '.png'

        plot_timeseries(meas_dates, meas_values, y_label=y_label, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                        errors=errors, img_filename=ybm_plot_path, plot_title=plot_title, y_axis_label=y_axis_label)


def plot_dby_timeseries(param_range, param_str, d2products_folder, lake, stats_str, blacklist):

    input_file_path = d2products_folder + '/Lake-' + lake + '/parameter-stats-yearly/Lake-' + lake + '_' + param_str + '.txt'
    if not os.path.exists(input_file_path):
        print('Missing ' + input_file_path)
        print('Please run extract_stats.py first')
        return
    else:
        print('Plotting decade_by_years time series for file Lake-' + lake + '_' + param_str + '.txt')

    dby_plot_folder = d2products_folder + '/Lake-' + lake + '/timeseries-yearly'
    dby_plot_path = dby_plot_folder + '/Lake-' + lake + '_' + param_str + '_' + stats_str + '_decade.png'
    if not os.path.exists(dby_plot_folder):
        os.makedirs(dby_plot_folder)

    plot_title = lake.upper() + ', spatial aggregate'
    x_min = datetime.date(2002, 1, 1)
    x_max = datetime.date(2012, 1, 1)
    if param_range:
        y_min = param_range[0]
        y_max = param_range[1]
    y_axis_label = divaux.get_legend_str(param_str)
    y_label = stats_str

    # Read the parameter-stats-yearly table
    input_table = open(input_file_path)
    line_number = 0
    meas_dates = []
    meas_values = []
    if stats_str == 'average':
        errors = []
    else:
        errors = None

    for line in input_table:
        line = line.lstrip()
        line = line.replace('\n', '')
        table_row_list = line.split('\t')
        if line_number == 0:
            stats_column = table_row_list.index(stats_str)
            error_column = table_row_list.index('sigma')
        else:
            if blacklist != '' and table_row_list[0] in blacklist:
                meas_values.append(np.nan)
                if stats_str == 'average':
                    errors.append(0)
            elif round(float(table_row_list[stats_column])) == -999:
                meas_values.append(np.nan)
                if stats_str == 'average':
                    errors.append(0)
            else:
                meas_values.append(float(table_row_list[stats_column]))
                if stats_str == 'average':
                    errors.append(float(table_row_list[error_column]))
            meas_dates.append(datetime.datetime.strptime(table_row_list[0], '%Y-%m-%d'))

        line_number += 1

    # Convert LSWT to centigrade
    if param_str in ['lswt_d_mean', 'lswt_n_mean']:
        meas_values = [meas_value - 273.15 for meas_value in meas_values]

    if not param_range:
        y_min = divaux.get_range_specs(max(meas_values)*0.5)[0][0]
        y_max = divaux.get_range_specs(max(meas_values)*0.5)[0][1]

    plot_timeseries(meas_dates, meas_values, y_label=y_label, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                    errors=errors, img_filename=dby_plot_path, plot_title=plot_title, y_axis_label=y_axis_label)


def main():

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ini', 'plot_timeseries.ini'))
    method = config['DEFAULT']['method']
    d2products_folder = config['DEFAULT']['products_path']
    param_range = False if config['DEFAULT']['param_range'] == 'False' else config['DEFAULT']['param_range'].split(',')
    if param_range:
        param_range = [float(param_range[0]), float(param_range[1])]
    param_str = config['DEFAULT']['param_str']
    lake = config['DEFAULT']['lake']
    stats_str = config['DEFAULT']['stats_str']
    blacklist = config['DEFAULT']['date_blacklist']

    if method.lower() == 'decade_by_months':
        plot_dbm_timeseries(param_range=param_range, param_str=param_str, d2products_folder=d2products_folder,
                            lake=lake, stats_str = stats_str, blacklist=blacklist)
    elif method.lower() == 'year_by_months':
        plot_ybm_timeseries(param_range=param_range, param_str=param_str, d2products_folder=d2products_folder,
                            lake=lake, stats_str = stats_str, blacklist=blacklist)
    elif method.lower() == 'decade_by_years':
        plot_dby_timeseries(param_range=param_range, param_str=param_str, d2products_folder=d2products_folder,
                            lake=lake, stats_str = stats_str, blacklist=blacklist)
    else:
        print('    method not known, please select decade_by_months/year_by_months/decade_by_years')
        return


if __name__ == "__main__":
    main()