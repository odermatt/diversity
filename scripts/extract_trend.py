#!python
__author__ = 'Helga'

import statsmodels.api as sm
import matplotlib.pyplot as plt
import os as os
import modules.auxiliary_functions as divaux
import configparser as configparser
import pandas as pd
import numpy as np
import pandas as pandas


def get_trend(param, d2products_folder, lake, config):
    base_path = d2products_folder + '/Lake-' + lake
    blacklist = get_blacklist(config, d2products_folder, lake, param)
    monthly_stats_path = base_path + '/parameter-stats-monthly/Lake-' + \
                         lake + '_' + param + '.txt'
    if os.path.exists(monthly_stats_path):
        meas_dates, meas_values, errors = divaux.read_statsmonthly(monthly_stats_path, 'p90_threshold',
                                                                   blacklist)
        return calculate_trend(meas_values)


def get_parameter_stats_monthly(param, d2products_folder, lake, config):
    base_path = d2products_folder + '/Lake-' + lake
    blacklist = get_blacklist(config, d2products_folder, lake, param)
    monthly_stats_path = base_path + '/parameter-stats-monthly/Lake-' + \
                         lake + '_' + param + '.txt'
    if os.path.exists(monthly_stats_path):
        meas_dates, meas_values, errors = divaux.read_statsmonthly(monthly_stats_path, 'p90_threshold',
                                                                   blacklist)
        return (meas_values, meas_dates)
    else:
        return ([], [])


def calculate_trend(values):
    coefficients, residuals, _, _, _ = np.polyfit(range(len(values)), values, 1, full=True)
    if len(residuals) > 0 and residuals[0] != 0:
        mse = residuals[0] / (len(values))
        nrmse = np.sqrt(mse) / (max(values) - min(values))
    else:
        nrmse = np.nan
    return (coefficients[0], nrmse)


def plots_seasonal_decompose(config, d2products_folder, lakes_list, params_list):
    for lake in lakes_list:
        base_path = d2products_folder + '/Lake-' + lake
        output_folder = base_path + '/decomposition-plots'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for param in params_list:
            blacklist = get_blacklist(config, d2products_folder, lake, param)

            monthly_stats_path = base_path + '/parameter-stats-monthly/Lake-' + \
                                 lake + '_' + param + '.txt'
            if os.path.exists(monthly_stats_path):
                meas_dates, meas_values, errors = divaux.read_statsmonthly(monthly_stats_path, 'average',
                                                                           blacklist)
                # statsmodels.seasonal_decompose: Plots saved in folder 'decomposition-plots'
                data = {'date': meas_dates, 'values': meas_values}
                df = pd.DataFrame(data, columns=['date', 'values'])
                df.index = df['date']
                del df['date']
                df = df.fillna(method='bfill')
                df = df.fillna(method='ffill')
                if df.isnull().sum().values[0] == 0:
                    decomposition = sm.tsa.seasonal_decompose(df, model='additive')
                    fig = decomposition.plot()
                    plt.title(param)
                    fig.savefig(output_folder + '/Lake-' + lake + '_decomposition_' + param + '.png')


def get_blacklist(config, d2products_folder, lake, param):
    blacklist_config = config['DEFAULT']['date_blacklist']
    if blacklist_config != 'False':
        if blacklist_config.startswith('blacklist_'):
            if param == 'num_obs':
                blacklist = ''
            else:
                blacklist_path = d2products_folder + '/Lake-' + lake + '/' + blacklist_config + '/blacklist_lake-' + lake + '_' + param + '.txt'
                if os.path.exists(blacklist_path):
                    blacklist = divaux.read_blacklist(blacklist_path)
                else:
                    blacklist = ''
        elif blacklist_config.startswith('20'):
            blacklist = blacklist_config.split(',')
    else:
        blacklist = False
    return blacklist


def main():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ini', 'extract_trend.ini'))
    d2products_folder = config['DEFAULT']['products_path']

    params = config['DEFAULT']['param_str']
    params_list = [param.lstrip() for param in params.split(',')]

    lakes = config['DEFAULT']['lakes']
    lakes_list = [lake.lstrip() for lake in lakes.split(',')]

    # statsmodels.seasonal_decompose: Plots saved in folder 'decomposition-plots'
    plots_seasonal_decompose(config, d2products_folder, lakes_list, params_list)

    # read product_stats_10y
    for lake in lakes_list:
        base_path = d2products_folder + '/Lake-' + lake
        product_stats_path = base_path + '/product-stats-10y' + '/Lake-' + lake + '_2003-01-01_2011-12-31.txt'
        if os.path.exists(product_stats_path):
            table = pandas.read_csv(product_stats_path, sep='\t', header=0, na_values=[''])
            params = table['Band']
            trend_monthly_mean = []
            trend_yearly_mean = []
            trend_seasonal = []
            for param in params:
                (values, dates) = get_parameter_stats_monthly(param, d2products_folder, lake, config)
                # monthly
                (t_monthly_mean, _) = calculate_trend(values)
                trend_monthly_mean.append(t_monthly_mean)
                # yearly
                data = {'date': dates, 'values': values}
                df = pd.DataFrame(data, columns=['date', 'values'])
                df.index = df['date']
                del df['date']
                values_yearly_mean = df.groupby(pd.TimeGrouper('A')).mean()
                (t_yearly_mean, _) = calculate_trend(values_yearly_mean['values'])
                trend_yearly_mean.append(t_yearly_mean)
                # seasonal decomposition
                df = df.fillna(method='bfill')
                df = df.fillna(method='ffill')
                if df.isnull().sum().values[0] == 0:
                    decomposition = sm.tsa.seasonal_decompose(df, model='additive')
                    idx = np.isfinite(decomposition.trend['values'])
                    (t_seasonal, _) = calculate_trend(decomposition.trend['values'][idx])
                    trend_seasonal.append(t_seasonal)
                else:
                    trend_seasonal.append(np.nan)

            table = table.assign(trend_monthly_mean=trend_monthly_mean)
            table = table.assign(trend_yearly_mean=trend_yearly_mean)
            table = table.assign(trend_seasonal_decomposition=trend_seasonal)
            table.to_csv(product_stats_path + '_trend.csv', '\t')


if __name__ == "__main__":
    main()
