#!python
__author__ = 'Helga, Daniel'

import modules.seasonal as sm
import matplotlib.pyplot as plt
import matplotlib
import os as os
import modules.auxiliary_functions as divaux
import configparser as configparser
import pandas as pd
import numpy as np
import datetime as datetime
import fnmatch as fnmatch


def get_trend(param, d2products_folder, lake, config):
    base_path = d2products_folder + '/Lake-' + lake
    blacklist = get_blacklist(config, d2products_folder, lake, param)
    monthly_stats_path = base_path + '/parameter-stats-monthly/Lake-' + \
                         lake + '_' + param + '.txt'
    if os.path.exists(monthly_stats_path):
        meas_dates, meas_values, errors = divaux.read_statsmonthly(monthly_stats_path, 'p90_threshold',
                                                                   blacklist)
        return calculate_trend(meas_values)


def plot(decomposition, param, lake):
    idx = np.isfinite(decomposition.trend['values'])
    trend = calculate_trend_line(decomposition.trend['values'][idx])
    p = np.poly1d(trend)
    # xp = np.linspace(0, len(decomposition.trend))
    data = {'date': decomposition.trend.index.values,
            'values': p(range(0, len(decomposition.trend)))}
    trend_df = pd.DataFrame(data, columns=['date', 'values'])
    trend_df.index = trend_df['date']
    del trend_df['date']
    matplotlib.rcParams['font.serif'] = "times new roman"
    matplotlib.rcParams['font.family'] = "serif"
    params = matplotlib.rcParams
    fig, axes = plt.subplots(4, 1, sharex=True)
    font_dict_label = {'family': 'times new roman'}
    font_dict_title = {'family': 'times new roman',
                       'size': 16}

    plt.rc('font', **font_dict_title)
    plt.suptitle(lake + ": " + param, weight='bold')

    if hasattr(decomposition.observed, 'plot'):  # got pandas use it
        decomposition.observed.plot(ax=axes[0], legend=False, colormap='Greys_r')
        axes[0].set_ylabel('Observed', fontdict=font_dict_label, color='k')
        axes[0].tick_params(axis='y', which='major', labelsize=8)
        axes[0].get_yaxis().set_label_coords(-0.09, 0.5)
        decomposition.trend.plot(ax=axes[1], legend=False, colormap='Greys_r')
        trend_df.plot(ax=axes[1], legend=False, colormap='Greys_r', linestyle='dashed')
        axes[1].legend(['seasonal decomp', 'trend yearly: ' + str(round(trend[0] * 12, 4))], loc=2, fontsize=8)
        axes[1].set_ylabel('Trend', fontdict=font_dict_label, color='k')
        axes[1].tick_params(axis='y', which='major', labelsize=8)
        axes[1].get_yaxis().set_label_coords(-0.09, 0.5)
        decomposition.seasonal.plot(ax=axes[2], legend=False, colormap='Greys_r')
        axes[2].set_ylabel('Seasonal', fontdict=font_dict_label, color='k')
        axes[2].tick_params(axis='y', which='major', labelsize=8)
        axes[2].get_yaxis().set_label_coords(-0.09, 0.5)
        decomposition.resid.plot(ax=axes[3], legend=False, colormap='Greys_r')
        axes[3].set_ylabel('Residual', fontdict=font_dict_label, color='k')
        axes[3].set_xlabel('', fontdict=font_dict_label, color='k')
        axes[3].get_yaxis().set_label_coords(-0.09, 0.5)
        axes[3].tick_params(axis='y', which='major', labelsize=8)
        axes[3].tick_params(axis='x', which='major', labelsize=14)
    else:
        axes[0].plot(decomposition.observed)
        axes[0].set_ylabel('Observed', fontdict=font_dict_label, color='k')
        axes[0].tick_params(axis='y', which='major', labelsize=8)
        axes[0].get_yaxis().set_label_coords(-0.09, 0.5)
        axes[1].plot(decomposition.trend)
        axes[1].set_ylabel('Trend', fontdict=font_dict_label, color='k')
        axes[1].tick_params(axis='y', which='major', labelsize=8)
        axes[1].get_yaxis().set_label_coords(-0.09, 0.5)
        axes[2].plot(decomposition.seasonal)
        axes[2].set_ylabel('Seasonal', fontdict=font_dict_label, color='k')
        axes[2].tick_params(axis='y', which='major', labelsize=8)
        axes[2].get_yaxis().set_label_coords(-0.09, 0.5)
        axes[3].plot(decomposition.resid)
        axes[3].set_ylabel('Residual', fontdict=font_dict_label, color='k')
        axes[3].set_xlabel('', fontdict=font_dict_label, color='k')
        axes[3].tick_params(axis='y', which='major', labelsize=8)
        axes[3].tick_params(axis='x', which='major', labelsize=12)
        axes[3].get_yaxis().set_label_coords(-0.09, 0.5)
        axes[3].set_xlim(0, decomposition.nobs)

    # fig.tight_layout()
    return fig


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
    if len(values) != 0:
        coefficients, residuals, _, _, _ = np.polyfit(range(len(values)), values, 1, full=True)
        if len(residuals) > 0 and residuals[0] != 0:
            mse = residuals[0] / (len(values))
            nrmse = np.sqrt(mse) / (max(values) - min(values))
        else:
            nrmse = np.nan
        coeff = coefficients[0]
    else:
        coeff = np.nan
        nrmse = np.nan
    return (coeff, nrmse)


def calculate_trend_line(values):
    if len(values) != 0:
        coefficients, residuals, _, _, _ = np.polyfit(range(len(values)), values, 1, full=True)
    else:
        coefficients = (np.nan, np.nan)
    return coefficients


def plots_seasonal_decompose(config, d2products_folder, lakes_list, params_list):
    for lake in lakes_list:
        print('Processing decomposition plots for ' + lake)
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

                if len(np.array(meas_values)[np.isnan(meas_values)]) != len(meas_values):
                    decomposition, decomposition_cleanup = seasonal_decompose(meas_dates, meas_values)
                    # fig = decomposition.plot()
                    # fig = plot(decomposition, param, lake)
                    # fig.savefig(output_folder + '/Lake-' + lake + '_decomposition_' + param + '.png')

                    fig = plot(decomposition_cleanup, param, lake)
                    fig.savefig(output_folder + '/Lake-' + lake + '_decomposition_' + param + '.png')


def seasonal_decompose(meas_dates, meas_values):
    # remove values of 2002 and 2012
    int_dates = []
    int_values = []
    for i, date in enumerate(meas_dates):
        if date >= datetime.datetime(2003, 1, 1, 0, 0) and date <= datetime.datetime(2011, 12, 1, 0, 0):
            int_dates.append(date)
            int_values.append(meas_values[i])
    # remove leading and trailing NaNs in the time series
    for i_value, value in enumerate(int_values):
        if not np.isnan(value):
            int_values = int_values[i_value:]
            int_dates = int_dates[i_value:]
            break
    for i_value, value in enumerate(reversed(int_values)):
        if not np.isnan(value):
            if i_value == 0:
                break
            else:
                int_values = int_values[: -i_value]
                int_dates = int_dates[: -i_value]
                break
    # statsmodels.seasonal_decompose: Plots saved in folder 'decomposition-plots'
    data = {'date': int_dates, 'values': int_values}
    df = pd.DataFrame(data, columns=['date', 'values'])
    df.index = df['date']
    del df['date']
    # df = df.fillna(method='bfill')
    # df = df.fillna(method='ffill')
    # if df.isnull().sum().values[0] == 0:
    decomposition = sm.seasonal_decompose(df, df, model='additive')

    df_cleanup = df.copy(deep=True)  # secure old data

    df_cleanup['date'] = pd.to_datetime(df_cleanup.index)
    df_cleanup['year'], df_cleanup['month'] = df_cleanup['date'].dt.year, df_cleanup['date'].dt.month
    for i in range(1, 13):
        valid_values = len(df_cleanup[df_cleanup['month'] == i]) - np.sum(
            np.isnan(df_cleanup[df_cleanup['month'] == i]['values']))
        if valid_values < 6:
            df_cleanup['values'][df_cleanup['month'] == i] = np.nan
    del df_cleanup['date']
    del df_cleanup['year']
    del df_cleanup['month']

    decomposition_cleanup = sm.seasonal_decompose(df, df_cleanup, model='additive')

    return decomposition, decomposition_cleanup


def get_blacklist(config, d2products_folder, lake, param):
    blacklist_config = config['DEFAULT']['date_blacklist']
    if blacklist_config != 'False':
        if blacklist_config.startswith('blacklist_'):
            if param == 'num_obs':
                blacklist = ''
            else:
                if '?' in blacklist_config:
                    blacklist_folder = \
                        fnmatch.filter(os.listdir(d2products_folder + '/Lake-' + lake), blacklist_config)[0]
                else:
                    blacklist_folder = blacklist_config
                blacklist_path = d2products_folder + '/Lake-' + lake + '/' + blacklist_folder + '/blacklist_lake-' + lake + '_' + param + '.txt'
                if os.path.exists(blacklist_path):
                    blacklist = divaux.read_blacklist(blacklist_path)
                else:
                    blacklist = ''
        elif blacklist_config.startswith('20'):
            blacklist = blacklist_config.split(',')
    else:
        blacklist = False
    return blacklist


def add_to_product_stats_10y(config, d2products_folder, lakes_list):
    for lake in lakes_list:
        print('Processing lake ' + lake)
        base_path = d2products_folder + '/Lake-' + lake
        product_stats_path = base_path + '/product-stats-9y' + '/Lake-' + lake + '_2003-01-01_2011-12-31.txt'
        if os.path.exists(product_stats_path):
            table = pd.read_csv(product_stats_path, sep='\t', header=0, na_values=[''])
            bands = table['Band']
            trend = []
            max_value = []
            max_month = []
            min_value = []
            min_month = []
            high_residual = []
            seasonal_variations = []
            baseline2_exceeded = []
            baselines = []
            seasonal_variances = []
            for band in bands:
                (values, dates) = get_parameter_stats_monthly(band, d2products_folder, lake, config)
                _, decomposition = seasonal_decompose(dates, values)
                if decomposition is None:
                    trend.append(np.nan)
                    max_value.append(np.nan)
                    max_month.append(np.nan)
                    min_value.append(np.nan)
                    min_month.append(np.nan)
                    baselines.append(np.nan)
                    baseline2_exceeded.append(np.nan)
                    seasonal_variations.append(np.nan)
                    high_residual.append(np.nan)
                    seasonal_variances.append(np.nan)
                    continue
                # trend
                idx = np.isfinite(decomposition.trend['values'])
                (t_seasonal, _) = calculate_trend(decomposition.trend['values'][idx])
                trend.append(round(t_seasonal * 12, 4))
                # min/max month and value seasonal
                one_year = decomposition.seasonal.loc['20040101':'20041201']
                max = one_year.max()[0]
                max_value.append(round(max, 4))
                if len(one_year) > 0:
                    idxmax = one_year.idxmax()[0]
                else:
                    idxmax = np.nan
                max_month.append(idxmax)
                min = one_year.min()[0]
                min_value.append(round(min, 4))
                if len(one_year) > 0:
                    idxmin = one_year.idxmin()[0]
                else:
                    idxmin = np.nan
                min_month.append(idxmin)

                # higher than baseline
                baseline = one_year.nsmallest(3, 'values').mean()[0]
                baselines.append(round(baseline, 4))
                idx = one_year['values'] > baseline + abs(baseline)
                over_baseline_list = one_year['values'][idx].index.strftime("%Y-%m-%d").tolist()
                one_year_non_nan = len(one_year[~np.isnan(one_year['values'])])
                over_baseline = len(over_baseline_list) / one_year_non_nan if one_year_non_nan != 0 else 0
                baseline2_exceeded.append(round(over_baseline, 4))

                # residuum higher than seasonal variation
                seasonal_variation = abs(max - min)
                seasonal_variations.append(round(seasonal_variation, 4))
                idx = decomposition.resid['values'] > seasonal_variation * 2
                high_list = decomposition.resid['values'][idx].index.strftime("%Y-%m-%d").tolist()
                observed_non_nan = len(decomposition.observed[~np.isnan(decomposition.observed)])
                high = len(high_list) / observed_non_nan if observed_non_nan != 0 else 0
                high_residual.append(round(high, 4))

                # seasonal variance
                seasonal_variance = np.nanvar(decomposition.seasonal)
                seasonal_variances.append(round(seasonal_variance, 4))

            table = table.assign(trend_per_year=trend)
            table = table.assign(max_value_seasonal=max_value)
            table = table.assign(max_month_seasonal=max_month)
            table = table.assign(min_value_seasonal=min_value)
            table = table.assign(min_month_seasonal=min_month)
            table = table.assign(baseline=baselines)
            table = table.assign(baseline2_exceeded=baseline2_exceeded)
            table = table.assign(seasonal_variations=seasonal_variations)
            table = table.assign(seasonal_variance=seasonal_variances)
            table = table.assign(high_residual=high_residual)
            table.to_csv(product_stats_path.replace('.txt', '_trend.txt'), '\t')


def main():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ini', 'extract_trend.ini'))
    d2products_folder = config['DEFAULT']['products_path']

    params = config['DEFAULT']['param_str']
    params_list = [param.lstrip() for param in params.split(',')]

    lakes = config['DEFAULT']['lakes']
    lakes_list = [lake.lstrip() for lake in lakes.split(',')]

    # statsmodels.seasonal_decompose: Plots saved in folder 'decomposition-plots'
    # plots_seasonal_decompose(config, d2products_folder, lakes_list, params_list)

    # add info to product_stats_10y for all parameters
    add_to_product_stats_10y(config, d2products_folder, lakes_list)


if __name__ == "__main__":
    main()
