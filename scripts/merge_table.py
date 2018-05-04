#!python
__author__ = 'Helga'

import pandas
import os
import fnmatch
import numpy as np
import modules.auxiliary_functions as divaux


def calculate_values(path, lake, param, values):
    base_path = path + '/Lake-' + lake
    blacklist = get_blacklist(path, lake, param)
    monthly_stats_path = base_path + '/parameter-stats-monthly/Lake-' + \
                         lake + '_' + param + '.txt'
    result = []
    if os.path.exists(monthly_stats_path):
        for value in values:
            dates, values, _ = divaux.read_statsmonthly(monthly_stats_path, value, blacklist)
            if value in ['average', 'median', 'p10_threshold', 'p90_threshold']:
                average = np.nanmean(values)
                median = np.nanmedian(values)
                result.append((value + '_average', average))
                result.append((value + '_median', median))
            elif value == 'maximum':
                maximum = np.nanmax(values)
                result.append((value, maximum))
            elif value == 'minimum':
                minimum = np.nanmin(values)
                result.append((value, minimum))
    return result


def get_blacklist(d2products_folder, lake, param):
    blacklist_folder = fnmatch.filter(os.listdir(d2products_folder + '/Lake-' + lake), 'blacklist_?0_??')[0]
    blacklist_path = d2products_folder + '/Lake-' + lake + '/' + blacklist_folder + '/blacklist_lake-' + lake + '_' + param + '.txt'
    if os.path.exists(blacklist_path):
        blacklist = divaux.read_blacklist(blacklist_path)
    else:
        blacklist = ''
    return blacklist


def main():
    # read shapefile csv into dataframe
    path_shapefile = 'C:/Users/helga/Dropbox/D2-for-WB/Shapefiles/all-lakes-joined.csv'
    targetpath = 'C:/Users/helga/Documents/Diversity-2/shapefile_merged.csv'
    shapefile = pandas.read_csv(path_shapefile, sep=';', header=0, na_values=[''])

    params = ['floating_cyanobacteria_mean', 'turbidity_cc_mean', 'owt_cc_dominant_class_mode', 'lswt_n_mean',
              'immersed_cyanobacteria_mean', 'floating_vegetation_mean', 'chl_mph_mean', 'tsm_cc_mean', 'cdom_fub_mean',
              'chl_fub_mean']
    values = ['trend_per_year', 'max_value_seasonal', 'max_month_seasonal', 'min_value_seasonal',
              'min_month_seasonal', 'baseline', 'baseline2_exceeded', 'seasonal_variations', 'seasonal_variance',
              'high_residual']
    values_calc = ['average', 'maximum', 'median', 'minimum', 'p10_threshold', 'p90_threshold']

    shapefile.index = shapefile['DIV2_NAME,C,50']
    length = len(shapefile['ABC_ID,N,9,0'])

    # read csv per lake and add data to dataframe
    base_lake = 'U:/Diversity-2/'
    file_mask = 'Lake-*_2003-01-01_2011-12-31_trend.txt'
    lake_mask = 'Lake-*'
    lakes = sorted(fnmatch.filter(os.listdir(base_lake), lake_mask))
    for lake in lakes:
        path = base_lake + lake + '/product-stats-9y/'
        if os.path.exists(path):
            files = fnmatch.filter(os.listdir(path), file_mask)
            if not files:
                print('No files for ' + lake)
            else:
                if len(files) > 1:
                    print('More than one file for ' + lake + ': ' + str(files))
                else:
                    lakefile = pandas.read_csv(path + '/' + files[0], sep='\t', header=0, na_values=[''])
                    lakefile.index = lakefile['Band']
                    for param in params:
                        for value in values:
                            shapefile.loc[lake.replace('Lake-', '').upper(), param + '_-_' + value] = lakefile.loc[
                                param, value]
                        results = calculate_values(base_lake, lake.replace('Lake-', ''), param, values_calc)
                        for (key, value) in results:
                            shapefile.loc[lake.replace('Lake-', '').upper(), param + '_-_' + key] = value
                    print('OK: ' + lake)

        else:
            print('No Folder for ' + lake)
    shapefile.to_csv(r'' + targetpath, index=None, sep='\t')

    # write csv


if __name__ == "__main__":
    main()
