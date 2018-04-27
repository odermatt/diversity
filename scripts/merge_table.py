#!python
__author__ = 'Helga'

import pandas
import os
import fnmatch
import numpy as np


def main():
    # read shapefile csv into dataframe
    path_shapefile = 'C:/Users/helga/Dropbox/D2-for-WB/Shapefiles/all-lakes-joined.csv'
    targetpath = 'C:/Users/helga/Documents/Diversity-2/shapefile_merged.csv'
    shapefile = pandas.read_csv(path_shapefile, sep=';', header=0, na_values=[''])

    params = ['floating_cyanobacteria_mean', 'turbidity_cc_mean', 'owt_cc_dominant_class_mode', 'lswt_n_mean',
              'immersed_cyanobacteria_mean', 'floating_vegetation_mean', 'chl_mph_mean', 'tsm_cc_mean', 'cdom_fub_mean',
              'chl_fub_mean']
    values = ['average', 'max_error', 'maximum', 'median', 'minimum', 'p10_threshold', 'p90_threshold', 'sigma',
              'total', 'trend_per_year', 'max_value_seasonal', 'max_month_seasonal', 'min_value_seasonal',
              'min_month_seasonal', 'baseline', 'baseline2_exceeded', 'seasonal_variations', 'seasonal_variance',
              'high_residual']

    shapefile.index = shapefile['DIV2_NAME,C,50']
    length = len(shapefile['ABC_ID,N,9,0'])
    for value in values:
        for param in params:
            shapefile.loc[:, param + '_-_' + value] = pandas.Series(np.zeros(length), index=shapefile.index)

    # read csv per lake and add data to dataframe
    base_lake = 'U:/Diversity-2/'
    file_mask = 'Lake-*_2003-01-01_2011-12-31_trend.txt'
    lake_mask = 'Lake-*'
    lakes = fnmatch.filter(os.listdir(base_lake), lake_mask)
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
                    for value in values:
                        for param in params:
                            shapefile[param + '_-_' + value][lake.replace('Lake-', '').upper()] = lakefile[value][param]
                    print('OK: ' + lake)

        else:
            print('No Folder for ' + lake)
    shapefile.to_csv(r'' + targetpath, index=None, sep='\t')

    # write csv


if __name__ == "__main__":
    main()
