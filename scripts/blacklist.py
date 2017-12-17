#!python
__author__ = 'Daniel'

import configparser as configparser
import os as os
import numpy as np
import pandas as pandas
import snappy as snappy
import scipy.stats as stats


def get_owt_mode(product_path):

    product = snappy.ProductIO.readProduct(product_path)
    owt_band = product.getBand('owt_cc_dominant_class_mode')
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    data_type = np.float32
    param_arr = np.zeros(width * height, dtype=data_type)
    owt_band.readPixels(0, 0, width, height, param_arr)

    return stats.mode(param_arr)[0][0]



def main():

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ini', 'blacklist.ini'))

    d2products_folder = config['DEFAULT']['products_path']
    threshold = config['DEFAULT']['threshold']

    source_band_names = config['DEFAULT']['param_str']
    source_band_list = [source_band_name.lstrip() for source_band_name in source_band_names.split(',')]

    if d2products_folder != '':
        if d2products_folder[-1] == '/':
           d2products_folder = d2products_folder[:-1]

    lake = config['DEFAULT']['lake']
    param_table_path = d2products_folder + '/Lake-' + lake + '/parameter-stats-monthly'

    owt_mode = get_owt_mode(d2products_folder + '/Lake-' + lake + '/l3-10y/Lake-' + lake + '_2003-01-01_2011-12-31.tif')

    blacklist_path = d2products_folder + '/Lake-' + lake + '/blacklist'

    if not os.path.exists(blacklist_path):
        os.makedirs(blacklist_path)

    for source_band in source_band_list:

        param_table = pandas.read_csv(param_table_path + '/Lake-' + lake + '_' + source_band + '.txt', sep='\t',
                                      header=0, na_values=[''])

        out_file = blacklist_path + '/blacklist_Lake-' + lake + '_' + source_band + '.txt'
        write_file = open(out_file, 'w')

        max_total = max(param_table['total'])

        for count, start_date in enumerate(param_table['stats_start_date']):
            if owt_mode in [4, 5, 6, 7] and source_band in ['chl_fub_mean', 'cdom_fub_mean']:
                write_file.write(start_date + '\n')
            elif owt_mode in [1, 2, 3] and source_band in ['chl_mph_mean']:
                write_file.write(start_date + '\n')
            elif param_table['total'][count] < (max_total * 0.2):
                write_file.write(start_date + '\n')

        write_file.close()

    print('\nLake ' + lake + ' blacklists written to:', blacklist_path)


if __name__ == "__main__":
    main()