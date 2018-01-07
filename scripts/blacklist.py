#!python
__author__ = 'Daniel'

import configparser as configparser
import os as os
import numpy as np
import pandas as pandas
import snappy as snappy


def get_binary_owt(product_path):

    product = snappy.ProductIO.readProduct(product_path)
    owt_band = product.getBand('owt_cc_dominant_class_mode')
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    data_type = np.float32
    param_arr = np.zeros(width * height, dtype=data_type)
    owt_band.readPixels(0, 0, width, height, param_arr)
    owts, counts = np.unique(param_arr, return_counts=True)

    clear_counts = 0
    turbid_counts = 0

    for count, owt in enumerate(owts[0:7]):
        if owt in [1, 2, 3]:
            clear_counts = clear_counts + counts[count]
        elif owt in [4, 5, 6, 7]:
            turbid_counts = turbid_counts + counts[count]

    if clear_counts >= turbid_counts:
        return 0
    else:
        return 1


def mode(data):
    counts = {}
    for x in data.flatten():
        counts[x] = counts.get(x,0) + 1
    maxcount = max(counts.values())
    modelist = []
    for x in counts:
        if counts[x] == maxcount:
            modelist.append(x)
    return modelist,maxcount


def main():

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ini', 'blacklist.ini'))

    d2products_folder = config['DEFAULT']['products_path']
    threshold = config['DEFAULT']['threshold']
    min_lswt_n = config['DEFAULT']['min_lswt_n']

    source_band_names = config['DEFAULT']['param_str']
    source_band_list = [source_band_name.lstrip() for source_band_name in source_band_names.split(',')]

    if d2products_folder != '':
        if d2products_folder[-1] == '/':
           d2products_folder = d2products_folder[:-1]

    lakes = config['DEFAULT']['lakes']
    lakes_list = [lake.lstrip() for lake in lakes.split(',')]

    for lake in lakes_list:
        param_table_path = d2products_folder + '/Lake-' + lake + '/parameter-stats-monthly'

        owt_mode = get_binary_owt(d2products_folder + '/Lake-' + lake + '/l3-10y/Lake-' + lake + '_2003-01-01_2011-12-31.tif')

        blacklist_path = d2products_folder + '/Lake-' + lake + '/blacklist_' + threshold

        if not os.path.exists(blacklist_path):
            os.makedirs(blacklist_path)

        lswt_available = False
        if min_lswt_n:
            if os.path.isfile(param_table_path + '/Lake-' + lake + '_lswt_n_mean.txt'):
                lswt_available = True
                lswt_n_table = pandas.read_csv(param_table_path + '/Lake-' + lake + '_lswt_n_mean.txt', sep='\t',
                                               header=0, na_values=[''])
            else:
                print('\nlswt_n stats file not available for lake ' + lake)

        for source_band in source_band_list:

            param_table = pandas.read_csv(param_table_path + '/Lake-' + lake + '_' + source_band + '.txt', sep='\t',
                                          header=0, na_values=[''])

            out_file = blacklist_path + '/blacklist_Lake-' + lake + '_' + source_band + '.txt'
            write_file = open(out_file, 'w')

            max_total = max(param_table['total'])

            for count, start_date in enumerate(param_table['stats_start_date']):
                if owt_mode == 1 and source_band in ['chl_fub_mean', 'cdom_fub_mean']:
                    write_file.write(start_date + '\n')
                elif owt_mode == 0 and source_band in ['chl_mph_mean']:
                    write_file.write(start_date + '\n')
                elif param_table['total'][count] < (max_total * int(threshold) / 100):
                    write_file.write(start_date + '\n')
                elif min_lswt_n and lswt_available:
                    if (lswt_n_table['average'][count] < (273.15 + int(min_lswt_n))) and (lswt_n_table['average'][count] != -999.0):
                        write_file.write(start_date + '\n')

            write_file.close()

        print('\nLake ' + lake + ' blacklists written to:', blacklist_path)


if __name__ == "__main__":
    main()