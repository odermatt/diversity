#!python
__author__ = 'Helga'

import configparser
import os
import modules.auxiliary_functions as divaux
import pandas as pd
import numpy as np

dates = ['2003-01-01', '2003-02-01', '2003-03-01', '2003-04-01', '2003-05-01', '2003-06-01', '2003-07-01', '2003-08-01',
         '2003-09-01', '2003-10-01', '2003-11-01', '2003-12-01',
         '2004-01-01', '2004-02-01', '2004-03-01', '2004-04-01', '2004-05-01', '2004-06-01', '2004-07-01', '2004-08-01',
         '2004-09-01', '2004-10-01', '2004-11-01', '2004-12-01', '2005-01-01', '2005-02-01', '2005-03-01', '2005-04-01',
         '2005-05-01', '2005-06-01', '2005-07-01', '2005-08-01', '2005-09-01', '2005-10-01', '2005-11-01', '2005-12-01',
         '2006-01-01', '2006-02-01', '2006-03-01', '2006-04-01', '2006-05-01', '2006-06-01', '2006-07-01', '2006-08-01',
         '2006-09-01', '2006-10-01', '2006-11-01', '2006-12-01', '2007-01-01', '2007-02-01', '2007-03-01', '2007-04-01',
         '2007-05-01', '2007-06-01', '2007-07-01', '2007-08-01', '2007-09-01', '2007-10-01', '2007-11-01', '2007-12-01',
         '2008-01-01', '2008-02-01', '2008-03-01', '2008-04-01', '2008-05-01', '2008-06-01', '2008-07-01', '2008-08-01',
         '2008-09-01', '2008-10-01', '2008-11-01', '2008-12-01', '2009-01-01', '2009-02-01', '2009-03-01', '2009-04-01',
         '2009-05-01', '2009-06-01', '2009-07-01', '2009-08-01', '2009-09-01', '2009-10-01', '2009-11-01', '2009-12-01',
         '2010-01-01', '2010-02-01', '2010-03-01', '2010-04-01', '2010-05-01', '2010-06-01', '2010-07-01', '2010-08-01',
         '2010-09-01', '2010-10-01', '2010-11-01', '2010-12-01', '2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01',
         '2011-05-01', '2011-06-01', '2011-07-01', '2011-08-01', '2011-09-01', '2011-10-01', '2011-11-01', '2011-12-01']


def main():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ini', 'blacklist_pooled.ini'))

    d2products_folder = config['DEFAULT']['products_path']
    params = config['DEFAULT']['param_str']
    params = [p.lstrip() for p in params.split(',')]

    for param in params:
        print(param)
        df_1 = pd.DataFrame(data={'month': dates})
        df_20 = pd.DataFrame(data={'month': dates})
        df_50 = pd.DataFrame(data={'month': dates})
        df_50_XY = pd.DataFrame(data={'month': dates})
        df_count = pd.DataFrame(data={'list': ['blacklist_1', 'blacklist_20', 'blacklist_50', 'blacklist_50_XY']})
        if d2products_folder != '':
            if d2products_folder[-1] == '/':
                d2products_folder = d2products_folder[:-1]
            directories = [f for f in os.listdir(d2products_folder) if
                           os.path.isdir(os.path.join(d2products_folder, f))]
            directories = np.sort(directories)
            for directory in directories:
                if 'blacklists' not in directory:
                    for dir in os.listdir(d2products_folder + '/' + directory):
                        if 'blacklist' in dir and 'blacklists' not in dir:
                            for file in os.listdir(d2products_folder + '/' + directory + '/' + dir):
                                if param in file:
                                    blacklist = divaux.read_blacklist(
                                        d2products_folder + '/' + directory + '/' + dir + '/' + file)
                                    lake = directory
                                    list = [lake]
                                    count = 0
                                    for date in dates:
                                        if date in blacklist:
                                            list.append('x')
                                            count += 1
                                        else:
                                            list.append('')
                                    list_df = pd.DataFrame({list[0]: list[1:]})
                                    if 'blacklist_1' in dir:
                                        df_1 = pd.concat([df_1, list_df], axis=1)
                                        if lake in df_count.columns:
                                            df_count.at[0, lake] = count
                                        else:
                                            list_count = pd.DataFrame({lake: [count, 0, 0, 0]})
                                            df_count = pd.concat([df_count, list_count], axis=1)
                                    if 'blacklist_20' in dir:
                                        df_20 = pd.concat([df_20, list_df], axis=1)
                                        if lake in df_count.columns:
                                            df_count.at[1, lake] = count
                                        else:
                                            list_count = pd.DataFrame({lake: [0, count, 0, 0]})
                                            df_count = pd.concat([df_count, list_count], axis=1)
                                    if 'blacklist_50' in dir and 'blacklist_50_' not in dir:
                                        df_50 = pd.concat([df_50, list_df], axis=1)
                                        if lake in df_count.columns:
                                            df_count.at[2, lake] = count
                                        else:
                                            list_count = pd.DataFrame({lake: [0, 0, count, 0]})
                                            df_count = pd.concat([df_count, list_count], axis=1)
                                    if 'blacklist_50_' in dir:
                                        df_50_XY = pd.concat([df_50_XY, list_df], axis=1)
                                        if lake in df_count.columns:
                                            df_count.at[3, lake] = count
                                        else:
                                            list_count = pd.DataFrame({lake: [0, 0, 0, count]})
                                            df_count = pd.concat([df_count, list_count], axis=1)

        df_1.to_csv(r'' + d2products_folder + '/blacklists/blacklist_1_' + param + '.txt', index=None, sep='\t')
        df_20.to_csv(r'' + d2products_folder + '/blacklists/blacklist_20_' + param + '.txt', index=None, sep='\t')
        df_50.to_csv(r'' + d2products_folder + '/blacklists/blacklist_50_' + param + '.txt', index=None, sep='\t')
        df_50_XY.to_csv(r'' + d2products_folder + '/blacklists/blacklist_50_XY_' + param + '.txt', index=None, sep='\t')
        df_count.to_csv(r'' + d2products_folder + '/blacklists/blacklist_count_' + param + '.txt', index=None, sep='\t')


if __name__ == "__main__":
    main()
