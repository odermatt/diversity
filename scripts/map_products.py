#!python
__author__ = 'Daniel'

import numpy as np
import math as math
import os as os
import beampy as bp
import cartopy.crs as ccrs
import cartopy.io.srtm as srtm
import cartopy.io.img_tiles as maps
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import package.auxiliary_functions as divaux
import package.colour_scales as colscales
import configparser as configparser


# basemap can be: srtm_hillshade, srtm_elevation, quadtree_rgb, none
# using srtm in areas above 55 deg N or S will produce an error


def plot_param(arranged_filepaths, arranged_labels, param_name, output_basename, basemap='srtm_elevation',
               crop_ext=False, param_range=False, grid=True, geogr_area=False):

    mpl.rc('font', family='Times New Roman')

    if 'monthly' in arranged_filepaths[0][0]:
        aggregate_type = 'monthly'
    elif 'yearly' in arranged_filepaths[0][0]:
        aggregate_type = 'yearly'
    else:
        aggregate_type = '10y'

    legend_extension = 1.2
    bar_orientation = 'vertical'
    nrows = len(arranged_filepaths)
    ncols = len(arranged_filepaths[0])

    if nrows + ncols < 3:
        linewidth = 0.8
        annotation_size = 20
        gridlabel_size = 5
    else:
        linewidth = 0.4
        annotation_size = 9
        gridlabel_size = 4

    for nth_row in range(nrows):
        for nth_col in range(ncols):

            print('Plotting ' + param_name + ' of ' + arranged_filepaths[nth_row][nth_col])

            # get product parameters
            product = bp.ProductIO.readProduct(arranged_filepaths[nth_row][nth_col])
            width = product.getSceneRasterWidth()
            height = product.getSceneRasterHeight()
            param_band = product.getBand(param_name)
            param_dt = param_band.getDataType()
            if param_dt <= 12:
                data_type = np.int32
            elif param_dt == 30:
                data_type = np.float32
            elif param_dt == 31:
                data_type = np.float64
            else:
                raise ValueError('cannot handle band of data_sh type \'' + param_dt + '\'')
            print('   image dimensions are ' + str(width) + ' by ' + str(height) + ' pixels')

            # read parameter band
            param_arr = np.zeros(width * height,  dtype=data_type)
            param_band.readPixels(0, 0, width, height, param_arr)
            param_arr = param_arr.reshape(height, width)

            # convert to centigrade if LSWT
            if param_name.startswith('lswt'):
                param_arr = [pixel - 273.15 for pixel in param_arr]

            # example recalibration for FUB_CHL in Lake Geneva
            # param_arr = [pixel/0.465+0.502 for pixel in param_arr]

            # merge shallow and extent layer into one parameter band
            if param_name in ['extent', 'shallow']:
                shallow_band = product.getBand('shallow')
                shallow_arr = np.zeros(width * height,  dtype=data_type)
                shallow_band.readPixels(0, 0, width, height, shallow_arr)
                shallow_arr = shallow_arr.reshape(height, width)
                param_arr = [shallow_arr[pixel] + param_arr[pixel] for pixel in range(len(param_arr))]
            if param_name == 'shallow':
                extent_band = product.getBand('extent')
                extent_arr = np.zeros(width * height,  dtype=data_type)
                extent_band.readPixels(0, 0, width, height, extent_arr)
                extent_arr = extent_arr.reshape(height, width)
                param_arr = [extent_arr[pixel] + param_arr[pixel] for pixel in range(len(param_arr))]

            # mask applicable pixels
            masked_param_arr = np.ma.masked_invalid(param_arr)
            masked_param_arr = np.ma.masked_where(masked_param_arr == -1, masked_param_arr)
            if param_name.startswith('lswt'):
                masked_param_arr = np.ma.masked_where(masked_param_arr < 0, masked_param_arr)
            if param_name in ['num_obs', 'shallow', 'extent']:
                masked_param_arr = np.ma.masked_where(masked_param_arr == 0, masked_param_arr)
            print('   applicable values are found in ' + str(masked_param_arr.count()) +
                  ' of ' + str(height * width) + ' pixels')

            # read lat and lon information
            geocoding = product.getGeoCoding()
            lowlef = geocoding.getGeoPos(bp.PixelPos(0, height - 1), None)
            upprig = geocoding.getGeoPos(bp.PixelPos(width - 1, 0), None)

            # determine appropriate map canvas
            if not geogr_area:

            # add map extent if desired
                if crop_ext:
                    lat_ext = (upprig.lat - lowlef.lat) / 8
                    lon_ext = (upprig.lon - lowlef.lon) / 8
                else:
                    lat_ext = 0
                    lon_ext = 0

            # increase the smaller portion of the map canvas until the image aspect ratio is at most 3:2
                x_dist = divaux.calculate_haversine(lowlef.lon - lon_ext, lowlef.lat,
                                                       upprig.lon + lon_ext, lowlef.lat)
                y_dist = divaux.calculate_haversine(lowlef.lon, lowlef.lat - lat_ext,
                                                       lowlef.lon, upprig.lat + lat_ext)
                if x_dist / y_dist < 2 / 3:
                    if lon_ext == 0:
                        lon_ext = (upprig.lon - lowlef.lon) / 20
                    while x_dist / y_dist < 2 / 3:
                        lon_ext = lon_ext * 1.1
                        x_dist = divaux.calculate_haversine(lowlef.lon - lon_ext, lowlef.lat,
                                                               upprig.lon + lon_ext, lowlef.lat)
                elif x_dist / y_dist > 3 / 2:
                    if lat_ext == 0:
                        lat_ext = (upprig.lat - lowlef.lat) / 20
                    while x_dist / y_dist > 3 / 2:
                        lat_ext = lat_ext * 1.1
                        y_dist = divaux.calculate_haversine(lowlef.lon, lowlef.lat - lat_ext,
                                                               lowlef.lon, upprig.lat + lat_ext)
                canvas_area = [[lowlef.lon - lon_ext, lowlef.lat - lat_ext],
                               [upprig.lon + lon_ext, upprig.lat + lat_ext]]

            # or use the predefined area's coordinates
            else:
                canvas_area = geogr_area

            # identify image aspect ratio
            x_dist = divaux.calculate_haversine(canvas_area[0][0], canvas_area[0][1], canvas_area[1][0], canvas_area[0][1])
            y_dist = divaux.calculate_haversine(canvas_area[0][0], canvas_area[0][1], canvas_area[0][0], canvas_area[1][1])
            if (0.83 < x_dist / y_dist) and (1.25 > x_dist / y_dist):
                orientation = 'square'
            elif x_dist < y_dist:
                orientation = 'portrait'
            else:
                orientation = 'landscape'

            # Determine parameter-dependent styles
            if param_name in ['owt_cc_dominant_class_mode']:
                legend_decimals = '%.0f'
                color_type = colscales.moores_seven_owt()
                param_range = [0.5, 7.5]
                ticks = list(range(1, 8, 1))

            elif param_name in ['num_obs']:
                color_type = colscales.num_obs_scale()
                if not param_range:
                    if masked_param_arr.count() != 0:
                        param_range = [np.percentile(masked_param_arr.compressed(), 10),
                                       np.percentile(masked_param_arr.compressed(), 90)]
                    else:
                        param_range = [1, 7]
                param_range[0] = math.floor(param_range[0])
                param_range[1] = math.ceil(param_range[1])
                if param_range[0] < 1:
                    param_range[0] = 1
                if param_range[1] < 7:
                    param_range[1] = 7
                interval = math.ceil((param_range[1] - param_range[0]) / 6)
                ticks = list(range(param_range[0], param_range[0] + (6 * interval), interval))
                param_range = [param_range[0] - (0.5 * interval), param_range[0] + (5.5 * interval)]
                legend_decimals = '%.0f'

            elif param_name.startswith('lswt'):
                color_type = 'rainbow'
                if not param_range:
                    if masked_param_arr.count() == 0:
                        param_range = [0, 10]
                    else:
                        param_range = [math.floor(9.8 * min(masked_param_arr.compressed())) / 10,
                                       math.ceil(10 * max(masked_param_arr.compressed())) / 9.8]
                tick_list = list(range(0, 11, 1))
                ticks = [param_range[0] + (tick * (param_range[1] - param_range[0]) / 10) for tick in tick_list]
                if ticks[-1] - ticks[0] in [10, 20, 30]:
                    legend_decimals = '%.0f'
                elif ticks[-1] - ticks[0] < 1:
                    legend_decimals = '%.2f'
                else:
                    legend_decimals = '%.1f'

            elif param_name in ['immersed_cyanobacteria_mean', 'floating_cyanobacteria_mean',
                                'floating_vegetation_mean']:
                if param_name == 'floating_vegetation_mean':
                    color_type = colscales.floating_portion()
                else:
                    color_type = colscales.cyano_portion()
                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                if (not param_range):
                    param_range = [0, 1]
                else:
                    if param_range[1] > 1:
                        param_range[1] = 1
                    if param_range[0] > param_range[1]:
                        param_range[0] = param_range[1] * 0.5
                if param_range[1] - param_range[0] > 0.5:
                    legend_decimals = '%.1f'
                else:
                    legend_decimals = '%.2f'
                ticks = [tick * param_range[1] / ticks[-1] for tick in ticks]

            elif param_name in ['extent', 'shallow']:
                color_type = colscales.extent_true()
                param_range = [-0.5, 2.5]
                ticks = [0, 1, 2]
                legend_decimals = '%.0f'

            # generic parameters for all water constituent concentrations and turbidity
            else:
                if param_name.endswith('sigma'):
                    color_type = colscales.rainbow_bright()
                else:
                    color_type = colscales.rainbow_king()

                if not param_range:
                    if masked_param_arr.count() != 0:
                        param_range, legend_decimals = divaux.get_range_specs(np.percentile(masked_param_arr.compressed(), 90))
                    else:
                        param_range = [0, 3]
                        legend_decimals = '%.1f'
                else:
                    forget, legend_decimals = divaux.get_range_specs((param_range[1] - param_range[0]))
                    legend_decimals = legend_decimals[0:2] + str(int(legend_decimals[2])+1) + legend_decimals[3:]

                rel_ticks = [0.00, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00]
                ticks = [rel_tick * param_range[1] for rel_tick in rel_ticks]

            # Skip if SRTM requested for area outside 55 deg NS
            if ((canvas_area[1][1] > 55) or (canvas_area[0][1] < -55)) and (basemap in ['srtm_hillshade', 'srtm_elevation']):
                print('   no SRTM data outside 55 deg N/S, skipping job')
                return

            ##############################
            ##### SRTM plot version ######
            ##############################
            elif canvas_area[1][1] <= 55 and canvas_area[0][1] >= -55 and basemap in ['srtm_hillshade', 'srtm_elevation']:

            # define the required srtm area
                srtm_area = [[math.floor(canvas_area[0][0]), math.floor(canvas_area[0][1])],
                             [math.ceil(canvas_area[1][0]),  math.ceil(canvas_area[1][1])]]
                srtm_tiles = [srtm_area[1][0] - srtm_area[0][0], srtm_area[1][1] - srtm_area[0][1]]

                # Get the int*int degree SRTM tile, fill data gaps
                elev, crs, extent = srtm.srtm_composite(srtm_area[0][0], srtm_area[0][1], srtm_tiles[1], srtm_tiles[0])
                elev_filled = srtm.fill_gaps(elev, 15)

                #  Add shading if requested (SAZ = 315 clockwise from South; SZN = 60)
                if basemap == 'srtm_hillshade':
                    print('   preparing SRTM hillshade basemap')
                    background = srtm.add_shading(elev_filled, 315.0, 30.0)
                    color_vals = [[0.6, 0.6, 0.6, 1], [1.0, 1.0, 1.0, 1]]
                    shade_grey = colors.LinearSegmentedColormap.from_list("ShadeGrey", color_vals)
                    base_cols = shade_grey
                elif basemap == 'srtm_elevation':
                    print('   preparing SRTM elevation basemap')
                    background = elev_filled
                    color_vals = [[0.6, 0.6, 0.6, 1], [1.0, 1.0, 1.0, 1]]
                    elev_grey = colors.LinearSegmentedColormap.from_list("ElevGrey", color_vals)
                    base_cols = elev_grey

                # Initialize plot once and subplots iteratively
                if nth_col + nth_row == 0:
                    x_canvas = (x_dist * ncols) / (y_dist * nrows)
                    fig = plt.figure(figsize = ((x_canvas * 3) + (2 * legend_extension), 3))
                map = fig.add_subplot(nrows, ncols, 1 + nth_col + (ncols * nth_row), projection=ccrs.Mercator())
                map.set_extent([canvas_area[0][0], canvas_area[1][0], canvas_area[0][1], canvas_area[1][1]])

                # Plot the background
                print('   adding background')
                background = map.imshow(background, extent=[srtm_area[0][0], srtm_area[1][0], srtm_area[0][1],
                            srtm_area[1][1]], cmap=base_cols, transform=crs, origin='lower')

                # Plot parameter
                print('   adding parameter content')
                parameter = map.imshow(masked_param_arr, extent=[lowlef.lon, upprig.lon, lowlef.lat, upprig.lat],
                           transform=crs, origin='upper', cmap=color_type, interpolation='none', vmin=param_range[0],
                           vmax=param_range[1], norm=None, zorder=10)

                # Add gridlines
                if grid:
                    print('   adding gridlines')
                    gridlines = map.gridlines(crs=crs, draw_labels=True, linewidth=linewidth, color='black', alpha=1.0,
                                              linestyle=':', zorder=13)
                    if orientation == 'square':
                        x_n_ticks = 4
                        y_n_ticks = 4
                    elif orientation == 'portrait':
                        x_n_ticks = 3
                        y_n_ticks = 4
                    else:
                        x_n_ticks = 4
                        y_n_ticks = 3
                    x_ticks = divaux.get_tick_positions(canvas_area[0][0], canvas_area[1][0], x_n_ticks)
                    y_ticks = divaux.get_tick_positions(canvas_area[0][1], canvas_area[1][1], y_n_ticks)
                    gridlines.xlocator = mpl.ticker.FixedLocator(x_ticks)
                    gridlines.ylocator = mpl.ticker.FixedLocator(y_ticks)
                    if nth_col != 0:
                        gridlines.ylabels_left = False
                    if nth_row != 0:
                        gridlines.xlabels_top = False
                    if nth_col + 1 != ncols:
                        gridlines.ylabels_right = False
                    if nth_row + 1 != nrows:
                        gridlines.xlabels_bottom = False

                    gridlines.xlabel_style = {'size': gridlabel_size, 'color': 'black'}
                    gridlines.ylabel_style = {'size': gridlabel_size, 'color': 'black'}

                # Annotate plot date
                print('   adding annotation')
                map.annotate(arranged_labels[nth_row][nth_col], xy = (0.05, 0.95), xycoords = 'axes fraction',
                                 fontsize = annotation_size, ha = 'left', va = 'top', zorder=14, )

            ##################################
            ##### non-SRTM plot version ######
            ##################################
            elif basemap in ['quadtree_rgb', 'none']:

            # Initialize plot once and subplots iteratively
                if nth_col + nth_row == 0:
                    x_canvas = (x_dist * ncols) / (y_dist * nrows)
                    fig = plt.figure(figsize = ((x_canvas * 3) + (2 * legend_extension), 3))
                map = fig.add_subplot(nrows, ncols, 1 + nth_col + (ncols * nth_row), projection=ccrs.Mercator())
                map.set_extent([canvas_area[0][0], canvas_area[1][0], canvas_area[0][1], canvas_area[1][1]])

                if basemap == 'none':
                    print('   proceeding without basemap')
                if basemap == 'quadtree_rgb':
                    print('   preparing Quadtree tiles basemap')
                    background = maps.QuadtreeTiles()

            # Add background
                    map.add_image(background, 10)

            # Plot parameter
                parameter = map.imshow(masked_param_arr, extent=[lowlef.lon, upprig.lon, lowlef.lat, upprig.lat],
                           transform=ccrs.PlateCarree(), origin='upper', cmap=color_type, interpolation='none',
                           vmin=param_range[0], vmax=param_range[1], norm=None, zorder=10)

            # Add gridlines
                if grid:
                    print('   adding gridlines')
                    gridlines = map.gridlines(draw_labels=True, linewidth=linewidth, color='black', alpha=1.0,
                                              linestyle=':', zorder=13)
                    if orientation == 'square':
                        x_n_ticks = 4
                        y_n_ticks = 4
                    elif orientation == 'portrait':
                        x_n_ticks = 3
                        y_n_ticks = 4
                    else:
                        x_n_ticks = 4
                        y_n_ticks = 3
                    x_ticks = divaux.get_tick_positions(canvas_area[0][0], canvas_area[1][0], x_n_ticks)
                    y_ticks = divaux.get_tick_positions(canvas_area[0][1], canvas_area[1][1], y_n_ticks)
                    gridlines.xlocator = mpl.ticker.FixedLocator(x_ticks)
                    gridlines.ylocator = mpl.ticker.FixedLocator(y_ticks)
                    if nth_col != 0:
                        gridlines.ylabels_left = False
                    if nth_row != 0:
                        gridlines.xlabels_top = False
                    if nth_col + 1 != ncols:
                        gridlines.ylabels_right = False
                    if nth_row + 1 != nrows:
                        gridlines.xlabels_bottom = False

                    gridlines.xlabel_style = {'size': gridlabel_size, 'color': 'black'}
                    gridlines.ylabel_style = {'size': gridlabel_size, 'color': 'black'}

            # Write date
                print('   adding annotation')
                map.annotate(arranged_labels[nth_row][nth_col], xy = (0.05, 0.95), xycoords = 'axes fraction',
                             fontsize = annotation_size, ha = 'left', va = 'top', zorder=14, color='white')

    # Create colorbar
    print('   creating colorbar')
    fig = plt.gcf()
    fig.subplots_adjust(top = 1, bottom = 0, left = 0, right = (x_canvas * 3) / ((x_canvas * 3) + (1.4 * legend_extension)),
                        wspace = 0.05, hspace = 0.05)
    cax = fig.add_axes([(x_canvas * 3) / ((x_canvas * 3) + (0.8 * legend_extension)), 0.15, 0.03, 0.7])
    fig.colorbar(parameter, cax=cax, ticks = ticks, format = legend_decimals, orientation=bar_orientation)
    legend_str = divaux.get_legend_str(param_name)
    cax.text(-1.2, 0.5, legend_str, rotation = 90, ha = 'center', va = 'center')

    # Save plot
    # Modifikation UL: Filename added.
    output_folder = os.path.dirname(arranged_filepaths[nth_row][nth_col]) + '/../maps-' + aggregate_type
    img_filename = output_folder + '/' + output_basename + '_' + param_name + '.png'
    print('   saving to PNG file', img_filename)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.savefig(img_filename, bbox_inches='tight', dpi=300)

    plt.close()


def plot_single_map(basemap, geogr_area, param_range, add_margin, param_str, d2products_folder, lake, aggregate_type):

    if d2products_folder == '':
        in_folder = 'Lake-' + lake + '/l3-' + aggregate_type
    else:
        in_folder = d2products_folder + '/Lake-' + lake + '/l3-' + aggregate_type
    products = []

    for file in os.listdir(in_folder):
        if file.endswith(".tif"):
            products.append(file)

    for product in products:
        arranged_filepaths = [[in_folder + '/' + product]]
        arranged_labels = divaux.get_single_label(product)
        output_basename = os.path.splitext(product)[0]

        plot_param(arranged_filepaths=arranged_filepaths, arranged_labels=arranged_labels,
                   output_basename=output_basename, param_range=param_range, param_name=param_str,
                   basemap=basemap, crop_ext=add_margin, geogr_area=geogr_area)

    print('   plot single map sequence finished')
    print('')


def plot_dby_map(basemap, geogr_area, param_range, add_margin, param_str, d2products_folder, lake):

    if d2products_folder == '':
        folder = 'Lake-' + lake + '/l3-yearly'
    else:
        folder = d2products_folder + '/Lake-' + lake + '/l3-yearly'

    arranged_filepaths, arranged_labels = divaux.get_dby_strings(folder)
    output_basename = 'Lake-' + lake + '_decade-by-year_subplots'

    plot_param(arranged_filepaths=arranged_filepaths, arranged_labels=arranged_labels,
               output_basename=output_basename, param_range=param_range, param_name=param_str,
               basemap=basemap, crop_ext=add_margin, geogr_area=geogr_area)

    print('   plot months by year sequence finished')
    print('')


def plot_ybm_map(basemap, geogr_area, param_range, add_margin, param_str, d2products_folder, lake):

    years = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']
    for year in years:
        if d2products_folder == '':
            folder = 'Lake-' + lake + '/l3-monthly'
        else:
            folder = d2products_folder + '/Lake-' + lake + '/l3-monthly'
        arranged_filepaths, arranged_labels = divaux.get_ybm_strings(folder, year)
        output_basename = 'Lake-' + lake + '_' + year + '_year-by-month_subplots'
        if arranged_filepaths != []:
            plot_param(arranged_filepaths=arranged_filepaths, arranged_labels=arranged_labels,
                       output_basename=output_basename, param_range=param_range, param_name=param_str,
                       basemap=basemap, crop_ext=add_margin, geogr_area=geogr_area)

    print('   plot years by month sequence finished')
    print('')


def main():

    config = configparser.ConfigParser()
    config.read('map_products.ini')
    method = config['DEFAULT']['method']
    d2products_folder = config['DEFAULT']['products_path']
    basemap = config['DEFAULT']['basemap']
    geogr_area = False if config['DEFAULT']['geogr_area'] == 'False' else config['DEFAULT']['geogr_area'].split(',')
    if geogr_area:
        geogr_area = [[float(geogr_area[0]), float(geogr_area[1])], [float(geogr_area[2]), float(geogr_area[3])]]
    add_margin = config['DEFAULT']['add_margin'] == 'True'
    param_range = False if config['DEFAULT']['param_range'] == 'False' else config['DEFAULT']['param_range'].split(',')
    if param_range:
        param_range = [float(param_range[0]), float(param_range[1])]
    param_str = config['DEFAULT']['param_str']
    lake = config['DEFAULT']['lake']

    if method.lower() == 'single':
        aggregate_type = config['SINGLE']['aggregate_type']
        plot_single_map(basemap=basemap, geogr_area=geogr_area, add_margin=add_margin, param_range=param_range,
                        param_str=param_str, aggregate_type=aggregate_type, d2products_folder=d2products_folder,
                        lake=lake)
    elif method.lower() == 'decade_by_years':
        plot_dby_map(basemap=basemap, geogr_area=geogr_area, add_margin=add_margin, param_range=param_range,
                     param_str=param_str, d2products_folder=d2products_folder, lake=lake)
    elif method.lower() == 'year_by_months':
        plot_ybm_map(basemap=basemap, geogr_area=geogr_area, add_margin=add_margin, param_range=param_range,
                     param_str=param_str, d2products_folder=d2products_folder, lake=lake)
    else:
        print('    method not known, please select single/decade_by_years/year_by_months')
        return


if __name__ == "__main__":
    main()