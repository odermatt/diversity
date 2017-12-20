#!python
__author__ = 'Daniel'

import numpy as np
import math as math
import os as os
import snappy as snappy
import cartopy.crs as ccrs
import cartopy.io.srtm as srtm
import cartopy.io.img_tiles as maps
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import modules.auxiliary_functions as divaux
import modules.colour_scales as colscales
import configparser as configparser
import urllib.request
from http.cookiejar import CookieJar
from cartopy.io import PostprocessedRasterSource, LocatedImage


# basemap can be: srtm_hillshade, srtm_elevation, quadtree_rgb, none
# using srtm in areas above 55 deg N or S will produce an error


def plot_param(arranged_filepaths, arranged_labels, param_name, output_basename, basemap='srtm_elevation',
               crop_ext=False, param_range=False, blacklist=False, grid=True, geogr_area=False):

    global canvas_area

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

    valid_months = False

    for nth_row in range(nrows):
        for nth_col in range(ncols):

            print('Plotting ' + param_name + ' of ' + arranged_filepaths[nth_row][nth_col])

            # get product parameters
            product_path = arranged_filepaths[nth_row][nth_col]
            product = snappy.ProductIO.readProduct(product_path)

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

            if product_path.split('_')[1] in blacklist:
                param_arr[:] = np.nan
            else:
                param_band.readPixels(0, 0, width, height, param_arr)
                if not np.isnan(np.nanmax(param_arr)):
                    valid_months = True

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
                obs_band = product.getBand('num_obs')
                obs_arr = np.zeros(width * height,  dtype=data_type)
                obs_band.readPixels(0, 0, width, height, obs_arr)
                obs_arr = obs_arr.reshape(height, width)
                extent_arr = (obs_arr >= 1).astype(int)
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
            geocoding = product.getSceneGeoCoding()
            lowlef = geocoding.getGeoPos(snappy.PixelPos(0, height - 1), None)
            upprig = geocoding.getGeoPos(snappy.PixelPos(width - 1, 0), None)

            # determine appropriate map canvas
            if not geogr_area:

            # add map extent if desired (larger gap north for date labels)
                if crop_ext:
                    south_ext = (upprig.lat - lowlef.lat) / 8
                    north_ext = (upprig.lat - lowlef.lat) / 5
                    lon_ext = (upprig.lon - lowlef.lon) / 8
                else:
                    south_ext = 0
                    north_ext = 0
                    lon_ext = 0

            # increase the smaller portion of the map canvas until the image aspect ratio is at most 3:2
                x_dist = divaux.calculate_haversine(lowlef.lon - lon_ext, lowlef.lat - south_ext,
                                                       upprig.lon + lon_ext, lowlef.lat + north_ext)
                y_dist = divaux.calculate_haversine(lowlef.lon, lowlef.lat - south_ext - south_ext,
                                                       lowlef.lon, upprig.lat + north_ext + north_ext)
                if x_dist / y_dist < 2 / 3:
                    if lon_ext == 0:
                        lon_ext = (upprig.lon - lowlef.lon) / 20
                    while x_dist / y_dist < 2 / 3:
                        lon_ext = lon_ext * 1.1
                        x_dist = divaux.calculate_haversine(lowlef.lon - lon_ext, lowlef.lat,
                                                               upprig.lon + lon_ext, lowlef.lat)
                elif x_dist / y_dist > 3 / 2:
                    if south_ext == 0:
                        south_ext = (upprig.lat - lowlef.lat) / 20
                        north_ext = (upprig.lat - lowlef.lat) / 20
                    while x_dist / y_dist > 3 / 2:
                        south_ext = south_ext * 1.1
                        north_ext = north_ext * 1.1
                        y_dist = divaux.calculate_haversine(lowlef.lon, lowlef.lat - south_ext,
                                                               lowlef.lon, upprig.lat + north_ext)
                canvas_area = [[lowlef.lon - lon_ext, lowlef.lat - south_ext],
                               [upprig.lon + lon_ext, upprig.lat + north_ext]]

                area_str =''

            # or use the predefined area's coordinates
            else:
                canvas_area = geogr_area #[[-1.1, 7.9], [0.1, 8.7]]# geogr_area

                area_str = '_' + str(geogr_area[0][0]) + '_' + str(geogr_area[0][1]) + '_' + str(geogr_area[1][0]) + '_' + \
                           str(geogr_area[1][1])
                area_str = area_str.replace('.', '-')

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

            # Try to get param_range from monthly stats
            elif not param_range:
                base_path = '/' + os.path.join(*product_path.split('/')[:-2])
                lake_name = product_path.split('/')[-3].split('-')[1]
                monthly_stats_path =  base_path + '/parameter-stats-monthly/Lake-' + \
                                     lake_name + '_' + param_name + '.txt'
                if os.path.exists(monthly_stats_path):
                    meas_dates, meas_values, errors = divaux.read_statsmonthly(monthly_stats_path, 'p90_threshold', blacklist)
                    range_max = np.nanpercentile(meas_values,90)
                    if np.isnan(range_max):
                        range_max = 999
                    if range_max < 0.1:
                        range_max = 0.1
                    if param_name.startswith('lswt'):
                        meas_dates, meas_values, errors = divaux.read_statsmonthly(monthly_stats_path, 'p10_threshold', blacklist)
                        range_min = np.nanmin(meas_values) - 273.15
                        range_max = np.nanmax(meas_values) - 273.15
                    else:
                        range_min = 0
                    param_range = [range_min, range_max]

            if param_name in ['num_obs']:
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
                if not param_range:
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
                    #legend_decimals = legend_decimals[0:2] + str(int(legend_decimals[2])+1) + legend_decimals[3:]

                rel_ticks = [0.00, 0.2, 0.4, 0.6, 0.8, 1.00]
                ticks = [rel_tick * param_range[1] for rel_tick in rel_ticks]

            # Skip if SRTM requested for area outside 55 deg NS
            if ((canvas_area[1][1] > 55) or (canvas_area[0][1] < -55)) and (basemap in ['srtm_hillshade', 'srtm_elevation']):
                print('   no SRTM data outside 55 deg N/S, skipping job')
                return

            ##############################
            ##### SRTM plot version ######
            ##############################
            elif canvas_area[1][1] <= 55 and canvas_area[0][1] >= -55 and basemap in ['srtm_hillshade', 'srtm_elevation']:

                if x_dist < 50 and y_dist < 50:
                    print('   larger image side is ' + str(round(max(x_dist, y_dist), 1)) + ' km, applying SRTM1')
                    source = srtm.SRTM1Source
                else:
                    print('   larger image side is ' + str(round(max(x_dist, y_dist), 1)) + ' km, applying SRTM3')
                    source = srtm.SRTM3Source

            # define the required srtm area
                srtm_area = [[math.floor(canvas_area[0][0]), math.floor(canvas_area[0][1])],
                             [math.ceil(canvas_area[1][0]),  math.ceil(canvas_area[1][1])]]
                srtm_tiles = [srtm_area[1][0] - srtm_area[0][0], srtm_area[1][1] - srtm_area[0][1]]

                #  Add shading if requested (SAZ = 315 clockwise from South; SZN = 60)
                if basemap == 'srtm_hillshade':
                    print('   preparing SRTM hillshade basemap')
                    srtm_raster = PostprocessedRasterSource(source(max_nx=8, max_ny=8), shade)
                    color_vals = [[0.8, 0.8, 0.8, 1], [1.0, 1.0, 1.0, 1]]
                    shade_grey = colors.LinearSegmentedColormap.from_list("ShadeGrey", color_vals)
                    base_cols = shade_grey
                elif basemap == 'srtm_elevation':
                    print('   preparing SRTM elevation basemap')
                    srtm_raster = PostprocessedRasterSource(source(max_nx=8, max_ny=8), elevate)
                    color_vals = [[0.7, 0.7, 0.7, 1], [0.90, 0.90, 0.90, 1], [0.97, 0.97, 0.97, 1], [1.0, 1.0, 1.0, 1]]
                    elev_grey = colors.LinearSegmentedColormap.from_list("ElevGrey", color_vals)
                    base_cols = elev_grey

                # Initialize plot once and subplots iteratively
                if nth_col + nth_row == 0:
                    x_canvas = (x_dist * ncols) / (y_dist * nrows)
                    fig = plt.figure(figsize = ((x_canvas * 3) + (2 * legend_extension), 3))
                map = fig.add_subplot(nrows, ncols, 1 + nth_col + (ncols * nth_row), projection=ccrs.PlateCarree())
                map.set_extent([canvas_area[0][0], canvas_area[1][0], canvas_area[0][1], canvas_area[1][1]])

                # Plot the background
                print('   adding background')
                map.add_raster(srtm_raster, cmap=base_cols)

            ##################################
            ##### non-SRTM plot version ######
            ##################################

            elif basemap in ['quadtree_rgb', 'nobasemap']:

                # Initialize plot once and subplots iteratively
                if nth_col + nth_row == 0:
                    x_canvas = (x_dist * ncols) / (y_dist * nrows)
                    fig = plt.figure(figsize = ((x_canvas * 3) + (2 * legend_extension), 3))
                map = fig.add_subplot(nrows, ncols, 1 + nth_col + (ncols * nth_row), projection=ccrs.PlateCarree())
                map.set_extent([canvas_area[0][0], canvas_area[1][0], canvas_area[0][1], canvas_area[1][1]])

                if basemap == 'nobasemap':
                    print('   proceeding without basemap')
                if basemap == 'quadtree_rgb':
                    print('   preparing Quadtree tiles basemap')

                    #background = maps.GoogleTiles(style='street')
                    #background = maps.GoogleTiles(style='satellite')
                    #background = maps.GoogleTiles(style='terrain')
                    #background = maps.MapQuestOpenAerial()
                    #background = maps.OSM()
                    background = maps.QuadtreeTiles()
                    #crs = maps.GoogleTiles().crs
                    #crs = maps.QuadtreeTiles().crs

            # Add background
                    print('   adding background')
                    map.add_image(background, 10)

            ##############################
            ##### both plot versions #####
            ##############################

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
            if basemap == 'quadtree_rgb':
                map.annotate(arranged_labels[nth_row][nth_col], xy = (0.05, 0.95), xycoords = 'axes fraction',
                             fontsize = annotation_size, ha = 'left', va = 'top', zorder=14, color='white')
            else:
                map.annotate(arranged_labels[nth_row][nth_col], xy = (0.05, 0.95), xycoords = 'axes fraction',
                             fontsize = annotation_size, ha = 'left', va = 'top', zorder=14, color='black')

    if not valid_months:
        print('   All products blacklisted, no plot produced for ' + output_basename + '_' + param_name)
        return

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
    output_folder = os.path.dirname(arranged_filepaths[nth_row][nth_col]) + '/../maps-' + aggregate_type
    img_filename = output_folder + '/' + output_basename + '_' + param_name + area_str + '.png'
    print('   saving to PNG file', img_filename)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.savefig(img_filename, bbox_inches='tight', dpi=400)

    plt.close()


def authenticate(username, password):

    # See discussion https://github.com/SciTools/cartopy/issues/789#issuecomment-245789751
    # Create a password manager to deal with the 401 reponse that is returned from
    # Earthdata Login

    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)

    # Create a cookie jar for storing cookies. This is used to store and return
    # the session cookie given to use by the data server (otherwise it will just
    # keep sending us back to Earthdata Login to authenticate).  Ideally, we
    # should use a file based cookie jar to preserve cookies between runs. This
    # will make it much more efficient.

    cookie_jar = CookieJar()

    # Install all the handlers.

    opener = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(password_manager),
        #urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
        #urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
        urllib.request.HTTPCookieProcessor(cookie_jar))
    urllib.request.install_opener(opener)

    return


def elevate(located_elevations):
    canvas_extent = (canvas_area[0][0], canvas_area[1][0], canvas_area[0][1], canvas_area[1][1])
    x_pixpdeg = len(located_elevations[0][0,:]) / (located_elevations.extent[1] - located_elevations.extent[0])
    y_pixpdeg = len(located_elevations[0][:,0]) / (located_elevations.extent[3] - located_elevations.extent[2])
    left_ind = math.floor(x_pixpdeg * (canvas_area[0][0] - located_elevations.extent[0]))
    righ_ind = math.floor(x_pixpdeg * (canvas_area[1][0] - located_elevations.extent[0]))
    lowe_ind = len(located_elevations[0][:,0]) - math.ceil(y_pixpdeg * (canvas_area[1][1] - located_elevations.extent[2]))
    uppe_ind = len(located_elevations[0][:,0]) - math.ceil(y_pixpdeg * (canvas_area[0][1] - located_elevations.extent[2]))

    #Rückgabe ganzer SRTM Tiles, macht Bildcanvas so gross wie unsichtbare SRTM Fläche
    #return LocatedImage(scaled_elevations, located_elevations.extent)
    return LocatedImage(located_elevations[0][lowe_ind:uppe_ind, left_ind:righ_ind], canvas_extent)


def shade(located_elevations):
    located_shades = srtm.add_shading(located_elevations.image, azimuth=135, altitude=15)
    canvas_extent = (canvas_area[0][0], canvas_area[1][0], canvas_area[0][1], canvas_area[1][1])
    x_pixpdeg = len(located_shades[0,:]) / (located_elevations.extent[1] - located_elevations.extent[0])
    y_pixpdeg = len(located_shades[:,0]) / (located_elevations.extent[3] - located_elevations.extent[2])
    left_ind = math.floor(x_pixpdeg * (canvas_area[0][0] - located_elevations.extent[0]))
    righ_ind = math.floor(x_pixpdeg * (canvas_area[1][0] - located_elevations.extent[0]))
    lowe_ind = len(located_shades[:,0]) - math.ceil(y_pixpdeg * (canvas_area[1][1] - located_elevations.extent[2]))
    uppe_ind = len(located_shades[:,0]) - math.ceil(y_pixpdeg * (canvas_area[0][1] - located_elevations.extent[2]))

    #Rückgabe ganzer SRTM Tiles, macht Bildcanvas so gross wie unsichtbare SRTM Fläche
    #return LocatedImage(located_shades, located_elevations.extent)
    return LocatedImage(located_shades[lowe_ind:uppe_ind, left_ind:righ_ind], canvas_extent)


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


def plot_ybm_map(basemap, geogr_area, param_range, add_margin, param_str, d2products_folder, lake, blacklist):

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
                       param_name=param_str, output_basename=output_basename, param_range=param_range,
                       blacklist=blacklist, basemap=basemap, crop_ext=add_margin, geogr_area=geogr_area)

    print('   plot years by month sequence finished')
    print('')


def main():

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ini', 'map_products.ini'))
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

    username = config['DEFAULT']['username']
    password = config['DEFAULT']['password']
    if username != 'None' and password != 'None':
        authenticate(username=username, password=password)

    params = config['DEFAULT']['param_str']
    params_list = [param.lstrip() for param in params.split(',')]

    lakes = config['DEFAULT']['lakes']
    lakes_list = [lake.lstrip() for lake in lakes.split(',')]

    for lake in lakes_list:
        for param in params_list:
            if method.lower() == 'single':
                aggregate_type = config['SINGLE']['aggregate_type']
                plot_single_map(basemap=basemap, geogr_area=geogr_area, add_margin=add_margin, param_range=param_range,
                                param_str=param, aggregate_type=aggregate_type, d2products_folder=d2products_folder,
                                lake=lake)
            elif method.lower() == 'decade_by_years':
                plot_dby_map(basemap=basemap, geogr_area=geogr_area, add_margin=add_margin, param_range=param_range,
                             param_str=param, d2products_folder=d2products_folder, lake=lake)
            elif method.lower() == 'year_by_months':
                blacklist_config = config['YEAR_BY_MONTH']['date_blacklist']

                if blacklist_config != 'False':
                    if blacklist_config.startswith('blacklist_'):
                        if param == 'num_obs':
                            blacklist = ''
                        else:
                            blacklist = divaux.read_blacklist(d2products_folder + '/Lake-' + lake + '/' + blacklist_config +
                                                  '/blacklist_lake-' + lake + '_' + param + '.txt')
                    elif blacklist_config.startswith('20'):
                        blacklist = blacklist_config.split(',')
                else:
                    blacklist = False

                plot_ybm_map(basemap=basemap, geogr_area=geogr_area, add_margin=add_margin, param_range=param_range,
                             param_str=param, d2products_folder=d2products_folder, lake=lake, blacklist=blacklist)
            else:
                print('    method not known, please select single/decade_by_years/year_by_months')
                return


if __name__ == "__main__":
    main()