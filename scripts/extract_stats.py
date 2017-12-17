#!python
__author__ = 'Daniel'

import configparser as configparser
import platform as platform
import os as os
import string as string
import pandas as pandas
import subprocess as subprocess


def create_xml(xml_path, input_shape_file, source_band_list, percentiles):

    xml = open(xml_path, 'w')

    xml.write('  <graph id="someGraphId">\n')
    xml.write('    <version>1.0</version>\n')
    xml.write('    <node id="someNodeId">\n')
    xml.write('      <operator>StatisticsOp</operator>\n')
    xml.write('      <parameters>\n')
    xml.write('        <sourceProductPaths>$inProduct</sourceProductPaths>\n')
    xml.write('        <shapefile>' + input_shape_file + '</shapefile>\n')
    xml.write('        <bandConfigurations>\n')

    for source_band_name in source_band_list:
        xml.write('          <bandConfiguration>\n')
        xml.write('            <sourceBandName>' + source_band_name + '</sourceBandName>\n')
        xml.write('          </bandConfiguration>\n')

    xml.write('        </bandConfigurations>\n')
    xml.write('        <percentiles>' + percentiles + '</percentiles>\n')
    xml.write('        <accuracy>4</accuracy>\n')
    xml.write('        <outputShapefile>$outShp</outputShapefile>\n')
    xml.write('        <outputAsciiFile>$outAscii</outputAsciiFile>\n')
    xml.write('      </parameters>\n')
    xml.write('    </node>\n')
    xml.write('  </graph>')

    xml.close()
    return xml_path


def iterate_statsop(beam_path, basis_xml_path, input_product_folder, output_folder):

    basis_xml = open(basis_xml_path, 'r').read()

    for file in os.listdir(input_product_folder):
        if file.endswith('.tif'):
            prod_name = os.path.basename(file)
            xml_modifications = string.Template(basis_xml).substitute(
                {'inProduct': input_product_folder + prod_name,
                 'outAscii': output_folder + '/' + os.path.splitext(prod_name)[0] + '.txt',
                 'outShp': output_folder + '/' + os.path.splitext(prod_name)[0] + '.shp'})

            modif_xml_path = output_folder + '/job_temp.xml'
            modif_xml = open(modif_xml_path, 'w')
            modif_xml.write(xml_modifications)
            modif_xml.close()

            print('Running StatisticsOp for product ' + file)

            if(platform.system() == 'Windows'):
                os.system('"' + beam_path + 'bin/gpt.bat" ' + modif_xml_path)
            else:
                #os.system(beam_path + 'bin/gpt.sh "' + modif_xml_path + '"')
                os.environ['JAVA_TOOL_OPTIONS'] = '-Djava.awt.headless=true'
                subprocess.call([beam_path + 'bin/gpt.sh', modif_xml_path])

            os.remove(modif_xml_path)
            for extension in ['_band_mapping.txt', '_metadata.txt', '.dbf', '.fix', '.prj', '.qix', '.shp', '.shx']:
                os.remove(output_folder + '/' + os.path.splitext(prod_name)[0] + extension)

        else:
            print('Skipping file ' + file)

    os.remove(basis_xml_path)


def merge_product_statistics(in_path, out_file, param):

    in_files = [f for f in os.listdir(in_path) if (f.endswith('.txt'))]
    in_files.sort()
    first_table = pandas.read_csv(in_path + '/' + in_files[0], sep='\t', header=0, na_values=[''])
    write_file = open(out_file, 'w')

    header = 'stats_start_date\t' + 'stats_end_date\t'
    for col_title in first_table.columns[2:]:
        header = header + col_title + '\t'
    header = header[:-1] +'\n'
    write_file.write(header)

    start_date = in_files[0][-25:-15]
    end_date = in_files[0][-14:-4]
    first_row = start_date + '\t' + end_date + '\t'
    param_row = first_table[first_table['Band'] == param].index.tolist()[0]
    param_stats = first_table.iloc[param_row]
    for col_value in param_stats[2:]:
        first_row = first_row + str(col_value) + '\t'
    first_row = first_row[:-1] +'\n'
    write_file.write(first_row)

    if len(in_files) > 1:
        for in_file in in_files[1:]:
            any_table = pandas.read_csv(in_path + '/' + in_file, sep='\t', header=0, na_values=[''])
            start_date = in_file[-25:-15]
            end_date = in_file[-14:-4]
            any_row = start_date + '\t' + end_date + '\t'
            param_row = any_table[any_table['Band'] == param].index.tolist()[0]
            param_stats = any_table.iloc[param_row]
            for col_value in param_stats[2:]:
                any_row = any_row + str(col_value) + '\t'
            any_row = any_row[:-1] +'\n'
            write_file.write(any_row)

    write_file.close()


def main():

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ini', 'extract_stats.ini'))
    aggregate_type = config['DEFAULT']['aggregate_type']
    beam_path = config['DEFAULT']['beam_path']
    d2products_folder = config['DEFAULT']['products_path']

    if d2products_folder != '':
        if d2products_folder[-1] == '/':
           d2products_folder = d2products_folder[:-1]

    shape_files = []
    shape_files_path = config['DEFAULT']['shapefiles_path']
    shape_files_allext = os.listdir(shape_files_path)
    for shape_file_allext in shape_files_allext:
        if shape_file_allext.endswith('.shp'):
            shape_files.append(shape_file_allext)

    lakes = config['DEFAULT']['lakes']
    lakes_list = [lake.lstrip() for lake in lakes.split(',')]

    for lake in lakes_list:
        if d2products_folder == '':
            input_product_folder = d2products_folder + 'Lake-' + lake + '/l3-' + aggregate_type + '/'
        else:
            input_product_folder = d2products_folder + '/Lake-' + lake + '/l3-' + aggregate_type + '/'

        for shape_file in shape_files:
            if '0_' + lake.lower() + '.shp' in shape_file.lower():
                lake_shape_file = shape_files_path + '/' + shape_file
                print(lake_shape_file + ' found')
        source_band_names = config['DEFAULT']['param_str']
        source_band_list = [source_band_name.lstrip() for source_band_name in source_band_names.split(',')]
        percentiles = config['DEFAULT']['percentiles']
        product_table_path = d2products_folder + '/Lake-' + lake + '/product-stats-' + aggregate_type
        param_table_path = d2products_folder + '/Lake-' + lake + '/parameter-stats-' + aggregate_type

        if not os.path.exists(product_table_path):
            os.makedirs(product_table_path)
        if not os.path.exists(param_table_path):
            os.makedirs(param_table_path)

        basis_xml_path = product_table_path + '/batch_temp.xml'

        create_xml(basis_xml_path, lake_shape_file, source_band_list, percentiles)

        iterate_statsop(beam_path=beam_path,
                        basis_xml_path=basis_xml_path,
                        input_product_folder=input_product_folder,
                        output_folder=product_table_path)

        for source_band in source_band_list:
            out_file = param_table_path + '/Lake-' + lake + '_' + source_band + '.txt'
            print('Merging ' + source_band + ' from Lake ' + lake + ' ' + aggregate_type + ' product statistics')
            merge_product_statistics(in_path=product_table_path, out_file=out_file, param=source_band)

            print('\nPlease find merged results in:', out_file)


if __name__ == "__main__":
    main()