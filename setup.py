__author__ = 'Daniel'

from distutils.core import setup

# python3 setup.py sdist
setup(name='Diversity',
      version='1.0',
      description='Diversity II inland waters post-processing tools',
      author='Daniel Odermatt',
      author_email='daniel.odermatt@odermatt-brockmann.ch',
      url='http://www.diversity2.info/products/',
      packages=['modules'],
      package_dir={'modules': 'modules'},
      data_files=[('diversity-ini', ['ini/extract_stats.ini', 'ini/map_products.ini',
                                      'ini/plot_timeseries.ini'])],
      scripts=['scripts/map_products', 'scripts/plot_timeseries',
               'scripts/extract_stats'])




