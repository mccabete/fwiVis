import setuptools

setuptools.setup(name='fwiVis',
version='0.1',
description='A package to load up and visualize FWI data next to fire objects genereated from VIIRS',
url='#',
author='Tempest McCabe',
install_requires=['s3fs', 'numpy', 'pandas', 'geopandas', 'matplotlib', 'xarray', 'rioxarray', 'rasterio', 'geocube', 'shapely', 'folium'],
author_email='tmccabe@umd.edu',
packages=setuptools.find_packages(),
zip_safe=False)
