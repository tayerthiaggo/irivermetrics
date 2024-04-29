from setuptools import setup, find_packages

setup(
    name='irivermetrics',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'dask',
        'geopandas',
        'numpy',
        'odc_algo',
        'odc_geo',
        'odc_stac',
        'pandas',
        'psutil',
        'pyproj',
        'rasterio',
        'rioxarray',
        'scipy',
        'Shapely',
        'skimage',
        'waterdetect',
        'xarray'
    ],
    # Optional metadata
    author='Thiaggo Tayer',
    author_email='thiaggo.tayer@uwa.edu.au',
    description='An open-source Python toolkit for identifying surface water and analyzing intermittent river patterns using remote sensing data.',
    license='MIT',
    keywords='water detection utilities',
    url='https://github.com/tayerthiaggo/irivermetrics',
)