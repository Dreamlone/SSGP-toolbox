import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

requires = [
    'gdal>=2.4',
    'numpy',
    'scikit-learn==0.21.3',
    'pandas',
    'scipy',
    'netCDF4',
    'pyproj'
]

setuptools.setup(
    name = "SSGP-toolbox",
    version = "1.0",
    author = "Mikhail Sarafanov",
    author_email = "mik_sar@mail.ru",
    description = "Simple Spatial Gapfilling Processor. Toolbox for filling gaps in spatial datasets (e.g. remote sensing data)",
    long_description = long_description,
    keywords = 'machine learning, spatial data, gapfilling',
    long_description_content_type = "text/markdown",
    url = "https://github.com/Dreamlone/SSGP-toolbox",
    install_requires = requires,
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
)
