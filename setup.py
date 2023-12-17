# coding: UTF-8
from setuptools import setup,find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# Package meta-data.
NAME = 'osmlocator'
DESCRIPTION = 'A tool for locating overlapping scatter marks.'
URL = 'https://github.com/ccqym/OsmLocator'
EMAIL = 'ccqiuym@126.com'
AUTHOR = 'Yuming Qiu'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '0.1.0'

REQUIRED = [
    'opencv_python>=3.3', 'numpy'
]

EXTRAS = {}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    #packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    packages=find_packages(exclude=["tests", 'datasets', "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    py_modules=['osmlocator'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MPL-2.0',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
