#
# This file is part of TransportMaps.
#
# TransportMaps is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TransportMaps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with TransportMaps.  If not, see <http://www.gnu.org/licenses/>.
#
# Transport Maps Library
# Copyright (C) 2015-2018 Massachusetts Institute of Technology
# Uncertainty Quantification group
# Department of Aeronautics and Astronautics
#
# Author: Transport Map Team
# Website: transportmaps.mit.edu
# Support: transportmaps.mit.edu/qa/
#

import os
import os.path
import sys, getopt, re
from setuptools import setup, find_packages
from Cython.Build import cythonize

global include_dirs
include_dirs = []

######################
# SCRIPTS
scripts_list = ['scripts/tmap-laplace',
                'scripts/tmap-max-likelihood',
                'scripts/tmap-postprocess',
                'scripts/tmap-tm',
                'scripts/tmap-adaptivity-postprocess',
                'scripts/tmap-sequential-tm',
                'scripts/tmap-sequential-postprocess',
                'scripts/tmap-deep-lazy-tm',
                'scripts/tmap-deep-lazy-trim',
                'scripts/tmap-load-dill']

######################
# DEPENDENCIES
# (mod_name, use_wheel)
setup_requires = []
install_requires = [
    'numpy>=1.15',
    'orthpol_light>=1.0.1',
    'scipy',
    'SpectralToolbox>=1.0.7',
    'dill',
    'scikit-sparse',
    'sortedcontainers',
    'semilattices',
    'tqdm',
    'fasteners'
]
opt_inst_req = {'SPHINX': ['Sphinx',
                           'sphinxcontrib-bibtex', 
                           'sphinx-prompt',
                           'robpol86-sphinxcontrib-googleanalytics',
                           # 'sphinxcontrib-googleanalytics',
                           'sphinxcontrib-contentui',
                           'nbsphinx',
                           'ipython', 
                           'sphinx_rtd_theme', 
                           'tabulate',
                           'pandoc'],
                'PLOT': ['matplotlib'],
                'MPI': ['mpi4py',
                        'mpi_map>=2.5'],
                'H5PY': ['mpi4py',
                         'h5py'],
                'PYHMC': ['cython',
                          'statsmodels',
                          'pyhmc']
}

#################################
# WRITE requirements.txt files
with open('requirements.txt','w') as f:
    for r in install_requires:
        f.write(r+"\n")
f.close()
for opt in opt_inst_req:
    with open('requirements-'+opt+'.txt', 'w') as f:
        for r in opt_inst_req[opt]:
            f.write(r+"\n")
    f.close()

# Cython files
ext_mod = ['TransportMaps/Distributions/Examples/Lorenz96/fast_eval.pyx']
    
# Get version string
local_path = os.path.split(os.path.realpath(__file__))[0]
version_file = os.path.join(local_path, 'TransportMaps/_version.py')
version_strline = open(version_file).read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, version_strline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))

# Check for optional packages in environment variables
for opt in opt_inst_req:
    val = os.getenv(opt)
    if val is not None:
        if val in ['TRUE', 'True', 'true']:
            install_requires += opt_inst_req[opt]

# Get optional pip flags
PIP_FLAGS = os.getenv('PIP_FLAGS')
if PIP_FLAGS is None:
    PIP_FLAGS = ''

setup(name = "TransportMaps",
      version = version,
      packages=find_packages(),
      include_package_data=True,
      url="http://transportmaps.mit.edu",
      author = "Transport Map Team - UQ Group - AeroAstro - MIT",
      author_email = "tmteam-support@mit.edu",
      license="COPYING.LESSER",
      description="Tools for the construction of transport maps",
      long_description=open("README.rst").read(),
      # cmdclass={'sdist': TransportMaps_sdist,
      #           'install': TransportMaps_install,
      #           'develop': TransportMaps_develop},
      include_dirs=include_dirs,
      scripts=scripts_list,
      setup_requires=setup_requires,
      install_requires=install_requires,
      zip_safe = False,         # I need this for debug purposes
      ext_modules=cythonize(ext_mod),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
          'Natural Language :: English',
          'Operating System :: POSIX',
          'Operating System :: POSIX :: Linux',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          ],
      )
