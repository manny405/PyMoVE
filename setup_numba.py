# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup
from pymove.molecules.marching_cubes_numba_compilation import cc

setup(
      name='pymove',
      version='0.0',
      packages=['pymove',
                'pymove/io',
                'pymove/molecules',
                'pymove/crystals',
                'pymove/libmpi',
                'pymove/models',
                'pymove/cli',
                ],
      #find_packages(exclude=[]),
      install_requires=['numpy', 'matplotlib', 'sklearn', 
                        'pandas','pymatgen', 'scipy', 'pymongo',
                        'torch', 'numba',
			"ase @ https://gitlab.com/ase/tarball/master",
                        "pycifrw"],
      ext_modules=[cc.distutils_extension()],
      entry_points={'console_scripts': ['pymove=pymove.cli.main:main']},
      )
