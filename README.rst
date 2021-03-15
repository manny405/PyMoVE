# PyMoVE
Library and utilities for training volume estimation models with PyMoVE. 


Installation
------------

It's highly recommended to start from an Anaconda distribution of Python, which can be found here_. 

Current full installation requires running two ``setup.py`` scripts. This will be modified in the near future. This is because the algorithm that enables the project marching cube algorithm relys on Numba compilation to run efficiently. The algorithm is still accessible and built in pure Python if there's a user cannot compile with Numba for any reason. But will require some changes to the current code base. 

Running following commands installs the library:

.. code-block:: bash

    python setup.py install
    python setup_numba.py install

.. _here: https://www.anaconda.com/products/individual


Examples
--------

The examples directory steps though all features of the PyMoVE library. These are

1. Finding molecules from the molecular crystal structure

2. Calculating the packing factor of molecular crystals

3. Calculating the topological fragment descriptor

.. figure:: static/Fragment_Image.png
    :scale: 60 %
    :align: center

4. Calculating the packing accessible surface

.. figure:: static/PAS.png
    :scale: 60 %
    :align: center

5. Evaluated the pre-trained model and model training & testing

.. figure:: static/Results.png
    :scale: 60 %
    :align: center
