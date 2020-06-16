==============
Transport Maps
==============

This package provides basic functionalities for the construction of monotonic transport maps.

Supported systems
-----------------

* \*nix like OS (Linux, Unix, ...)
* Mac OS

Other operating systems have not been tested and they likely require a more complex procedure for the installation (this includes the Microsoft Windows family..).

We reccommend to work in a virtual environment using `virtualenv <https://virtualenv.readthedocs.io/en/latest/>`_ or `Anaconda <https://www.continuum.io/why-anaconda>`_.

Installation requirements
-------------------------

* `gcc <https://gcc.gnu.org/>`_ (or an alternative C/C++ compiler)
* `gfortran <https://gcc.gnu.org/fortran/>`_ (or an alternative Fortran compiler)

Automatic installation
----------------------

First of all make sure to have the latest version of `pip <https://pypi.python.org/pypi/pip>`_ installed

 $ pip install --upgrade pip

The package and its python dependencies can be installed running the command:

 $ pip install --upgrade numpy
 $ pip install --upgrade TransportMaps

If one whish to enable some of the optional dependencies:

 $ MPI=True SPHINX=True PLOT=True H5PY=True pip install --upgrade TransportMaps

These options will install the following modules:

* MPI -- parallelization routines (see the `tutorial <mpi-usage.html>`_). It requires the separate installation of an MPI backend (`openMPI <https://www.open-mpi.org/>`_, `mpich <https://www.mpich.org/>`_, etc.). The following Python modules will be installed:
  * `mpi4py <https://pypi.python.org/pypi/mpi4py>`_
  * `mpi_map <https://pypi.python.org/pypi/mpi_map>`_

* PLOT -- plotting capabilities:

  * `MatPlotLib <https://pypi.python.org/pypi/matplotlib/>`_

* SPHINX -- documentation generation routines:

  * `sphinx <https://pypi.python.org/pypi/Sphinx>`_
  * `sphinxcontrib-bibtex <https://pypi.python.org/pypi/sphinxcontrib-bibtex/>`_
  * `ipython <https://pypi.python.org/pypi/ipython>`_
  * `nbsphinx <https://pypi.python.org/pypi/nbsphinx>`_

* H5PY -- routines for the storage of big data-set. It requires the separate installation of the `hdf5 <https://www.hdfgroup.org/>`_ backend.

  * `mpi4py <https://pypi.python.org/pypi/mpi4py>`_
  * `h5py <http://www.h5py.org/>`_

* PYHMC -- routines for Hamiltonian Markov Chain Monte Carlo

  * `pyhmc <http://pythonhosted.org/pyhmc/>`_

Manual installation
-------------------

If anything goes wrong with the automatic installation you can try to install manually the following packages.

Mandatory Back-end packages (usually installed with `numpy <https://pypi.python.org/pypi/numpy>`_):

* `BLAS <http://www.netlib.org/blas/>`_ (with development/header files)
* `LAPACK <http://www.netlib.org/lapack/>`_ (with development/header files)

Mandatory Python packages:

* `pip <https://pypi.python.org/pypi/pip>`_
* `numpy <https://pypi.python.org/pypi/numpy>`_ >= 1.10
* `scipy <https://pypi.python.org/pypi/scipy>`_
* `orthpol_light <https://pypi.python.org/pypi/orthpol-light>`_
* `SpectralToolbox <https://pypi.python.org/pypi/SpectralToolbox>`_
* `dill <https://pypi.python.org/pypi/dill>`_

Finally install TransportMaps:

 $ pip install TransportMaps

Running the Unit Tests
----------------------

Unit tests are available and can be run through the command:

   >>> import TransportMaps as TM
   >>> TM.tests.run_all()

There are >3500 unit tests, and it will take some time to run all of them.

Dependencies of the software "TransportMaps" are listed in the "setup.py" file.
The only dependency that is not yet openly available is "semilattices" which is attached here.
The folder "data", contains the problem settings and results appearing in the paper.
All python serialized objects are stored using the package "dill".
Big data sets are stored using "h5py"


Code for sections 4.4: 
The finite element discretization in the "poisson" example is carried out using "dolfin" and "dolfin-adjoint" version 2017.2.0
The finite element discretization in the "cantilever-timoshenko" example is carried out using "dolfin" and "dolfin-adjoint" version 2018.1.0
We stress that for reproducibility the version matters when it comes to "dolfin" and "dolfin-adjoint" (for reasons connected to the meshing ordering differences between the two versions).

Credits removed for anonymous submission to NeurIPS 2020
-------
