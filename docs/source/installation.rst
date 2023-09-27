Installation
=====

.. _Prequisite Packages:

Quick installation via Anaconda (recommended)
------------

To use ``Banksy_py``, we recommend setting up a ``conda`` environment and installing the prequisite packages, then cloning this repository.

.. code-block:: console

   (base) $ conda create --name banksy
   (base) $ conda activate banksy
   (banksy) $ conda install -c conda-forge scanpy python-igraph leidenalg
   (banksy) $ git clone https://github.com/prabhakarlab/Banksy_py.git

Alternatively, users can install the prerequisite packages using ``pip``. 

Additional Packages (Optional)
----------------

By default, we use the leiden-algorithm implemented via the ``leidenalg`` as the community-detection algorithm. Alternatively, users may be interested in using ``louvain`` and ``mclust``.

To run ``louvain`` clustering (another resolution-based clustering algorithm) via ``sc.tl.louvain``, install additional packages via:

.. code-block:: console

   (banksy) $ conda install -c conda-forge louvain

To run ``mclust`` (a Gaussian-mixture modelling based clustering method), install additional packages via:

.. code-block:: console

   (banksy) $ conda install -c conda-forge rpy2
   (banksy) $ conda install -c conda-forge r-mclust


Get started with some examples
----------------
Users can get started by trying out a few example tutorials here: :doc:`slideseqv1_analysis`.
