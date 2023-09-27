Usage
=====

.. _installation:

Quick installation via Anaconda (recommended)
------------

To use the python-implementation of ``Banksy``, we recommend setting up a ``conda`` environment and installing the prequisite packages. 

.. code-block:: console

   (base) $ conda create --name banksy
   (base) $ conda activate banksy
   (banksy) $ conda install -c conda-forge scanpy python-igraph leidenalg

Alternatively, users can install the prerequisite packages using `pip`. 

Additional Packages (Optional)
----------------

By default, we use the leiden-algorithm implemented via the ``leidenalg`` as the community-detection algorithm. Alternatively, users may be interested in using ``louvain`` and ``mclust``.

To run ``louvain`` clustering (another resolution-based clustering algorithm) via ``sc.tl.louvain``, install additional packages via:

.. code-block:: console

   (banksy) $ conda install -c conda-forge louvain

To run ``mclust``, install additional packages via:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

