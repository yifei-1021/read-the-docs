Usage
=====

.. _installation:

Quick installation via Anaconda (recommended)
------------

To use the python-implementation of Banksy, we recommend setting up a ``conda`` environment and installing the prequisite packages. 
.. code-block:: console

   (base) $ conda create --name banksy
   (base) $ conda activate banksy
   (banksy) $ conda install -c conda-forge scanpy python-igraph leidenalg

Alternatively, users can install the prerequisite packages using `pip`. 

Additional Packages (Optional)
----------------

By default, we use the leiden-algorithm implemented via the ``leidenalg`` as the community-detection algorithm. Alternatively, users may be interested in using ``louvain`` and ``mclust``.

For ``louvain``, install additional packages via:

.. code-block:: console

   (banksy) $ conda install -c conda-forge louvain

you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

