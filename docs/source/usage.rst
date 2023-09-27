Usage
=====

.. _installation:

Quick installation
------------

To use the python-implementation of Banksy, we recommend setting up a `conda` environment and installing the prequisite packages.

.. code-block:: console

   (base) $ conda create --name banksy
   (base) $ conda activate banksy
   (banksy) $ conda install -c conda-forge scanpy python-igraph leidenalg

Creating recipes
----------------

To retrieve a list of random ingredients,
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

