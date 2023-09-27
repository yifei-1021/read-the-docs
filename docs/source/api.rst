Useful functions 
===================================

Under the main ``banksy`` module


``banksy.initialize_banksy``

Under banksy.embed_banksy
-------
`` initialize_banksy(adata: anndata.AnnData,
                      coord_keys: Tuple[str],
                      num_neighbours: int = 15,
                      nbr_weight_decay: str = 'scaled_gaussian',
                      max_m: int = 1,
                      plt_edge_hist: bool = True,
                      plt_nbr_weights: bool = True,
                      plt_agf_angles: bool = False,
                      plt_theta: bool = True ) -> dict:`` 

This is the main Function that initializes the BANKSY Object as a dictionary
    
Input Args:
    adata (AnnData): AnnData object containing the data matrix
    
    coord_keys (Tuple[str]): A

    num_neighbours or k_geom (int) : The number of neighbours in which the edges, weights and theta graph are constructed. By default, we use k_geom = 15 as we have done for all results in our manuscript.

    nbr_weight_decay (str): Type of neighbourhood decay function, can be 'scaled_gaussian' or 'reciprocal'. By default, we use ``scaled_gaussian``.

    max_m (int): Maximum order of azimuthal gabor filter, we use a default of 1.

    
Optional Args:
    plt_edge (bool): Visualize the edge histogram*

    plt_weight (bool): Visualize the weights graph

    plt_agf_weights (bool): Visualize the AGF weights

    plt_theta (bool): Visualize angles around random cell


``generate_banksy_matrix`` banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                    banksy_dict,
                                                    lambda_list,
                                                    max_m)

Under the utilities ``utils`` module
--------

.. autosummary::
   :toctree: generated

   BANKSY\_py
