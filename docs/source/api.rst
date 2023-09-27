Useful functions 
===================================

Under the main ``banksy`` module


``banksy.initialize_banksy``

Functions under banksy.initialize_banksy
-------
**initialize_banksy**: ``initialize_banksy(adata: anndata.AnnData, coord_keys: Tuple[str], num_neighbours: int = 15, nbr_weight_decay: str = 'scaled_gaussian', max_m: int = 1,plt_edge_hist: bool = True, plt_nbr_weights: bool = True,plt_agf_angles: bool = False, plt_theta: bool = True ) -> dict:`` 

    This is the main Function that initializes the BANKSY Object as a dictionary
        
    Input Args:
        ``adata`` (AnnData): AnnData object containing the data matrix.

        ``coord_keys`` (Tuple[str]): A tuple containing 3 keys to access the `x`, `y` and `xy` coordinates of the cell positions under ``data.obs``. For example, ``coord_keys = ('x','y','xy')``, in which ``adata.obs['x']`` and ``adata.obs['y']`` are 1-D numpy arrays, and ``adata.obs['xy']`` is a 2-D numpy array.
    
        ``num_neighbours`` (int) a.k.a k_geom: The number of neighbours in which the edges, weights and theta graph are constructed. By default, we use k_geom = 15 as we have done for all results in our manuscript.
    
        ``nbr_weight_decay`` (str): Type of neighbourhood decay function, can be ``scaled_gaussian`` or ``reciprocal``. By default, we use ``scaled_gaussian``.
    
        ``max_m`` (int): Maximum order of azimuthal gabor filter, we use a default of 1.
    
        
    Optional Args:
        ``plt_edge`` (bool): Visualize the edge histogram*
    
        ``plt_weight`` (bool): Visualize the weights graph
    
        ``plt_agf_weights`` (bool): Visualize the AGF weights
    
        ``plt_theta`` (bool): Visualize angles around a random cell

    Returns:
        ``banksy_dict`` (dict): A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``


Functions under banksy.embed_banksy
-------
**generate_banksy_matrix**: ``generate_banksy_matrix(adata: anndata.AnnData, banksy_dict: dict, lambda_list: list, max_m: int, plot_std: bool = False, save_matrix: bool = False, save_folder: str = './data', variance_balance: bool = False, verbose: bool = True) -> Tuple[dict, np.ndarray]`` 

    Creates the banksy matrices with the set hyperparameters given. Stores the computed banksy matrices in the ``banksy_dict`` object, also returns the *last* ``banksy matrix`` that was computed
        
    Input Args:
        ``adata`` (AnnData): AnnData object containing the data matrix
        ``banksy_dict`` (Tuple[str]): A tuple containing 3 keys to access the `x`, `y` and `xy` coordinates of the cell positions under ``data.obs``. For example, ``coord_keys = ('x','y','xy')``, in which ``adata.obs['x']`` and ``adata.obs['y']`` are 1-D numpy arrays, and ``adata.obs['xy']`` is a 2-D numpy array.
    
        ``num_neighbours`` (int) a.k.a k_geom: The number of neighbours in which the edges, weights and theta graph are constructed. By default, we use k_geom = 15 as we have done for all results in our manuscript.
    
        ``nbr_weight_decay`` (str): Type of neighbourhood decay function, can be ``scaled_gaussian`` or ``reciprocal``. By default, we use ``scaled_gaussian``.
    
        ``max_m`` (int): Maximum order of azimuthal gabor filter, we use a default of 1.
    
        
    Optional Args:
        ``plt_edge`` (bool): Visualize the edge histogram*
    
        ``plt_weight`` (bool): Visualize the weights graph
    
        ``plt_agf_weights`` (bool): Visualize the AGF weights
    
        ``plt_theta`` (bool): Visualize angles around a random cell

    Returns:
        ``banksy_dict`` (dict): A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``


.. autosummary::
   :toctree: generated

   BANKSY\_py
