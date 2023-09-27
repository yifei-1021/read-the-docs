Useful functions 
===================================


Functions under **banksy.initialize_banksy**
-------
**initialize_banksy**: ``initialize_banksy(adata: anndata.AnnData, coord_keys: Tuple[str], num_neighbours: int = 15, nbr_weight_decay: str = 'scaled_gaussian', max_m: int = 1,plt_edge_hist: bool = True, plt_nbr_weights: bool = True,plt_agf_angles: bool = False, plt_theta: bool = True ) -> dict:`` 

    This is the main Function that initializes the BANKSY Object as a dictionary
        
    **Input Args**:
        ``adata`` (AnnData): AnnData object containing the data matrix.

        ``coord_keys`` (Tuple[str]): A tuple containing 3 keys to access the `x`, `y` and `xy` coordinates of the cell positions under ``data.obs``. For example, ``coord_keys = ('x','y','xy')``, in which ``adata.obs['x']`` and ``adata.obs['y']`` are 1-D numpy arrays, and ``adata.obs['xy']`` is a 2-D numpy array.
    
        ``num_neighbours`` (int) a.k.a k_geom: The number of neighbours in which the edges, weights and theta graph are constructed. By default, we use k_geom = 15 as we have done for all results in our manuscript.
    
        ``nbr_weight_decay`` (str): Type of neighbourhood decay function, can be ``scaled_gaussian`` or ``reciprocal``. By default, we use ``scaled_gaussian``.
    
        ``max_m`` (int): Maximum order of azimuthal gabor filter, we use a default of 1.
    
        
    **Optional Args**:
        ``plt_edge`` (bool): Visualize the edge histogram*
    
        ``plt_weight`` (bool): Visualize the weights graph
    
        ``plt_agf_weights`` (bool): Visualize the AGF weights
    
        ``plt_theta`` (bool): Visualize angles around a random cell

    **Returns**:
        ``banksy_dict`` (dict): A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``


Functions under **banksy.embed_banksy**
-------
**generate_banksy_matrix**: ``generate_banksy_matrix(adata: anndata.AnnData, banksy_dict: dict, lambda_list: list, max_m: int, plot_std: bool = False, save_matrix: bool = False, save_folder: str = './data', variance_balance: bool = False, verbose: bool = True) -> Tuple[dict, np.ndarray]`` 

    Creates the banksy matrices with the set hyperparameters given. Stores the computed banksy matrices in the ``banksy_dict`` object, also returns the *last* ``banksy matrix`` that was computed
        
    **Input Args**:
        ``adata`` (AnnData): AnnData object containing the data matrix

        ``banksy_dict`` (dict): The banksy_dict object generated from ``initialize_banksy`` function. Note that this function also returns the same ``banksy_dict`` object, it appends computed ``banksy_matrix`` for each hyperparameter under ``banksy_dict[nbr_weight_decay][lambda_param]``.
    
        ``lambda_list`` (List[int]): A list of ``lambda`` parameters that the users can specify. We recommend ``lambda = 0.2`` for cell-typing and ``lambda = 0.8`` for domain segemntation. 
    
        ``max_m`` (int): The maximum order of the AGF transform. 
    
        
    **Optional Args**:
        ``plot_std`` (bool): Visualize the standard  deivation per gene in the dataset, Defaults to ``False``.

        ``save_matrix`` (bool): Option to save all ``banksy_matrix`` generated as a ``csv`` file named ``f"adata_{nbr_weight_decay}_l{lambda_param}_{time_str}.csv"``. Defaults to ``False``.

        ``save_folder`` (bool): Path to folder for saving the ``banksy_matrix``. Defaults to ``./data``.
    
        ``variance_balance`` (bool): Balance the variance between the ``gene-expression``, ``neighboorhood`` and ``AGF`` matrices. Defaults to False.
    
        ``plt_theta`` (bool): Visualize angles around a random cell

    **Returns**:
        ``banksy_dict`` (dict): A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``

        ``banksy_matrix`` (dict): The last ``banksy_matrix`` generated, useful if the use is simply running one set of parameters.

Functions under **utils.umap_pca**
-------

**pca_umap**: ``pca_umap(banksy_dict: dict,pca_dims: List[int] = [20,], plt_remaining_var: bool = True, add_umap: bool = False, **kwargs) -> Tuple[dict, np.ndarray]`` 

    Applies dimensionality reduction via PCA (which is used for clustering), optionally applies UMAP to cluster the groups. Note that UMAP is used for visualization.

    **Args**:
        ``banksy_dict`` (dict): The processing dictionary containing info about the banksy matrices.
    
        pca_dims (List of integers): A list of integers which the PCA will reduce to. For example, specifying `pca_dims = [10,20]` will generate two sets of reduced `pca_embeddings` which can be accessed by first retreiving the adata object: `` adata = banksy_dictbanksy_dict[{nbr_weight_decay}][{lambda_param}]["adata"]``. Then taking the pca embedding from ``pca_embeddings = adata.obsm[reduced_pc_{pca_dim}]``. Defaults to ``[20]``

        ``plt_remaining_var`` (bool): generate a scree plot of remaining variance. Defaults to False.

        ``add_umap`` (bool): Whether to apply ``UMAP`` for visualization later. Note this is required for plotting the ``full-figure`` option used in ``plot_results``.

    **Returns**:       
        ``banksy_dict`` (dict): A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``

        ``banksy_matrix`` (dict): The last ``banksy_matrix`` generated, useful if the use is simply running one set of parameters.

Functions under **banksy.cluster_methods**
-------

**run_Leiden_partition**: ``run_Leiden_partition(banksy_dict: dict, resolutions: list, num_nn: int = 50, num_iterations: int = -1, partition_seed: int = 1234, match_labels: bool = True, annotations = None, max_labels: int = None,**kwargs) -> dict:`` 

    Main driver function that runs Leiden partition across the banksy matrices stored in banksy_dict. See the original leiden package: https://leidenalg.readthedocs.io/en/stable/intro.html

    **Arg**s:
        banksy_dict (dict): The processing dictionary containing:
        |__ nbr weight decay
          |__ lambda_param
            |__ anndata  

        ``resolutions``: Resolution of the partition. We recommend users to try to adjust resolutions to match the number of clusters that they need.
            
        ``num_nn (int)``: Number of nearest neighrbours for Leiden-parition. Also refered to as ``k_expr`` in our manuscript, default = 50.

        ``num_iterations (int)``: Number of iterations in which the paritition is conducted, default = -1:

        ``partition_seed (int)``: seed for partitioning (Leiden) algorithm, default = 1234.
        
        ``match_labels (bool)``: Determines if labels are kept consistent across different hyperparameter settings,  default = True.

        ``annotations``: If manual annotations for the labels are provided under ``adata.obsm[{annotation}]". If so, we also compute the ``adjusted rand index`` for BANKSY's performance under ``results_df[param_name]['ari']`` 

    Optional args (kwargs):
        Other parameters to the Leiden Partition:

        shared_nn_max_rank (int), default = 3

        shared_nn_min_shared_nbrs (int), default = 5
    
    Returns:
        results_df (pd.DataFrame): A pandas dataframe containing the results of the partition.

        The results can be accessed via: 
            ``
            param_str = f"{nbr_weight_decay}_pc{pca_dim}_nc{lambda_param:0.2f}_r{resolution:0.2f}" # A unique id for current hyperparameters
            results_df[param_str] = {
                "decay": nbr_weight_decay, ### Type of weight decay function used
                "lambda_param": lambda_param, ### 
                "num_pcs":pca_dim,
                "resolution":resolution,
                "num_labels": label.num_labels,
                "labels": label,
                "adata": banksy_dict[nbr_weight_decay][lambda_param]["adata"]
            }``


.. autosummary::
   :toctree: generated

   BANKSY\_py
