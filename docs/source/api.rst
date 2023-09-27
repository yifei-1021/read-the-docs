Documentation for ``banksy_py`` package
===================================

``banksy.initialize_banksy`` module
-------
**initialize_banksy**: This is the main Function that initializes the **banksy_dict** object as a dictionary
   
   ``initialize_banksy(adata: anndata.AnnData, coord_keys: Tuple[str],   num_neighbours: int = 15,   nbr_weight_decay: str = 'scaled_gaussian',   max_m: int = 1,  plt_edge_hist: bool = True, plt_nbr_weights: bool = True,  plt_agf_angles: bool = False,  plt_theta: bool = True ) -> dict:`` 
   
   
      **Input Args**:
         ``adata (AnnData)``: AnnData object containing the data matrix.
         
         ``coord_keys (Tuple[str])``: A tuple containing 3 keys to access the `x`, `y` and `xy` coordinates of the cell positions under ``data.obs``. For example, ``coord_keys = ('x','y','xy')``, in which ``adata.obs['x']`` and ``adata.obs['y']`` are 1-D numpy arrays, and ``adata.obs['xy']`` is a 2-D numpy array.
         
         ``num_neighbours (int)`` a.k.a k_geom: The number of neighbours in which the edges, weights and theta graph are constructed. By default, we use k_geom = 15 as we have done for all results in our manuscript.
         
         ``nbr_weight_decay (str)``: Type of neighbourhood decay function, can be ``scaled_gaussian`` or ``reciprocal``. By default, we use ``scaled_gaussian``.
         
         ``max_m (int)``: Maximum order of azimuthal gabor filter, we use a default of 1.
      
      
      **Optional Args**:
      ``plt_edge (bool)``: Visualize the edge histogram*
      
      ``plt_weight (bool)``: Visualize the weights graph
      
      ``plt_agf_weights (bool)``: Visualize the AGF weights
      
      ``plt_theta (bool)``: Visualize angles around a random cell
      
      **Returns**:
      ``banksy_dict (dict)``: A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``
   

``banksy.embed_banksy`` module
-------
**generate_banksy_matrix**: Creates the banksy matrices with the set hyperparameters given. Stores the computed banksy matrices in the ``banksy_dict`` object, also returns the *last* ``banksy matrix`` that was computed

   
 ``generate_banksy_matrix(adata: anndata.AnnData, banksy_dict: dict, lambda_list: list, max_m: int, plot_std: bool = False, save_matrix: bool = False, save_folder: str = './data', variance_balance: bool = False, verbose: bool = True) -> Tuple[dict, np.ndarray]`` 

    **Input Args**:
     ``adata (AnnData)``: AnnData object containing the data matrix

     ``banksy_dict (dict)``: The banksy_dict object generated from ``initialize_banksy`` function. Note that this function also returns the same ``banksy_dict`` object, it appends computed ``banksy_matrix`` for each hyperparameter under ``banksy_dict[nbr_weight_decay][lambda_param]``.
 
     ``lambda_list (List[int])``: A list of ``lambda`` parameters that the users can specify. We recommend ``lambda = 0.2`` for cell-typing and ``lambda = 0.8`` for domain segemntation. 
 
     ``max_m (int)``: The maximum order of the AGF transform. 
    
        
    **Optional Args**:
     ``plot_std (bool)``: Visualize the standard  deivation per gene in the dataset, Defaults to ``False``.

     ``save_matrix (bool)``: Option to save all ``banksy_matrix`` generated as a ``csv`` file named ``f"adata_{nbr_weight_decay}_l{lambda_param}_{time_str}.csv"``. Defaults to ``False``.

     ``save_folder (bool)``: Path to folder for saving the ``banksy_matrix``. Defaults to ``./data``.
 
     ``variance_balance (bool)``: Balance the variance between the ``gene-expression``, ``neighboorhood`` and ``AGF`` matrices. Defaults to False.
 
     ``plt_theta (bool)``: Visualize angles around a random cell

    **Returns**:
     ``banksy_dict (dict)``: A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``

     ``banksy_matrix (np.ndarray)``: The last ``banksy_matrix`` generated, useful if the use is simply running one set of parameters.

``banksy.cluster_methods`` module
-------

**run_Leiden_partition**: Main driver function that runs Leiden partition across the banksy matrices stored in banksy_dict. See the original leiden package: https://leidenalg.readthedocs.io/en/stable/intro.html

   ``run_Leiden_partition(banksy_dict: dict, resolutions: list, num_nn: int = 50, num_iterations: int = -1, partition_seed: int = 1234, match_labels: bool = True, annotations = None, max_labels: int = None,**kwargs) -> dict:`` 

    **Args**:
     ``banksy_dict (dict)``: The processing dictionary containing:

      |__ ``nbr weight decay``

         |__ ``lambda_param``

             |__ ``adata``
   
     ``resolutions``: Resolution of the partition. We recommend users to try to adjust resolutions to match the number of clusters that they need.
         
     ``num_nn (int)``: Number of nearest neighrbours for Leiden-parition. Also refered to as ``k_expr`` in our manuscript, default = 50.
   
     ``num_iterations (int)``: Number of iterations in which the paritition is conducted, default = -1:
   
     ``partition_seed (int)``: seed for partitioning (Leiden) algorithm, default = 1234.
     
     ``match_labels (bool)``: Determines if labels are kept consistent across different hyperparameter settings,  default = True.
   
     ``annotations (str)``: If manual annotations for the labels are provided under ``adata.obsm[{annotation}]". If so, we also compute the ``adjusted rand index`` for BANKSY's performance under ``results_df[param_name]['ari']`` 

    **Optional args**: other parameters to the Leiden Partition:

     ``shared_nn_max_rank (int)``, default = 3

     ``shared_nn_min_shared_nbrs (int)``, default = 5
    
    Returns:
     ``results_df (pd.DataFrame)``: A pandas dataframe containing the results of the partition.

     The results can be accessed via: 
         
         ``param_str = f"{nbr_weight_decay}_pc{pca_dim}_nc{lambda_param:0.2f}_r{resolution:0.2f}"`` # A unique id for current hyperparameters

         ``results_df[param_str] = {``

             ``"decay": nbr_weight_decay,`` - Type of weight decay function used

             ``"lambda_param": lambda_param,`` - Lambda Parameter specified

             ``"num_pcs":pca_dim,`` - number of PCs specified

             ``"resolution":resolution,`` - Resolution specified for clustering

             ``"num_labels": label.num_labels``, - Number of Labels generated by banksy

             ``"labels": label,`` - Labels generated by Banksy

             ``"adata": banksy_dict[nbr_weight_decay][lambda_param]["adata"]`` - original ``AnnData`` object
         ``}``

**run_mclust_partition**: Main driver function that runs ``mclust`` partition across the banksy matrices stored in banksy_dict. Note that we need to specify the number of clusters  ``num_labels`` for mclust, which is applicable for datasets in we know the number of clusters to look for (e.g., DLPFC).  This is based on ``STAGATE``'s implementation of the ``mclust`` package.  see https://github.com/zhanglabtools/STAGATE/blob/main/STAGATE/utils.py

 ``run_mclust_partition(banksy_dict: dict, partition_seed: int = 1234, annotations = None, num_labels: int = None, **kwargs  ) -> dict:``

     Args:
      ``banksy_dict (dict)``: The processing dictionary containing:

         |__ ``nbr weight decay``

            |__ ``lambda_param``

                |__ ``adata``

      ``partition_seed (int)``: Seed used for mclust partition
          
      ``annotations (str)``: If manual annotations for the labels are provided under ``adata.obsm[{annotation}]". If so, we also compute the ``adjusted rand index`` for BANKSY's performance under ``results_df[param_name]['ari']`` 

      ``num_labels (int)``: Number of labels required for ``mclust`` model.

     Returns:
      ``results_df (pd.DataFrame)``: A pandas dataframe containing the results of the partition

``banksy.plot_banksy`` module
-------
**plot_results**: Plot and visualize the results of Banksy, including the full-figure.
    
   ``plot_results(results_df: pd.DataFrame, weights_graph: Union[csc_matrix, csr_matrix], c_map: str,  match_labels: bool, coord_keys: Tuple[str], max_num_labels: int = 20, save_fig: bool = False, save_fullfig: bool = False, save_path: str = None, plot_dot_plot: bool = False, plot_heat_map: bool = False, n_genes: int = 5, color_list: List[str] = [], dataset_name: str = "", main_figsize: Tuple[float, float] = (15, 9),**kwargs) -> None``
   
       **Args**:
        ``results_df (pd.DataFrame)``: DataFrame containing all the results after running ``leiden`` clustering algorithm.

        ``weight_graph (csc_matrix)``: weight_graph object in a dictionary

        ``max_num_labels (int)``: Maximum number of labels

        ``match_labels (bool)``: If the match labels options was previously indicated. THe outpug figures will match the clusters generated from BANKSY using different hypeparameters.

        ``max_num_labels (int)``: Number of labels used to match labels (if ``match_labels=True``).

        ``coord_keys (Tuple(str))``: keys to access the coordinates for ``x``, ``y`` and ``xy`` accessed under ``adata.obsm``. 
   
       **Optional args**:
        ``save_fig (bool)``: Save the figure containing clusters generated by BANKSY. All figure are saved via the name ``f"{nbr_weight_decay}_pc{pca_dim}_nc{lambda_param:0.2f}_r{resolution:0.2f}".png``
        
        ``save_fullfig (bool)``: Save full figure, including UMAP and PCA plots along with clusters.

        ``c_map (str)``: Colour map used for clustering, such as ``tab20``

        ``save_all_h5ad (bool)``: to save a copy of the temporary anndata object as ``.h5ad`` format

        ``file_path (str)``: file path for saving the output figure/files. default file path is 'data'
       
       **Returns**:
        The main figure for visualization using banksy


``banksy.run_banksy`` module
-------
**run_banksy_multiparam**: Combines the (1) ``generate_banksy_matrix``, (2) ``pca_umap``, (3) ``run_cluster_partition`` and (4) ``plot_banksy`` functions to run banksy for multiple parameters (``lambda``, ``resolution`` and ``pca_dims``), and generate its figure in one step. Note the user still has to initalize the ``banksy_dict`` via ``initialize_banksy``.

   ``run_banksy_multiparam(adata: anndata.AnnData, banksy_dict: dict,lambda_list: List[int],resolutions: List[int],color_list: Union[List, str],max_m: int,filepath: str, key: Tuple[str], match_labels: bool = False, pca_dims: List[int] = [20, ], savefig: bool = True, annotation_key: str = "cluster_name", max_labels: int = None, variance_balance: bool = False, cluster_algorithm: str = 'leiden', partition_seed: int = 1234, add_nonspatial: bool = True, **kwargs) ``

     **Args**:
      ``adata (AnnData)``: AnnData object containing the data matrix
      
      ``banksy_dict (dict)``: The banksy_dict object generated from ``initialize_banksy`` function. Note that this function also returns the same ``banksy_dict`` object, it appends computed ``banksy_matrix`` for each hyperparameter under ``banksy_dict[nbr_weight_decay][lambda_param]``.
      
      ``lambda_list (List[int])``: A list of ``lambda`` parameters that the users can specify. We recommend ``lambda = 0.2`` for cell-typing and ``lambda = 0.8`` for domain segemntation.
      
      ``resolutions (List[int])``: Resolution of the partition. We recommend users to try to adjust resolutions to match the number of clusters that they need.
      
      ``color_list (Union[List, str])``: Color map or list to plot figure, e.g., ``tab20``
      
      ``max_m (int)``: The maximum order of the AGF transform. 
      
      ``key (str)`` a.k.a ``coord_keys``: A tuple containing 3 keys to access the `x`, `y` and `xy` coordinates of the cell positions under ``data.obs``. For example, ``coord_keys = ('x','y','xy')``, in which ``adata.obs['x']`` and ``adata.obs['y']`` are 1-D numpy arrays, and ``adata.obs['xy']`` is a 2-D numpy array.
      
      ``filepath (str)``: file path for saving the output figure/files. default file path is 'data'
          
      ``annotation_key (str)``: If manual annotations for the labels are provided under ``adata.obsm[{annotation}]". If so, we also compute the ``adjusted rand index`` for BANKSY's performance under ``results_df[param_name]['ari']`` 


      **Optional args**:
      ``match_labels (bool)``: Whether to match labels between runs of ``banksy`` using different hyperparameters.
      
      ``pca_dims (List of integers)``: A list of integers which the PCA will reduce to. For example, specifying `pca_dims = [10,20]` will generate two sets of reduced `pca_embeddings` which can be accessed by first retreiving the adata object: `` adata = banksy_dictbanksy_dict[{nbr_weight_decay}][{lambda_param}]["adata"]``. Then taking the pca embedding from ``pca_embeddings = adata.obsm[reduced_pc_{pca_dim}]``. Defaults to ``[20]``
      
      ``max_labels (int)``: Maximum number of labels used for ``mclust`` or ``leiden``. For ``leiden``, if ``max_label`` is set and ``resolution`` is left as an empty ``list``, it will try to search for a resolution that matches the same number of ``max_num_labels``.
      
      ``savefig (bool)``: To save the figures generated from ``banksy``, default = True
      
      ``partition_seed (int)``: Seed used for Clustering algorithm, default = 1234
      
      ``variance_balance (bool)``: Balance the variance between the ``gene-expression``, ``neighboorhood`` and ``AGF`` matrices. defaults to False.
      
      ``cluster_algorithm (str)``: Type of clustering algorithm to use: either ``leiden`` or ``mclust``. default to ``leiden``

      ``add_nonspatial (bool)``: Whether to add results for ``nonspatial`` clustering, defaults to True

     **Returns**:
      ``results_df (pd.DataFrame)``: A pandas dataframe containing the results of the partitions
   


``utils.umap_pca`` module
-------

**pca_umap**: Applies dimensionality reduction via ``PCA`` (which is used for clustering), optionally applies ``UMAP`` to cluster the groups. Note that ``UMAP`` is used for visualization.

 ``pca_umap(banksy_dict: dict,pca_dims: List[int] = [20,], plt_remaining_var: bool = True, add_umap: bool = False, **kwargs) -> Tuple[dict, np.ndarray]`` 
    
    **Args**:
     ``banksy_dict (dict)``: The processing dictionary containing info about the banksy matrices.
 
     ``pca_dims (List of integers)``: A list of integers which the PCA will reduce to. For example, specifying `pca_dims = [10,20]` will generate two sets of reduced `pca_embeddings` which can be accessed by first retreiving the adata object: `` adata = banksy_dictbanksy_dict[{nbr_weight_decay}][{lambda_param}]["adata"]``. Then taking the pca embedding from ``pca_embeddings = adata.obsm[reduced_pc_{pca_dim}]``. Defaults to ``[20]``

     ``plt_remaining_var (bool)``: generate a scree plot of remaining variance. Defaults to False.

     ``add_umap (bool)``: Whether to apply ``UMAP`` for visualization later. Note this is required for plotting the ``full-figure`` option used in ``plot_results``.

    **Returns**:       
     ``banksy_dict (dict)``: A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``

     ``banksy_matrix (np.ndarray)``: The last ``banksy_matrix`` generated, useful if the use is simply running one set of parameters.


.. autosummary::
   :toctree: generated

   BANKSY\_py
