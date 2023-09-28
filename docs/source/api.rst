Documentation for ``banksy_py`` package
===================================

``banksy.initialize_banksy`` module
-------

.. py:function:: initialize_banksy(adata: anndata.AnnData, coord_keys: Tuple[str],   num_neighbours: int = 15,   nbr_weight_decay: str = 'scaled_gaussian',   max_m: int = 1,  plt_edge_hist: bool = True, plt_nbr_weights: bool = True,  plt_agf_angles: bool = False,  plt_theta: bool = True ) -> dict: 

   Initializes the ``banksy_dict`` dictionary containing the weights graphs.

   :param AnnData adata: AnnData object containing the data matrix.
   :param Tuple[str] coord_keys: A tuple containing 3 keys to access the `x`, `y` and `xy` coordinates of the cell positions under ``data.obs``. For example, ``coord_keys = ('x','y','xy')``, in which ``adata.obs['x']`` and ``adata.obs['y']`` are 1-D numpy arrays, and ``adata.obs['xy']`` is a 2-D numpy array.
   :param int num_neighbours: (a.k.a ``k_geom``) The number of neighbours in which the edges, weights and theta graph are constructed. By default, we use ``k_geom = 15``.
   :param str nbr_weight_decay: Type of neighbourhood decay function, can be ``scaled_gaussian`` or ``reciprocal``. By default, we use ``scaled_gaussian``.
   :param int max_m: Maximum order of azimuthal gabor filter, we use a default of ``max_m = 1``.

   :param bool plt_edge: Visualize the edge histogram, defaults to ``True``.
   :param bool plt_weight: Visualize the weights graph, defaults to ``True``.
   :param bool plt_agf_weights: Visualize the AGF weights, defaults to ``False``.
   :param bool plt_theta: Visualize angles around a random cell, defaults to ``True``.

   :return: ``banksy_dict``: A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``.
   :rtype: ``dict``
   

``banksy.embed_banksy`` module
-------
.. py:function:: generate_banksy_matrix(adata: anndata.AnnData, banksy_dict: dict, lambda_list: List[float], max_m: int, plot_std: bool = False, save_matrix: bool = False, save_folder: str = './data', variance_balance: bool = False, verbose: bool = True) -> Tuple[dict, np.ndarray]:

   Creates the banksy matrices with the set hyperparameters given. Stores the computed banksy matrices in the ``banksy_dict`` object, also returns the *last* ``banksy matrix`` that was computed.

   :param AnnData adata: AnnData object containing the data matrix.
   :param dict banksy_dict: The banksy_dict object generated from ``initialize_banksy`` function. Note that this function also returns the same ``banksy_dict`` object, it appends computed ``banksy_matrix`` for each hyperparameter under ``banksy_dict[nbr_weight_decay][lambda_param]``.
   :param List[float] lambda_list: A list of ``lambda`` parameters that the users can specify, e.g., ``lambda_list = [0.2, 0.8]``. We recommend ``lambda_list = [0.2]`` for cell-typing and ``lambda_list = [0.8]`` for domain segemntation. 
   :param int num_neighbours: (a.k.a k_geom) The number of neighbours in which the edges, weights and theta graph are constructed. By default, we use ``k_geom = 15``.
   :param str nbr_weight_decay: Type of neighbourhood decay function, can be ``scaled_gaussian`` or ``reciprocal``. By default, we use ``scaled_gaussian``.
   :param int max_m: Maximum order of azimuthal gabor filter, we use a default of ``max_m = 1``.

   :param bool plot_std: Visualize the standard  deivation per gene in the dataset, Defaults to ``False``.
   :param bool save_matrix: Option to save all ``banksy_matrix`` generated as a ``csv`` file named ``f"adata_{nbr_weight_decay}_l{lambda_param}_{time_str}.csv"``. Defaults to ``False``.
   :param bool save_folder: Path to folder for saving the ``banksy_matrix``. Defaults to ``./data``.
   :param bool variance_balance: (Not used in paper) Balance the variance between the ``gene-expression``, ``neighboorhood`` and ``AGF`` matrices. Defaults to ``False``.

   :return: ``banksy_dict``: Updated dictionary object containing computed ``banksy_matrix`` for each hyperparameter under ``banksy_dict[nbr_weight_decay][lambda_param]``.
   :return: ``banksy_matrix``: The last ``banksy_matrix`` generated, useful if the use is simply running one set of parameters.
   :rtype: Tuple(``dict``, ``np.ndarray``)

.. note::

   The ``banksy_dict`` is now organized as such:

   ``banksy_dict (dict)``: contains all processed results from ``banksy``
         
         |__ ``nbr weight decay``: type of decay function used ``scaled_gaussian`` by default.
         
            |__ ``lambda_param``: ``lambda_param``used in clustering
         
                |__ (``banksy_matrix`` | ``adata``) : ``banksy_matrix`` generated from the current parameters.
      

``banksy.cluster_methods`` module
-------

.. py:function:: run_Leiden_partition(banksy_dict: dict, resolutions: list, num_nn: int = 50, num_iterations: int = -1, partition_seed: int = 1234, match_labels: bool = True, annotations = None, max_labels: int = None,**kwargs) -> dict:

   Main driver function that runs Leiden partition across the banksy matrices stored in ``banksy_dict``. We use the original implementation from the ``leiden`` package: https://leidenalg.readthedocs.io/en/stable/intro.html
 
   :param dict banksy_dict: The banksy_dict object containing the ``banksy_matrices`` generated from ``embed_banksy`` function. 
   :param Union[List[float], None] resolutions: A list of ``resolution`` parameters that is used for leiden clustering, e.g., ``resolution = [0.2, 0.8]``.  We recommend users to try to adjust resolutions to match the number of clusters that they need. An iterative search for the ``resolution`` that matches the number of ``max_labels`` is conducted if the user set ``resolution = []`` and ``max_labels`` to their desired cluster number. 
   :param int num_nn: (a.k.a ``k_expr``)  Number of nearest neighrbours for Leiden-parition. Also refered to as ``k_expr`` in our manuscript, default = 50.
   :param int num_iterations:  Number of iterations in which the paritition is conducted, default = -1

   :param int partition_seed: Numerical seed for partitioning (Leiden) algorithm, default = ``1234``.
   :param bool match_labels: Determines if labels annotated to each cluster are matched across different hyperparameter settings,  default = ``True``.
   :param Optional[str, None] annotations:  Key (``str``) to access manual annotations if provided provided under ``adata.obsm[{annotation}]``, otherwise set ``annotations = None``. If so, we also compute the ``adjusted rand index`` for BANKSY's performance under ``results_df[param_name]['ari']`` 
   :param Optional[int] shared_nn_max_rank: An optional argument for  ``leiden-alg``, defaults to 3.
   :param Optional[int] shared_nn_min_shared_nbrs: An optional argument for  ``leiden-alg``, defaults to 5.

   :param Optional[int, None] max_labels: Maximum number of cluster labels to be identified. E.g., setting ``resolution = []`` and ``max_label = 5`` will searches for the resolution that yields 5 clusters. Defaults to ``None``.

   :return: ``results_df``: A pandas dataframe containing the results of the partition
   :rtype: ``pd.DataFrame``

.. note::

   Using ``run_Leiden_partition``, the results in ``results_df`` can be accessed via: 
      
      ``param_str = f"{nbr_weight_decay}_pc{pca_dim}_nc{lambda_param:0.2f}_r{resolution:0.2f}"`` # A unique id for current hyperparameters

      ``results_df[param_str] = {``

            ``"decay": nbr_weight_decay,`` - Type of weight decay function used

            ``"lambda_param": lambda_param,`` - Lambda Parameter specified

            ``"num_pcs": pca_dim,`` - number of PCs specified

            ``"resolution": resolution,`` - Resolution specified for clustering

            ``"num_labels": label.num_labels``, - Number of Labels generated by banksy

            ``"labels": label,`` - Labels generated by Banksy

            ``"adata": banksy_dict[nbr_weight_decay][lambda_param]["adata"]`` - original ``AnnData`` object
      ``}``


.. py:function:: run_mclust_partition(banksy_dict: dict, partition_seed: int = 1234, annotations = None, num_labels: int = None, **kwargs  ) -> dict:

   Main driver function that runs ``mclust`` partition across the banksy matrices stored in banksy_dict. Note that we need to specify the number of clusters  ``num_labels`` for mclust, which is applicable for datasets in we know the number of clusters to look for (e.g., DLPFC).  This is based on ``STAGATE``'s implementation of the ``mclust`` package.  see https://github.com/zhanglabtools/STAGATE/blob/main/STAGATE/utils.py
 
   :param dict banksy_dict: The banksy_dict object containing the ``banksy_matrices`` generated from ``embed_banksy`` function. 

   :param int partition_seed: Numerical seed for for ``mclust`` partition, default = ``1234``.
   :param Optional[str, None] annotations:  Key (``str``) to access manual annotations if provided provided under ``adata.obsm[{annotation}]``, otherwise set ``annotations = None``. If so, we also compute the ``adjusted rand index`` for BANKSY's performance under ``results_df[param_name]['ari']`` 
   :param int num_labels:  Number of labels required for ``mclust`` model.

   :return: ``results_df``: A pandas dataframe containing the results of the partition
   :rtype: ``pd.DataFrame``

.. note::

   Using ``run_mclust_partition``, the results in ``results_df`` can be accessed via: 
   
   
      ``param_str = f"{nbr_weight_decay}_pc{pca_dim}_nc{lambda_param:0.2f}_labels{num_labels:0.2f}_mclust"`` # A unique id for current hyperparameters

      ``results_df[param_str] = {``

            ``"decay": nbr_weight_decay,`` - Type of weight decay function used

            ``"lambda_param": lambda_param,`` - Lambda Parameter specified

            ``"num_labels": label.num_labels``, - Number of Labels specified by users

            ``"labels": label,`` - Labels generated by Banksy

            ``"adata": banksy_dict[nbr_weight_decay][lambda_param]["adata"]`` - original ``AnnData`` object
      ``}``


``banksy.plot_banksy`` module
-------
.. py:function:: plot_results(results_df: pd.DataFrame, weights_graph: Union[csc_matrix, csr_matrix], c_map: str,  match_labels: bool, coord_keys: Tuple[str], max_num_labels: int = 20, save_fig: bool = False, save_fullfig: bool = False, save_path: str = None, plot_dot_plot: bool = False, plot_heat_map: bool = False, n_genes: int = 5, main_figsize: Tuple[float, float] = (15, 9),**kwargs) -> None

   Plot and visualize the results of Banksy, including the full-figure.
 
   :param pd.DataFrame results_df: DataFrame containing all the results after running the clustering algorithm.

   :param Union[csc_matrix, csr_matrix] weight_graph: ``weight_graph`` generated from ``initalize_banksy``.
   :param str c_map: Color map for plotting figure if required. We recommend ``tab20``.
   :param bool match_labels:  If the match labels options was previously indicated. THe output figures will match the clusters generated from BANKSY using different hypeparameters.
   :param str  coord_keys (Tuple[str]): keys to access the coordinates for ``x``, ``y`` and ``xy`` accessed under ``adata.obsm`` 


   :param int max_num_labels: umber of labels used to match labels (if ``match_labels=True``). Defaults to 20.
   :param bool save_fig: Whether to save the ``figure`` containing (only) spatial clusters generated by BANKSY. All figure are saved via the name ``f"{nbr_weight_decay}_pc{pca_dim}_nc{lambda_param:0.2f}_r{resolution:0.2f}".png``.
   :param bool save_fullfig: Save ``full figure``, including spatial clusters, UMAP and PCA plots along with clusters. Note, if ``True`` then requires ``UMAP`` embeddings that can be obtained by ``add_umap = True `` under ``umap_pca`` function.
   
   :param Optional[str, None] annotations:  Key (``str``) to access manual annotations if provided provided under ``adata.obsm[{annotation}]``, otherwise set ``annotations = None``. If so, we also compute the ``adjusted rand index`` for BANKSY's performance under ``results_df[param_name]['ari']`` 
   :param str save_path: file path for saving the output figure/files. default file path is ``'./data'``
   :param Optional[bool] plot_dot_plot: Plot dot plot for genes expressions. default = ``False``.
   :param Optional[bool] plot_heat_map: Plot heatmap plot for genes expressions. default = ``False``.
   :param Optional[int, None] n_genes: Number of genes used to generate ``heat_map``, default = 5.

   :return: None


``banksy.run_banksy`` module
-------

.. py:function:: run_banksy_multiparam(adata: anndata.AnnData, banksy_dict: dict, lambda_list: List[int], resolutions: List[int], color_list: Union[List, str], max_m: int, filepath: str, key: Tuple[str], match_labels: bool = False, pca_dims: List[int] = [20, ], savefig: bool = True, annotation_key: str = "cluster_name", max_labels: int = None, variance_balance: bool = False, cluster_algorithm: str = 'leiden', partition_seed: int = 1234, add_nonspatial: bool = True, **kwargs) -> None

   Combines the (1) ``generate_banksy_matrix``, (2) ``pca_umap``, (3) ``run_cluster_partition`` and (4) ``plot_banksy`` functions to run banksy for multiple parameters (``lambda``, ``resolution`` and ``pca_dims``), and generate its figure in one step. Note the user still has to initalize the ``banksy_dict`` via ``initialize_banksy``.
 
   :param AnnData adata: AnnData object containing the data matrix
   :param dict banksy_dict: The banksy_dict object generated from ``initialize_banksy`` function. Note that this function also returns the same ``banksy_dict`` object, it appends computed ``banksy_matrix`` for each hyperparameter under ``banksy_dict[nbr_weight_decay][lambda_param]``.
   :param List[float] lambda_list: A list of ``lambda`` parameters that the users can specify, e.g., ``lambda_list = [0.2, 0.8]``. We recommend ``lambda_list = [0.2]`` for cell-typing and ``lambda_list = [0.8]`` for domain segemntation. 
   :param List[float] resolutions: Resolution used for ``leiden`` clustering. We recommend users to try to adjust resolutions to match the number of clusters that they need. 
   :param Union[List, str] color_list: Color map or list to plot figure, e.g., ``tab20``
   :param int max_m: Maximum order of azimuthal gabor filter, we use a default of ``max_m = 1``.
   :param str filepath: file path for saving the output figure/files. default file path is ``'./data'``
   :param str keys (Tuple[str]): a.k.a ``coord_keys``: A tuple containing 3 keys to access the `x`, `y` and `xy` coordinates of the cell positions under ``data.obs``. For example, ``coord_keys = ('x','y','xy')``, in which ``adata.obs['x']`` and ``adata.obs['y']`` are 1-D numpy arrays, and ``adata.obs['xy']`` is a 2-D numpy array.
   :param bool match_labels:  If the match labels options was previously indicated. The output figures will match the clusters generated from BANKSY using different hypeparameters.
   :param List[int] pca_dims: A list of integers which the PCA will reduce to. For example, specifying `pca_dims = [10,20]` will generate two sets of reduced `pca_embeddings` which can be accessed by first retreiving the adata object: `` adata = banksy_dictbanksy_dict[{nbr_weight_decay}][{lambda_param}]["adata"]``. Then taking the pca embedding from ``pca_embeddings = adata.obsm[reduced_pc_{pca_dim}]``. Defaults to ``[20]``
   :param bool savefig: Whether to save the ``figure`` containing (only) spatial clusters generated by BANKSY. All figure are saved via the name ``f"{nbr_weight_decay}_pc{pca_dim}_nc{lambda_param:0.2f}_r{resolution:0.2f}".png``.
   :param Optional[str, None] annotation_key:  If manual annotations for the labels are provided under ``adata.obsm[{annotation}]``. If so, we also compute the ``adjusted rand index`` for each ``param`` under ``results_df[param_name]['ari']`` 
   :param Optional[int] max_labels: Maximum number of labels used for ``mclust`` or ``leiden``. For ``leiden``, if ``max_label`` is set and ``resolution`` is left as an empty ``list``, it will try to search for a resolution that matches the same number of ``max_num_labels``.
   :param Optional[bool] variance_balance: (not used in manuscript) Balance the variance between the ``gene-expression``, ``neighboorhood`` and ``AGF`` matrices. defaults to ``False``.
   :param Optional[bool] add_nonspatial: Whether to add results for ``nonspatial`` clustering, defaults to ``True``
   :param Optional[int] partition_seed:  Seed used for Clustering algorithm, default = ``1234``.
   
   :return: ``results_df``: Pandas dataframe containing the results of the from running ``banksy`` using various parameters.
   :rtype: ``pd.DataFrame``


``utils.umap_pca`` module
-------

.. py:function:: pca_umap(banksy_dict: dict,pca_dims: List[int] = [20,], plt_remaining_var: bool = True, add_umap: bool = False, **kwargs) -> Tuple[dict, np.ndarray]
   
   Applies dimensionality reduction via ``PCA`` (which is used for clustering), optionally applies ``UMAP`` to cluster the groups. Note that ``UMAP`` is used for visualization.

   :param dict banksy_dict: The processing dictionary containing info about the banksy matrices.
   :param List[int] pca_dims: A list of integers which the PCA will reduce to. For example, specifying `pca_dims = [10,20]` will generate two sets of reduced `pca_embeddings` which can be accessed by first retreiving the adata object: ``adata = banksy_dict[{nbr_weight_decay}][{lambda_param}]["adata"]``. Then taking the pca embedding from ``pca_embeddings = adata.obsm[reduced_pc_{pca_dim}]``. Defaults to ``[20]``
   :param bool plt_remaining_var: Generate a scree plot of remaining variance. Defaults to ``False``.
   :param bool add_umap: Whether to apply ``UMAP`` for visualization later. Note this is required for plotting the ``full-figure`` option used in ``plot_results``.

   :return ``banksy_dict``:  A dictionary object containing the graph of weights obtained from the neigbhourhood weight decay function. The graph data can be accessed via ``banksy['weights']``
   :return ``banksy_matrix``: The last ``banksy_matrix`` generated, useful if the use is simply running one set of parameters.
   :rtype: ``Tuple[dict, np.ndarray]``
      


``utils.refine_clusters`` module
-------

.. py:function:: refine_clusters(adata: anndata.AnnData, results_df: pd.DataFrame, coord_keys: tuple, color_list: list = spagcn_color, savefig: bool = False, output_folder: str = "",  refine_method: str = "once", refine_iterations: int = 1, annotation_key: str = "manual_annotations", num_neigh: int = 6, verbose: bool = False) -> pd.DataFrame:
   
   Function to refine (a.k.a ``label smoothening``) predicted labels based on nearest neighbours based on ``SpaGCN``'s implementation of this ``label smoothening`` procedure: https://github.com/jianhuupenn/SpaGCN

   :param AnnData adata:  Original anndata object
   :param Tuple[str] coord_keys: A tuple containing 3 keys to access the `x`, `y` and `xy` coordinates of the cell positions under ``data.obs``. For example, ``coord_keys = ('x','y','xy')``, in which ``adata.obs['x']`` and ``adata.obs['y']`` are 1-D numpy arrays, and ``adata.obs['xy']`` is a 2-D numpy array.
   :param pd.DataFrame results_df: ``DataFrame`` object containing the results from ``run_banksy``.
   :param list color_list: List in which colors are used to plot the figures. default = ``spagcn``, which uses ``SpaGCN``'s color palette to generate cluster images.
   :param bool savefig: Whether to save images (containing refined clusters) generated by refinement procedure, default = ``False``.
   :param str output_folder: Path to folder in which figures are saved.
   :param Optional[str] refine_method: Options - ``("auto" | "once" | "iter_num" )``. To refine clusters once only or iteratively refine multiple times. If ``auto`` is specified, the refinement procedure completes iteratively until only 0.5% of the nodes are changed. If ``iter_num`` is specified, specify the 'refine_iterations' parameter. default = ``once``. 
   :param Optional[int] refine_iterations: Number of iterations to ``refine`` if ``refine_method =  iter_num``. default = 1 (same as setting ``refine_method = "once"``)
   :param Optional[str] annotation_key: The key in which the ground truth annotations are accessed under ``adata.obs[annotation_key]``. If so, the ``ari`` of the refined clusters are also calculated. If no ground truth is present, then set ``annotation_key = None``.
   :param Optional[int] num_neigh: Number of nearest-neighbours in which refinement is conducted over. By default, we use ``num_neigh = 6`` same as ``SpaGCN``'s implementation.
   :param Optional[bool] verbose: Whether to print steps conducted during each iteration process.
   :return ``results_df``: DataFrame Object containing the results.
   :rtype: ``pd.DataFrame``

.. autosummary::
   :toctree: generated

   BANKSY\_py
