Useful functions 
--------

Under the main ``banksy`` module
--------

``banksy.initialize_banksy``

``banksy.embed_banksy`` banksy_dict = initialize_banksy(adata,
                                coord_keys,
                                k_geom,
                                nbr_weight_decay=nbr_weight_decay,
                                max_m=max_m,
                                plt_edge_hist=False,
                                plt_nbr_weights=True,
                                plt_agf_angles=False,
                                plt_theta=False
                                )


``generate_banksy_matrix`` banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                    banksy_dict,
                                                    lambda_list,
                                                    max_m)

.. autosummary::
   :toctree: generated

   BANKSY\_py
