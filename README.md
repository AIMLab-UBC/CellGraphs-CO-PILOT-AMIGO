## Run

```
python3 multi_modal_train.py --data_dir path_to_data 
```


Below are the available configs.

```
--data_dir: (default='./data', type=str) - Path to the dataset.
--cuda: (default=True, type=bool) - Use CUDA for training.
--save_dir: (default='./checkpoint', type=str) - Path to save model checkpoints.
--seed: (default=0, type=int) - Random seed for reproducibility.
--job_name: (default="", type=str) - Job name for identification.
--slurm_job_id: (default="", type=str) - Slurm job ID for cluster jobs.
--optim: (default='adam', type=str) - Optimizer name.
--lr: (default=0.01, type=float) - Learning rate.
--weight_decay: (default=0.0004, type=float) - Weight decay for regularization.
--scheduler: (default='cosine', type=str) - Learning rate scheduler type.
--iteration: (default=30000, type=int) - Number of training iterations.
--fold: (default=5, type=int) - Number of cross-validation folds.
--fold_number: (default=None, type=int) - Fold number to run (None means all folds).
--batch_size: (default=512, type=int) - Batch size for training.
--validation_interval: (default=1000, type=int) - Interval for validation.
--worker_count: (default=0, type=int) - Number of data loader workers.
--hetero_type_count: (default=None, type=int) - Number of node types for heterogeneous graph.
--cache: (default=True, type=bool) - Use dataset cache.
--arch: (default=None, type=str) - Model architecture (pathomic | simple).
--input_dim: (default=None, type=int) - Input dimension of the graph.
--neptune: (default=True, type=bool) - Monitor with Neptune.
--wandb: (default=True, type=bool) - Monitor with WandB.
--weight_sharing: (default=False, type=bool) - Use weight sharing in the model.
--multi_modal_weight_sharing: (default=False, type=bool) - Use weight sharing across modalities.
--distributed: (default=False, type=bool) - Use multi-GPU training.
--tags: (default=None, type=list) - Monitoring tags.
--dataset_unique_name: (default=None, type=str) - Dataset unique name.
--dataset_save_dir: (default=None, type=str) - Directory to save dataset.
--modalities: (default=None, type=list) - Modality names.
--combinator_weight_sharing: (default=False, type=bool) - Use weight sharing for the combinator.
--single_combinator_weight_sharing: (default=False, type=bool) - Use weight sharing for single combinator.
--combinator_type: (default='simple', type=str) - Combinator type for modality combination.
--transformer_cls_token: (default=False, type=bool) - Use cls token in transformer.
--transformer_pool: (default='mean', type=str) - Transformer pooling type.
--transformer_n_head: (default=1, type=int) - Transformer number of attention heads.
--transformer_pos: (default=False, type=bool) - Use position encoding in transformer.
--single_modal_combinator_type: (default='simple', type=str) - Combinator mode for single modal aggregation.
--subtypes: (default=None, type=list) - Subtypes.
--edge_weighting: (default=False, type=str) - Use edge information: similarity or distance.
--conv_type: (default='sage', type=str) - Type of convolution.
--distance_feature: (default=False, type=bool) - Distance features.
--batch_censor_portion: (default=None, type=float) - Portion of censor cases in a batch.
--dcl: (default=False, type=bool) - Use decouple contrastive loss.
--multi_stage_loss: (default=0, type=float) - Multi-stage loss factor.
--auto_gpu: (default=False, type=bool) - Automatically select GPUs.
--interval_loss: (default=None, type=str) - Interval loss type: chen, hierarchical, confidence, none.
--bin_count: (default=4, type=int) - Number of interval bins.
--learnable_skip_connection: (default=False, type=bool) - Use learnable_skip_connection.
--temprature: (default=False, type=bool) - Use temperature.
--node_drop: (default=None, type=float) - Probability of node drop.
--train_only_transform: (default=False, type=bool) - Use transform in train only.
--multi_mode_norm: (default=None, type=str) - Normalization for single mode: none, batch, layer, instance.
--mid_relu: (default=False, type=bool) - Enable mid relu.
--dataset_name: (default="tumorbank", type=str) - Dataset name.
--rotation_aug: (default=False, type=bool) - Enable rotation augmentation.
--position_encoding: (default=None, type=str) - Position encoding for point cloud. Options: none, concat, randla.
--preconv: (default=False, type=bool) - Enable preconv.
--dynamic_graph: (default=None, type=int) - Dynamic graph knn.
--dynamic_graph_expansion_scale: (default=0, type=float) - Dynamic graph knn expansion scale.
--expansion_pos_encoding: (default=False, type=bool) - Expansion pos encoding.
--pair_norm: (default=False, type=bool) - Enable pair normalization.
--pooling_type: (default='sag', type=str) - Pooling type.
--pooling_ratio: (default=0.2, type=float) - Pooling ratio.
--layer_sharing: (default=False, type=bool) - Layer sharing.
--gated_attention: (default='', type=str) - Enable gated attention: mean, var{n_head}.
--intra_sharing: (default=False, type=bool) - Enable intra_sharing.
--heldout: (default=False, type=bool) - Enable heldout.
--extra_layer: (default=False, type=bool) - Enable extra_layer.
--expansion_dim_factor: (default=2, type=float) - Expansion dim factor.
--plot_graph: (default=False, type=bool) - Plot graph.
--similarity_encoding: (default=False, type=bool) - Similarity encoding.
--hierarchical_attention: (default=False, type=bool) - Hierarchical attention.
--n_layers: (default=3, type=int) - Number of GNN layers.
--fully_connected: (default=False, type=bool) - Fully connected.
--nhid: (default=256, type=int) - Hidden dimension of GNN.
--graph_dim: (default=128, type=int) - Output dimension of GNN.
--multi_block: (default=0, type=int) - Shared block length.
--gradient_accumulation: (default=1, type=int) - Gradient accumulation.
--single_layer_preconv: (default=False, type=bool) - Single layer preconv.
```