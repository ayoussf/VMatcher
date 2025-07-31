import dataclasses

@dataclasses.dataclass
class BackboneConfig:
    backbone_type: str = 'VGG'
    initial_dim: int = 128
    block_dims: tuple[int, ...] = (64, 128, 256)
    resolution: tuple[int, ...] = (8, 1)
    resolution_ratio: float = resolution[0] // resolution[1]
    num_blocks: tuple[int, ...] = (2, 4, 14)
    width_multiplier: tuple[float, ...] = (1, 1, 1)
    override_groups_map: type = None
    deploy: bool = False
    use_se: bool = False
    use_checkpoint: bool = False

@dataclasses.dataclass
class MambaConfig:
    model_type: str = "mamba_v"
    num_layers: int = 24
    att_ratio: float = 0.17
    mlp_ratio: float = 0.40
    cross_only: bool = False
    switch: bool = True
    mlp_type: str = "gated_mlp"
    mlp_expand: int = 2
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 3
    expand: int = 1
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    bidirectional: bool = False
    divide_output: bool = False
    fused_add_norm: bool = True
    rmsnorm: bool = True
    norm_eps: float = 1e-5
    residual_in_fp32: bool = True
    drop_path_rate: float = 0.0

@dataclasses.dataclass
class MambaAttention:
    embed_dim: int = 256
    num_heads: int = 8
    num_heads_kv:int = None
    head_dim: int = None  # If None, use embed_dim // num_heads
    mlp_dim: int = 0
    qkv_proj_bias: bool = False
    out_proj_bias: bool = False
    softmax_scale: float = None
    causal: bool = False
    layer_idx: int = None
    d_conv: int = 0
    rotary_emb_dim: int = int(embed_dim/num_heads)
    rotary_emb_base: float = 10000.0
    rotary_emb_interleaved: bool = False
    cat: bool = False
    rmsnorm: bool = True
    norm_eps: float = 1e-5
    downsample: bool = True
    down_scale: int = 4
    flash: bool = True

@dataclasses.dataclass
class Coarse_matching:
    thr: float = 0.1
    border_rm: int = 2
    dsmax_temperature: float = 0.1
    train_coarse_percent: float = 0.3
    train_pad_num_gt_min: int = 200
    coarse_type: str = 'focal'
    sparse_spvs: bool = True
    skip_softmax: bool = False
    fp16matmul: bool = False

@dataclasses.dataclass
class Fine_preprocess:
    fine_window_size: int = 8
    replace_nan: bool = True

@dataclasses.dataclass
class Fine_matching:
    local_regress_temperature: float = 10.0
    local_regress_slicedim: int = 8
    half: bool = False
    sparse_spvs: bool = True

@dataclasses.dataclass
class vmatcher_loss:
    align_corner: bool = False
    
    # Coarse loss settings
    coarse_type: str = 'focal'
    coarse_weight: float = 1.0
    coarse_sigmoid_weight: float = 1.0
    coarse_overlap_weight: bool = True
    
    # Focal loss settings
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    pos_weight: float = 1.0
    neg_weight: float = 1.0

    # Fine loss settings
    fine_type: str = 'l2'
    fine_overlap_weight: bool = True
    fine_overlap_weight2: bool = False
    fine_weight: float = 1.0
    fine_correct_thr: float = 1.0
    local_weight: float = 0.25

@dataclasses.dataclass
class dataset_settings_train:
    # Data settings
    train_base_path: str = "data/megadepth/index/megadepth_indices"
    trainval_data_source: str = 'MegaDepth'
    train_data_root: str = 'data/megadepth/train'
    train_pose_root: str = None
    train_npz_root: str = f"{train_base_path}/scene_info_0.1_0.7_no_sfm"
    train_list_path: str = f"{train_base_path}/trainvaltest_list/train_list.txt"
    train_intrinsic_path: str = None
    
    val_base_path: str = "data/megadepth/index/megadepth_indices"
    val_data_root: str = 'data/megadepth/test'
    val_pose_root: str = None
    val_npz_root: str = f"{val_base_path}/scene_info_val_1500_no_sfm"
    val_list_path: str = f"{val_base_path}/trainvaltest_list/val_list.txt"
    val_intrinsic_path: str = None
    fp16: bool = False

    # Testing settings
    test_base_path = '/assets/megadepth_1500_scene_info'
    test_data_source: str = 'MegaDepth'
    test_data_root: str = 'data/megadepth/test/megadepth_1500'
    test_pose_root: str = None
    test_npz_root: str = f'{test_base_path}'
    test_list_path: str = f'{test_base_path}/megadepth_1500.txt'
    test_intrinsic_path: str = None
    
    # General options
    min_overlap_score_train: float = 0.0
    min_overlap_score_test: float = 0.0
    augmentation_type: str = None

    # ScanNet options
    scan_img_resizex: int = 640
    scan_img_resizey: int = 480

    # MegaDepth options
    mgdpt_img_resize: int = 832
    mgdpt_img_pad: bool = True
    mgdpt_depth_pad: bool = True
    mgdpt_df: int = 8

    # Hpatches options
    ignore_scenes: bool = True

    # Sampler settings
    data_sampler: str = 'scene_balance'
    n_samples_per_subset: int = 100
    sb_subset_sample_replacement: bool = True
    sb_subset_shuffle: bool = True
    sb_repeat: int = 1

    rdm_replacement: bool = True
    rdm_num_samples: int = None

@dataclasses.dataclass
class dataset_settings_test:
    # Data settings
    train_base_path: str = None
    trainval_data_source: str = None
    train_data_root: str = None
    train_pose_root: str = None
    train_npz_root: str = None
    train_list_path: str = None
    train_intrinsic_path: str = None
    
    test_base_path: str = None
    val_data_root: str = None
    val_pose_root: str = None
    val_npz_root: str = None
    val_list_path: str = None
    val_intrinsic_path: str = None
    fp16: bool = False

    # Testing settings
    test_base_path = 'assets/megadepth_test_1500_scene_info'
    test_data_source: str = 'MegaDepth'
    test_data_root: str = 'data/megadepth/test/megadepth_test_1500'
    test_pose_root: str = None
    test_npz_root: str = f'{test_base_path}'
    test_list_path: str = f'{test_base_path}/megadepth_test_1500.txt'
    test_intrinsic_path: str = None
    
    # General options
    min_overlap_score_train: float = 0.0
    min_overlap_score_test: float = 0.0
    augmentation_type: str = None

    # ScanNet options
    scan_img_resizex: int = 640
    scan_img_resizey: int = 480

    # MegaDepth options
    mgdpt_img_resize: int = 1184
    mgdpt_img_pad: bool = False
    mgdpt_depth_pad: bool = False
    mgdpt_df: int = 8

    # Hpatches options
    ignore_scenes: bool = True

    # Sampler settings
    data_sampler: str = 'scene_balance'
    n_samples_per_subset: int = 100
    sb_subset_sample_replacement: bool = True
    sb_subset_shuffle: bool = True
    sb_repeat: int = 1

    rdm_replacement: bool = True
    rdm_num_samples: int = None

@dataclasses.dataclass
class plotting:
    enable_plotting: bool = True
    n_val_pairs_to_plot: int = 32
    plot_mode: str = 'evaluation'
    plot_matches_alpha: str = 'dynamic'

@dataclasses.dataclass
class metrics:
    epi_err_thr: float = 1e-4
    pose_geo_model: str = 'E'
    pose_estimation_method: str = 'RANSAC'
    homography_estimation_method: str = 'Poselib'
    topk: int = 1024
    ransac_pixel_thr: float = 0.5
    ransac_conf: float = 0.99999
    ransac_max_iters: int = 10000
    use_magsacpp: bool = False
    eval_times: int = 5