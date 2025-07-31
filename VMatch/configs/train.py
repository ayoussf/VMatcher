from .conf_blocks import *
from VMatch.src.utils.misc import setup_gpus
import math

@dataclasses.dataclass
class train_settings:
    
    # Data settings
    batch_size: int = 1
    accumulate_grad_batches: int = 8
    acc_batch_size = batch_size * accumulate_grad_batches
    canonical_bs: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    parallel_load_data: bool = False

    # Trainer settings
    gpus: int = setup_gpus(1)
    num_nodes: int = 1
    world_size: int = gpus * num_nodes
    true_batch_size: int = world_size * acc_batch_size
    scaling: float = true_batch_size / canonical_bs
    canonical_lr: float = 6.4e-3
    true_lr: float = canonical_lr * scaling
    find_lr: bool = False
    profiler: str = 'inference'
    max_epochs: int = 30
    check_val_every_n_epoch: int = 1
    benchmark: bool = True
    num_sanity_val_steps: int = 0
    limit_val_batches: int = 1.
    flush_logs_every_n_steps: int = 1000
    log_every_n_steps: int = 50

    # Optimizer settings
    optimizer: str = 'adamw'
    adamw_decay: float = 0.1
    adam_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    gradient_clipping: float = 0.0
    fp16_optimizer: bool = False

    # Warmup settings
    warmup_type: str = 'linear'
    warmup_ratio: float = 0.1
    warmup_step: int = math.floor(1875/scaling)

    # Scheduler settings
    scheduler: str = 'MultiStepLR'
    scheduler_interval: str = 'epoch'
    mslr_milestones: tuple = (8, 12, 16, 20, 24)
    mslr_gamma: float = 0.5
    cosa_tmax: int = 30
    elr_gamma: float = 0.999992

    # seed settings
    seed: int = 66
        
    # Logging settings
    exper_name: str = 'VMatcher'

    # Checkpoint settings
    disable_ckpt: bool = False

@dataclasses.dataclass(frozen=True)
class train_config:
    backbone: BackboneConfig
    mamba_config: MambaConfig
    atten_config: MambaAttention
    match_coarse: Coarse_matching
    fine_preprocess: Fine_preprocess
    match_fine: Fine_matching
    train_settings: train_settings
    vmatch_loss: vmatcher_loss
    dataset_settings: dataset_settings_train
    plotting: plotting
    metrics: metrics
    half: bool = False
    mp: bool = False
    deter: bool = True