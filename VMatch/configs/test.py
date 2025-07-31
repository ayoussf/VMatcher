import dataclasses
from .conf_blocks import *
from VMatch.src.utils.misc import setup_gpus

@dataclasses.dataclass
class test_settings:
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    parallel_load_data: bool = False
    gpus: int = setup_gpus(1)
    num_nodes: int = 1
    world_size: int = gpus * num_nodes
    seed: int = 66

@dataclasses.dataclass(frozen=False)
class test_config:
    backbone: BackboneConfig
    mamba_config: MambaConfig
    atten_config: MambaAttention
    match_coarse: Coarse_matching
    fine_preprocess: Fine_preprocess
    match_fine: Fine_matching
    metrics: metrics
    dataset_settings: dataset_settings_test
    plotting: plotting
    test_settings: test_settings
    mp: bool = True
    half: bool = False
    fp32: bool = False
    deter: bool = True
    profiler: str = 'inference'
    ckpt_path: str = ''
    dump_dir: str = ''