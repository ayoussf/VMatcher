import torch
from pytorch_lightning.plugins import DDPPlugin
import tyro
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from loguru import logger as loguru_logger
from pathlib import Path
from VMatch.configs import main_config
from VMatch.src.utils.profiler import build_profiler
from VMatch.src.lightning.VMatcher_lightning import PL_VMatcher
from VMatch.src.lightning.data import MultiSceneDataModule
from VMatch.src.utils.misc import get_rank_zero_only_logger

loguru_logger = get_rank_zero_only_logger(loguru_logger)

@tyro.conf.configure(tyro.conf.FlagConversionOff)
class main():
    """main class, script backbone.
    Args:
        config_path: Path to configuration.
        task: The task to be performed.
    """
    def __init__(self, 
                 task:str,
                 config: main_config):
                
        if task == 'train':
            
            config = config.train

            if config.deter:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            pl.seed_everything(config.train_settings.seed)
            
            profiler = build_profiler(config.train_settings.profiler)
            
            model = PL_VMatcher(config, profiler=profiler)
            
            loguru_logger.info(f"VMatcher LightningModule initialized!")
            
            data_module = MultiSceneDataModule(config)
            
            loguru_logger.info(f"DataModule initialized!")
            
            logger = TensorBoardLogger(save_dir='logs/tb_logs', name=config.train_settings.exper_name, default_hp_metric=False)
            
            ckpt_dir = Path(logger.log_dir) / 'checkpoints'
            
            ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=5, mode='max',
                                            save_last=True,
                                            dirpath=str(ckpt_dir),
                                            filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
            
            lr_monitor = LearningRateMonitor(logging_interval='step')
            
            callbacks = [lr_monitor]
            
            if not config.train_settings.disable_ckpt:
                callbacks.append(ckpt_callback)
            
            trainer = pl.Trainer(plugins=[DDPPlugin(find_unused_parameters=False,
                                                    num_nodes=config.train_settings.num_nodes,
                                                    sync_batchnorm=config.train_settings.world_size > 0)],
                                gradient_clip_val=config.train_settings.gradient_clipping,
                                callbacks=callbacks,
                                logger=logger,
                                sync_batchnorm=config.train_settings.world_size > 0,
                                replace_sampler_ddp=False,
                                reload_dataloaders_every_epoch=False,
                                weights_summary='full',
                                profiler=profiler,
                                gpus=config.train_settings.gpus,
                                accelerator='ddp',
                                num_nodes=config.train_settings.num_nodes,
                                check_val_every_n_epoch=config.train_settings.check_val_every_n_epoch,
                                log_every_n_steps=config.train_settings.log_every_n_steps,
                                flush_logs_every_n_steps=config.train_settings.flush_logs_every_n_steps,
                                limit_val_batches=config.train_settings.limit_val_batches,
                                num_sanity_val_steps=config.train_settings.num_sanity_val_steps,
                                benchmark=config.train_settings.benchmark,
                                max_epochs=config.train_settings.max_epochs,
                                accumulate_grad_batches=config.train_settings.accumulate_grad_batches,
                                )
            loguru_logger.info(f"Trainer initialized!")
            
            loguru_logger.info(f"Start training!")
            trainer.fit(model, datamodule=data_module)
            loguru_logger.info(f"Training finished!")
        
        elif task == 'test':
            
            config = config.test

            if config.deter:
                torch.backends.cudnn.deterministic = True
            
            pl.seed_everything(config.test_settings.seed)

            if config.fp32:
                config.mp = False
                config.half = False
                config.dataset_settings.fp16 = False
                config.mamba_config.residual_in_fp32 = True
            
            if config.half:
                config.half = True
                config.dataset_settings.fp16 = True
            else:
                config.half = False
                config.dataset_settings.fp16 = False
            
            loguru_logger.info(f"Config initialized!")

            profiler = build_profiler(config.profiler)
            model = PL_VMatcher(config, pretrained_ckpt=config.ckpt_path, profiler=profiler, dump_dir=config.dump_dir)
            loguru_logger.info(f"VMatcher_lightning initialized!")
            
            data_module = MultiSceneDataModule(config)
            loguru_logger.info(f"DataModule initialized!")

            trainer = pl.Trainer(replace_sampler_ddp=False,
                                 logger=False,
                                 gpus=-1,
                                 num_nodes=config.test_settings.num_nodes,
                                 accelerator='ddp',
                                 benchmark=True,)
            loguru_logger.info(f"Start testing!")
            trainer.test(model, datamodule=data_module, verbose=False)

if __name__ == '__main__':
    tyro.cli(main)