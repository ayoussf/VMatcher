# Code adapted from EfficientLoFTR [https://github.com/zju3dv/EfficientLoFTR/]

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from collections import defaultdict
import pprint
from loguru import logger
from src.VMatcher.VMatcher import VMatcher
from VMatch.src.losses.vmatcher_loss import VMatchLoss
from VMatch.src.VMatcher.model_utils.supervision.supervision import compute_supervision_coarse, compute_supervision_fine
from VMatch.src.optimisers import build_optimizer, build_scheduler
from VMatch.src.utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors, aggregate_metrics, estimate_homography
from VMatch.src.utils.plotting import make_matching_figures
from VMatch.src.utils.comm import gather, all_gather
from VMatch.src.utils.misc import flattenList
from VMatch.src.utils.profiler import PassThroughProfiler

def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
    for modules in [matcher.backbone.layer1, matcher.backbone.layer2, matcher.backbone.layer3]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    return matcher

class PL_VMatcher(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        if hasattr(config, "test_settings"):
            self.n_vals_plot = max(config.plotting.n_val_pairs_to_plot // config.test_settings.world_size, 1)
        else:
            self.n_vals_plot = max(config.plotting.n_val_pairs_to_plot // config.train_settings.world_size, 1)
        
        self.matcher = VMatcher(self.config, profiler=self.profiler)
        if hasattr(config, "train_settings"):
            self.loss = VMatchLoss(self.config)
        
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            msg=self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        self.warmup = False
        self.reparameter = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config.train_settings)
        scheduler = build_scheduler(self.config.train_settings, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, 
                       optimizer_idx,optimizer_closure, 
                       on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.train_settings.warmup_step
        if self.trainer.global_step < warmup_step:
            if self.config.train_settings.warmup_type == 'linear':
                base_lr = self.config.train_settings.warmup_ratio * self.config.train_settings.true_lr
                lr = base_lr + (self.trainer.global_step / self.config.train_settings.warmup_step) * abs(self.config.train_settings.true_lr - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.train_settings.warmup_type == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.train_settings.warmup_type}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_coarse(batch, self.config)
        
        with self.profiler.profile("VMatcher forward"):
            with torch.autocast(enabled=self.config.mp, device_type='cuda'):
                self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_fine(batch, self.config, self.logger)
            
        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.mp, device_type='cuda'):
                self.loss(batch)
    
    def _compute_metrics(self, batch):
        if self.config.dataset_settings.test_data_source == 'HPatches':
            estimate_homography(batch, self.config.metrics)
            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                       'mean_dist': batch['mean_dist']}
            ret_dict = {'metrics': metrics}
        else:
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, self.config.metrics)

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [(batch['epi_errs'].reshape(-1,1))[batch['m_bids'] == b].reshape(-1).cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers'],
                'num_matches': [batch['mconf'].shape[0]],
                }
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)
            
            if self.config.plotting.enable_plotting:
                compute_symmetrical_epipolar_errors(batch)
                figures = make_matching_figures(batch, self.config.plotting, self.config.plotting.plot_mode)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)

    def on_validation_epoch_start(self):
        self.matcher.fine_matching.validate = True

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        ret_dict, _ = self._compute_metrics(batch)
        
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.plotting.plot_mode: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config.plotting, mode=self.config.plotting.plot_mode)

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        
    def validation_epoch_end(self, outputs):
        self.matcher.fine_matching.validate = False
        
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.sanity_checking:
                cur_epoch = -1

            
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

            val_metrics_4tb = aggregate_metrics(metrics, self.config.metrics.epi_err_thr, config=self.config.metrics)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            
            
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                
                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))

    def test_step(self, batch, batch_idx):
        if (self.config.backbone.backbone_type in ['RepVGG', 'VGG']) and not self.reparameter:
            if self.config.backbone.backbone_type == 'RepVGG':
                self.matcher = reparameter(self.matcher)
            elif self.config.backbone.backbone_type == 'VGG':
                self.matcher.backbone.fuse_model()
            else:
                raise NotImplementedError(f'Backbone not implemented')
            if self.config.half:
                self.matcher = self.matcher.eval().half()
            self.reparameter = True

        if not self.warmup:
            if self.config.half:
                for i in range(50):
                    self.matcher(batch)
            else:
                with torch.autocast(enabled=self.config.mp, device_type='cuda'):
                    for i in range(50):
                        self.matcher(batch)
            self.warmup = True
            torch.cuda.synchronize()

        if self.config.half:
            self.start_event.record()
            self.matcher(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
        else:
            with torch.autocast(enabled=self.config.mp, device_type='cuda'):
                self.start_event.record()
                self.matcher(batch)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)
        ret_dict, rel_pair_names = self._compute_metrics(batch)
        return ret_dict

    def test_epoch_end(self, outputs):
        
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        if self.trainer.global_rank == 0:
            hpatches = True if self.config.dataset_settings.test_data_source == 'HPatches' else False
            if hpatches:
                len = 540 if self.config.dataset_settings.ignore_scenes == True else 580
                print('Averaged Matching time over {} pairs: {:.2f} ms'.format(len, self.total_ms / len))
            else:
                print('Averaged Matching time over 1500 pairs: {:.2f} ms'.format(self.total_ms / 1500))
            val_metrics_4tb = aggregate_metrics(metrics, self.config.metrics.epi_err_thr, config=self.config.metrics, hpatches=hpatches)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))