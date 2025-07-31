import torch
import torch.nn as nn
import math
from einops import rearrange
from functools import partial
from timm.models.layers import DropPath
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from VMatch.src.VMatcher.model_utils.vm.build_blocks import create_block, allocate_auto
from VMatch.src.VMatcher.model_utils.backbone.backbone import RepVGG_8_1_align
from VMatch.src.VMatcher.model_utils.backbone.VGG import VGG_BACKBONE
from VMatch.src.VMatcher.model_utils.matching.coarse_matching import CoarseMatching
from VMatch.src.VMatcher.model_utils.matching.fine_preprocess import FinePreprocess
from VMatch.src.VMatcher.model_utils.matching.fine_matching import FineMatching
from VMatch.src.VMatcher.model_utils import detect_NaN, mask_features, unmask_features

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class VMatcher(nn.Module):
    def __init__(self, config: dict, profiler=None):
        super(VMatcher, self).__init__()

        self.config = config
        self.profiler = profiler
        device = "cuda"
        dtype = None
        
        # Backbone.
        if config.backbone.backbone_type == 'RepVGG':
            self.backbone = RepVGG_8_1_align(config=config.backbone)
        else:
            self.backbone = VGG_BACKBONE(config=config.backbone, device=device, dtype=dtype)
        
        # Mamba and Mamba related layers.
        pattern = allocate_auto(config.mamba_config.model_type,
                                config.mamba_config.num_layers,
                                config.mamba_config.att_ratio,
                                config.mamba_config.mlp_ratio,
                                config.mamba_config.cross_only,
                                config.mamba_config.switch,)
        d_p = [x.item() for x in torch.linspace(0, config.mamba_config.drop_path_rate, len(pattern)+1)]
        self.vim_layers = nn.ModuleList([create_block(config.mamba_config, config.atten_config, d_p[i], i, device=device, dtype=dtype,
                                                      pattern=pattern) for i in range(len(pattern))])
        
        self.drop_path = DropPath(config.mamba_config.drop_path_rate) if config.mamba_config.drop_path_rate > 0. else nn.Identity()
        self.norm_i = (nn.LayerNorm if not config.mamba_config.rmsnorm else RMSNorm)(config.mamba_config.d_model, eps=config.mamba_config.norm_eps,
                                                                                     device=device, dtype=dtype)
        self.norm_f = (nn.LayerNorm if not config.mamba_config.rmsnorm else RMSNorm)(config.mamba_config.d_model, eps=config.mamba_config.norm_eps,
                                                                                      device=device, dtype=dtype)
        
        # Matching & Refinement layers.
        self.coarse_matching = CoarseMatching(config.match_coarse)
        self.fine_preprocess = FinePreprocess(config)
        self.fine_matching = FineMatching(config.match_fine)
        
        self.apply(partial(_init_weights, n_layer=len(pattern)))

    def forward(self, data: dict) -> torch.Tensor:
        '''
        Input:
            data: dict
        Output:
            data: dict
        '''
        data.update({'bs': data['image0'].size(0),
                     'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]})
        
        if data['hw0_i'] == data['hw1_i']:
            ret_dict = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            feats_c = ret_dict['feats_c']
            data.update({'feats_x2': ret_dict['feats_x2'],
                         'feats_x1': ret_dict['feats_x1'],})
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
        else:
            ret_dict0, ret_dict1 = self.backbone(data['image0']), self.backbone(data['image1'])
            feat_c0 = ret_dict0['feats_c']
            feat_c1 = ret_dict1['feats_c']
            data.update({
                'feats_x2_0': ret_dict0['feats_x2'],
                'feats_x1_0': ret_dict0['feats_x1'],
                'feats_x2_1': ret_dict1['feats_x2'],
                'feats_x1_1': ret_dict1['feats_x1'],
            })
        
        mul = self.config.backbone.resolution_ratio
        data.update({'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
                     'hw0_f': [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul], 
                     'hw1_f': [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul]})
        
        mask_c0 = mask_c1 = None
        H0, W0 = feat_c0.size(-2), feat_c0.size(-1) 
        H1, W1 = feat_c1.size(-2), feat_c1.size(-1)
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'], data['mask1']
            feat_c0, feat_c1 = mask_features(feat_c0, feat_c1, mask_c0, mask_c1)
            H0, W0 = feat_c0.size(-2), feat_c0.size(-1) 
            H1, W1 = feat_c1.size(-2), feat_c1.size(-1)

        feat_c0, feat_c1 = rearrange(feat_c0, 'b c h w -> b (h w) c').contiguous(), rearrange(feat_c1, 'b c h w -> b (h w) c').contiguous()
        
        feat_c0, feat_c1 = self.norm_i(feat_c0), self.norm_i(feat_c1)
        residual_0, residual_1 = None, None
        for layer in self.vim_layers:
            feat_c0, feat_c1, residual_0, residual_1 = layer(feat_c0=feat_c0,
                                                             feat_c1=feat_c1,
                                                             residual_0=residual_0,
                                                             residual_1=residual_1,
                                                             dims=(H0, W0, H1, W1))
        
        if not self.config.mamba_config.fused_add_norm:
            residual_0 = residual_0 + self.drop_path(feat_c0)
            residual_1 = residual_1 + self.drop_path(feat_c1)
            feat_c0 = self.norm_f(residual_0.to(dtype=self.norm_f.weight.dtype))
            feat_c1 = self.norm_f(residual_1.to(dtype=self.norm_f.weight.dtype))
        else:
            feat_c0 = layer_norm_fn(self.drop_path(feat_c0),
                                    self.norm_f.weight,
                                    self.norm_f.bias,
                                    eps=self.norm_f.eps,
                                    residual=residual_0,
                                    prenorm=False,
                                    residual_in_fp32=self.config.mamba_config.residual_in_fp32,
                                    is_rms_norm=isinstance(self.norm_f, RMSNorm))
            
            feat_c1 = layer_norm_fn(self.drop_path(feat_c1),
                                    self.norm_f.weight,
                                    self.norm_f.bias,
                                    eps=self.norm_f.eps,
                                    residual=residual_1,
                                    prenorm=False,
                                    residual_in_fp32=self.config.mamba_config.residual_in_fp32,
                                    is_rms_norm=isinstance(self.norm_f, RMSNorm))
        
        if 'mask0' in data:
           feat_c0, feat_c1 = rearrange(feat_c0,"b (H W) c -> b c H W", H=H0, W=W0).contiguous(), rearrange(feat_c1,"b (H W) c -> b c H W", H=H1, W=W1).contiguous()
           feat_c0, feat_c1 = unmask_features(feat_c0, feat_c1, mask_c0, mask_c1)
           feat_c0, feat_c1 = rearrange(feat_c0, 'b c h w -> b (h w) c').contiguous(), rearrange(feat_c1, 'b c h w -> b (h w) c').contiguous()
        
        if self.config.fine_preprocess.replace_nan and (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))):
            detect_NaN(feat_c0, feat_c1, 'vm')
        
        self.coarse_matching(feat_c0, feat_c1, data,
                             mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0,
                             mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1)
        
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])
        
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_c0, feat_c1, data)

        if self.config.fine_preprocess.replace_nan and (torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold))):
            detect_NaN(feat_f0_unfold, feat_f1_unfold, 'fine refinement')
        
        del feat_c0, feat_c1, mask_c0, mask_c1

        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)