
import torch
import torch.nn as nn
from dataclasses import asdict
from functools import partial
from timm.models.layers import DropPath
from einops import rearrange
from VMatch.src.VMatcher.model_utils.vm.models.mamba_vision import MambaVision
from VMatch.src.VMatcher.model_utils.vm.models.v_mha import MHA
from VMatch.src.VMatcher.model_utils.vm.models.mlps import GatedMLP, FFN
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

def allocate_auto(model_type: str,
                  total_layers_count: int,
                  target_attention_ratio: float,
                  target_mlp_ratio: float,
                  cross_only: bool = False,
                  switch: bool = False) -> list:
    
    attention_layers_count: int = round(total_layers_count * target_attention_ratio)
    mamba_layers_count: int = total_layers_count - attention_layers_count
    mamba_sections_count: int = attention_layers_count + 1
    mamba_section_length: float = mamba_layers_count / mamba_sections_count

    layer_type_list = [model_type] * total_layers_count
    x: float = mamba_section_length
    self: bool = True
    cross: bool = False
    for l in range(total_layers_count):
        if x < 0.5:
            if cross_only:
                layer_type_list[l] = "cross"
                x += mamba_section_length
            else:
                if self:
                    layer_type_list[l] = "self"
                    x += mamba_section_length
                    self = False
                    cross = True
                elif cross:
                    layer_type_list[l] = "cross"
                    x += mamba_section_length
                    self = True
                    cross = False
        else:
            x -= 1
    
    mlp_layers_count: int = round(total_layers_count * target_mlp_ratio)
    if mlp_layers_count > 0:
        mamba_layers_count -= mlp_layers_count
        mamba_to_mlp_ratio: float = mamba_layers_count / mlp_layers_count

        x: float = mamba_to_mlp_ratio
        for l in range(total_layers_count):
            if layer_type_list[l] == model_type:
                if x < 0.5:
                    layer_type_list[l] = "mlp"
                    x += mamba_to_mlp_ratio
                else:
                    x -= 1
    
    if switch:
        # remove every second mamba layer and replace with mamba_s
        counter = 1
        for l in range(total_layers_count):
            if model_type == layer_type_list[l]:
                if counter == 0:
                    layer_type_list[l] = model_type+"_s"
                    counter = 1
                else:
                    counter = 0
    return layer_type_list

class Block(nn.Module):
    def __init__(self, config, model, norm, drop_path, device="cuda", dtype=torch.float16, switch=False):
        super().__init__()
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        self.model = model()
        self.norm = norm(config.d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.switch = switch
        
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, feat_c0, residual_0, feat_c1, residual_1, **kwargs):
        
        if not self.fused_add_norm:
            residual_0 = residual_0 + self.drop_path(feat_c0) if residual_0 is not None else feat_c0
            feat_c0 = self.norm(residual_0.to(dtype=self.norm.weight.dtype))
            
            residual_1 = residual_1 + self.drop_path(feat_c1) if residual_1 is not None else feat_c1
            feat_c1 = self.norm(residual_1.to(dtype=self.norm.weight.dtype))

            if self.residual_in_fp32:
                residual_0 = residual_0.to(torch.float32)
                residual_1 = residual_1.to(torch.float32)
        else:
            if residual_0 is None:
                feat_c0, residual_0 = layer_norm_fn(feat_c0, 
                                                    self.norm.weight,
                                                    self.norm.bias,
                                                    residual=residual_0,
                                                    prenorm=True,
                                                    residual_in_fp32=self.residual_in_fp32,
                                                    eps=self.norm.eps,
                                                    is_rms_norm=isinstance(self.norm, RMSNorm))
                feat_c1, residual_1 = layer_norm_fn(feat_c1,
                                                    self.norm.weight,
                                                    self.norm.bias,
                                                    residual=residual_1,
                                                    prenorm=True,
                                                    residual_in_fp32=self.residual_in_fp32,
                                                    eps=self.norm.eps,
                                                    is_rms_norm=isinstance(self.norm, RMSNorm))   
            else:
                feat_c0, residual_0 = layer_norm_fn(self.drop_path(feat_c0),
                                                    self.norm.weight,
                                                    self.norm.bias,
                                                    residual=residual_0,
                                                    prenorm=True,
                                                    residual_in_fp32=self.residual_in_fp32,
                                                    eps=self.norm.eps,
                                                    is_rms_norm=isinstance(self.norm, RMSNorm))
                feat_c1, residual_1 = layer_norm_fn(self.drop_path(feat_c1),
                                                    self.norm.weight,
                                                    self.norm.bias,
                                                    residual=residual_1,
                                                    prenorm=True,
                                                    residual_in_fp32=self.residual_in_fp32,
                                                    eps=self.norm.eps,
                                                    is_rms_norm=isinstance(self.norm, RMSNorm))
        if self.switch:
            H0, W0, H1, W1 = kwargs.get('dims')
            feat_c0 = rearrange(feat_c0, "b (H W) d -> b (W H) d", H=H0, W=W0).contiguous() 
            feat_c1 = rearrange(feat_c1, "b (H W) d -> b (W H) d", H=H1, W=W1).contiguous()
        
        features = self.model(u_0=feat_c0, u_1=feat_c1, **kwargs)
        
        if self.switch:
            return rearrange(features[0], 'b (W H) d -> b (H W) d', H=H0, W=W0).contiguous(), \
                   rearrange(features[1], 'b (W H) d -> b (H W) d', H=H1, W=W1).contiguous(), \
                   residual_0, residual_1
        else:
            return features[0], features[1], residual_0, residual_1

def create_block(
    config,
    atten_config,
    drop_path,
    layer_idx,
    pattern,
    device="cuda",
    dtype=torch.float16,):
    
    norm = partial(nn.LayerNorm if not config.rmsnorm else RMSNorm, eps=config.norm_eps, device=device, dtype=dtype)
    
    if (pattern[layer_idx] == "mamba_v") or (pattern[layer_idx] == "mamba_v_s"):
        mamba = partial(MambaVision, device=device, dtype=dtype, **asdict(config))
        block = Block(config=config,
                      model=mamba,
                      norm=norm,
                      drop_path=drop_path,
                      device=device,
                      dtype=dtype,
                      switch=True if pattern[layer_idx].endswith("_s") else False)
        block.layer_idx = layer_idx
        return block
    
    elif pattern[layer_idx] == "self":
        self_attention = partial(MHA, device=device, dtype=dtype, **asdict(atten_config))
        block = Block(config=config,
                      model=self_attention,
                      norm=norm,
                      drop_path=drop_path,
                      device=device,
                      dtype=dtype)
        block.layer_idx = layer_idx
        return block
    
    elif pattern[layer_idx] == "cross":
        cross_attention = partial(MHA, device=device, dtype=dtype, cross=True, **asdict(atten_config))
        block = Block(config=config,
                      model=cross_attention,
                      norm=norm,
                      drop_path=drop_path,
                      device=device,
                      dtype=dtype)
        block.layer_idx = layer_idx
        return block
    
    elif pattern[layer_idx] == "mlp":
        if config.mlp_type == "gated_mlp":
            mlp = partial(GatedMLP, in_features=config.d_model, hidden_features=config.mlp_expand*config.d_model,
                          out_features=config.d_model, bias=config.bias, device=device, dtype=dtype)
        else:
            mlp = partial(FFN, in_features=config.d_model, hidden_features=config.mlp_expand*config.d_model,
                          out_features=config.d_model, bias=config.bias, device=device, dtype=dtype)
        block = Block(config=config,
                      model=mlp,
                      norm=norm,
                      drop_path=drop_path,
                      device=device,
                      dtype=dtype)
        block.layer_idx = layer_idx
        return block
    
    else:
        raise NotImplementedError(f'Layer type {pattern[layer_idx]} not implemented')