import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.backends.cuda import sdp_kernel

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

from mamba_ssm.ops.triton.layer_norm import RMSNorm

class MHA(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        head_dim=None,  # If None, use embed_dim // num_heads
        mlp_dim=0,
        qkv_proj_bias=True,
        out_proj_bias=True,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        d_conv=0,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_interleaved=False,
        device=None,
        dtype=None,
        cross=False,
        cat=False,
        rmsnorm=True,
        norm_eps=1e-5,
        downsample=False,
        down_scale=4,
        flash=False,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.rotary_emb_dim = rotary_emb_dim
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.cross = cross
        self.cat = cat
        self.downsample = downsample
        self.down_scale = down_scale
        self.flash = flash

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        if head_dim is None:
            assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = head_dim if head_dim is not None else self.embed_dim // num_heads
        self.mlp_dim = math.ceil(mlp_dim / 256) * 256
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads

        if self.rotary_emb_dim > 0 and not self.cross:
            assert RotaryEmbedding is not None, "rotary requires flash_attn to be installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        self.in_proj = nn.Linear(embed_dim, qkv_dim + self.mlp_dim, bias=qkv_proj_bias, **factory_kwargs)
        if self.d_conv > 0:
            self.conv1d = nn.Conv1d(
                qkv_dim, qkv_dim, kernel_size=self.d_conv, padding=self.d_conv - 1, groups=qkv_dim,
                **factory_kwargs
            )
        self.out_proj = nn.Linear(out_dim + self.mlp_dim // 2, embed_dim, bias=out_proj_bias, **factory_kwargs)
        if self.cat:
            self.fc1 = nn.Linear(embed_dim*2, embed_dim*2, bias=out_proj_bias, **factory_kwargs)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(embed_dim*2, embed_dim, bias=out_proj_bias, **factory_kwargs)
            self.norm_mlp = (nn.LayerNorm if not rmsnorm else RMSNorm)(embed_dim, eps=norm_eps,
                                                                       device=device, dtype=dtype)
    
    def forward(self, u_0, u_1, **kwargs):
        
        if self.downsample:
            H0, W0, H1, W1 = kwargs.get('dims')
            u_0_d = rearrange(u_0, "b (h w) c -> b c h w", h=H0, w=W0).contiguous()
            u_1_d = rearrange(u_1, "b (h w) c -> b c h w", h=H1, w=W1).contiguous()
            u_0_d = F.interpolate(u_0_d, size=(H0//self.down_scale, W0//self.down_scale), mode='bilinear', align_corners=False)
            u_1_d = F.interpolate(u_1_d, size=(H1//self.down_scale, W1//self.down_scale), mode='bilinear', align_corners=False)
            H0_d, W0_d, H1_d, W1_d = u_0_d.shape[-2], u_0_d.shape[-1], u_1_d.shape[-2], u_1_d.shape[-1]
            u_0_d = rearrange(u_0_d, "b c h w -> b (h w) c").contiguous()
            u_1_d = rearrange(u_1_d, "b c h w -> b (h w) c").contiguous()
            qkv_0 = self.in_proj(u_0_d)
            qkv_1 = self.in_proj(u_1_d)
        else:
            qkv_0 = self.in_proj(u_0)
            qkv_1 = self.in_proj(u_1)

        if self.mlp_dim > 0:
            qkv_0, x_mlp_0 = qkv_0.split([qkv_0.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up_0, x_mlp_gate_0 = x_mlp_0.chunk(2, dim=-1)
            x_mlp_0 = x_mlp_up_0 * F.silu(x_mlp_gate_0)
            qkv_1, x_mlp_1 = qkv_1.split([qkv_1.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up_1, x_mlp_gate_1 = x_mlp_1.chunk(2, dim=-1)
            x_mlp_1 = x_mlp_up_1 * F.silu(x_mlp_gate_1)
        
        if self.d_conv > 0:
            if causal_conv1d_fn is None:
                qkv_0 = rearrange(
                    self.conv1d(rearrange(qkv_0, "b s d -> b d s"))[..., :-(self.d_conv - 1)], "b d s -> b s d"
                ).contiguous()
                qkv_1 = rearrange(
                    self.conv1d(rearrange(qkv_1, "b s d -> b d s"))[..., :-(self.d_conv - 1)], "b d s -> b s d"
                ).contiguous()
                
            else:
                qkv_0 = causal_conv1d_fn(
                    qkv_0.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias
                ).transpose(1, 2)
                qkv_1 = causal_conv1d_fn(
                    qkv_1.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias
                ).transpose(1, 2)
        
        q_0, kv_0 = qkv_0.split([self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim], dim=-1)
        q_0 = rearrange(q_0, "... (h d) -> ... h d", d=self.head_dim)
        kv_0 = rearrange(kv_0, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
        q_1, kv_1 = qkv_1.split([self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim], dim=-1)
        q_1 = rearrange(q_1, "... (h d) -> ... h d", d=self.head_dim)
        kv_1 = rearrange(kv_1, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

        if self.rotary_emb_dim > 0 and not self.cross:
            q_0, kv_0 = self.rotary_emb(
                q_0, kv_0, seqlen_offset=0, max_seqlen=None
            )
            q_1, kv_1 = self.rotary_emb(
                q_1, kv_1, seqlen_offset=0, max_seqlen=None
            )
        
        k_0, v_0 = kv_0.unbind(dim=-3)
        k_1, v_1 = kv_1.unbind(dim=-3)
        k_0, v_0, k_1, v_1 = map(lambda x: torch.repeat_interleave(x, dim=2, repeats=self.num_heads // self.num_heads_kv), [k_0, v_0, k_1, v_1])
        
        if not self.cross:
            if self.flash:
                if q_0.dtype == torch.float16:
                    with sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                        context_0 = F.scaled_dot_product_attention(
                            q_0.transpose(1, 2), k_0.transpose(1, 2), v_0.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                        ).transpose(1, 2)
                        context_1 = F.scaled_dot_product_attention(
                            q_1.transpose(1, 2), k_1.transpose(1, 2), v_1.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                        ).transpose(1, 2)
                else:
                    context_0 = F.scaled_dot_product_attention(
                            q_0.transpose(1, 2), k_0.transpose(1, 2), v_0.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                        ).transpose(1, 2)
                    context_1 = F.scaled_dot_product_attention(
                        q_1.transpose(1, 2), k_1.transpose(1, 2), v_1.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                    ).transpose(1, 2)
            else:
                qk0 = torch.einsum("nlhd,nshd->nlsh", q_0, k_0)
                qk1 = torch.einsum("nlhd,nshd->nlsh", q_1, k_1)       
                softmax_temp_0 = 1. / q_0.size(3)**.5
                softmax_temp_1 = 1. / q_1.size(3)**.5
                con0 = torch.softmax(softmax_temp_0 * qk0, dim=2)
                con1 = torch.softmax(softmax_temp_1 * qk1, dim=2)
                context_0 = torch.einsum("nlsh,nshd->nlhd", con0, v_0)
                context_1 = torch.einsum("nlsh,nshd->nlhd", con1, v_1)
        else:
            if self.flash:
                if q_0.dtype == torch.float16:
                    with sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                        context_0 =  F.scaled_dot_product_attention(
                            q_0.transpose(1, 2), k_1.transpose(1, 2), v_1.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                        ).transpose(1, 2)
                        context_1 = F.scaled_dot_product_attention(
                            q_1.transpose(1, 2), k_0.transpose(1, 2), v_0.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                        ).transpose(1, 2)
                else:
                    context_0 =  F.scaled_dot_product_attention(
                            q_0.transpose(1, 2), k_1.transpose(1, 2), v_1.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                        ).transpose(1, 2)
                    context_1 = F.scaled_dot_product_attention(
                        q_1.transpose(1, 2), k_0.transpose(1, 2), v_0.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                    ).transpose(1, 2)
            else:
                qk0 = torch.einsum("nlhd,nshd->nlsh", q_0, k_1)
                qk1 = torch.einsum("nlhd,nshd->nlsh", q_1, k_0) 
                softmax_temp_0 = 1. / q_0.size(3)**.5
                softmax_temp_1 = 1. / q_1.size(3)**.5
                con0 = torch.softmax(softmax_temp_0 * qk0, dim=2)
                con1 = torch.softmax(softmax_temp_1 * qk1, dim=2)
                context_0 = torch.einsum("nlsh,nshd->nlhd", con0, v_1)
                context_1 = torch.einsum("nlsh,nshd->nlhd", con1, v_0)
                
        context_0 = rearrange(context_0, "... h d -> ... (h d)")
        context_1 = rearrange(context_1, "... h d -> ... (h d)")

        if self.mlp_dim > 0:
            context_0 = torch.cat([context_0, x_mlp_0], dim=-1)
            context_1 = torch.cat([context_1, x_mlp_1], dim=-1)
        
        out_0 = self.out_proj(context_0)
        out_1 = self.out_proj(context_1)

        if self.downsample:
            out_0 = rearrange(out_0, "b (h w) c -> b c h w", h=H0_d, w=W0_d).contiguous()
            out_1 = rearrange(out_1, "b (h w) c -> b c h w", h=H1_d, w=W1_d).contiguous()
            out_0 = F.interpolate(out_0, size=(H0, W0), mode='bilinear', align_corners=False)
            out_1 = F.interpolate(out_1, size=(H1, W1), mode='bilinear', align_corners=False)
            out_0 = rearrange(out_0, "b c h w -> b (h w) c").contiguous()
            out_1 = rearrange(out_1, "b c h w -> b (h w) c").contiguous()
        
        if self.cat:
            out_0 = self.fc1(torch.cat([u_0, out_0],dim=-1))
            out_0 = self.act(out_0)
            out_0 = self.fc2(out_0)
            out_0 = self.norm_mlp(out_0)
                        
            out_1 = self.fc1(torch.cat([u_1, out_1],dim=-1))
            out_1 = self.act(out_1)
            out_1 = self.fc2(out_1)
            out_1 = self.norm_mlp(out_1)

        return out_0, out_1