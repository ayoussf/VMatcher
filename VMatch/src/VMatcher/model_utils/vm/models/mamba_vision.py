import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class MambaVision(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_state=16,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
        bidirectional=False,
        divide_output=False,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.bidirectional = bidirectional
        self.divide_output = divide_output
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        if self.bidirectional:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner//2,
            ).contiguous()
            A_log_b = torch.log(A_b)
            self.A_log_b = nn.Parameter(A_log_b)
            self.A_log_b._no_weight_decay = True
            
            self.conv1d_x_b = nn.Conv1d(
                in_channels=self.d_inner//2,
                out_channels=self.d_inner//2,
                bias=conv_bias//2,
                kernel_size=d_conv,
                groups=self.d_inner//2,
                **factory_kwargs,
            )
            self.conv1d_z_b = nn.Conv1d(
                in_channels=self.d_inner//2,
                out_channels=self.d_inner//2,
                bias=conv_bias//2,
                kernel_size=d_conv,
                groups=self.d_inner//2,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner//2, device=device))
            self.D_b._no_weight_decay = True


    def forward(self, u_0, u_1, **kwargs):
        
        A = -torch.exp(self.A_log.float())
        
        _, seqlen0, _ = u_0.shape
        xz0 = self.in_proj(u_0)
        xz0 = rearrange(xz0, "b l d -> b d l")
        x0, z0 = xz0.chunk(2, dim=1)
        x0_f = F.silu(F.conv1d(input=x0, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z0_f = F.silu(F.conv1d(input=z0, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x0_dbl = self.x_proj(rearrange(x0_f, "b d l -> (b l) d"))
        dt0, B0, C0 = torch.split(x0_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt0 = rearrange(self.dt_proj(dt0), "(b l) d -> b d l", l=seqlen0)
        B0 = rearrange(B0, "(b l) dstate -> b dstate l", l=seqlen0).contiguous()
        C0 = rearrange(C0, "(b l) dstate -> b dstate l", l=seqlen0).contiguous()
        y0 = selective_scan_fn(x0_f, 
                               dt0, 
                               A, 
                               B0, 
                               C0, 
                               self.D.float(), 
                               z=None, 
                               delta_bias=self.dt_proj.bias.float(), 
                               delta_softplus=True, 
                               return_last_state=None)
        y0 = torch.cat([y0, z0_f], dim=1)
        y0 = rearrange(y0, "b d l -> b l d")

        _, seqlen1, _ = u_1.shape
        xz1 = self.in_proj(u_1)
        xz1 = rearrange(xz1, "b l d -> b d l")
        x1, z1 = xz1.chunk(2, dim=1)
        x1_f = F.silu(F.conv1d(input=x1, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z1_f = F.silu(F.conv1d(input=z1, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x1_dbl = self.x_proj(rearrange(x1_f, "b d l -> (b l) d"))
        dt1, B1, C1 = torch.split(x1_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt1 = rearrange(self.dt_proj(dt1), "(b l) d -> b d l", l=seqlen1)
        B1 = rearrange(B1, "(b l) dstate -> b dstate l", l=seqlen1).contiguous()
        C1 = rearrange(C1, "(b l) dstate -> b dstate l", l=seqlen1).contiguous()
        y1 = selective_scan_fn(x1_f, 
                               dt1, 
                               A, 
                               B1, 
                               C1, 
                               self.D.float(), 
                               z=None, 
                               delta_bias=self.dt_proj.bias.float(), 
                               delta_softplus=True, 
                               return_last_state=None)
        y1 = torch.cat([y1, z1_f], dim=1)
        y1 = rearrange(y1, "b d l -> b l d")
        
        if self.bidirectional:
            A_b = -torch.exp(self.A_log_b.float())
            
            x0_b = x0.flip([-1])
            z0_b = z0.flip([-1])
            x0_b = F.silu(F.conv1d(input=x0_b, weight=self.conv1d_x_b.weight, bias=self.conv1d_x_b.bias, padding='same', groups=self.d_inner//2))
            z0_b = F.silu(F.conv1d(input=z0_b, weight=self.conv1d_z_b.weight, bias=self.conv1d_z_b.bias, padding='same', groups=self.d_inner//2))
            x0_dbl_b = self.x_proj_b(rearrange(x0_b, "b d l -> (b l) d"))
            dt0_b, B0_b, C0_b = torch.split(x0_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt0_b = rearrange(self.dt_proj_b(dt0_b), "(b l) d -> b d l", l=seqlen0)
            B0_b = rearrange(B0_b, "(b l) dstate -> b dstate l", l=seqlen0).contiguous()
            C0_b = rearrange(C0_b, "(b l) dstate -> b dstate l", l=seqlen0).contiguous()
            y0_b = selective_scan_fn(x0_b, 
                                     dt0_b, 
                                     A_b, 
                                     B0_b, 
                                     C0_b, 
                                     self.D_b.float(), 
                                     z=None, 
                                     delta_bias=self.dt_proj_b.bias.float(), 
                                     delta_softplus=True, 
                                     return_last_state=None)
            y0_b = torch.cat([y0_b, z0_b], dim=1)
            y0_b = rearrange(y0_b, "b d l -> b l d")
            y0_b = y0_b.flip([1])

            x1_b = x1.flip([-1])
            z1_b = z1.flip([-1])
            x1_b = F.silu(F.conv1d(input=x1_b, weight=self.conv1d_x_b.weight, bias=self.conv1d_x_b.bias, padding='same', groups=self.d_inner//2))
            z1_b = F.silu(F.conv1d(input=z1_b, weight=self.conv1d_z_b.weight, bias=self.conv1d_z_b.bias, padding='same', groups=self.d_inner//2))
            x1_dbl_b = self.x_proj_b(rearrange(x1_b, "b d l -> (b l) d"))
            dt1_b, B1_b, C1_b = torch.split(x1_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt1_b = rearrange(self.dt_proj_b(dt1_b), "(b l) d -> b d l", l=seqlen1)
            B1_b = rearrange(B1_b, "(b l) dstate -> b dstate l", l=seqlen1).contiguous()
            C1_b = rearrange(C1_b, "(b l) dstate -> b dstate l", l=seqlen1).contiguous()
            y1_b = selective_scan_fn(x1_b, 
                                     dt1_b, 
                                     A_b, 
                                     B1_b, 
                                     C1_b, 
                                     self.D_b.float(), 
                                     z=None, 
                                     delta_bias=self.dt_proj_b.bias.float(), 
                                     delta_softplus=True, 
                                     return_last_state=None)
            y1_b = torch.cat([y1_b, z1_b], dim=1)
            y1_b = rearrange(y1_b, "b d l -> b l d")
            y1_b = y1_b.flip([1])

            if self.divide_output:
                out0 = self.out_proj((y0 + y0_b) / 2)
                out1 = self.out_proj((y1 + y1_b) / 2)
            else:
                out0 = self.out_proj(y0 + y0_b)
                out1 = self.out_proj(y1 + y1_b)
        else:
            out0 = self.out_proj(y0)
            out1 = self.out_proj(y1)
        
        return out0, out1