import torch
from loguru import logger


def detect_NaN(feat_0, feat_1, stage):
    logger.info(f'NaN detected in feature {stage}.')
    logger.info(f"#NaN in feat_0: {torch.isnan(feat_0).int().sum()}, #NaN in feat_1: {torch.isnan(feat_1).int().sum()}")
    feat_0[torch.isnan(feat_0)] = 0
    feat_1[torch.isnan(feat_1)] = 0

def get_cache(B, config, device):
    return (None, torch.zeros(B, 
                              config.d_conv-1, 
                              config.d_model* config.expand_factor,
                              device=device))

def mask_features(feat0, feat1, mask0, mask1):
    agg_size0, agg_size1 = 4, 4
    mask_h0, mask_w0, mask_h1, mask_w1 = mask0[0].sum(-2)[0], mask0[0].sum(-1)[0], mask1[0].sum(-2)[0], mask1[0].sum(-1)[0]
    mask_h0, mask_w0, mask_h1, mask_w1 = mask_h0//agg_size0*agg_size0, mask_w0//agg_size0*agg_size0, mask_h1//agg_size1*agg_size1, mask_w1//agg_size1*agg_size1
    feat0 = feat0[:, :, :mask_h0, :mask_w0]
    feat1 = feat1[:, :, :mask_h1, :mask_w1]
    return feat0, feat1

def unmask_features(feat0, feat1, mask0, mask1):
    
    b, c, h0, w0 = feat0.size()
    _, H0, W0 = mask0.size()
    if h0 != H0:
        feat0 = torch.cat([feat0, torch.zeros(b, c, H0-h0, W0, device=feat0.device, dtype=feat0.dtype)], dim=-2)
    elif w0 != W0:
        feat0 = torch.cat([feat0, torch.zeros(b, c, H0, W0-w0, device=feat0.device, dtype=feat0.dtype)], dim=-1)

    b, c, h1, w1 = feat1.size()
    _, H1, W1 = mask1.size()
    if h1 != H1:
        feat1 = torch.cat([feat1, torch.zeros(b, c, H1-h1, W1, device=feat1.device, dtype=feat1.dtype)], dim=-2)
    elif w1 != W1:
        feat1 = torch.cat([feat1, torch.zeros(b, c, H1, W1-w1, device=feat1.device, dtype=feat1.dtype)], dim=-1)
    
    return feat0, feat1

def pad_to_len(feat0, feat1, H0, W0, H1, W1):
    padded_0 = False
    padded_1 = False
    og_len = 0
    if feat0.size(1) > feat1.size(1):
        padded_1 = True
        og_len = feat1.size(1)
        pad = feat0.size(1) - feat1.size(1)
        feat1 = torch.cat([feat1, torch.zeros(feat1.size(0), pad, feat1.size(2), dtype=feat1.dtype ,device=feat1.device)], dim=1)
        H1, W1 = H0, W0
    elif feat0.size(1) < feat1.size(1):
        padded_0 = True
        og_len = feat0.size(1)
        pad = feat1.size(1) - feat0.size(1)
        feat0 = torch.cat([feat0, torch.zeros(feat0.size(0), pad, feat0.size(2), dtype=feat0.dtype, device=feat0.device)], dim=1)
        H0, W0 = H1, W1
    
    return feat0, feat1, padded_0, padded_1, og_len, H0, W0, H1, W1

def unpad_to_original(feat0, feat1, padded_0, padded_1, og_len):
    if padded_0:
        feat0 = feat0[:, :og_len, :]
    if padded_1:
        feat1 = feat1[:, :og_len, :]
    return feat0, feat1