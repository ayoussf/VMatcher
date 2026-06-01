import torch
import numpy as np
import cv2
import tyro
import dataclasses
import os
import requests
import re
from VMatch.src.VMatcher.VMatcher import VMatcher
from VMatch.configs.conf_blocks import (BackboneConfig,
                                        MambaConfig,
                                        MambaAttention,
                                        Coarse_matching,
                                        Fine_preprocess,
                                        Fine_matching,)
from VMatch.src.utils.dataset import imread_gray, get_resized_wh, get_divisible_wh
from VMatch.src.utils.plotting import (make_matching_figure,
                                       dynamic_alpha,
                                       error_colormap,)

@dataclasses.dataclass
class config:
    model_type: str  # base or tiny
    backbone: BackboneConfig  # Backbone Config
    mamba_config: MambaConfig  # Mamba Config
    atten_config: MambaAttention  # Attention Config
    match_coarse: Coarse_matching  # Coarse Matching Config
    fine_preprocess: Fine_preprocess  # Fine Preprocess Config
    match_fine: Fine_matching  # Fine Matching Config
    optimised: bool = False  # Model's optimised mode
    mp: bool = True  # Mixed precision
    flash: bool = False  # Flash Attention
    image_path_0: str = "assets/demo/image_0.jpg"  # Path to the first image
    image_path_1: str = "assets/demo/image_1.jpg"  # Path to the second image
    resize: int | None = None  # Resize image's longer side
    df: int = 8  # Image size divisibility factor

class Demo:
    def __init__(self, config):
        self.config = config
        assert torch.cuda.is_available(), "CUDA is required but not available"
        self.device = torch.device("cuda")
        self.VMatcher = VMatcher(self.config)
        if not os.path.exists(f"checkpoints/{config.model_type}.ckpt"):
            self.download_weights(config.model_type)
        state_dict = torch.load(f"checkpoints/{config.model_type}.ckpt",
                                map_location="cpu")["state_dict"]
        self.VMatcher.load_state_dict(state_dict, strict=True)
        self.VMatcher.to(self.device).eval()
        self.VMatcher.backbone.fuse_model()
    
    def download_weights(self, model_type):
        print("Downloading weights for model type", model_type)
        if model_type not in ("base", "tiny"):
            raise ValueError(f"model_type must be 'base' or 'tiny', got '{model_type}'")
        file_id = ("1ENP_DhAihiv5WJrRWoAXmOHUdrddJaxU" if model_type == "base"
                   else "1TRiKdPhGjpQ1F2_O9KJ8CyaPffWfNf61")
        output_path = f"checkpoints/{model_type}.ckpt"
        try:
            import gdown
            gdown.download(id=file_id, output=output_path)
            print("Checkpoint downloaded")
            return
        except Exception as e:
            print(f"gdown failed ({e}), falling back to requests...")
        response = requests.get("https://drive.usercontent.google.com/download",
                                params={"id": file_id, "export": "download", "confirm": "t"},
                                stream=True,)
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r{downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
        print("\nCheckpoint downloaded")
    
    def read_image(self, path):
        image = imread_gray(path)
        w, h = image.shape[1], image.shape[0]
        w_new, h_new = get_resized_wh(w, h, self.config.resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, self.config.df)
        image = cv2.resize(image, (w_new, h_new))
        scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)[None]
        image = torch.from_numpy(image).float()[None][None] / 255
        return image, scale
    
    def plot(self, data):
        img0 = (data["image0"][0][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data["image1"][0][0].cpu().numpy() * 255).round().astype(np.int32)

        b_mask = data["m_bids"] == 0
        mkpts0 = data["mkpts0_f"][b_mask].cpu().numpy()
        mkpts1 = data["mkpts1_f"][b_mask].cpu().numpy()

        if "scale0" in data:
            mkpts0 = mkpts0 / data["scale0"][0].cpu().numpy()[[1, 0]]
            mkpts1 = mkpts1 / data["scale1"][0].cpu().numpy()[[1, 0]]

        n_matches = len(mkpts0)
        alpha = dynamic_alpha(n_matches)
        color = error_colormap(np.zeros(n_matches), thr=1.0, alpha=alpha)

        make_matching_figure(img0,
                             img1,
                             mkpts0,
                             mkpts1,
                             color,
                             text=[f"#Matches {n_matches}"],
                             path=f"assets/demo/results_vmatcher_{self.config.model_type}.png",)
        print(f"Saved to assets/demo/results_vmatcher_{self.config.model_type}.png ({n_matches} matches)")

    @torch.inference_mode()
    def run(self):
        image0, scale0 = self.read_image(self.config.image_path_0)
        image1, scale1 = self.read_image(self.config.image_path_1)
        data = {"image0": image0.to(self.device),
                "image1": image1.to(self.device),
                "scale0": scale0.to(self.device),
                "scale1": scale1.to(self.device),}
        with torch.autocast(enabled=self.config.mp, device_type=self.device.type):
            self.VMatcher(data)
        self.plot(data)

def setup_cfg(cfg):
    assert cfg.model_type in ("base", "tiny"), (
           f"Model type must be 'base' or 'tiny', got: '{cfg.model_type}'")
    if cfg.model_type.lower() == "tiny":
        cfg.mamba_config.num_layers = 14
    if cfg.optimised:
        assert cfg.mp, "Optimised mode requires mp"
        assert cfg.match_coarse.thr > 10.0, \
               "Optimised mode requires thr > 10.0 (best at 15.0 or 20.0)"
        cfg.match_coarse.skip_softmax = True
        cfg.match_coarse.fp16matmul = True
    if cfg.mp:
        cfg.mamba_config.residual_in_fp32 = False
    if cfg.flash:
        cfg.atten_config.flash = True
    return cfg

if __name__ == "__main__":
    cfg = tyro.cli(config)
    cfg = setup_cfg(cfg)
    demo = Demo(cfg)
    demo.run()