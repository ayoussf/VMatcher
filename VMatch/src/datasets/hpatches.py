import torch
import kornia.geometry.transform as K
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path

class HPatchesDataset(Dataset):
    default_config = {'name': 'HPatches',
                      'alteration': 'all',
                      'resize': 480,
                      'resize_side': 'short',
                      'ignore_scenes':True}
    def __init__(self, root_dir, ignore_scenes) -> None:
        super(HPatchesDataset,self).__init__()

        self.root_dir = root_dir 
        self.config = self.default_config
        self.config['ignore_scenes'] = ignore_scenes
        self.samples = self._init_dataset()
    
    def _init_dataset(self):
        ignored_scenes = (
                "i_contruction",
                "i_crownnight",
                "i_dc",
                "i_pencils",
                "i_whitebuilding",
                "v_artisans",
                "v_astronautis",
                "v_talent",
            )

        data_dir = Path(self.root_dir)#, self.config["name"])
        folder_dirs = [x for x in data_dir.iterdir() if x.is_dir()]

        image_paths = []
        warped_image_paths = []
        homographies = []
        names = []

        for folder_dir in folder_dirs:
            if (folder_dir.stem in ignored_scenes) and self.config["ignore_scenes"]:
                continue
            if self.config["alteration"] == 'i' != folder_dir.stem[0] != 'i':
                continue
            if self.config["alteration"] == 'v' != folder_dir.stem[0] != 'v':
                continue

            num_images = 5
            file_ext = '.ppm' 

            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(folder_dir, "1" + file_ext)))
                warped_image_paths.append(str(Path(folder_dir, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(folder_dir, "H_1_" + str(i)))))
                names.append(f"{folder_dir.stem}_{1}_{i}")
        
        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'homography': homographies,
                 'names': names} 
        
        return files

    def __len__(self):
        return len(self.samples['image_paths'])
    
    def read_image(self, image):
        """
        Read image from path.
        Input:
            image: path to image
        Output:
            image: image as torch tensor (float32)
        """
        image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        image = image[None]
        image = torch.as_tensor(image/255., dtype=torch.float32)
        return image  

    def get_resize_shape(self, H, W):
        """
        Get new image shape after resizing.
        Input:
            H: height of image
            W: width of image
        Output:
            size: new size of image
        """
        side = self.config["resize_side"]
        side_size = self.config["resize"]
        aspect_ratio = W / H

        if isinstance(side_size, list):
            size = side_size[0], side_size[1]
            return size
        
        if side == "vert":
            size = side_size, int(side_size * aspect_ratio)
        elif side == "horz":
            size = int(side_size / aspect_ratio), side_size
        elif (side == "short") ^ (aspect_ratio < 1.0):
            size = side_size, int(side_size * aspect_ratio)
        else:
            size = int(side_size / aspect_ratio), side_size
        return size
    
    def get_resized_wh(self, w, h, resize=None):
        if resize is not None:  # resize the longer edge
            scale = resize / min(h, w)
            w_new, h_new = int(round(w*scale)), int(round(h*scale))
        else:
            w_new, h_new = w, h
        return w_new, h_new
    
    def get_divisible_wh(self, w, h, df=None):
        if df is not None:
            w_new, h_new = map(lambda x: int(x // df * df), [w, h])
        else:
            w_new, h_new = w, h
        return w_new, h_new
     
    def preprocess(self, image):
        """
        Preprocess image.
        Input:
            image: image as torch tensor (float32)
        Output:
            image: image as torch tensor (float32)
            T: scale matrix
        """
        H, W = image.shape[-2:]
        H_new, W_new = self.get_resize_shape(H, W)
        W_new, H_new = self.get_divisible_wh(W_new, H_new, 8)
        size = (H_new, W_new)
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=size, mode='bilinear', align_corners=False, antialias=True).squeeze(0)
        scale = torch.Tensor([image.shape[-1] / W, image.shape[-2] / H]).to(torch.float32)
        T = np.diag([scale[0], scale[1], 1.0])
        return image, T

    def __getitem__(self, index):

        image = self.read_image(self.samples['image_paths'][index])
        warped_image = self.read_image(self.samples['warped_image_paths'][index])
        
        image, T0 = self.preprocess(image)
        warped_image, T1 = self.preprocess(warped_image)
        
        homography = self.samples['homography'][index].astype(np.float32)
        homography = T1 @ homography @ np.linalg.inv(T0)

        name = self.samples['names'][index]
        size = image.shape[-2:]
        data = {"image0": image,
                "image1": warped_image,
                "H": homography.astype(np.float32),
                "pair_names": name,
                "HPatches_size": [size[0], size[1]]}
        return data