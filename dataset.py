import os
from typing import List, Tuple, Optional
import cv2
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class ResizeAndNormalize:
    def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD, thr=0.5):
        self.size = size  # (H, W)
        self.mean = mean
        self.std = std
        self.thr = thr

    def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = TF.resize(
            img_pil, self.size,
            interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        img_t = TF.to_tensor(img_resized)                   # Convert image to tensor
        img_t = TF.normalize(img_t, self.mean, self.std)    # Normalize with ImageNet mean and std
        H, W = self.size
        mask_ts = []
        for c in range(mask_hwc.shape[2]):
            m = Image.fromarray(mask_hwc[..., c])
            m = TF.resize(m, (H, W), interpolation=InterpolationMode.NEAREST)
            # Convert to numpy array to handle instance segmentation masks properly
            m_np = np.array(m)  # uint8 array, values: 0 (background), 1,2,3... (instances)
            # For instance segmentation: any non-zero value should be foreground (1)
            # Convert to binary: 0 -> 0, any value > 0 -> 1
            m_binary = (m_np > 0).astype(np.float32)
            mask_ts.append(torch.from_numpy(m_binary))
        mask_t = torch.stack(mask_ts, dim=0).float()   # Stack into (C,H,W)
        return img_t, mask_t


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def _list_files(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    out.sort()
    return out

def _swap_dir_keep_name(path: str, src_dir_name: str, dst_dir_name: str, dst_ext: Optional[str]) -> str:
    parts = path.replace("\\", "/").split("/")
    try:
        i = parts.index(src_dir_name)
    except ValueError:
        raise RuntimeError(f"Directory name '{src_dir_name}' not found in path: {path}")
    parts[i] = dst_dir_name
    dst_path = "/".join(parts)
    if dst_ext is not None:
        base, _ = os.path.splitext(dst_path)
        dst_path = base + dst_ext
    return dst_path

class FolderDataset(data.Dataset):
    """
    Directory structure (two modes):
    
    Mode 1 (with train/test split):
      root/
        train/
          img/
          label/
        test/
          img/
          label/
    
    Mode 2 (flat structure, auto-split):
      root/
        Images/
        Masks/
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dir_name: str = "img",
        label_dir_name: str = "label",
        img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
        mask_ext: Optional[str] = None,  
        transform: Optional[ResizeAndNormalize] = None,
        strict_pair: bool = True,
        auto_split: bool = False,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir_name = img_dir_name
        self.label_dir_name = label_dir_name
        self.mask_ext = mask_ext
        self.transform = transform
        self.strict_pair = strict_pair
        self.auto_split = auto_split

        # Determine directory structure
        if auto_split:
            # Flat structure: root/Images, root/Masks
            self.img_dir = os.path.join(root, img_dir_name)
            self.label_dir = os.path.join(root, label_dir_name)
        else:
            # Traditional structure: root/split/img, root/split/label
            self.img_dir = os.path.join(root, split, img_dir_name)
            self.label_dir = os.path.join(root, split, label_dir_name)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        all_img_paths = _list_files(self.img_dir, img_exts)
        if len(all_img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")

        # Build all pairs first
        all_pairs: List[Tuple[str, str]] = []
        for ip in all_img_paths:
            mp = _swap_dir_keep_name(ip, self.img_dir_name, self.label_dir_name, self.mask_ext)
            if not os.path.isfile(mp) and self.mask_ext is None:
                base, _ = os.path.splitext(mp)
                found = False
                for ext in SUPPORTED_EXTS:
                    cand = base + ext
                    if os.path.isfile(cand):
                        mp = cand
                        found = True
                        break
                if not found and self.strict_pair:
                    raise FileNotFoundError(f"Label file not found: {base}.(any extension with same name)")
                elif not found:
                    continue
            elif not os.path.isfile(mp) and self.strict_pair:
                raise FileNotFoundError(f"Label file not found: {mp}")
            elif not os.path.isfile(mp):
                continue

            all_pairs.append((ip, mp))

        if len(all_pairs) == 0:
            raise RuntimeError(f"No valid (img, label) pairs found!")

        # Split data if auto_split is enabled
        if auto_split:
            import random
            random.seed(seed)
            random.shuffle(all_pairs)
            split_idx = int(len(all_pairs) * train_ratio)
            if split == "train":
                self.pairs = all_pairs[:split_idx]
            elif split == "test" or split == "val":
                self.pairs = all_pairs[split_idx:]
            else:
                raise ValueError(f"Unknown split: {split}. Use 'train' or 'test'/'val'")
        else:
            self.pairs = all_pairs

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid (img, label) pairs in {self.split} set!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
        if mask is None:
            raise RuntimeError(f"Failed to read label: {mask_path}")
        # (H,W) -> (H,W,1)
        mask_hwc = mask[..., None]

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask_hwc)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            # For instance segmentation masks: any non-zero value is foreground
            # mask is uint8: 0 (background), 1,2,3... (instances)
            m_np = mask_hwc[..., 0]  # (H, W) uint8
            m_binary = (m_np > 0).astype(np.float32)  # Convert to binary: 0->0, >0->1
            mask_t = torch.from_numpy(m_binary).unsqueeze(0).float()  # (1, H, W)

        meta = {
            "img_path": img_path,
            "mask_path": mask_path,
            "id": os.path.splitext(os.path.basename(img_path))[0],
        }
        return img_t, mask_t, meta
