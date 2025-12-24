import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image
import torch


def _ensure_hwc3(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] == 4:
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        blended = rgb * alpha + (1 - alpha) * 255.0
        return blended.clip(0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported channel count {image.shape[2]}")


def load_info(info: dict) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Retrieve image path and extrinsics (c2w / w2c) describing the current frame."""
    img_path = info["data_path"]
    c2w = torch.tensor(info["sensor2lidar_transform"], dtype=torch.float32)

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = torch.from_numpy(lidar2cam_r.T.astype(np.float32))
    w2c[3, :3] = torch.from_numpy(-lidar2cam_t.astype(np.float32))

    return img_path, c2w, w2c


def load_conditions(
    image_paths: Sequence[str],
    resolution: Tuple[int, int],
    *,
    is_input: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load OmniScene RGB images, normalized intrinsics, and dynamic masks."""

    def resize_if_needed(img: Image.Image, ck: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        resize_flag = img.height != resolution[0] or img.width != resolution[1]
        if not resize_flag:
            return np.array(img), ck, False

        fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
        scale_h = resolution[0] / img.height
        scale_w = resolution[1] / img.width
        new_ck = np.array(
            [
                [fx * scale_w, 0, cx * scale_w],
                [0, fy * scale_h, cy * scale_h],
                [0, 0, 1],
            ]
        )
        resized = img.resize((resolution[1], resolution[0]))
        return np.array(resized), new_ck, True

    images = []
    masks = []
    intrinsics = []

    for path in image_paths:
        param_path = (
            path.replace("samples", "samples_param_small")
            .replace("sweeps", "sweeps_param_small")
            .replace(".jpg", ".json")
        )
        params = json.loads(Path(param_path).read_text())
        ck = np.array(params["camera_intrinsic"], dtype=np.float32)

        image_path = (
            path.replace("samples", "samples_small")
            .replace("sweeps", "sweeps_small")
        )
        pil_image = Image.open(image_path)
        rgb, ck, resized = resize_if_needed(pil_image, ck)
        ck[0, :] /= resolution[1]
        ck[1, :] /= resolution[0]
        images.append(_ensure_hwc3(rgb))
        intrinsics.append(ck)

        if is_input:
            mask = np.ones(resolution, dtype=np.float32)
        else:
            mask_path = (
                image_path.replace("sweeps_small", "sweeps_mask_small")
                .replace("samples_small", "samples_mask_small")
                .replace(".jpg", ".png")
            )
            mask_image = Image.open(mask_path).convert("L")
            if resized:
                mask_image = mask_image.resize((resolution[1], resolution[0]), Image.BILINEAR)
            mask = np.asarray(mask_image, dtype=np.float32) / 255.0
        masks.append(mask)

    image_tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
    mask_tensor = torch.from_numpy(np.stack(masks)).bool()
    intrinsic_tensor = torch.from_numpy(np.stack(intrinsics)).float()

    return image_tensor, mask_tensor, intrinsic_tensor
