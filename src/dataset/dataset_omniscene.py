import copy
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import repeat
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler
from .utils_omniscene import load_conditions, load_info


@dataclass
class DatasetOmniSceneCfg(DatasetCfgCommon):
    name: Literal["omniscene"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False


class DatasetOmniScene(Dataset):
    data_version: str = "interp_12Hz_trainval"
    dataset_prefix: str = "/datasets/nuScenes"
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        cfg: DatasetOmniSceneCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        load_rel_depth: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.resolution = tuple(cfg.image_shape)
        self.data_root = str(cfg.roots[0])
        self.near = 0.1 if cfg.near == -1 else cfg.near
        self.far = 1000.0 if cfg.far == -1 else cfg.far
        self.load_rel_depth = stage == "test" if load_rel_depth is None else load_rel_depth
        if stage != "test":
            self.load_rel_depth = False

        self.bin_tokens = self._load_bin_tokens(stage)

    def _load_json_bins(self, split: str) -> list[str]:
        path = Path(self.data_root) / self.data_version / split
        return json.loads(path.read_text())["bins"]

    def _load_bin_tokens(self, stage: Stage) -> list[str]:
        if stage == "train":
            return self._load_json_bins("bins_train_3.2m.json")
        if stage == "val":
            bins = self._load_json_bins("bins_val_3.2m.json")
            return bins[:30000:3000][:10]
        if stage == "test":
            bins = self._load_json_bins("bins_val_3.2m.json")
            return bins[0::14][:2048]
        raise ValueError(f"Unsupported stage {stage}")

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def __getitem__(self, index: int):
        bin_token = self.bin_tokens[index]
        bin_info_path = (
            Path(self.data_root)
            / self.data_version
            / "bin_infos_3.2m"
            / f"{bin_token}.pkl"
        )
        with bin_info_path.open("rb") as f:
            bin_info = pickle.load(f)

        sensor_info_center = {
            sensor: bin_info["sensor_info"][sensor][0]
            for sensor in self.camera_types + ["LIDAR_TOP"]
        }

        context_paths, context_c2w = [], []
        for cam in self.camera_types:
            info = copy.deepcopy(sensor_info_center[cam])
            img_path, c2w, _ = load_info(info)
            context_paths.append(img_path.replace(self.dataset_prefix, self.data_root))
            context_c2w.append(c2w)
        context_c2w = torch.stack(context_c2w)

        context_imgs, context_masks, context_intrinsics, context_rel_depths = load_conditions(
            context_paths,
            self.resolution,
            is_input=True,
            load_rel_depth=self.load_rel_depth,
        )

        render_paths, render_c2w = [], []
        frame_count = len(bin_info["sensor_info"]["LIDAR_TOP"])
        assert frame_count >= 3, f"Bin {bin_token} has insufficient frames."
        render_indices = [[1, 2]] * len(self.camera_types)
        for cam, indices in zip(self.camera_types, render_indices):
            for ind in indices:
                info = copy.deepcopy(bin_info["sensor_info"][cam][ind])
                img_path, c2w, _ = load_info(info)
                render_paths.append(img_path.replace(self.dataset_prefix, self.data_root))
                render_c2w.append(c2w)
        render_c2w = torch.stack(render_c2w)

        target_imgs, target_masks, target_intrinsics, target_rel_depths = load_conditions(
            render_paths,
            self.resolution,
            is_input=False,
            load_rel_depth=self.load_rel_depth,
        )

        target_imgs = torch.cat([target_imgs, context_imgs], dim=0)
        target_masks = torch.cat([target_masks, context_masks], dim=0)
        target_c2w = torch.cat([render_c2w, context_c2w], dim=0)
        target_intrinsics = torch.cat([target_intrinsics, context_intrinsics], dim=0)
        if target_rel_depths is not None and context_rel_depths is not None:
            target_rel_depths = torch.cat([target_rel_depths, context_rel_depths], dim=0)

        context = {
            "extrinsics": context_c2w,
            "intrinsics": context_intrinsics,
            "image": context_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=len(context_c2w)),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=len(context_c2w)),
            "index": torch.arange(len(context_c2w)),
        }
        target = {
            "extrinsics": target_c2w,
            "intrinsics": target_intrinsics,
            "image": target_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=len(target_c2w)),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=len(target_c2w)),
            "index": torch.arange(len(target_c2w)),
            "masks": target_masks,
        }
        if target_rel_depths is not None:
            target["rel_depth"] = target_rel_depths

        return {
            "context": context,
            "target": target,
            "scene": bin_token,
        }
