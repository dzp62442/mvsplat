# OmniScene 数据集实验说明

## 配置概览
- **数据集配置**：计划在 `config/dataset/omniscene.yaml` 中复刻 depthsplat 的结构，将 `name` 设为 `omniscene`、`roots` 指向 `datasets/omniscene`，`defaults.view_sampler` 选择 `all`（无需再做帧间子采样）。基础参数与 re10k 配置保持一致：黑色背景、`cameras_are_circular=false`、`baseline_scale_bounds=true`。近远平面、baseline 归一化保持 depthsplat 经验（`near=0.5`、`far=100.`、`make_baseline_1=false`）。
- **实验配置**：参照 README 中 `python -m src.main +experiment=re10k …` 的用法，新增 `config/experiment/omniscene_112x200.yaml` 与 `config/experiment/omniscene_224x400.yaml`。两份文件都将覆盖：
  - `defaults`：沿用 `re10k` 的编码器/解码器/损失组合。
  - `dataset.image_shape`：分别设为 `[112, 200]` 与 `[224, 400]`；其它字段（near/far、train_times_per_scene）同 depthsplat。
  - `data_loader.train.batch_size=1`，并把 `data_loader.val/test.batch_size` 也改为 1，使训练/验证/测试全程 batch size=1。
  - `trainer.max_steps=100_001`，`trainer.val_check_interval=0.01`（Hydra 会覆盖 `config/main.yaml` 的 0.5）。
  - `optimizer`、`loss`、`model.encoder` 仍沿用本项目的默认参数，以保证与 main 分支的 re10k 结果一致。
  - README 目前没有自动测试的开关；若需要模拟 depthsplat 的 `eval_model_every_n_val=10`，只能在训练脚本外手动触发测试，因此文档中会注明「本仓库暂无该字段，可忽略」。
- **运行命令**：在 README 的训练/测试模版基础上新增示例，例如：
  ```bash
  # 224x400 训练
  python -m src.main +experiment=omniscene_224x400 data_loader.train.batch_size=1
  # 112x200 测试
  python -m src.main +experiment=omniscene_112x200 mode=test \
    checkpointing.load=checkpoints/omniscene_112x200.ckpt \
    dataset/view_sampler=all test.compute_scores=true
  ```
  其中 `dataset/view_sampler=all` 允许我们沿用 omniscene 配置，也可通过命令行指定 `dataset.image_shape` 在两个分辨率间切换。

## 数据加载流程
1. **注册入口**：扩展 `src/dataset/__init__.py` 的 `DATASETS` 和 `DatasetCfg` 联合类型，新增 `DatasetOmniScene` 与配置体，使 Hydra 读取 `dataset.name=omniscene` 时能够创建新的数据集。
2. **数据类实现**：新增 `src/dataset/dataset_omniscene.py` 与 `src/dataset/utils_omniscene.py`，整体逻辑直接参考 depthsplat：
   - 在 `__init__` 中根据 stage 读取 `bins_train_3.2m.json`/`bins_val_3.2m.json`，对 val/test 继续沿用 “每隔 N 个 bin 抽样” 的策略（train 全量、val `[:30000:3000][:10]`、test `[0::14][:2048]`）。
   - `__getitem__` 读取 `bin_infos_3.2m/<token>.pkl`，为 6 个环视相机收集 key-frame（上下文）与 2 帧非关键帧（监督目标），再调用 `load_conditions` 完成 resize、归一化内参、载入动态掩码。
   - 输出的 `context` / `target` 字典结构保持与 re10k 一致，包括 `extrinsics`、`intrinsics`、`image`、`near/far`、`index`。`target` 额外附带 `masks`（布尔张量），以便后续在损失中屏蔽动态区域。
   - 与 depthsplat 相同，view sampler 仅为占位（无需采样），但仍会把 `ViewSampler` 实例传入以满足接口。
3. **工具复用**：`utils_omniscene.py` 将直接沿用 `load_info`、`load_conditions` 等函数，只需把路径前缀 `dataset_prefix` 改为 `datasets/omniscene`，并保证 PIL/NumPy 依赖写在 `requirements.txt` 中已经满足。
4. **差异与兼容性**：
   - mvsplat 当前只有 `DatasetRE10k`（`IterableDataset` 实现），我们新增的 `DatasetOmniScene` 将继承 `Dataset`，但 `DataModule` 会在 `shuffle=not isinstance(dataset, IterableDataset)` 的判断下自动对 train loader 启用 shuffle，无需额外流程。
   - `src/dataset/types.py` 尚未声明 `masks` 字段，需要附带扩展（可定义为可选字段），避免类型提示缺失。
   - `apply_patch_shim` 默认只裁剪图像，需要参考 depthsplat 的实现，为包含 `masks` 的视图补充同步裁剪，避免 mask 与图像错位。

## 主程序调用方式
- **Hydra 流程**：主入口仍是 `python -m src.main`。当实验文件将 `dataset: omniscene` 写入配置后，`load_typed_root_config` 会把 `DatasetOmniSceneCfg` 解析成 dataclass，再由 `DataModule` 初始化对应数据集。训练/测试命令与 README 中 re10k 的范式一致，唯一差别是 `+experiment=omniscene_*` 和新的 checkpoint 路径。
- **模型调用**：`ModelWrapper.training_step` 目前只依赖 `context`/`target` 的 `image/near/far` 等字段，没有动态掩码逻辑。为了在 OmniScene 上对齐 depthsplat，我们计划：
  1. 在 `TrainCfg` 中添加一个布尔标志（例如 `use_dynamic_mask`），默认 `false`，在 OmniScene 实验文件中设为 `true`。
  2. 当该标志开启且 `batch["target"]` 含 `masks` 时，对 `prediction.color` 与 GT 计算损失前应用 mask，跳过动态区域。
  3. 测试阶段同样可在保存指标之前把掩码作用于 `compute_psnr/ssim/lpips`，保持训练测试一致性。
  这样即可最大化复用 depthsplat 的调用方式，同时确保 main 分支的其它实验不受影响。
- **W&B/评估节奏**：mvsplat 没有 `train.eval_model_every_n_val` 机制，因而无法像 depthsplat 那样自动触发测试；文档中会说明「按照 README 的做法，若需要定期测试，可以在训练脚本之外按固定步数恢复 checkpoint 后运行 `mode=test` 命令」。

## 复用与差异总结
- **可直接复用**：OmniScene 的数据解析、mask 加载、bin 抽样流程与 depthsplat 完全一致，可以在本仓库中基本照搬相应模块，只需调整 import 路径与配置对象。
- **需要适配**：
  - 本仓库的类型定义、patch shim 以及 `ModelWrapper` 需额外支持 `masks`；否则虽然能训练，但会失去 OmniScene 原本的动态物体过滤优势。
  - README 中的新命令与实验配置需要补充（酷似 re10k 部分），以确保团队成员知道如何切换 112x200 / 224x400 版本并使用 batch size=1、`val_check_interval=0.01`、`max_steps=100_001` 等设定。
- **后续实现步骤**：先落地配置与文档，再把 depthsplat 的 `dataset_omniscene.py`/`utils_omniscene.py` 迁移到 mvsplat，接着扩展 `DatasetCfg`/`ModelWrapper`/`types`/`patch_shim` 等文件，使训练流程与 depthsplat 对齐。完成后即可在 comp_svfgs 分支上对 OmniScene 运行与自有方法一致的实验基线。
