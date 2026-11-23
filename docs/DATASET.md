# 草莓成熟度数据集准备

原实验配置曾引用外部目录，目前该目录不存在。项目自带的 `datasets/coco128` 是 COCO 示例集，不是草莓成熟度训练集。仓库不发布原始照片和标注，也不自动下载未知来源数据。

## 文件结构

将你有权使用的数据整理为 YOLO 检测格式：

```text
datasets/strawberry/
├── images/
│   ├── train/     # 训练照片
│   └── val/       # 验证照片
└── labels/
    ├── train/     # 与照片同名的 .txt
    └── val/
```

例如 `images/train/001.jpg` 对应 `labels/train/001.txt`。每行标注一个草莓：

```text
class_id center_x center_y width height
3 0.50 0.45 0.20 0.30
```

坐标按图片宽高归一化到 0–1，类别编号是 0–5。真正没有草莓的背景图片可以使用空标注；有草莓但漏标会影响训练。训练前应人工抽查框的位置、类别和漏标情况。

## 固定类别顺序

| ID | 模型类别 | 中文解释 |
|---|---|---|
| 0 | Early-Turning | 早期转色 |
| 1 | Green | 青果 |
| 2 | Late-Turning | 后期转色 |
| 3 | Red | 红熟 |
| 4 | Turning | 转色 |
| 5 | White | 白果 |

顺序来自现有 `best.pt`，必须保持一致。以上中文为阅读辅助；具体转色边界需要依据原始标注标准确认，模型类别也不等同于糖度或食用品质测量。

修改 `configs/strawberry.yaml` 的 `path` 可使用其他数据位置。通过 `strawberry.py` 运行时，相对路径基于 YAML 所在目录解析；绝对路径也受支持。`train`、`val` 相对于 `path`，支持目录或图片清单文件。清单中的图片应使用绝对路径，或使用 YOLOv5 支持的相对清单路径格式。程序在运行时写入 `runs/configs/*.resolved.yaml`，避免 YOLOv5 自身的相对路径规则产生歧义。

## 训练和验证

```powershell
# 从发布的草莓模型继续微调；会开启一个新实验，不是恢复旧优化器状态
python strawberry.py train --epochs 100 --batch-size 8 --device cpu

# NVIDIA CUDA 环境可以使用 --device 0
python strawberry.py train --device 0 --batch-size 16 --workers 0

# 重新计算验证集指标
python strawberry.py validate --weights runs/train/strawberry/weights/best.pt --device cpu
```

输出目录存在时 YOLOv5 会自动增加编号，例如 `strawberry2`，请将验证命令的权重路径改为实际目录。

同一视频、同一采集批次的相似图片应放在同一划分，避免训练集与验证集重复。建议另外保留独立测试集，覆盖不同光照、遮挡、背景和相机距离。类别不平衡时需检查每类召回率和混淆矩阵，不能只看整体 mAP。

原始数据及其划分未恢复之前，无法严格复现历史指标。
