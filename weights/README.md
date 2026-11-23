# 模型文件

执行 `python strawberry.py download-weights` 下载 `strawberry_best.pt`。

权重作为 GitHub Release 附件发布，Git 只保存元数据及 SHA-256 校验值。
`model.json` 中记录类别顺序、文件大小、下载地址与历史训练指标。
下载入口会校验 SHA-256，不会覆盖内容不同的现有模型。

本地原 `runs/train/exp3/weights/best.pt` 已复制到此目录，原文件继续保留。
