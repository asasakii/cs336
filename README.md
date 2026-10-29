# cs336

Stanford CS336 Spring 2026 Assignment 1 学习实现与实验记录。

基于 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)。保留上游 LICENSE、测试与讲义；上游 README 见 [README.upstream.md](README.upstream.md)。这是当前完成状态的上传快照，不是“整份作业全部通过”的声明。

## 当前实现

- 字节级 BPE 训练与 Tokenizer，支持特殊 token、encode/decode 和 iterable 编码。
- Embedding、无 bias Linear、RMSNorm、SiLU、SwiGLU、稳定 softmax、causal multihead attention、RoPE、pre-norm TransformerBlock 与 TransformerLM。
- Cross entropy、AdamW、cosine schedule、梯度裁剪、batch 采样、checkpoint 保存/恢复。
- prepare_tokens.py：按完整文档边界编码、uint16 存储、分块读回校验。
- train.py：CPU/CUDA、JSONL 日志、采样验证、checkpoint 与恢复。
- generation.py：交互式英文故事续写，temperature、top-p、结束 token、最大长度与 /continue。代码已保存；实际生成效果尚未确认。

## 已验证的 TinyStories 正式实验

| 项目 | 结果 |
|---|---|
| 参数量 | 22,696,448 |
| 模型 | vocab 10000 / context 256 / width 512 / 4 layers / 16 heads / FFN 1344 |
| 设备 | RTX 3060 系列 6 GiB，PyTorch 2.11.0+cu128 |
| Batch / updates | 8 / 160000 |
| 训练 token 预算 | 327,680,000 |
| Peak / minimum learning rate | 0.001 / 0.0001 |
| Warmup | 1600 updates |
| 最终采样验证 loss | **1.3394585400819778** |
| 首次记录 loss ≤ 1.45 | step 84000 |
| 日志经过时间 | 约 5 小时 18 秒 |

验证使用固定 32 个 batch，共 65,536 个采样 token，并非全量验证。train_loss 是当前单 batch，不是区间平均。计时不包括启动时数据扫描与模型创建。Checkpoint 已读回并确认 iteration=160000、模型张量有限；没有保存 RNG 状态，不能保证恢复后逐步复现不中断训练。

## 文件导航

- [cs336_basics/](cs336_basics/)：实现及中文代码阅读笔记。
- [tests/](tests/)：上游测试、夹具及当前适配器。test_tokenizer.py 的本地改动是 Windows 下条件导入 resource，未更改测试断言。
- [data/](data/)：已训练的 TinyStories 10K 词表、merge 规则与编码元数据。
- [runs/](runs/)：真实实验配置和 metrics.jsonl；不包含训练权重。
- [手机版剩余题详解 PDF](docs/CS336_Assignment1_剩余题详解_手机版.pdf)：33 页学习资料，明确区分实测、理论和预测，含真实学习曲线。
- [上传文件清单](docs/UPLOAD_MANIFEST.json)：原始文件大小与 Git blob SHA-1，便于核对快照。

## 本地运行

在实际仓库根目录使用已经配置好的解释器。以下是 PowerShell 命令：

```powershell
# 运行官方测试；本次上传没有重新执行全套测试。
& ".\.venv\Scripts\python.exe" -m pytest tests -v

# 已有本地 checkpoint 时，启动故事续写。
# 生成题要求至少 256 个 token，或者遇到第一个结束 token 停止。
& ".\.venv\Scripts\python.exe" -m cs336_basics.generation --device cuda --temperature 0.8 --top-p 0.9 --max-new-tokens 256
```

输入英文故事开头开始新故事，输入 /continue 继续，输入 /quit 退出。模型是 TinyStories 续写模型，未经过指令对话训练；超过上下文长度时只使用最近 256 个 token 预测。

依赖声明见 pyproject.toml；torch 要求约为 2.11.0。CUDA 版 PyTorch 已在原环境安装，不要未经核对用依赖同步替换 CUDA 构建。此仓库不携带 .venv。

## 未上传的大文件

训练文本、完整 .npy 编码数组、正式训练的 last.pt、缓存与 IDE 配置没有上传。官方测试使用的小模型 tests/fixtures/ts_tests/model.pt 是测试夹具，予以保留。

因此新克隆本仓库后，不能直接假定拥有训练数据或训练好的模型权重。config.json 与编码元数据保留了实验时的本地路径；换机器需按新路径设置参数。

## 完成范围

已有反馈确认 BPE 核心测试、Tokenizer 23 passed / 2 skipped，以及多个模型组件测试；完整模型与部分训练工具的全部官方测试通过反馈尚未齐全。Windows Tokenizer 内存测试未通过实际执行确认。

其他学习率扫描、发散运行、batch 比较、架构消融、OWT 和排行榜尚未完成实测。PDF 中相关结论是标注的预测，不是实验记录。后续不应把它们报告为已运行结果。

上传日期：2026-10-29。
