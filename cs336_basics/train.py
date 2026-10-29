import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.losses import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.schedules import get_lr_cosine_schedule
from cs336_basics.nn_utils import gradient_clipping
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    # argparse 将 PowerShell 中的 --参数名 转成 Python 中的属性。
    parser = argparse.ArgumentParser(
        description="CS336 小规模语言模型训练"
    )

    # 不提供数据文件时，使用下面生成的人工序列。
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--valid-data", type=str, default=None)

    # 模型配置：这些默认值只用于 CPU 调试。
    parser.add_argument("--vocab-size", type=int, default=6)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # 训练和优化器配置。
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--cosine-end", type=int, default=200)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # 日志、验证和保存频率。
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)

    # 这个入门版本固定使用 CPU。
    # 允许通过命令行选择 CPU 或 NVIDIA GPU。
    # 默认仍使用 CPU；传入 --device cuda 时才使用 GPU。
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    # parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    # 每次实验使用自己的输出目录。
    parser.add_argument(
        "--run-dir", type=str, default="runs/debug_loop"
    )
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()

def validate_args(args):
    # 这些参数都必须是正整数。
    positive_names = [
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "batch_size",
        "steps",
        "log_every",
        "eval_every",
        "eval_batches",
        "save_every",
    ]

    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} 必须大于 0")

    if args.vocab_size < 2:
        raise ValueError("本调试脚本要求 vocab_size 至少为 2")

    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model 必须能被 num_heads 整除")

    d_head = args.d_model // args.num_heads
    if d_head % 2 != 0:
        raise ValueError("使用 RoPE 时，每个 head 的维度必须为偶数")

    if args.rope_theta <= 0:
        raise ValueError("rope_theta 必须大于 0")

    if not 0 <= args.min_lr <= args.max_lr:
        raise ValueError("需要满足 0 <= min_lr <= max_lr")

    if not 0 <= args.warmup_steps < args.cosine_end:
        raise ValueError("需要满足 0 <= warmup_steps < cosine_end")

    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm 必须大于 0")

    # 训练和验证路径要么都提供，要么都不提供。
    if (args.train_data is None) != (args.valid_data is None):
        raise ValueError("请同时提供 train-data 和 valid-data")


def load_token_array(path, vocab_size, context_length):
    # mmap_mode="r"：按需读取，避免一次性把整个数组装进内存。
    # allow_pickle=False：这里需要普通数值数组，不需要 Python 对象。
    tokens = np.load(path, mmap_mode="r", allow_pickle=False)

    if not isinstance(tokens, np.ndarray):
        raise ValueError("数据必须是单个 .npy 数组")

    if tokens.ndim != 1:
        raise ValueError("token 数据必须是一维数组")

    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("token 数据必须使用整数 dtype")

    if len(tokens) <= context_length:
        raise ValueError("至少需要 context_length + 1 个 token")

    # 分块检查所有 ID，避免为了检查而复制整个大数组。
    chunk_size = 1_000_000

    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start:start + chunk_size]

        if chunk.min() < 0 or chunk.max() >= vocab_size:
            raise ValueError(
                f"{path} 中存在超出词表范围的 token ID"
            )

    return tokens


def prepare_datasets(args):
    if args.train_data is None:
        # 人工规律序列：0, 1, 2, ..., V-1, 0, 1, ...
        # % 是取余运算，用来让 ID 周期性回到 0。
        train_tokens = (
            np.arange(4096, dtype=np.int64) % args.vocab_size
        )
        valid_tokens = (
            np.arange(1024, dtype=np.int64) % args.vocab_size
        )

        if min(len(train_tokens), len(valid_tokens)) <= args.context_length:
            raise ValueError("人工数据长度不足，请减小 context_length")

        print("使用人工周期序列：仅用于检查训练流程。")
        return train_tokens, valid_tokens

    train_tokens = load_token_array(
        args.train_data,
        args.vocab_size,
        args.context_length,
    )
    valid_tokens = load_token_array(
        args.valid_data,
        args.vocab_size,
        args.context_length,
    )

    print("训练 token 数：", len(train_tokens))
    print("验证 token 数：", len(valid_tokens))

    return train_tokens, valid_tokens

def train_step(model, optimizer, x, y, args, iteration):
    # iteration 表示“目前已经完成多少次更新”。
    # 第一次更新之前 iteration = 0。
    learning_rate = get_lr_cosine_schedule(
        it=iteration,
        max_learning_rate=args.max_lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_steps,
        cosine_cycle_iters=args.cosine_end,
    )

    # 一个优化器可以有多个参数组。
    # 这里给每组设置同样的本步学习率。
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    # 清除上一轮梯度，避免梯度意外累积。
    optimizer.zero_grad()

    # x:      (batch_size, context_length)
    # logits: (batch_size, context_length, vocab_size)
    logits = model(x)

    # 你的交叉熵实现支持三维 logits 和二维 targets。
    # 返回整个 batch、所有位置的平均损失。
    loss = cross_entropy(logits, y)

    # 遇到 NaN 或 inf 时立即停止，而不是继续污染参数。
    if not torch.isfinite(loss).item():
        raise RuntimeError(
            f"iteration={iteration} 的 loss 不是有限值"
        )

    # 根据 loss 计算每个参数的梯度，写入 parameter.grad。
    loss.backward()

    # 这是调试检查：先确认梯度本身没有 NaN / inf。
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(
                    f"iteration={iteration} 出现非有限梯度"
                )

    # 对所有参数的梯度做全局裁剪。
    gradient_clipping(
        model.parameters(),
        max_l2_norm=args.max_grad_norm,
    )

    # 到这里才真正更新参数。
    optimizer.step()

    # item() 将标量张量转换为 Python 数字，方便打印和写日志。
    return loss.item(), learning_rate

def evaluate(model, dataset, args):
    # 保存进入函数前的模式。
    was_training = model.training

    # 你的 get_batch 使用 NumPy 随机采样。
    # 保存随机状态，避免验证取样改变后续训练取样的随机序列。
    numpy_state = np.random.get_state()

    try:
        # 每次验证使用同一组随机起点，方便比较损失变化。
        # 这是导师选择的调试方案，不是官方强制规则。
        np.random.seed(args.seed + 1)

        model.eval()
        total_loss = 0.0

        # 验证只计算结果，不创建用于反向传播的计算图。
        with torch.no_grad():
            for _ in range(args.eval_batches):
                x, y = get_batch(
                    dataset,
                    args.batch_size,
                    args.context_length,
                    args.device,
                )

                logits = model(x)
                loss = cross_entropy(logits, y)

                if not torch.isfinite(loss).item():
                    raise RuntimeError("验证损失不是有限值")

                total_loss += loss.item()

        # 每个验证 batch 的大小和长度相同，
        # 因此平均 batch loss 等于这些采样 token 的平均 loss。
        return total_loss / args.eval_batches

    finally:
        # 无论验证是否报错，都恢复原来的模式和随机状态。
        model.train(was_training)
        np.random.set_state(numpy_state)

def prepare_run_directory(args):
    run_dir = Path(args.run_dir)

    # 转成字典，方便保存 JSON。
    config = vars(args).copy()

    # 这些值允许恢复训练时变化：
    # steps：可以从训练到 50 步，改成训练到 100 步。
    # resume：仅表示本次从哪里恢复。
    # run_dir：输出位置单独处理。
    for name in ["steps", "resume", "run_dir"]:
        config.pop(name)

    # 将数据路径记录为绝对路径，减少相对路径引起的混淆。
    for name in ["train_data", "valid_data"]:
        if config[name] is not None:
            config[name] = str(Path(config[name]).resolve())

    config_path = run_dir / "config.json"

    if args.resume is None:
        # 新实验使用新目录，避免覆盖已有实验。
        if run_dir.exists():
            raise ValueError(
                f"{run_dir} 已存在：新实验请换一个 run-dir；"
                "恢复训练请提供 --resume"
            )

        run_dir.mkdir(parents=True)

        with config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

    else:
        if not config_path.exists():
            raise ValueError("恢复训练时，run-dir 中必须存在 config.json")

        checkpoint_path = Path(args.resume)

        if not checkpoint_path.is_file():
            raise ValueError("指定的检查点文件不存在")

        if checkpoint_path.resolve().parent != run_dir.resolve():
            raise ValueError("本入门脚本要求从同一 run-dir 内的检查点恢复")

        with config_path.open("r", encoding="utf-8") as file:
            previous_config = json.load(file)

        # 确保恢复时模型和调度等设置没有意外变化。
        if previous_config != config:
            raise ValueError(
                "恢复配置与 config.json 不一致。"
                "本脚本只允许改变 --steps，不自动改变模型或学习率方案。"
            )

    return run_dir


def write_log(log_path, record):
    # JSONL：每次追加一行，方便恢复训练时继续写。
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(record)

def main():
    args = parse_args()
    validate_args(args)

    # 固定随机种子，让从头启动的调试实验更容易比较。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_tokens, valid_tokens = prepare_datasets(args)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # iteration 始终表示“已完成的更新次数”。
    iteration = 0

    if args.resume is not None:
        iteration = load_checkpoint(
            src=args.resume,
            model=model,
            optimizer=optimizer,
        )

    if iteration >= args.steps:
        raise ValueError(
            f"检查点已完成 {iteration} 步，"
            f"--steps 必须大于 {iteration}"
        )

    run_dir = prepare_run_directory(args)
    log_path = run_dir / "metrics.jsonl"
    checkpoint_path = run_dir / "last.pt"

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("模型参数量：", parameter_count)
    print("开始时已完成步数：", iteration)
    print("本次目标总步数：", args.steps)

    # elapsed_seconds 是本次进程内的耗时。
    # 恢复训练后重新计时，不冒充从实验最开始累计的时间。
    start_time = time.perf_counter()

    initial_valid_loss = evaluate(model, valid_tokens, args)

    write_log(
        log_path,
        {
            "event": "start",
            "step": iteration,
            "resumed": args.resume is not None,
            "valid_loss": initial_valid_loss,
            "elapsed_seconds": time.perf_counter() - start_time,
        },
    )

    model.train()

    while iteration < args.steps:
        x, y = get_batch(
            train_tokens,
            args.batch_size,
            args.context_length,
            args.device,
        )

        # update_index 是本次参数更新使用的调度索引。
        update_index = iteration

        train_loss, learning_rate = train_step(
            model=model,
            optimizer=optimizer,
            x=x,
            y=y,
            args=args,
            iteration=update_index,
        )

        # optimizer.step() 已执行，因此完成次数增加 1。
        iteration += 1

        if (
            iteration == 1
            or iteration % args.log_every == 0
            or iteration == args.steps
        ):
            write_log(
                log_path,
                {
                    "event": "train",
                    "step": iteration,
                    "update_index": update_index,
                    "lr": learning_rate,
                    "train_loss": train_loss,
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
            )

        if (
            iteration % args.eval_every == 0
            or iteration == args.steps
        ):
            valid_loss = evaluate(model, valid_tokens, args)

            write_log(
                log_path,
                {
                    "event": "validation",
                    "step": iteration,
                    "valid_loss": valid_loss,
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
            )

        if (
            iteration % args.save_every == 0
            or iteration == args.steps
        ):
            # 每次覆盖这个实验目录中的“最近检查点”。
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                out=checkpoint_path,
            )

            write_log(
                log_path,
                {
                    "event": "checkpoint",
                    "step": iteration,
                    "path": str(checkpoint_path),
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
            )

    print("本次训练结束，已完成更新次数：", iteration)


if __name__ == "__main__":
    main()