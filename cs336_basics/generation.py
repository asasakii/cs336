import argparse
import json
import math
from pathlib import Path

import torch

from cs336_basics.activations import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM


# 根据当前文件的位置定位仓库，不依赖 PyCharm 的工作目录。
ROOT = Path(__file__).resolve().parents[1]
END_TOKEN = "<|endoftext|>"


def parse_args():
    parser = argparse.ArgumentParser(
        description="TinyStories 交互式故事续写"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "tinystories_lr_1e-3_b8_01",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=ROOT / "data" / "tinystories_vocab_10000.json",
    )
    parser.add_argument(
        "--merges",
        type=Path,
        default=ROOT / "data" / "tinystories_merges_10000.json",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if (
        not math.isfinite(args.temperature)
        or args.temperature < 0
    ):
        parser.error("temperature 必须是有限的非负数")

    if not 0 < args.top_p <= 1:
        parser.error("top-p 必须满足 0 < top-p <= 1")

    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens 必须大于 0")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前解释器无法使用 CUDA")

    return args


def load_for_generation(args):
    config_path = args.run_dir / "config.json"
    checkpoint_path = args.run_dir / "last.pt"

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=[END_TOKEN],
    )

    # 模型输出的每个 ID 必须能由这份 tokenizer 解码。
    expected_ids = set(range(config["vocab_size"]))
    if set(tokenizer.vocab) != expected_ids:
        raise ValueError("Tokenizer 的 ID 集合与模型词表大小不匹配")

    eos_id = tokenizer.token_to_id[END_TOKEN.encode("utf-8")]

    # 使用训练时的结构配置，但设备由本次命令指定。
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=args.device,
        dtype=torch.float32,
    )

    # 先读到 CPU，避免把不用于生成的优化器状态占到显存中。
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # 生成只需恢复模型权重，不需要创建或恢复优化器。
    model.load_state_dict(checkpoint["model"], strict=True)
    iteration = checkpoint["iteration"]
    del checkpoint

    model.eval()
    return model, tokenizer, eos_id, iteration


def sample_next_token(logits, temperature, top_p):
    """
    logits: (vocab_size,)
    返回值: 一个 Python int，表示采样得到的 token ID。
    """
    logits = logits.float()

    if not torch.isfinite(logits).all().item():
        raise RuntimeError("模型输出包含 NaN 或 inf")

    # 温度为 0 时使用 greedy decoding。
    if temperature == 0:
        return int(torch.argmax(logits).item())

    # 先减最大值，保持概率不变，同时提高数值稳定性。
    scaled_logits = (logits - logits.max()) / temperature
    probabilities = softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        # sorted_ids 保存排序前的原始词表 ID。
        sorted_probs, sorted_ids = torch.sort(
            probabilities, descending=True
        )

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 当前 token 之前的累计概率。
        # 若它仍小于 top_p，就必须保留当前 token。
        # 这样会保留首次使累计概率达到阈值的那个 token。
        previous_mass = torch.cat(
            [
                torch.zeros_like(cumulative_probs[:1]),
                cumulative_probs[:-1],
            ],
            dim=-1,
        )
        keep = previous_mass < top_p
        keep[0] = True

        filtered_probs = torch.where(
            keep,
            sorted_probs,
            torch.zeros_like(sorted_probs),
        )
        filtered_probs = filtered_probs / filtered_probs.sum()

        # 这里抽到的是“排序后的位置”。
        sampled_position = torch.multinomial(
            filtered_probs, num_samples=1
        )

        # 再把排序后的位置映射回原始词表 ID。
        return int(sorted_ids[sampled_position].item())

    sampled_id = torch.multinomial(
        probabilities, num_samples=1
    )
    return int(sampled_id.item())
#根据top_p截断，累计概率小于top_p

@torch.inference_mode() #调用这个函数时，让整个函数体在推理模式下执行，减少自动求导相关记录与开销。
def generate(
    model,
    prompt_ids,
    eos_id,
    max_new_tokens,
    temperature,
    top_p,
):
    """
    prompt_ids: 非空的 Python ID 列表。
    返回:
        all_ids: prompt 加上新增内容，不包含结束 token。
        sampled_count: 本轮采样次数，包含结束 token（若生成）。
        stop_reason: 停止原因。
    """
    if not prompt_ids:
        raise ValueError("输入编码后不能为空")

    # 复制列表，避免修改调用者保存的原始 prompt。
    all_ids = list(prompt_ids)
    device = next(model.parameters()).device

    for index in range(max_new_tokens):
        # 仅用最近 context_length 个 token 预测。
        context_ids = all_ids[-model.context_length:]

        # 外层方括号增加 batch 维：
        # Python 列表 (T,) -> 张量 (1, T)。
        x = torch.tensor(
            [context_ids],
            dtype=torch.long,
            device=device,
        )

        # logits: (1, T, vocab_size)
        logits = model(x)

        # 0：唯一的 batch；
        # -1：最后一个输入位置；
        # :：保留词表中的全部候选。
        # next_logits: (vocab_size,)
        next_logits = logits[0, -1, :]

        next_id = sample_next_token(
            logits=next_logits,
            temperature=temperature,
            top_p=top_p,
        )

        if next_id == eos_id:
            return all_ids, index + 1, "生成了结束 token"

        all_ids.append(next_id)

    return all_ids, max_new_tokens, "达到最大新增 token 数"


def main():
    args = parse_args()

    # 只在程序启动时设一次 seed。
    # 每轮不会重置随机数，因此可生成不同续写。
    torch.manual_seed(args.seed)

    model, tokenizer, eos_id, iteration = load_for_generation(args)

    print(f"已加载 checkpoint：step={iteration}")
    print(f"设备：{next(model.parameters()).device}")
    print(f"上下文窗口：{model.context_length}")
    print(f"结束 token ID：{eos_id}")
    print(
        f"temperature={args.temperature}, "
        f"top_p={args.top_p}, "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print("\n输入英文故事开头，开始一段新故事。")
    print("/continue：继续上一段故事")
    print("/quit：退出")
    print("这是故事续写模型，不是指令问答模型。")

    history_ids = None
    story_ended = False

    while True:
        try:
            text = input("\n故事开头> ")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        command = text.strip()

        if command == "/quit":
            print("已退出。")
            break

        if command == "/continue":
            if history_ids is None:
                print("请先输入一个故事开头。")
                continue

            if story_ended:
                print("上一段已生成结束 token，请输入新的故事开头。")
                continue

            prompt_ids = history_ids

        else:
            if not command:
                print("输入不能为空。")
                continue

            prompt_ids = tokenizer.encode(text)

            if eos_id in prompt_ids:
                print("故事开头请不要包含 <|endoftext|>。")
                continue

        if len(prompt_ids) > model.context_length:
            print(
                f"当前故事有 {len(prompt_ids)} 个 token，"
                f"本轮从最近 {model.context_length} 个 token 开始预测。"
            )

        print("正在续写……")

        all_ids, sampled_count, stop_reason = generate(
            model=model,
            prompt_ids=prompt_ids,
            eos_id=eos_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        visible_new_tokens = len(all_ids) - len(prompt_ids)
        history_ids = all_ids
        story_ended = stop_reason == "生成了结束 token"

        # 将全部 bytes 拼起来后再解码，避免逐 token 解码破坏 UTF-8。
        print("\n完整故事：")
        print(tokenizer.decode(all_ids))
        print(
            f"\n[本轮新增正文 token：{visible_new_tokens}；"
            f"采样次数：{sampled_count}；"
            f"停止原因：{stop_reason}]"
        )


if __name__ == "__main__":
    main()