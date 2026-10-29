import argparse
import hashlib
import json
import shutil
import time
from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


# 当前脚本固定保留这个文档分隔符。
SPECIAL_TOKEN = "<|endoftext|>"

# < 表示小端字节序，u 表示无符号整数，2 表示每项 2 字节。
TOKEN_DTYPE = np.dtype("<u2")


def parse_args():
    """读取 PowerShell 中传入的参数。"""
    parser = argparse.ArgumentParser(
        description="按完整文档边界，将 TinyStories 文本编码为 .npy"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)

    parser.add_argument("--vocab-size", type=int, default=10000)

    # 文本模式下的读取数量是“字符数”，不是字节数。
    parser.add_argument("--chunk-chars", type=int, default=65536)

    # 不提供时处理完整文件。
    # 提供 3 时只处理前三个文档段，用于小规模试运行。
    parser.add_argument("--max-documents", type=int, default=None)

    # 每完成这么多文档段，打印一次进度。
    parser.add_argument("--log-every", type=int, default=1000)

    return parser.parse_args()

#读取文件的全部字节，计算一个 SHA-256 摘要字符串。每次只读取一块，避免把整个大文件一次装进内存。
def sha256_file(path):
    """分块计算文件摘要，不把整个文件读入内存。"""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()#根据全部原始数据计算出来的一小段固定长度的值，用来代表这份数据。


def iter_text_chunks(file, chunk_chars):
    """从已经打开的文本文件中，逐次读取字符块。"""
    while True:
        chunk = file.read(chunk_chars)

        if chunk == "":
            # 空字符串表示文件已经结束。
            break

        yield chunk


def iter_documents(chunks, separator=SPECIAL_TOKEN):
    """
    将任意读取块拼成完整文档段。

    每个正常文档段包含末尾的 separator。
    文件结束后，未带 separator 的尾部文本也会被保留。
    """
    pending = ""

    for chunk in chunks:
        pending += chunk

        # start 表示 pending 中尚未交出的文本起点。
        start = 0

        while True:
            position = pending.find(separator, start)

            if position == -1:
                # 当前剩余部分没有完整分隔符。
                break

            end = position + len(separator)

            # 包含原有分隔符，不删除正文空白。
            document = pending[start:end]
            start = end

            yield document

        # 只保留尚未结束的文档或分隔符片段。
        pending = pending[start:]

    # chunks 已经耗尽，此时才处理最终尾部。
    # 不能使用 pending.strip()，否则会丢掉纯空白尾部。
    if pending:
        yield pending

def format_duration(seconds):
    """将秒数转换为 时:分:秒；未知时间显示 --:--:--。"""
    if seconds is None:
        return "--:--:--"

    # round：四舍五入。
    # int：转成整数，避免显示很多小数位。
    # max：避免由于计时误差出现负数。
    seconds = max(0, int(round(seconds)))

    # divmod(a, b) 同时返回“整数商”和“余数”。
    # 例如 divmod(3665, 3600) 得到 (1, 65)。
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # :02d 表示用至少两位整数显示，不足两位时在前面补 0。
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_encoding_progress(
    completed,
    total,
    start_time,
    token_count,
    finished=False,
):
    """
    在同一行显示编码进度。

    completed、total 必须使用相同单位：
    - 完整文件：UTF-8 字节数。
    - 限定文档数：文档段数。

    finished=True 表示编码循环已结束，打印后换行。
    """
    elapsed = time.perf_counter() - start_time

    if total > 0:
        fraction = completed / total
    else:
        fraction = 0.0

    # 将显示比例限制在 0～1，防止进度条长度越界。
    fraction = min(1.0, max(0.0, fraction))

    # 进度条总共使用 20 个字符。
    width = 20
    filled = int(width * fraction)

    # 字符串乘整数表示重复，例如 "#" * 3 得到 "###"。
    bar = "#" * filled + "-" * (width - filled)

    if finished:
        remaining = 0.0
    elif completed > 0 and elapsed > 0:
        speed = completed / elapsed
        remaining = max(0, total - completed) / speed
    else:
        # 刚开始还没有完成任何文档，暂时无法估计速度。
        remaining = None

    message = (
        f"编码 [{bar}] "
        f"{fraction:6.2%} | "
        f"token={token_count:,} | "
        f"已用 {format_duration(elapsed)} | "
        f"剩余约 {format_duration(remaining)}"
    )

    # \r：将光标移回当前行开头。
    # end=""：本次不换行，下次打印会更新这一行。
    # ljust(120)：在末尾补空格，清除上一次较长文本的残留。
    # flush=True：立即显示，不等待输出缓冲区积满。
    print(
        "\r" + message.ljust(120),
        end="\n" if finished else "",
        flush=True,
    )

def encode_to_raw(
    source_path,
    raw_path,
    tokenizer,
    vocab_size,
    chunk_chars,
    max_documents,
    log_every,
):
    """
    第一阶段：
    每次编码一个文档段，将 ID 连续写入原始二进制文件。

    返回累计统计，不返回全量 ID 列表。
    """
    special_id = tokenizer.token_to_id[
        SPECIAL_TOKEN.encode("utf-8")
    ]

    total_documents = 0
    total_tokens = 0
    total_characters = 0
    total_text_bytes = 0

    source_eot_count = 0
    encoded_eot_count = 0

    min_id = None
    max_id = None

    # 记录实际编码文本和 ID 字节的摘要。
    text_digest = hashlib.sha256()
    ids_digest = hashlib.sha256()

    # 完整运行按原文 UTF-8 字节数计算进度。
    # 小检查按请求处理的文档段数计算进度。
    progress_total = (
        source_path.stat().st_size
        if max_documents is None
        else max_documents
    )

    start_time = time.perf_counter()

    # 用于控制刷新频率，避免每编码一个小文档都刷新终端。
    last_progress_time = start_time

    print_encoding_progress(
        completed=0,
        total=progress_total,
        start_time=start_time,
        token_count=0,
    )

    # newline="" 保留原始换行形式。
    with source_path.open(
        "r", encoding="utf-8", newline=""
    ) as source:
        # xb：只允许新建，已有同名文件时立即报错。
        with raw_path.open("xb") as raw_file:
            chunks = iter_text_chunks(source, chunk_chars)
            documents = iter_documents(chunks)

            if max_documents is not None:
                # 最多消费指定数量的文档段，不遍历整个文件。
                documents = islice(documents, max_documents)

            for document in documents:
                # 关键：编码完整文档段，而不是任意读取块。
                ids = tokenizer.encode(document)

                # 当前 document 非空，正常情况下不应编码为空。
                if not ids:
                    raise ValueError("非空文档段被编码为空")

                current_min = min(ids)
                current_max = max(ids)

                # 必须在转换成 uint16 之前检查范围。
                if current_min < 0 or current_max >= vocab_size:
                    raise ValueError(
                        "发现非法 ID："
                        f"min={current_min}, max={current_max}"
                    )

                # 检查当前文本是否能由这些 ID 完整还原。
                # 这会增加一些运行时间，但有助于检查正式数据准备。
                if tokenizer.decode(ids) != document:
                    raise ValueError(
                        f"第 {total_documents + 1} 个文档段解码不一致"
                    )

                text_eot = document.count(SPECIAL_TOKEN)
                ids_eot = ids.count(special_id)

                if text_eot != ids_eot:
                    raise ValueError(
                        f"第 {total_documents + 1} 个文档段"
                        "的特殊分隔符数量不一致"
                    )

                # 当前文档的 ID 转成紧凑数组。
                # 不累计到整个训练集的大列表中。
                array = np.asarray(ids, dtype=TOKEN_DTYPE)
                payload = array.tobytes()

                written = raw_file.write(payload)

                if written != len(payload):
                    raise OSError("本次写入的字节数不足")

                total_documents += 1
                total_tokens += len(ids)
                total_characters += len(document)

                document_bytes = document.encode("utf-8")
                total_text_bytes += len(document_bytes)

                source_eot_count += text_eot
                encoded_eot_count += ids_eot

                text_digest.update(document_bytes)
                ids_digest.update(payload)

                if min_id is None:
                    min_id = current_min
                    max_id = current_max
                else:
                    min_id = min(min_id, current_min)
                    max_id = max(max_id, current_max)

                now = time.perf_counter()

                # 满足任意一个条件就刷新：
                # 1. 距离上次刷新已经过去至少 1 秒；
                # 2. 完成了 log_every 个文档段。
                #
                # 保留 --log-every 参数：
                # 小检查使用 --log-every 1 时，每段都能看到更新。
                if (
                    now - last_progress_time >= 1.0
                    or total_documents % log_every == 0
                ):
                    completed = (
                        total_text_bytes
                        if max_documents is None
                        else total_documents
                    )

                    print_encoding_progress(
                        completed=completed,
                        total=progress_total,
                        start_time=start_time,
                        token_count=total_tokens,
                    )
                    # 本次刷新完成，重新记录刷新时间。
                    # 下一次用当前时间减去它，判断是否又过去了 1 秒。
                    last_progress_time = now

    completed = (
        total_text_bytes
        if max_documents is None
        else total_documents
    )

    print_encoding_progress(
        completed=completed,
        total=progress_total,
        start_time=start_time,
        token_count=total_tokens,
        finished=True,
    )

    # max_documents 表示“最多处理多少段”。
    # 如果源文件本身没有这么多段，实际数量会小于请求数量。
    # 不强行把这种情况显示为 100%，而是明确说明提前到达文件末尾。
    if (
            max_documents is not None
            and total_documents < max_documents
    ):
        print(
            f"已到达文件末尾：请求最多 {max_documents} 段，"
            f"实际处理 {total_documents} 段。",
            flush=True,
        )

    if total_tokens == 0:
        raise ValueError("没有生成任何 token，请检查输入文件")

    expected_bytes = total_tokens * TOKEN_DTYPE.itemsize
    actual_bytes = raw_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"原始 ID 文件大小错误："
            f"预期 {expected_bytes}，实际 {actual_bytes}"
        )

    return {
        "document_segments": total_documents,
        "token_count": total_tokens,
        "character_count": total_characters,
        "encoded_text_utf8_bytes": total_text_bytes,
        "min_id": min_id,
        "max_id": max_id,
        "source_eot_count": source_eot_count,
        "encoded_eot_count": encoded_eot_count,
        "encoded_text_sha256": text_digest.hexdigest(),
        "id_payload_sha256": ids_digest.hexdigest(),
        "encoding_seconds": time.perf_counter() - start_time,
    }


def build_npy(raw_path, partial_path, token_count):
    """
    第二阶段：
    写正规的 .npy 文件头，再复制已经编码好的 ID 字节。
    不重新运行 tokenizer。
    """
    header = {
        # 描述数组元素的存储类型和字节序。
        "descr": TOKEN_DTYPE.str,#小端 unit16 <u2

        # 一维数组按普通连续顺序存储。
        "fortran_order": False,

        # (token_count,) 是一维形状元组。
        # 逗号不能省略，否则不是单元素元组。
        "shape": (token_count,),
    }

    with partial_path.open("xb") as output:
        # 由 NumPy 写入规范的 .npy 头部。
        np.lib.format.write_array_header_1_0(output, header)

        with raw_path.open("rb") as raw_file:
            # 每次最多复制约 8 MiB，不把整个文件读入内存。
            shutil.copyfileobj(
                raw_file,
                output,
                length=8 * 1024 * 1024,
            )


def verify_npy(raw_path, partial_path, token_count):
    """
    重新打开 .npy，分块比较它与原始 ID 文件。
    这一步检查的是全部 ID，而不只是开头几个。
    """
    loaded = np.load(
        partial_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    try:
        if loaded.shape != (token_count,):
            raise ValueError(
                f".npy 形状错误：{loaded.shape}"
            )

        if loaded.dtype != TOKEN_DTYPE:
            raise ValueError(
                f".npy dtype 错误：{loaded.dtype}"
            )

        chunk_tokens = 1_000_000

        with raw_path.open("rb") as raw_file:
            for start in range(0, token_count, chunk_tokens):
                end = min(start + chunk_tokens, token_count)
                count = end - start

                raw_bytes = raw_file.read(
                    count * TOKEN_DTYPE.itemsize
                )

                if len(raw_bytes) != count * TOKEN_DTYPE.itemsize:
                    raise ValueError("原始 ID 文件提前结束")

                expected = np.frombuffer(
                    raw_bytes,
                    dtype=TOKEN_DTYPE,
                )

                if not np.array_equal(loaded[start:end], expected):
                    raise ValueError(
                        f"读回 ID 不一致，检查区间 [{start}, {end})"
                    )

            if raw_file.read(1) != b"":
                raise ValueError("原始 ID 文件存在多余字节")

    finally:
        # Windows 上重命名前，需要释放文件的内存映射引用。
        del loaded


def main():
    args = parse_args()

    source_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    vocab_path = Path(args.vocab).resolve()
    merges_path = Path(args.merges).resolve()

    if args.chunk_chars <= 0:
        raise ValueError("chunk-chars 必须大于 0")

    if args.log_every <= 0:
        raise ValueError("log-every 必须大于 0")

    if args.max_documents is not None and args.max_documents <= 0:
        raise ValueError("max-documents 必须大于 0")

    if not 1 <= args.vocab_size <= 65536:
        raise ValueError("uint16 要求 vocab-size 在 1～65536 之间")

    if output_path.suffix.lower() != ".npy":
        raise ValueError("输出文件应使用 .npy 扩展名")

    for path in [source_path, vocab_path, merges_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"输出目录不存在：{output_path.parent}"
        )

    # 所有中间文件放在输出文件旁边。
    raw_path = output_path.with_name(
        output_path.name + ".ids.tmp"
    )
    partial_path = output_path.with_name(
        output_path.name + ".partial"
    )
    metadata_path = output_path.with_name(
        output_path.name + ".json"
    )

    # 不覆盖旧结果，也不自动删除失败运行留下的文件。
    for path in [
        output_path,
        raw_path,
        partial_path,
        metadata_path,
    ]:
        if path.exists():
            raise FileExistsError(
                f"文件已存在，请先检查或改用新的输出名称：{path}"
            )

    # 记录输入大小和修改时间，结束时检查文件是否发生变化。
    source_before = source_path.stat()

    # 先记录词表文件摘要，后续确认来源时可以使用。
    vocab_sha256 = sha256_file(vocab_path)
    merges_sha256 = sha256_file(merges_path)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=[SPECIAL_TOKEN],
    )

    # 当前训练模型假设 ID 连续，范围为 0～vocab_size-1。
    expected_vocab_ids = set(range(args.vocab_size))

    if set(tokenizer.vocab.keys()) != expected_vocab_ids:
        raise ValueError(
            "实际词表 ID 与指定 vocab-size 不匹配"
        )

    special_id = tokenizer.token_to_id[
        SPECIAL_TOKEN.encode("utf-8")
    ]

    total_start = time.perf_counter()

    print("输入：", source_path)
    print("输出：", output_path)
    print("词表大小：", len(tokenizer.vocab))
    print("特殊 token ID：", special_id)
    print(
        "处理范围：",
        "完整文件"
        if args.max_documents is None
        else f"最多 {args.max_documents} 个文档段",
    )

    try:
        stats = encode_to_raw(
            source_path=source_path,
            raw_path=raw_path,
            tokenizer=tokenizer,
            vocab_size=args.vocab_size,
            chunk_chars=args.chunk_chars,
            max_documents=args.max_documents,
            log_every=args.log_every,
        )

        source_after = source_path.stat()

        if (
            source_before.st_size != source_after.st_size
            or source_before.st_mtime_ns != source_after.st_mtime_ns
        ):
            raise RuntimeError("编码期间输入文件发生变化")

        if args.max_documents is None:
            # UTF-8 严格读取、保留换行，再编码回 UTF-8，
            # 完整文件处理时字节计数应与输入文件大小一致。
            if stats["encoded_text_utf8_bytes"] != source_before.st_size:
                raise ValueError("完整文本的字节计数与源文件不一致")

        print("编码结束，开始构建 .npy 文件。", flush=True)

        build_npy(
            raw_path,
            partial_path,
            stats["token_count"],
        )

        print("开始分块核对全部 ID。", flush=True)

        verify_npy(
            raw_path,
            partial_path,
            stats["token_count"],
        )

        # 验证成功之后，再使用最终文件名。
        # 当前目标环境是 Windows，同名目标存在时 rename 会报错。
        if output_path.exists():
            raise FileExistsError(output_path)

        partial_path.rename(output_path)

        metadata = {
            "source_path": str(source_path),
            "source_file_bytes": source_before.st_size,
            "output_path": str(output_path),
            "vocab_path": str(vocab_path),
            "merges_path": str(merges_path),
            "vocab_sha256": vocab_sha256,
            "merges_sha256": merges_sha256,
            "vocab_size": args.vocab_size,
            "special_token": SPECIAL_TOKEN,
            "special_token_id": special_id,
            "dtype": TOKEN_DTYPE.str,
            "chunk_chars": args.chunk_chars,
            "max_documents": args.max_documents,
            "full_file_run": args.max_documents is None,
            "roundtrip_checked_for_every_segment": True,
            "raw_to_npy_all_ids_equal": True,
            "npy_file_bytes": output_path.stat().st_size,
            **stats,
        }

        metadata["total_seconds"] = (
            time.perf_counter() - total_start
        )

        # x 模式同样禁止覆盖已有记录。
        with metadata_path.open("x", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # 最终数组和记录都已生成，再删除本次创建的原始临时 ID 文件。
        raw_path.unlink()

        print("完成。")
        print("文档段数：", stats["document_segments"])
        print("token 数：", stats["token_count"])
        print("dtype：", TOKEN_DTYPE)
        print("ID 范围：", stats["min_id"], stats["max_id"])
        print(
            "分隔符数量：",
            stats["source_eot_count"],
            stats["encoded_eot_count"],
        )
        print("全部 ID 读回一致：True")
        print("输出文件：", output_path)
        print("编码记录：", metadata_path)
        print(
            "总耗时秒数：",
            round(metadata["total_seconds"], 2),
        )

    except Exception:
        # 如果报错时进度条还占着当前行，先换行，
        # 避免错误说明紧接在进度条后面。
        print(flush=True)
        # 失败时不假装完成，也不自动删除可能有助于排查的文件。
        print(
            "编码未正常完成。已生成的文件会保留，"
            "请先检查报错，不要把 .partial 当成训练数据。",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()