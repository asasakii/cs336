import regex
import os
from typing import BinaryIO
from collections import Counter,defaultdict

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def train_bpe(input_path: str,vocab_size: int,special_tokens:list[str]) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    #初始化256个单字节token
    vocab = {i:bytes([i]) for i in range(256)}
    #例如 ID 65  → b"A"

    #添加特殊token  输入的特殊tokens用字节序列表示 也就是UTF-8 然后加入vocab
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    #合并记录
    merges = []

    #读取训练文本
    with open(input_path,"r",encoding="utf-8") as f:
        text = f.read()

    #如果存在特殊token，需要按特殊token split文档
    if special_tokens:
        special_pattern = "|".join(regex.escape(token) for token in sorted(special_tokens,key=len,reverse=True))
        documents = regex.split(special_pattern,text)
        #special_pattern:特殊token或特殊token或特殊token...
    else:
        documents = [text] #列表，按照特殊符号分隔

    #正则预分词
    #Counter 预测分词 “字典”
    pretoken_counts  = Counter()
    for document in documents:
        #document: low low  match: low  按照PAT的规则得到 match
        for match in regex.finditer(PAT,document): #匹配对象，包含匹配文本match.group()、位置match.span()
            piece = match.group()
            pretoken_counts[piece] +=1
    #这一步得到{"low": 1," low": 2,"lower": 1,}

    #转化为单字节token序列
    byte_counts = Counter()

    for piece,frequency in pretoken_counts.items():
        sequence = tuple(bytes([value]) for value in piece.encode("utf-8"))
        byte_counts[sequence] +=frequency
    # {
    #     (b'l', b'o', b'w'): 1,
    #     (b' ', b'l', b'o', b'w'): 2,
    #     (b'l', b'o', b'w', b'e', b'r'): 1,
    # }

    #统计相邻 token对
    pair_counts = Counter()
    pair_to_sequences = defaultdict(set)

    for sequence,frequency in byte_counts.items():
        local_pairs = Counter(zip(sequence,sequence[1:]))
        for pair,occurences in local_pairs.items():
            pair_counts[pair] += occurences * frequency
            pair_to_sequences[pair].add(sequence)

    #接下来只处理受影响的片段也就是 pair_to_sequence
    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(pair_counts,key= lambda pair:(pair_counts[pair],pair))

        left,right =best_pair
        merged_token = left + right
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

        #固定本轮需要修改的片段
        affected_sequences = list(pair_to_sequences[best_pair])

        for sequence in affected_sequences:
            frequency = byte_counts.pop(sequence)

            old_pairs = Counter(zip(sequence,sequence[1:]))
            # 1. 移除旧片段对计数和索引的贡献
            for pair,occurences in old_pairs.items():
                pair_counts[pair] -= occurences * frequency
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                pair_to_sequences[pair].discard(sequence)
                if not pair_to_sequences[pair]:
                    del pair_to_sequences[pair]
            # 2. 合并当前片段
            new_sequence = []
            i = 0
            while i < len(sequence):
                if (
                        i + 1 < len(sequence)
                        and (sequence[i], sequence[i + 1]) == best_pair
                ):
                    new_sequence.append(merged_token)
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1
            new_sequence = tuple(new_sequence)
            assert b"".join(sequence) == b"".join(new_sequence)
            byte_counts[new_sequence] += frequency
            # 3. 加入新片段对计数和索引的贡献
            new_pair = Counter(zip(new_sequence,new_sequence[1:]))
            for pair,occurences in new_pair.items():
                pair_counts[pair] += occurences * frequency
                pair_to_sequences[pair].add(new_sequence)

    return vocab,merges

if __name__ == "__main__":
    import time

    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=r"D:\aaa_code\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.perf_counter() - start
    longest_token = max(vocab.values(), key=len)

    print("训练耗时（秒）：", round(elapsed, 2))
    print("词表大小：", len(vocab))
    print("合并次数：", len(merges))
    print("最长 token：", longest_token)
    print("最长 token 的字节数：", len(longest_token))
    ordinary_tokens = [
        token_bytes
        for token_bytes in vocab.values()
        if token_bytes != b"<|endoftext|>"
    ]

    for token in sorted(ordinary_tokens, key=len, reverse=True)[:5]:
        print("字节数：", len(token), "token：", token)
