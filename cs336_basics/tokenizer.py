import json
import regex
from collections.abc import Iterable, Iterator
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = (list(special_tokens) if special_tokens is not None else [])
        existing_tokens = set(self.vocab.values())
        next_id = max(self.vocab, default=-1) + 1

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in existing_tokens:
                self.vocab[next_id] = token_bytes
                existing_tokens.add(token_bytes)
                next_id += 1
        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

    def _encode_piece(self, piece: str) -> list[int]:#把一个普通文字片段编码成 tokens_ids
        # 1. 将片段转换成单字节 token 序列
        tokens = [
            bytes([value])
            for value in piece.encode("utf-8")
        ]

        while len(tokens) > 1:
            # 2. 找到当前可用且排名最小的合并规则
            candidates = [
                (left, right)
                for left, right in zip(tokens, tokens[1:])
                if (left, right) in self.merge_ranks
            ]

            if not candidates:
                break

            best_pair = min(
                candidates,
                key=lambda pair: self.merge_ranks[pair],
            )

            # 3. 从左到右，不重叠地合并该 pair
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        # 4. 将最终的字节 token 转换成 ID
        return [self.token_to_id[token] for token in tokens]

    def encode_iterable(#逐段编码可迭代文本
        self, iterable: Iterable[str]
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def encode(self, text: str) -> list[int]:#把一整段文字编码成 ID
        special_set = set(self.special_tokens)

        if special_set:
            special_pattern = "|".join(
                regex.escape(token)
                for token in sorted(
                    special_set, key=len, reverse=True
                )
            )

            # 捕获组让 split 保留匹配到的特殊 token
            parts = regex.split(f"({special_pattern})", text)
        else:
            parts = [text]

        ids = []

        for part in parts:
            if not part:
                continue

            if part in special_set:
                ids.append(
                    self.token_to_id[part.encode("utf-8")]
                )
            else:
                for match in regex.finditer(PAT, part):
                    ids.extend(
                        self._encode_piece(match.group())
                    )

        return ids

    def decode(self,ids:list[int]) -> str:
        pieces = []
        for token_id in ids:
            pieces.append(self.vocab[token_id])
        all_bytes = b"".join(pieces)
        return all_bytes.decode("utf-8",errors="replace")

    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None,):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            saved_vocab = json.load(f)

        with open(merges_filepath, "r", encoding="utf-8") as f:
            saved_merges = json.load(f)

        vocab = {
            int(token_id): bytes(byte_values)
            for token_id, byte_values in saved_vocab.items()
        }

        merges = [
            (bytes(left), bytes(right))
            for left, right in saved_merges
        ]

        return cls(vocab, merges, special_tokens)



if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        r"D:\aaa_code\assignment1-basics\data\tinystories_vocab_10000.json",
        r"D:\aaa_code\assignment1-basics\data\tinystories_merges_10000.json",
        special_tokens=["<|endoftext|>"],
    )
    text = "Hello, 世界!<|endoftext|>Hello again."
    ids = tokenizer.encode(text)

    special_id = tokenizer.token_to_id[b"<|endoftext|>"]

    parts = ["Hello, 世界!\n", "Another story.\n"]

    stream = tokenizer.encode_iterable(parts)
    ids = list(stream)

    expected = [
        token_id
        for part in parts
        for token_id in tokenizer.encode(part)
    ]

    print("逐段编码一致：", ids == expected)
    print("还原文本一致：",
          tokenizer.decode(ids) == "".join(parts))
    print("空输入：", list(tokenizer.encode_iterable([])))