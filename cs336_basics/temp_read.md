vocab = {
    # 原来的 256 个单字节 token 都还在
    # ……
    256: b"<|endoftext|>",
    257: b"ow",
    258: b"low",
    259: b" low",
    260: b"er",
    261: b" lower",
}

merges = [
    (b"o", b"w"),
    (b"l", b"ow"),
    (b" ", b"low"),
    (b"e", b"r"),
    (b" low", b"er"),
]

