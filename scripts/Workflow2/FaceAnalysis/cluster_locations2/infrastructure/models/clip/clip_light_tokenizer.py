import html
import json
from pathlib import Path

import numpy as np
import regex as re


def _bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    char_values = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            char_values.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, [chr(value) for value in char_values]))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs = set()
    previous = word[0]
    for char in word[1:]:
        pairs.add((previous, char))
        previous = char
    return pairs


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _basic_clean(text: str) -> str:
    try:
        import ftfy
    except ImportError:
        ftfy = None

    if ftfy is not None:
        text = ftfy.fix_text(text)
    return html.unescape(html.unescape(text)).strip()


class ClipLightTokenizer:
    model_max_length = 77

    def __init__(self, tokenizer_path: Path):
        self.tokenizer_path = Path(tokenizer_path)
        self.vocab_path = self.tokenizer_path / "vocab.json"
        self.merges_path = self.tokenizer_path / "merges.txt"

        if not self.vocab_path.exists() or not self.merges_path.exists():
            raise FileNotFoundError(
                f"CLIP tokenizer files not found: {self.vocab_path}, {self.merges_path}"
            )

        with self.vocab_path.open("r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        merges = self.merges_path.read_text(encoding="utf-8").splitlines()
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merge_pairs = [tuple(merge.split()) for merge in merges if merge.strip()]
        self.bpe_ranks = dict(zip(merge_pairs, range(len(merge_pairs))))

        self.byte_encoder = _bytes_to_unicode()
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pattern = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"
            r"[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+",
            re.IGNORECASE,
        )

        self.bos_token_id = self.encoder["<|startoftext|>"]
        self.eos_token_id = self.encoder["<|endoftext|>"]
        self.pad_token_id = self.eos_token_id

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                    new_word.extend(word[index:next_index])
                    index = next_index
                except ValueError:
                    new_word.extend(word[index:])
                    break

                if (
                    word[index] == first
                    and index < len(word) - 1
                    and word[index + 1] == second
                ):
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        value = " ".join(word)
        self.cache[token] = value
        return value

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        text = _whitespace_clean(_basic_clean(text)).lower()
        for token in re.findall(self.pattern, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self._bpe(token).split(" ")
            )
        return bpe_tokens

    def tokenize(self, texts: list[str]):
        input_rows = []
        mask_rows = []

        for text in texts:
            tokens = self.encode(str(text))[: self.model_max_length - 2]
            row = [self.bos_token_id, *tokens, self.eos_token_id]
            mask = [1] * len(row)

            padding = self.model_max_length - len(row)
            if padding > 0:
                row.extend([self.pad_token_id] * padding)
                mask.extend([0] * padding)

            input_rows.append(row)
            mask_rows.append(mask)

        return (
            np.array(input_rows, dtype=np.int64),
            np.array(mask_rows, dtype=np.int64),
        )
