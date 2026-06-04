"""Decode the local GPT-2-tokenized wikitext (train.bin/val.bin, uint16) back to raw
UTF-8 bytes, so we can train a cheap byte-level LM (vocab 256). The non-adiabatic-lag
phenomenon is an optimizer effect, independent of tokenization; byte-level is ~10x faster
on the output head than 50k-vocab and needs no network. Saves data/wiki_train.u8 / wiki_val.u8.
"""
import os, numpy as np, tiktoken

SRC = r"C:\Users\21100\Desktop\清理文件\llm_pretrain\data\wikitext"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUT, exist_ok=True)
N_TRAIN_TOK = 40_000_000   # ~40M tokens -> ~160MB text (plenty for ~1 epoch)

enc = tiktoken.get_encoding("gpt2")


def decode_to_bytes(bin_path, n_tok, out_path):
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n = min(n_tok, len(arr)) if n_tok else len(arr)
    toks = np.asarray(arr[:n]).astype(int).tolist()
    text = enc.decode(toks)
    b = text.encode("utf-8", errors="replace")
    with open(out_path, "wb") as f:
        f.write(b)
    print(f"{os.path.basename(bin_path)}: {n} tok -> {len(b)/1e6:.1f}MB bytes -> {out_path}", flush=True)
    return len(b)


if __name__ == "__main__":
    decode_to_bytes(os.path.join(SRC, "train.bin"), N_TRAIN_TOK, os.path.join(OUT, "wiki_train.u8"))
    decode_to_bytes(os.path.join(SRC, "val.bin"), None, os.path.join(OUT, "wiki_val.u8"))
    print("done", flush=True)
