import json
from pathlib import Path

if __name__ == "__main__":
    vocab = Path("./vocabulary.txt").read_text().split("\n")
    vocab = dict(zip(vocab, range(len(vocab))))
    Path("./vocabulary.json").write_text(json.dumps(vocab, indent=2))
