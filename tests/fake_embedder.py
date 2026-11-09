"""Deterministic stand-in for MiniLM in tests: hashed bag of words, L2-normalised.
Lets retrieval tests run offline, fast and repeatably (no model download)."""
import hashlib
import math
import re


class FakeEmbedder:
    name = "fake-hash-embedder"

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def embed(self, texts):
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                if len(w) < 3:
                    continue
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
