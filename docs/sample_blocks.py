"""Sample candidate blocks for query authoring.

Sampling is seeded and happens BEFORE any retrieval is run, so the query set
cannot be shaped by what the bi-encoder happens to find. Book furniture is
filtered with the same boilerplate score the app ships (quality.js), because a
table-of-contents block has no answerable question in it.
"""
import json, re, numpy as np

def boilerplate_score(t):                       # port of app/src/quality.js
    L = max(len(t), 1); per1k = 1000 / L
    n = lambda p: len(re.findall(p, t))
    digits = sum(c.isdigit() for c in t); commas = t.count(",")
    return (n(r"\[\s*\d+\s*\]") * 2.0 + n(r"\b(?:19|20)\d{2}\b") * 1.5
            + n(r"\bet al\.") * 3.0 + len(re.findall(r"https?://|www\.|\.com|\.org|doi:", t, re.I)) * 1.5
            + n(r"[“”]") * 0.8 + n(r"\. \. \.") * 4.0) * per1k \
           + max(0, digits / L - 0.04) * 60 + max(0, (commas / L) * 100 - 3.0) * 1.2

blocks = json.load(open("ddia_blocks.json"))
rng = np.random.default_rng(20260906)
order = rng.permutation(len(blocks))
picked = [i for i in order if len(blocks[i]) > 900 and boilerplate_score(blocks[i]) <= 25][:110]
json.dump([int(i) for i in picked], open("ddia_candidates.json", "w"))
print(f"blocks {len(blocks)}  candidates after furniture filter {len(picked)}")
for rank, i in enumerate(picked[:55]):
    print(f"\n--- cand {rank}  block {i} ---\n{blocks[i]}")
