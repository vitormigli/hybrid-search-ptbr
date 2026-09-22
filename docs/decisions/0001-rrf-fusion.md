# 1. Fuse BM25 and dense retrieval with Reciprocal Rank Fusion

## Status

Accepted

## Context

Lexical (BM25) and dense (embedding) retrieval fail on different queries: BM25 misses
paraphrases and synonyms, dense retrieval misses exact-match keyword queries (IDs,
codes, rare terms). We want to combine both without training a learned re-ranker.

## Decision

Use Reciprocal Rank Fusion (RRF): `score(doc) = sum(1 / (k + rank))` across each
ranker's result list, with `k=60` (a common default that avoids over-weighting rank-1
results from any single ranker). No score normalization needed since RRF only uses
rank position, not raw scores — which sidesteps BM25 and cosine-similarity scores
living on completely different scales.

## Consequences

- Works without any training data or tuning beyond `k`.
- A document that ranks well in either list (not necessarily both) still surfaces.
- Loses some information compared to a learned re-ranker (e.g. a cross-encoder), which
  is the natural next step if hybrid retrieval alone isn't accurate enough.
