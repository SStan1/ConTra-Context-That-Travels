"""Reward terms and the VERL adapter."""

import json
import math
import os
from urllib.request import Request, urlopen

from contra.context import parse_prefix

INVALID = -1.5
RETRIEVAL_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
CONTEXT_INSTRUCTION = (
    "Retrieve a passage answering the question with the correct identities and "
    "relationships. The source introduction resolves implicit document context; "
    "it is not itself the question."
)


def detailed_query(instruction, query):
    return f"Instruct: {instruction}\nQuery: {query}"


def contextual_query(query, opening_chunk):
    opening = " ".join(opening_chunk.split()[:128])
    return detailed_query(
        CONTEXT_INSTRUCTION,
        f"QUESTION: {query}\nSOURCE INTRODUCTION: {opening}\nQUESTION: {query}",
    )


def paper_reward(raw_cos, prefixed_cos, raw_logit, prefixed_logit, support, length):
    if not all(map(math.isfinite, (raw_cos, prefixed_cos, raw_logit, prefixed_logit, support))):
        raise ValueError("Reward model scores must be finite")
    if not 0 <= support <= 1 or length < 0:
        raise ValueError("Invalid factual confidence or prefix length")
    embedding = max(-1.0, min(1.0, prefixed_cos - raw_cos))
    reranking = 0.025 * math.tanh(prefixed_logit - raw_logit)
    factual = -0.25 * max(0.75 - support, 0.0) / 0.75
    brevity = -0.2 * min(max(length - 16, 0) / 96, 1.0)
    return embedding + reranking + factual + brevity


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    prefix = parse_prefix(solution_str)
    if prefix is None:
        return INVALID
    endpoint = os.environ.get("CONTRA_REWARD_URL", "http://127.0.0.1:8765/score")
    request = Request(
        endpoint,
        data=json.dumps({
            "prefix": prefix,
            "example": {
                "query": str(extra_info["query"]),
                "chunks": list(extra_info["chunks"]),
                "target_index": int(extra_info["target_index"]),
            },
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=3600) as response:
        score = float(json.load(response)["score"])
    if not math.isfinite(score):
        raise ValueError("The reward service returned a nonfinite score")
    return score
