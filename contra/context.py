"""Shared training and inference prompt."""

import json
import re

SYSTEM = """You are a contextualization model for retrieval-augmented generation (RAG).

Given a document and one target chunk from that document, generate a concise contextual prefix containing only the necessary information that helps the chunk remain understandable and retrievable outside its original document.

Do not repeat the target chunk, introduce unsupported information, or add unnecessary details.

Output only the contextual prefix."""

WRAPPER = re.compile(
    r"</?(?:document|target_chunk|chunk|think|analysis|final|answer)\b[^>]*>"
    r"|<\|[^>]*>|\x60\x60\x60|<NO_CONTEXT", re.IGNORECASE
)


def source_window(chunks, target_index, tokenizer, budget):
    """Keep contiguous whole chunks around the target, in source order."""
    if not chunks or not 0 <= target_index < len(chunks):
        raise ValueError("The target index must identify a source chunk")
    if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
        raise ValueError("Every source chunk must contain text")

    left, right = target_index, target_index + 1

    def fits(start, stop):
        text = "\n\n".join(chunks[start:stop])
        return len(tokenizer.encode(text, add_special_tokens=False)) <= budget

    if fits(0, len(chunks)):
        return list(enumerate(chunks))
    if not fits(left, right):
        raise ValueError("Target chunk exceeds the source token budget")
    while True:
        candidates = []
        if left > 0:
            candidates.append((target_index - left + 1, left - 1, right))
        if right < len(chunks):
            candidates.append((right - target_index, left, right + 1))
        for _, start, stop in sorted(candidates):
            if fits(start, stop):
                left, right = start, stop
                break
        else:
            return [(i, chunks[i]) for i in range(left, right)]


def messages(chunks, target_index, tokenizer, budget=8000):
    window = source_window(chunks, target_index, tokenizer, budget)
    blocks = []
    for index, text in window:
        marker = ' target="true"' if index == target_index else ""
        blocks.append(f'<CHUNK index="{index}"{marker}>\n{text}\n</CHUNK>')
    document = "\n\n".join(blocks)
    user = f"<DOCUMENT>\n{document}\n</DOCUMENT>\n\n<TARGET_CHUNK>\n{chunks[target_index]}\n</TARGET_CHUNK>"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_prefix(text):
    value = text.strip()
    if not value or WRAPPER.search(value):
        return None
    if value.startswith(("{", "[")):
        try:
            structured = json.loads(value)
        except ValueError:
            pass
        else:
            if isinstance(structured, (dict, list)):
                return None
    return value


def augment(prefix, chunk):
    return f"{prefix}\n\n{chunk}"
