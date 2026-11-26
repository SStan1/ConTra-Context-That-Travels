"""Generate a contextual prefix for each chunk of a document."""

import argparse
import json
from pathlib import Path

from contra.context import augment, messages, parse_prefix
from contra.train import MAX_RESPONSE_TOKENS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--document", type=Path, required=True, help="JSON array of ordered chunk texts")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    chunks = json.loads(args.document.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    model_type = AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = model_type.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    for index, chunk in enumerate(chunks):
        inputs = tokenizer.apply_chat_template(
            messages(chunks, index, tokenizer), tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_RESPONSE_TOKENS)
        tokens = output[0, inputs["input_ids"].shape[1]:]
        prefix = parse_prefix(tokenizer.decode(tokens, skip_special_tokens=True))
        if prefix is None or len(tokens) >= MAX_RESPONSE_TOKENS:
            raise ValueError(f"No complete prefix for chunk {index}")
        print(json.dumps({
            "chunk_id": index, "prefix": prefix, "text": augment(prefix, chunk),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
