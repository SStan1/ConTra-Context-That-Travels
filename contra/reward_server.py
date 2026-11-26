"""Serve the frozen reward models separately from the rollout GPUs."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from contra.scorers import FrozenScorers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Same Qwen3.5 tokenizer as the policy")
    parser.add_argument("--embedding", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--reranker", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--judge", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    scorers = FrozenScorers(args.policy, args.embedding, args.reranker, args.judge, args.device)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/score":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                score = scorers.score(payload["example"], payload["prefix"])
                result = json.dumps({"score": score}).encode()
            except (ValueError, KeyError, TypeError, IndexError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

    class RewardServer(HTTPServer):
        request_queue_size = 512

    with RewardServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Reward server listening on 127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
