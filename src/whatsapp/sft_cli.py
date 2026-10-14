"""Build SFT JSONL files from an indexed, approved WhatsApp chat."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .sft import build_sft
from .topic_memory import LazyLocalQwen, OpenAICompatibleTopicModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Write SFT pairs with real target replies")
    parser.add_argument("workspace", type=Path, help="indexed WhatsApp review workspace")
    parser.add_argument("--output", type=Path, help="output directory (default: WORKSPACE/sft)")
    parser.add_argument("--context-messages", type=int, default=6)
    parser.add_argument("--limit", type=int, help="maximum synthetic-query model calls")
    parser.add_argument("--real-only", action="store_true",
                        help="use actual preceding chat without generating queries")
    parser.add_argument("--api-url", help="OpenAI-compatible /v1 endpoint")
    parser.add_argument("--model", help="model available at --api-url")
    args = parser.parse_args()
    try:
        if bool(args.api_url) != bool(args.model):
            raise ValueError("--api-url and --model must be supplied together")
        if args.real_only:
            if args.limit is not None or args.api_url:
                raise ValueError("--real-only cannot be combined with --limit or a query model")
            model = None
        elif args.api_url:
            model = OpenAICompatibleTopicModel(args.api_url, args.model)
        else:
            if importlib.util.find_spec("mlx_vlm") is None:
                raise RuntimeError("local Qwen is unavailable; pass --api-url and --model "
                                   "or use --real-only")
            model = LazyLocalQwen()
        result = build_sft(args.workspace, output=args.output, model=model,
                           query_limit=args.limit, context_messages=args.context_messages)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
