"""命令行问答（无 LLM key 时用 mock 模式验证检索链路）。

用法:
    python scripts/query.py 净化器滤芯多久换一次
    python scripts/query.py --mode naive 年假有几天
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="LiteRAG 命令行问答")
    parser.add_argument("question")
    parser.add_argument("--mode", default="full", choices=["naive", "hybrid", "rerank", "full"])
    args = parser.parse_args()

    pipe = RAGPipeline()
    result = pipe.ask(args.question, mode=args.mode)
    trace = result.trace
    print("=== 检索轨迹 ===")
    print(f"改写: {trace.rewritten or '（未启用/未变化）'}")
    if trace.sub_queries:
        print(f"子查询: {trace.sub_queries}")
    print(f"向量命中 {trace.vector_hits} | BM25 命中 {trace.bm25_hits} | 融合候选 {trace.fused_candidates}")
    print(f"耗时: {json.dumps(trace.timings, ensure_ascii=False)}")
    print("\n=== 引用片段 ===")
    for c in result.citations:
        print(f"[{c['ref']}] {c['source']} > {c['section_path']} (score={c['score']})")
    print("\n=== 答案 ===")
    print(result.answer)


if __name__ == "__main__":
    main()
