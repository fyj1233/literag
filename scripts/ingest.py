"""摄入 CLI：把 data/docs（或指定目录/文件）灌入向量库 + BM25 索引。

用法:
    python scripts/ingest.py                # 灌入默认 data/docs 目录
    python scripts/ingest.py path/to/file.pdf
    python scripts/ingest.py path/to/dir
    python scripts/ingest.py --reset        # 清空索引后重新灌入
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="LiteRAG 文档摄入")
    parser.add_argument("path", nargs="?", default="data/docs", help="文件或目录，默认 data/docs")
    parser.add_argument("--reset", action="store_true", help="清空现有索引后灌入")
    parser.add_argument("--visibility", default="L1", choices=["L1", "L2", "L3", "L4"],
                        help="文档密级（默认 L1 公开；机密文档请单独指定）")
    args = parser.parse_args()

    pipe = RAGPipeline()
    if args.reset:
        pipe.vector_store.reset()
        pipe.bm25_store.rebuild([])
        print("[reset] 索引已清空")

    target = Path(args.path)
    t0 = time.perf_counter()
    if target.is_dir():
        results = pipe.ingest_directory(target, visibility=args.visibility)
    else:
        results = [pipe.ingest_file(target, visibility=args.visibility)]

    print(f"\n{'文档':<44}{'chunk数':>8}{'耗时(s)':>10}")
    print("-" * 62)
    for r in results:
        print(f"{r['source']:<44}{r['chunks']:>8}{r['elapsed']:>10}")
    total = sum(r["chunks"] for r in results)
    print("-" * 62)
    print(f"共 {len(results)} 份文档 / {total} 个 chunk / 总耗时 {time.perf_counter() - t0:.1f}s")
    print(f"向量库: {pipe.vector_store.collection_name} ({pipe.vector_store.count()} 条)")
    print("完成。启动服务: uv run uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
