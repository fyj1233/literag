"""下载真实公开语料（hello-algo《Hello 算法》，CC BY-NC-SA 4.0）到 data/real_docs/。

    python scripts/fetch_real_corpus.py

用途：在"真实技术文档"上验证检索质量（详见 eval/real_corpus_report.md）。
默认走 jsdelivr CDN，失败自动切换 ghproxy 镜像；文件仅存于本地，不进入 git。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FILES = [
    "docs/chapter_computational_complexity/time_complexity.md",
    "docs/chapter_computational_complexity/space_complexity.md",
    "docs/chapter_computational_complexity/iteration_and_recursion.md",
    "docs/chapter_array_and_linkedlist/array.md",
    "docs/chapter_array_and_linkedlist/linked_list.md",
    "docs/chapter_array_and_linkedlist/list.md",
    "docs/chapter_stack_and_queue/stack.md",
    "docs/chapter_stack_and_queue/queue.md",
    "docs/chapter_hashing/hash_map.md",
    "docs/chapter_hashing/hash_collision.md",
    "docs/chapter_searching/binary_search.md",
    "docs/chapter_sorting/quick_sort.md",
    "docs/chapter_sorting/merge_sort.md",
    "docs/chapter_sorting/bubble_sort.md",
]

MIRRORS = [
    "https://cdn.jsdelivr.net/gh/krahets/hello-algo@main/{path}",
    "https://ghproxy.cn/https://raw.githubusercontent.com/krahets/hello-algo/main/{path}",
]

OUT = Path(__file__).resolve().parent.parent / "data" / "real_docs"


def fetch(path: str) -> str | None:
    import httpx

    for tpl in MIRRORS:
        try:
            r = httpx.get(tpl.format(path=path), timeout=30, follow_redirects=True)
            if r.status_code == 200 and not r.text.lstrip().startswith("<!DOCTYPE"):
                return r.text
        except httpx.HTTPError:
            continue
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for rel in FILES:
        target = OUT / Path(rel).name
        if target.exists() and target.stat().st_size > 1000:
            print(f"[skip] {rel}")
            ok += 1
            continue
        text = fetch(rel)
        if text is None:
            print(f"[fail] {rel}")
            continue
        target.write_text(text, encoding="utf-8")
        print(f"[ok]   {rel} ({len(text.encode('utf-8')) // 1024}KB)")
        ok += 1
    print(f"\n完成：{ok}/{len(FILES)} 个文件已就位于 {OUT}")


if __name__ == "__main__":
    main()
