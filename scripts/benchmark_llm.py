"""LLM 端点流式测速：对比不同 API 的首字延迟 / 增量间隔 / 总耗时。

    python scripts/benchmark_llm.py                    # 测 .env 当前配置（2 轮）
    python scripts/benchmark_llm.py --rounds 3         # 多测几轮取稳定值
    python scripts/benchmark_llm.py --base-url https://open.bigmodel.cn/api/paas/v4 \
        --key 你的key --model glm-4-flash              # 直接对比另一个端点

结论口径：首字延迟（TTFT）决定"等多久开始出字"；增量间隔决定"打字机顺不顺滑"。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

PROMPT = "请用大约100字介绍一下空气滤芯的作用"


def measure(client: OpenAI, model: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    first = None
    last = t0
    gaps = []
    chars = 0
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
        max_tokens=max_tokens,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            if first is None:
                first = now - t0
            else:
                gaps.append(now - last)
            last = now
            chars += len(chunk.choices[0].delta.content)
    total = time.perf_counter() - t0
    return {
        "ttft": first if first is not None else total,
        "deltas": len(gaps) + 1,
        "avg_gap_ms": (sum(gaps) / len(gaps) * 1000) if gaps else 0,
        "total": total,
        "chars": chars,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    from app.config import settings   # noqa: E402 —— 读取 .env 兜底配置

    base_url = args.base_url or settings.llm_base_url
    api_key = args.key or settings.llm_api_key
    model = args.model or settings.llm_model
    client = OpenAI(base_url=base_url, api_key=api_key)
    print(f"端点: {base_url}\n模型: {model}\n")

    rows = []
    for i in range(1, args.rounds + 1):
        tag = "预热" if i == 1 else f"第{i}轮"
        try:
            r = measure(client, model, args.max_tokens)
            rows.append(r)
            print(f"[{tag}] 首字 {r['ttft']:.2f}s | {r['deltas']} 个增量 | "
                  f"平均间隔 {r['avg_gap_ms']:.0f}ms | 总耗时 {r['total']:.2f}s | {r['chars']} 字")
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] 失败: {e}")
            return

    best = min(rows[1:] or rows, key=lambda r: r["ttft"])
    print(f"\n结论: 首字延迟 {best['ttft']:.2f}s（去掉预热后最优一轮）——"
          + ("属于快端点（<3s），日常体验流畅" if best["ttft"] < 3
             else "偏慢（>3s），问答前会有可感知的等待；项目侧已有阶段提示兜底"))


if __name__ == "__main__":
    main()
