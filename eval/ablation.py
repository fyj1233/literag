"""消融实验：在同一个评测集上对比四种检索配置的检索质量。

    python eval/ablation.py            # 检索指标（Hit@5 / MRR@10 / 关键词覆盖），离线可复现
    python eval/ablation.py --judge    # 追加 LLM-as-judge 生成指标（忠实度/答案相关性，需配置 key）

四级配置即四种 mode（对应 app 的功能开关）：
  naive   仅向量召回            —— baseline
  hybrid  + BM25 稀疏召回 + RRF 融合
  rerank  + CrossEncoder 精排
  full    + LLM 查询改写与子查询分解
说明：LLM_PROVIDER=mock 时改写为恒等（full 与 rerank 结果相同），
配置真实 LLM 后重跑即可得到改写带来的增益。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from eval.metrics import hit_at_k, keyword_coverage, mrr_at_k  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "eval" / "dataset.jsonl"
OUT_DIR = ROOT / "eval" / "results"

MODES = ["naive", "hybrid", "rerank", "full"]
MODE_LABEL = {
    "naive": "baseline（仅向量召回）",
    "hybrid": "+ BM25 混合召回 + RRF 融合",
    "rerank": "+ CrossEncoder 精排",
    "full": "+ LLM 查询改写 / 子查询分解",
}

JUDGE_PROMPT = """请评估以下 RAG 回答质量，只输出 JSON：
{{"faithfulness": 1-5整数, "relevancy": 1-5整数}}
faithfulness: 回答中的每个论断是否都能在【参考资料】中找到依据（5=完全可溯源）
relevancy: 回答对【用户问题】的直接回答程度（5=完全切题）

【用户问题】{question}
【参考资料】
{context}
【回答】{answer}"""


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_retrieval(pipe, items: list[dict], mode: str) -> dict:
    hit5 = mrr10 = kwcov = 0.0
    times: list[float] = []
    per_item = []
    for it in items:
        t0 = time.perf_counter()
        hits, _trace = pipe.retriever.retrieve([it["question"]], mode=mode, top_k=10)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        hit_dicts = [h.to_dict() for h in hits]
        h5 = hit_at_k(hit_dicts, it["source"], 5)
        m10 = mrr_at_k(hit_dicts, it["source"], 10)
        kc = keyword_coverage(hit_dicts, it.get("keywords", []), 5)
        hit5 += h5
        mrr10 += m10
        kwcov += kc
        per_item.append({"id": it["id"], "hit@5": h5, "mrr@10": m10, "ms": round(elapsed)})
    n = len(items)
    return {
        "hit@5": round(hit5 / n, 4),
        "mrr@10": round(mrr10 / n, 4),
        "keyword_coverage@5": round(kwcov / n, 4),
        "avg_retrieval_ms": round(sum(times) / n),
        "per_item": per_item,
    }


def eval_judge(pipe, items: list[dict], mode: str) -> dict:
    """LLM-as-judge：忠实度（可溯源）与答案相关性，1-5 分。"""
    faith = rel = 0.0
    done = 0
    for it in items:
        result = pipe.ask(it["question"], mode=mode)
        if "没有检索到" in result.answer or "没有找到" in result.answer:
            continue  # mock 兜底话术不参与打分
        context = "\n".join(f"[{i}] {h.text}" for i, h in enumerate(result.hits, 1))
        try:
            data = pipe.llm.chat_json([{
                "role": "user",
                "content": JUDGE_PROMPT.format(question=it["question"], context=context, answer=result.answer),
            }])
            faith += float(data.get("faithfulness", 0))
            rel += float(data.get("relevancy", 0))
            done += 1
        except Exception:  # noqa: BLE001 —— 单条评估失败跳过
            continue
    if not done:
        return {"faithfulness": None, "relevancy": None, "note": "judge 未运行（需配置真实 LLM key）"}
    return {"faithfulness": round(faith / done, 2), "relevancy": round(rel / done, 2), "n": done}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="追加 LLM-as-judge 生成指标（需真实 LLM）")
    parser.add_argument("--judge-modes", default=",".join(MODES),
                        help="对哪些模式跑 judge（逗号分隔，默认全部；LLM 慢/省额度时可只传 full）")
    args = parser.parse_args()
    judge_modes = [m.strip() for m in args.judge_modes.split(",") if m.strip() in MODES]

    from app.rag.pipeline import RAGPipeline

    settings.enable_corrective = False  # 消融只测检索四层本身，纠正重查不参与
    pipe = RAGPipeline()
    items = load_dataset()
    print(f"评测集 {len(items)} 条 · 知识库 {pipe.vector_store.count()} chunks\n")

    OUT_DIR.mkdir(exist_ok=True)
    # 模型热身：避免首个配置承担冷启动耗时，保证各模式计时口径一致
    pipe.retriever.retrieve(["热身查询"], mode="rerank", top_k=1)

    results: dict[str, dict] = {}
    for mode in MODES:
        print(f"[{mode}] {MODE_LABEL[mode]} 评测中…")
        results[mode] = eval_retrieval(pipe, items, mode)
        print(f"  Hit@5={results[mode]['hit@5']:.2%}  MRR@10={results[mode]['mrr@10']:.4f}  "
              f"平均耗时={results[mode]['avg_retrieval_ms']}ms")

    if args.judge:
        for mode in judge_modes:
            print(f"[{mode}] LLM-as-judge 评估中…")
            results[mode]["judge"] = eval_judge(pipe, items, mode)

    (OUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_docs = len({s for it in items for s in ([it["source"]] if isinstance(it["source"], str) else it["source"])})
    n_multi = sum(1 for it in items if isinstance(it["source"], list))
    lines = [
        "# LiteRAG 消融实验结果",
        "",
        f"- 评测集：{len(items)} 条中文 QA（自建，覆盖 {n_docs} 份文档，其中 {n_multi} 条跨文档/对比题，"
        f"含字面相近的干扰文档用于拉开区分度）",
        f"- 知识库：{pipe.vector_store.count()} chunks · Embedding: {pipe.embedding.name}",
        "",
    has_judge = any((results[m].get("judge") or {}).get("faithfulness") for m in MODES)
    lines.append("| 配置 | Hit@5 ↑ | MRR@10 ↑ | 关键词覆盖@5 ↑ | 平均检索耗时 ↓ |"
                 + (" 忠实度↑ | 答案相关性↑ |" if has_judge else ""))
    lines.append("| --- | --- | --- | --- | --- |" + (" --- | --- |" if has_judge else ""))
    for mode in MODES:
        r = results[mode]
        row = (f"| {MODE_LABEL[mode]} | {r['hit@5']:.2%} | {r['mrr@10']:.4f} | "
               f"{r['keyword_coverage@5']:.2%} | {r['avg_retrieval_ms']}ms |")
        if has_judge:
            j = r.get("judge") or {}
            fv, rel = j.get("faithfulness"), j.get("relevancy")
            row += f" {fv if fv is not None else '—'} | {rel if rel is not None else '—'} |"
        lines.append(row)
    lines += ["", "> 复现：`python eval/ablation.py`（检索指标离线可复现；"
              "`--judge` 需在 .env 配置真实 LLM 后追加忠实度/相关性评分）"]
    (OUT_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已写入 {OUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
