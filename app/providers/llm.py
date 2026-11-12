"""LLM Provider 抽象与实现：

- OpenAICompatibleLLM: 任意 OpenAI 兼容接口（DeepSeek / 智谱 / Qwen / 中转站），
  支持流式与非流式，key 为空时在调用时报错（而不是启动时）。
- MockLLM: 离线演示模式——从检索到的资料中抽取与问题最相关的句子组成答案，
  并带上 [n] 引用标注。让整个系统在没有任何 key 时也能端到端跑通。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.config import settings

CONTEXT_MARKER_BEGIN = "【参考资料】"
CONTEXT_MARKER_END = "【资料结束】"


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """非流式对话。"""

    @abstractmethod
    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        """流式对话，逐段 yield 文本增量。"""

    # ---- 便捷方法：带重试与健壮解析的 JSON 输出 ----
    def chat_json(self, messages: list[dict]) -> dict:
        text = self.chat(messages)
        return parse_json_loose(text)


def parse_json_loose(text: str) -> dict:
    """从 LLM 输出中尽力提取 JSON（容忍 ```json 包裹与前后废话）。"""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"LLM 输出中未找到 JSON: {text[:200]}")
    return json.loads(m.group(0))


class OpenAICompatibleLLM(LLMProvider):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model = model or settings.llm_model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            if not self.api_key:
                raise RuntimeError(
                    "LLM_API_KEY 为空：请在 .env 中填入 key，或设置 LLM_PROVIDER=mock 以离线演示"
                )
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def chat(self, messages: list[dict]) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        client = self._get_client()
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class MockLLM(LLMProvider):
    """离线演示 LLM：解析 prompt 中的【参考资料】块，抽取关键词命中句子拼成答案。

    足以验证 检索→组装→引用→流式 全链路，替换为真实 LLM 只需改 .env。
    """

    def _extract_context(self, messages: list[dict]) -> tuple[str, list[tuple[int, str]]]:
        """返回 (原始问题, [(编号, 资料文本)])。以最后一条 user 消息为准。"""
        question = ""
        blocks: list[tuple[int, str]] = []
        for msg in messages:
            if msg["role"] != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):   # 多模态消息：只取文本部分
                content = " ".join(p.get("text", "") for p in content
                                   if isinstance(p, dict) and p.get("type") == "text")
            if "【用户问题】" in content:
                question = content.split("【用户问题】")[-1].strip()
            elif "用户问题：" in content:
                question = content.rsplit("用户问题：", 1)[-1].strip()
            elif "以下问题" in content:
                question = content.rsplit("：", 1)[-1].strip()
            else:
                question = content.strip()
        full = "\n".join(m["content"] for m in messages)
        m = re.search(
            re.escape(CONTEXT_MARKER_BEGIN) + r"(.*?)" + re.escape(CONTEXT_MARKER_END),
            full,
            re.DOTALL,
        )
        if m:
            content = m.group(1)
            for match in re.finditer(r"^\[(\d+)\]\s*(.+?)(?=^\[\d+\]|\Z)", content, re.S | re.M):
                blocks.append((int(match.group(1)), match.group(2).strip()))
        return question, blocks

    def _keywords(self, question: str) -> list[str]:
        import jieba

        stopwords = {"的", "是", "了", "吗", "什么", "怎么", "如何", "多少", "哪些", "请问", "一下", "我", "你", "他", "在", "有", "和", "与", "或", "对", "关于", "介绍"}
        tokens = jieba.lcut(question)
        # 相邻单字合并为双字候选（兜底 jieba 词典外的新词，如"年假"被拆成"年"+"假"）
        merged = [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)
                  if len(tokens[i]) == 1 and len(tokens[i + 1]) == 1]
        return [w for w in tokens + merged if len(w) >= 2 and w not in stopwords]

    @staticmethod
    def _kw_score(sentence: str, keywords: list[str]) -> float:
        """完整词命中记 1 分；复合词按词根（前两字）部分命中记 0.5 分。"""
        score = 0.0
        for kw in keywords:
            if kw in sentence:
                score += 1.0
            elif len(kw) >= 2 and kw[:2] in sentence:
                score += 0.5
        return score

    def _compose(self, question: str, blocks: list[tuple[int, str]]) -> str:
        kws = self._keywords(question)
        scored: list[tuple[float, int, str]] = []
        for no, text in blocks:
            sentences = re.split(r"(?<=[。！？!?])", text)
            for sent in sentences:
                if len(sent.strip()) < 6:
                    continue
                score = self._kw_score(sent, kws)
                if score > 0:
                    scored.append((score, no, sent.strip()))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return (
                "（演示模式）知识库中没有检索到与该问题直接相关的内容。"
                "请在 .env 中配置真实 LLM（如 DeepSeek / 智谱 GLM-4-Flash）以获得生成式回答。"
            )
        picked, seen, used = [], set(), set()
        for _, no, sent in scored:
            key = sent[:40]
            if key in seen:
                continue
            seen.add(key)
            picked.append(f"{sent}[{no}]")
            used.add(no)
            if len(picked) >= 3:
                break
        return "根据知识库资料：" + "".join(picked)

    def chat(self, messages: list[dict]) -> str:
        question, blocks = self._extract_context(messages)
        if "输出 JSON" in str(messages) or "json" in str(messages[-1].get("content", "")).lower():
            # 查询改写等结构化任务：演示模式下返回恒等改写
            return json.dumps({"rewritten": question, "sub_queries": []}, ensure_ascii=False)
        return self._compose(question, blocks)

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        answer = self.chat(messages)
        step = 8
        for i in range(0, len(answer), step):
            yield answer[i:i + step]


def get_llm_provider() -> LLMProvider:
    kind = settings.llm_provider.lower()
    if kind == "mock":
        return MockLLM()
    if kind in ("openai", "openai_compatible", "compatible", "deepseek", "zhipu", "qwen"):
        return OpenAICompatibleLLM()
    raise ValueError(f"未知的 LLM_PROVIDER: {settings.llm_provider}")
