"""多轮对话指代消解：history 必须参与查询改写的 Prompt 构造。"""
from app.providers.llm import LLMProvider
from app.rag.query import optimize_query


class RecordingLLM(LLMProvider):
    """记录收到的 prompt 并返回固定 JSON。"""

    def __init__(self):
        self.last_messages = None

    def chat(self, messages):
        self.last_messages = messages
        return '{"rewritten": "云帆X1净化器的整机保修期是多久", "sub_queries": []}'

    def chat_stream(self, messages):
        yield self.chat(messages)


def test_history_participates_in_rewrite_prompt():
    llm = RecordingLLM()
    history = [
        {"role": "user", "content": "净化器保修几年？"},
        {"role": "assistant", "content": "整机保修 2 年。"},
    ]
    optimize_query("那加湿器呢？", llm, enable=True, history=history)
    prompt = llm.last_messages[-1]["content"]
    assert "对话历史" in prompt
    assert "净化器保修几年" in prompt          # 历史轮次进入 prompt
    assert "指代" in prompt                     # 指代消解指令进入 prompt
    assert "那加湿器呢" in prompt               # 当前问题仍在


def test_no_history_keeps_prompt_clean():
    llm = RecordingLLM()
    optimize_query("保修多久", llm, enable=True, history=None)
    assert "对话历史" not in llm.last_messages[-1]["content"]


def test_history_truncated_to_two_turns():
    llm = RecordingLLM()
    history = [{"role": "user", "content": f"旧问题{i}"} for i in range(6)]
    optimize_query("现在呢", llm, enable=True, history=history)
    prompt = llm.last_messages[-1]["content"]
    assert "旧问题0" not in prompt and "旧问题1" not in prompt  # 只保留最近两轮
    assert "旧问题4" in prompt and "旧问题5" in prompt
