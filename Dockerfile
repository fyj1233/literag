FROM python:3.12-slim

WORKDIR /app

# 国内网络可取消注释以加速 pip
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY eval ./eval
COPY data/docs ./data/docs

# 国内网络下模型走 hf-mirror（首个请求自动下载约 1.2GB 并缓存在卷中）
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/app/data/hf-cache
# 镜像内不含模型权重，按 repo id 自动下载（环境变量优先级高于 .env）
ENV LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
ENV LOCAL_RERANKER_MODEL=BAAI/bge-reranker-base
ENV LLM_PROVIDER=mock

VOLUME ["/app/data", "/app/uploads"]
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
