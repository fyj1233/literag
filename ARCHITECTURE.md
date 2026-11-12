# LiteRAG 架构设计文档

> 本文描述系统的分层结构、请求生命周期、数据模型与权限模型。设计取舍的"为什么"见 README「关键设计决策」；面向面试的逐环节讲解见《项目详解.html》。

## 一、系统分层总览

```mermaid
flowchart TB
    subgraph 接入层
        WEB[web/index.html 单文件前端<br/>登录 / 流式渲染 / 引用卡片 / 模式切换]
        API[FastAPI 路由层 app/api<br/>auth + chat + search + documents + audit]
    end
    subgraph 业务编排层
        PIPE[app/rag/pipeline.py<br/>ingest / ask / ask_stream 编排]
        AUTH[app/auth.py<br/>SQLite 用户表 + PBKDF2 + HS256 JWT + 审计日志]
    end
    subgraph 检索引擎层
        QUERY[query.py 查询优化<br/>改写 / 子查询 / HyDE]
        RETRI[retrieval.py 检索编排<br/>双路召回 + RRF + 精排 + ACL]
        GEN[generation.py 生成层<br/>Prompt 组装 / 引用对齐]
        CHUNK[chunking.py 切分<br/>递归切分 + 上下文块头]
    end
    subgraph 供应层（可插拔 Provider）
        LLM[llm.py<br/>OpenAI 兼容 / Mock]
        EMB[embedding.py<br/>本地BGE / API / hash]
        RER[reranker.py<br/>CrossEncoder / API / 直通]
    end
    subgraph 存储层
        VS[(ChromaDB<br/>向量, 持久化)]
        BM[(rank_bm25<br/>jieba 稀疏索引, pickle)]
        PS[(ParentStore<br/>章节父块, pickle)]
        UDB[(users.db<br/>用户 + 审计, SQLite)]
    end
    WEB --> API --> PIPE
    API --> AUTH --> UDB
    PIPE --> QUERY --> RETRI --> VS
    RETRI --> BM
    RETRI --> RER
    PIPE --> GEN --> LLM
    PIPE --> CHUNK --> EMB --> VS
    PIPE --> PS
```

分层原则：**上层只依赖下一层的抽象**（Provider 基类）、**权限过滤在检索引擎层内完成**（不暴露给上层重试）、**HTTP 细节不进入业务模块**（auth.py 不依赖 FastAPI）。

## 二、两条请求生命周期

### 摄入（POST /api/documents，boss/employee）

```
上传文件 + 密级(visibility) → 校验格式/密级 → load_file 解析(章节+元数据)
  → chunk_document 递归切分(500/80 + 上下文块头) → 每 chunk 记 parent_id + visibility
  → Embedding 批量向量化 → Chroma upsert + ParentStore 写父块 + BM25 全量重建
  → 同名文档先删后灌（幂等）
```

### 问答（POST /api/chat，SSE）

```
Authorization: Bearer JWT → verify_token(签名+过期) → 角色 → 可见密级集合 allowed
  → optimize_query（LLM 改写 + 子查询分解；多轮时结合历史消解指代；失败降级原问题）
  → Retriever.retrieve(queries, mode, allowed):
       向量路: query embedding → Chroma where{visibility ∈ allowed} → Top-K
       稀疏路: jieba 分词 → BM25 打分 → 候选补全元数据 → 按 visibility 过滤   ← 过滤必须在补全后
       → RRF 融合(k=60) → 候选 Top-20 → CrossEncoder 精排 → Top-5
       → 低置信(<0.35)触发 CRAG-lite：扩大候选重查一次，取两次较优
  → _context_blocks：Parent-Child 开启时，同章节命中合并为父块全文(截断 1200 字)
  → SSE: meta(检索轨迹+是否纠正) → delta*(流式) → done(答案+[n]引用对齐)
  → audit 写入审计日志（仅 boss 可查 /api/audit）
```

## 三、数据模型

| 存储 | 内容 | 关键字段/结构 |
| --- | --- | --- |
| ChromaDB collection（按 embedding 模型命名） | chunk 正文 + 向量 | metadata: `source / title / section_path / chunk_index / page / parent_id / visibility` |
| BM25（pickle） | chunk_id → 分词后的 embedding_text | 正文以 Chroma 为准，BM25 只存 id 与索引 |
| ParentStore（pickle） | parent_id → 章节全文 | `{source, title, section_path, page, text}`；密级继承自文档，存于 chunk |
| users.db（SQLite） | 用户 + 审计 | users(username, pw=PBKDF2 盐$哈希, role)；audit_log(ts, username, endpoint, question, sources) |
| data/ 目录 | docs 演示语料 / confidential 机密语料 / chroma / bm25 / uploads | 索引与语料均在项目内，可整目录迁移 |

**权限模型**：角色 `boss→[L1..L4]`、`employee→[L1,L2]`、`guest→[L1]`；文档级标密（chunk/父块继承）；**默认拒绝**——未知角色按最低权限；过滤点在召回阶段（向量 where + BM25 补全后过滤），无权限内容不进入融合/精排/Prompt 的任何环节。

## 四、模块职责一览

| 模块 | 职责 | 不负责 |
| --- | --- | --- |
| app/api/routes.py | HTTP 协议、鉴权依赖注入、SSE 编码、审计埋点 | 业务逻辑 |
| app/auth.py | 密码哈希、JWT 签发校验、用户与审计存储 | HTTP 解析、角色到密级的映射（在 config） |
| app/rag/pipeline.py | 两条主链路的编排与进程级单例 | 具体检索算法 |
| app/rag/retrieval.py | 双路召回、RRF、精排、ACL 过滤、检索轨迹 | 向量库实现细节 |
| app/rag/chunking.py | 切分算法与上下文块头 | 向量化 |
| app/providers/* | LLM/Embedding/Reranker 的抽象与多实现 | 业务流程 |
| eval/* | 评测集、指标、消融与真实语料评测 | — |

## 五、扩展点

1. **换 LLM/Embedding/Reranker**：在 providers/ 加一个实现类 + `.env` 一行切换；向量模型切换后按模型名隔离 collection，重灌即可。
2. **生产化认证**：把 `auth.verify_token` 换成 PyJWT/SSO 校验、`ROLE_LEVELS` 换成 DB/配置中心下发——检索层 ACL 代码不变。
3. **规模化**：Chroma → Milvus/ES（ANN + 增量）、BM25 → Elasticsearch、精排 GPU/API 化；摄入队列化。
4. **新文档格式**：loaders.py 增加一个 loader 函数注册进 LOADERS 即可。

## 六、运行视图

```
一键启动.bat → 自检环境(uv) → 自检模型(models/) → 自检索引 → uvicorn :8000
uvicorn 生命周期: lifespan → auth.seed()（建表+内置账号，幂等）
进程内单例: RAGPipeline（向量库连接 / BM25 / 模型懒加载各一份）
```
