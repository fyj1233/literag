"""下载本地模型到项目 models/ 目录（约 1.2GB，只需执行一次）。

    python scripts/download_models.py

说明：
- 默认走 .env 中的 HF_ENDPOINT 镜像（国内网络友好）；
- 只下载运行所需的 safetensors 权重与配置/词表，跳过冗余的 bin/onnx 格式；
- 下载后模型以普通文件夹存放在 models/ 下，项目离线加载，不依赖 HuggingFace 缓存。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR, settings  # noqa: E402  （导入即设置 HF_ENDPOINT/HF_HOME）

MODELS = {
    "bge-small-zh-v1.5": [
        "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
        "vocab.txt", "special_tokens_map.json", "modules.json",
        "sentence_bert_config.json", "config_sentence_transformers.json", "1_Pooling/config.json",
    ],
    "bge-reranker-base": [
        "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
        "sentencepiece.bpe.model", "special_tokens_map.json",
    ],
}


def main() -> None:
    from huggingface_hub import hf_hub_download

    for model, files in MODELS.items():
        out_dir = BASE_DIR / "models" / model
        marker = out_dir / "model.safetensors"
        if marker.exists():
            print(f"[skip] {model} 已存在（{out_dir}）")
            continue
        print(f"[download] {model} -> {out_dir}")
        for f in files:
            path = hf_hub_download(repo_id=f"BAAI/{model}", filename=f)
            target = out_dir / f
            target.parent.mkdir(parents=True, exist_ok=True)
            if Path(path) != target:
                shutil.copyfile(path, target)
            print(f"  ok: {f}")
    print("\n完成。模型已就位于 models/ 目录。")


if __name__ == "__main__":
    import shutil  # noqa: E402

    main()
