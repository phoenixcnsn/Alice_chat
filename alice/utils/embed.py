"""
嵌入工具 — 为 ChromaDB 风格检索提供 embedding 函数

支持:
  - sentence-transformers (本地模型，推荐)
  - OpenAI embeddings API
  - 简单 TF-IDF fallback (无需安装任何库)
"""

from __future__ import annotations

from typing import List, Optional


def create_embed_fn(provider: str = "tfidf", **kwargs) -> Optional[callable]:
    """
    创建 embedding 函数。

    Args:
        provider: "sentence_transformers" | "openai" | "tfidf"
        **kwargs: 传给具体实现的参数

    Returns:
        async function (texts: List[str]) -> List[List[float]]
        如果 provider 不可用则返回 None
    """
    if provider == "sentence_transformers":
        return _create_st_embed_fn(**kwargs)
    elif provider == "openai":
        return _create_openai_embed_fn(**kwargs)
    elif provider == "tfidf":
        return _create_tfidf_embed_fn(**kwargs)
    else:
        return None


# ------------------------------------------------------------
# Sentence-Transformers (推荐: 本地运行，免费，效果好)
# ------------------------------------------------------------
_ST_MODEL = None


def _create_st_embed_fn(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """
    使用 sentence-transformers 的本地模型。
    首次运行会下载模型 (~120MB)。

    pip install sentence-transformers
    """
    global _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        if _ST_MODEL is None:
            _ST_MODEL = SentenceTransformer(model_name)

        async def embed(texts: List[str]) -> List[List[float]]:
            return _ST_MODEL.encode(texts, normalize_embeddings=True).tolist()

        return embed
    except ImportError:
        print("[embed_utils] sentence-transformers 未安装，回退到 TF-IDF")
        return None


# ------------------------------------------------------------
# OpenAI Embeddings
# ------------------------------------------------------------
def _create_openai_embed_fn(
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small",
    base_url: Optional[str] = None,
):
    """
    使用 OpenAI embeddings API。

    pip install openai
    """
    try:
        from openai import AsyncOpenAI

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)

        async def embed(texts: List[str]) -> List[List[float]]:
            response = await client.embeddings.create(
                model=model,
                input=texts,
            )
            return [d.embedding for d in response.data]

        return embed
    except ImportError:
        print("[embed_utils] openai 未安装，回退到 TF-IDF")
        return None


# ------------------------------------------------------------
# TF-IDF Fallback (零依赖，始终可用)
# ------------------------------------------------------------
import math
import re
from collections import Counter


class _TfidfEmbedder:
    """简单的 TF-IDF 嵌入器，作为无依赖回退方案。

    注意: 这是非常基础的实现，向量维度可能不一致。
    仅用于 ChromaDB 不可用或完全没有 embedding 库时的 fallback。
    建议安装 sentence-transformers 获得更好的效果。
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.idf: dict = {}
        self.vocab: list = []
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        # 中文: 按字 + 双字组合
        text = text.lower()
        tokens = []
        # 中文单字
        chinese = re.findall(r'[一-鿿]', text)
        for c in chinese:
            tokens.append(c)
        # 中文双字
        for i in range(len(chinese) - 1):
            tokens.append(chinese[i] + chinese[i + 1])
        # 英文/数字词
        words = re.findall(r'[a-z0-9]+', text)
        tokens.extend(words)
        return tokens

    def fit(self, texts: List[str]):
        """构建 IDF 表"""
        doc_count = len(texts)
        doc_freq = Counter()
        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                doc_freq[token] += 1

        # IDF
        self.idf = {
            token: math.log((doc_count + 1) / (freq + 1)) + 1
            for token, freq in doc_freq.items()
        }
        # 取前 dim 个最高 IDF 的词作为 vocab
        self.vocab = [t for t, _ in sorted(self.idf.items(), key=lambda x: x[1], reverse=True)[:self.dim]]
        self._fitted = True

    def transform(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)

        vectors = []
        for text in texts:
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            vec = [0.0] * self.dim
            for i, word in enumerate(self.vocab):
                vec[i] = tf.get(word, 0) * self.idf.get(word, 0)
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


_tfidf_embedder = None


def _create_tfidf_embed_fn(dim: int = 128):
    """创建 TF-IDF embedding 函数（始终可用）"""
    global _tfidf_embedder
    _tfidf_embedder = _TfidfEmbedder(dim=dim)

    async def embed(texts: List[str]) -> List[List[float]]:
        return _tfidf_embedder.transform(texts)

    print(f"[embed_utils] 使用 TF-IDF fallback (dim={dim})，建议 pip install sentence-transformers")
    return embed
