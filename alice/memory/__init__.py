"""
alice.memory — 记忆系统框架

子模块:
  chroma_client  — 共享 ChromaDB 客户端（MemoryRAG + StyleStore 共用）
  store          — SQLite 存储层（全量留存 + FTS5 + 摘要 + 记忆碎片 + 衰减）
  rag            — 三路召回 + RRF 融合 + 异步提取
  manager        — 三层记忆编排器（摘要 + 检索 + 维护）
"""
