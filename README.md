# Alice Chat —— 基于积温情绪引擎开发的人格训练Agent

一个有"人格"的桌面 AI 聊天应用。基于**五轴微分方程情绪引擎**驱动角色情感状态，支持 LLM 对话、图片生成、人格训练。

---

## 功能

- **情绪模拟**：五轴积温引擎 —— 思念(Connection)、骄傲(Pride)、愉悦(Valence)、唤醒(Arousal)、沉浸(Immersion)
- **角色人格**：基于源文本自动提取角色画像（语气、用词、行为模式），RAG 检索注入
- **多 LLM 支持**：DeepSeek / OpenAI / Anthropic（纯文本接口，自动安装 SDK）
- **图片生成**：4 种后端 —— Replicate / DALL-E / SD WebUI / 本地 Diffusers
- **人格工坊**：从对话文本中提取角色 → 训练风格 → 增量学习 → 反馈优化
- **记忆系统**：SQLite + ChromaDB 双层记忆，支持对话摘要和语义检索

---

## 环境配置

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

### 2. 安装依赖

```bash
.venv\Scripts\pip install -r requirements.txt
```

### 3. 启动

```
双击 启动.bat
```

或终端运行：

```bash
python run.py
```

---

## 连接 LLM

打开设置面板 → **LLM API** 区域：

| 提供商 | 需要 |
|--------|------|
| **DeepSeek** | 注册 [deepseek.com](https://platform.deepseek.com) 获取 API Key，Base URL 留空 |
| **OpenAI** | API Key，Base URL 可留空 |
| **Anthropic** | API Key |

点击 **连接 LLM**，验证通过后状态栏显示 🟢。

---

## 图片生成（可选）

打开设置面板 → **🖼 图片生成** 区域，选择引擎：

### Replicate（推荐云端方案）
- 注册 [replicate.com](https://replicate.com) 获取 API Key
- 引擎选 `Replicate`，输入 Key，点击连接
- 默认模型：`black-forest-labs/flux-2-pro`

### OpenAI DALL-E 3
- 使用 OpenAI API Key
- 引擎选 `OpenAI (DALL-E)`

### SD WebUI（本地已有服务）
- 先启动 AUTOMATIC1111 / ComfyUI
- 引擎选 `SD WebUI`，填写地址（默认 `http://localhost:7860`）

### Diffusers 本地模型（需要自行下载模型文件）

**首先安装 CUDA 版 PyTorch**（普通 `pip install torch` 装的是 CPU 版，会极慢）：

```bash
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**然后下载 Stable Diffusion 模型文件**：

1. 访问 [huggingface.co/runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
2. 下载所有文件到本地文件夹（默认 `alice/model/AI-ModelScope/stable-diffusion-v1-5/`）
3. 必须包含：`model_index.json`、`unet/`、`vae/`、`text_encoder/`、`tokenizer/`、`scheduler/`、`safety_checker/`、`feature_extractor/`

**文件结构示例**：

```
alice/model/AI-ModelScope/stable-diffusion-v1-5/
├── model_index.json
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── text_encoder/
│   ├── config.json
│   └── model.safetensors
├── tokenizer/
│   ├── vocab.json
│   ├── merges.txt
│   └── ...
├── scheduler/
│   └── scheduler_config.json
├── safety_checker/
│   ├── config.json
│   └── model.safetensors
└── feature_extractor/
    └── preprocessor_config.json
```

> **注意**：模型文件约 8GB，`.safetensors` 和 `.bin` 文件较大，确保网络稳定。项目中 `.gitignore` 已排除 `alice/model/` 目录。

引擎选 `Diffusers (本地)` → 点击 📁 浏览选择上述文件夹 → 点击连接。首次加载需数十秒。

---

## 人格训练

1. 打开 **角色工坊** 面板
2. 输入角色名，粘贴对话文本（支持 `.txt` / `.md` 批量导入）
3. 点击 **多阶段提取** → LLM 自动分析角色画像 + 引擎参数 + 风格示例
4. 点击 **保存角色** → 写入 `presets/` 目录
5. 后续对话中 RAG 自动检索风格示例注入提示词

详见 [人格训练框架技术文档](docs/人格训练框架技术文档.md)

---

## 项目结构

```
alice/
├── main.py              # 桌面应用入口
├── run.py               # 启动器
├── requirements.txt     # Python 依赖
├── presets/             # 角色预设（JSON 配置）
├── docs/                # 技术文档
├── alice/
│   ├── core/
│   │   ├── agent.py     # 人格对话代理 + Agent Loop
│   │   └── engine.py    # 五轴积温情绪引擎
│   ├── llm/
│   │   ├── adapters.py  # LLM 适配器（DeepSeek/OpenAI/Anthropic）
│   │   └── prompts.py   # 系统提示词 + 工具定义
│   ├── image/           # 图片生成适配器
│   ├── training/        # 人格提取 + 训练 + RAG
│   ├── memory/          # 对话记忆系统
│   ├── storage/         # 预设/存档管理
│   └── ui/              # PyQt5 界面
└── data/                # 本地数据（不提交 git）
    ├── settings.json    # 用户配置（含 API Key）
    ├── style_store.db   # 风格示例库
    └── chroma/          # 向量数据库
```

---

## 角色预设

内置三个示例预设：

| 预设 | 角色 | 来源 |
|------|------|------|
| 默认 | Clara（通用助手） | 内置 |
| alysia | 爱莉希雅 | 崩坏3 |
| 流萤_liuying | 流萤（萨姆） | 崩坏：星穹铁道 |

---

## 技术要点

- **情绪引擎**：五轴微分方程系统，思念对数增长、骄傲回归静息值、愉悦二阶弹簧阻尼、唤醒资源模型
- **图片生成**：LLM 在文本中输出 `[IMG: 描述]` 标记，Agent 解析后调用图片 API 生成
- **人格 RAG**：SQLite + ChromaDB 双层存储，对话时语义检索角色风格示例注入系统提示词
- **Agent Loop**：LLM 输出 → 解析 ` ```tool:send_image``` ` 工具块 → 执行 → 反馈结果 → 循环（最多 5 次）

---

## 常见问题

**Q: 连接 LLM 时提示"需要安装 openai SDK"？**
自动安装失败时手动运行：`pip install openai`

**Q: DeepSeek 连接超时？**
检查网络。Base URL 留空会自动填 `https://api.deepseek.com/v1`

**Q: 说"看看自拍"后只有文字描述？**
需要在设置面板连接图片引擎（LLM 和图片生成是独立配置的）

**Q: 本地模型 CPU 占用高？**
装了 CPU 版 PyTorch，按上方 Diffusers 步骤替换为 CUDA 版
