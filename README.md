# epubTranslatorWithLLM

> LLM-based EPUB translator with resumable pipeline, reference-volume consistency, OpenAI-compatible backends, and a local Web UI.

`epubTranslatorWithLLM` 是一个面向长篇 EPUB 的翻译工具。它不是“一次把整章丢给模型机翻”的脚本，而是一条完整流水线：

1. 按 `spine` 顺序读取正文，先做全书上下文整理
2. 冻结章节上下文后并行翻译批次
3. 对每个批次做校对和低分重试
4. 在保留原书 XHTML、目录、资源和大部分元数据的前提下重建输出 EPUB

当前版本同时支持：

- `progress.json` 断点续跑
- 任务失败后的自动续跑
- 前作精翻 EPUB 作为软参考，提取系列译名和文风
- DeepSeek / 阿里云 / 通用 OpenAI-compatible 接口
- `mock` 模式本地联调
- 本地 Web UI：选书、上传、调参、看日志、下载结果

## 适合什么场景

- 翻译轻小说、网文、长篇小说这类上下文强依赖 EPUB
- 系列作品翻译，希望尽量继承前作译名和语气
- 任务很长，不能接受中断后从头再跑
- 需要反复试模型、试参数，希望直接用本地网页操作

## 核心能力

- 按 EPUB `spine` 顺序处理正文，避免章节顺序错乱
- 摘要阶段串行，翻译阶段按批次并行，兼顾上下文稳定和速度
- 支持 `--reference-epub`，先抽取前作参考画像，再参与摘要、翻译和校对
- 支持批次级 review；分数过低会按反馈自动重翻
- 支持 `--provider mock`，方便离线跑通流程和测试
- 支持任务失败后自动复用已有 `progress.json` 再续跑
- 尽量保留原始 XHTML 结构，而不是全部重建成纯文本模板
- Web UI 内置文件发现、上传、实时进度、日志和结果下载

## 工作流概览

### 1. 可选 reference phase

如果提供 `--reference-epub`，程序会先读取一册已经翻译好的前作，提取系列中的惯用译名、表达习惯和风格提示，作为后续阶段的软参考。

### 2. summary phase

按正文真实顺序为每章生成摘要和上下文状态，避免后续翻译时因为章节乱序导致角色关系或设定漂移。

### 3. translation phase

每章会被切成多个片段批次；每个批次都在“当前章节冻结上下文”上并发翻译，而不是在翻译过程中不断漂移上下文。

### 4. review and rebuild

每个批次翻译后会做结构化校对。低于阈值时自动重试。全部完成后，把译文回写进原 EPUB 结构并生成新书。

## 安装

推荐使用项目本地虚拟环境：

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

如果暂时不用 `uv`：

```bash
python3 -m pip install -r requirements.txt
```

推荐 Python `3.9+`。

## 环境变量与 Provider

默认的自动探测顺序如下：

1. `EPUB_TRANSLATOR_API_KEY`
2. `OPENAI_API_KEY`
3. `ALIYUN_API_KEY`
4. `DEEPSEEK_API_KEY`

常用可选变量：

```bash
export EPUB_TRANSLATOR_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export EPUB_TRANSLATOR_MODEL=qwen-max
```

也兼容旧变量：

```bash
export ALIYUN_API_KEY=your_aliyun_api_key
export DEEPSEEK_API_KEY=your_deepseek_api_key
```

支持的运行模式：

- `--provider auto`：按环境变量自动选择可用后端
- `--provider openai-compatible`：显式走 OpenAI-compatible 接口
- `--provider mock`：不调用真实模型，适合联调和测试

## 快速开始

### Web UI

推荐日常直接用 Web UI：

```bash
uv run --python .venv/bin/python webui.py
```

默认地址：

```text
http://127.0.0.1:7860
```

页面里可以直接：

- 从项目目录里选择已有 `.epub`
- 手动填写绝对路径
- 上传一个新的 `.epub`
- 再额外指定一个前作参考 EPUB
- 选择 `DeepSeek / 阿里云 / OpenAI Compatible / Mock`
- 调整模型、并发、批次大小、校对阈值等参数
- 实时查看摘要进度、翻译进度、活动 worker 和日志
- 任务完成后直接下载输出 EPUB

### CLI

如果你更偏向命令行，也可以直接运行 `main.py`。不传 `--input` 时，会默认读取 `testBook/` 下找到的第一本 EPUB。

最小示例：

```bash
uv run --python .venv/bin/python main.py \
  --input testBook/yourbook.epub \
  --source-lang 日语 \
  --target-lang 中文
```

带前作参考：

```bash
uv run --python .venv/bin/python main.py \
  --input testBook/yourbook.epub \
  --reference-epub /path/to/previous-volume.zh.epub \
  --source-lang 日语 \
  --target-lang 中文 \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen-max
```

本地联调，不请求真实模型：

```bash
uv run --python .venv/bin/python main.py \
  --input testBook/yourbook.epub \
  --source-lang 日语 \
  --target-lang 中文 \
  --provider mock \
  --title-suffix "（中文译本）"
```

默认输出路径：

```text
epubOutput/<原文件名>.<目标语言>.epub
```

## 常用参数

- `--input`：待翻译 EPUB 路径
- `--reference-epub`：前作精翻参考 EPUB 路径
- `--output`：输出文件路径或输出目录
- `--source-lang` / `--target-lang`：源语言和目标语言
- `--model`：统一指定默认模型
- `--summary-model` / `--translation-model` / `--review-model`：分阶段指定模型
- `--translation-workers 4`：翻译阶段 worker 数
- `--auto-resume-retries 2`：任务失败后自动续跑次数
- `--progress-file progress.json`：断点续跑文件
- `--reset-progress`：忽略现有 progress，从头开始
- `--max-batch-chars 3500`：单批次最大字符数
- `--max-batch-segments 64`：单批次最大片段数
- `--max-review-retries 2`：校对失败时重试次数
- `--min-review-score 85`：低于该分数触发重翻
- `--recent-summary-limit 5`：向后续批次传递的最近摘要数量
- `--title-suffix "（中文译本）"`：给输出书名加后缀

## 输出与状态文件

- `epubOutput/`：默认输出目录
- `progress.json`：CLI 默认断点文件
- `.webui/uploads/`：Web UI 上传文件缓存
- `.webui/progress/`：Web UI 每个任务的独立进度文件

更换输入书、源/目标语言或参考 EPUB 时，旧进度会自动失效并重新开始，避免错误复用。

## 项目结构

```text
epubTranslatorWithLLM/
├── main.py
├── webui.py
├── requirements.txt
├── translator/
│   ├── cli.py
│   ├── config.py
│   ├── epub_utils.py
│   ├── llm.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── state.py
│   ├── webapp.py
│   ├── static/
│   └── templates/
├── tests/
│   ├── test_epub_utils.py
│   ├── test_llm.py
│   ├── test_pipeline.py
│   ├── test_state.py
│   └── test_webapp.py
├── epubTemplates/
└── README.md
```

模块说明：

- `main.py`：CLI 入口
- `webui.py`：本地 Web UI 入口
- `translator/pipeline.py`：完整翻译流水线
- `translator/llm.py`：OpenAI-compatible 和 mock 客户端封装
- `translator/epub_utils.py`：EPUB 解析、分批、内容回写
- `translator/state.py`：进度文件与上下文状态管理
- `translator/webapp.py`：Flask UI、后台任务和下载接口
- `tests/`：无网络的本地回归测试

## 测试

```bash
uv run --python .venv/bin/python -m unittest discover -s tests -q
```

## 许可证

本项目使用 [MIT 许可证](https://opensource.org/licenses/MIT)，你可以自由使用、修改和分发本项目。
