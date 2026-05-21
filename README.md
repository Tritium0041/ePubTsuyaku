# epubTranslatorWithLLM

这是一个基于 LLM 的 EPUB 电子书翻译工具。当前版本已经从早期单文件脚本重构为完整流水线，按你博客里的方法实现了：

1. `全书顺序定调/要素抽取`
2. `按冻结章节上下文并行初翻`
3. `逐批校对`
4. `断点恢复与结果重建`

同时补上了断点续跑、OpenAI 兼容接口接入、章节顺序修复、目录与资源保留、本地 `mock` 联调模式，以及一个本地可直接使用的 Web UI。

## 现在支持的能力

- 按 EPUB `spine` 顺序处理正文，避免早期版本的章节顺序错乱。
- 先按章节顺序完成全书摘要，再基于每章冻结好的上下文快照并行翻译批次。
- 可选接入一册已经精翻完成的前作 EPUB，先抽取系列惯用译名和文风，再作为软参考参与后续摘要、翻译和校对。
- 翻译后自动做一轮校对；如果分数过低或发现明显问题，会按反馈重翻。
- DeepSeek 官方接口会优先走严格 schema / JSON mode，并关闭 thinking 以提高结构化输出稳定性。
- 尽量保留原始 XHTML 结构，而不是把章节全部重建成纯段落模板。
- 保留原书的资源、目录和大部分元数据，并输出新的 EPUB。
- 使用 `progress.json` 做断点恢复；摘要阶段、已完成批次和已完成章节都会被保留。
- 支持任意 OpenAI 兼容接口，也保留了对 `ALIYUN_API_KEY` / `DEEPSEEK_API_KEY` 的兼容。
- 支持 `--provider mock`，方便本地跑通流程和做测试。
- 支持 `Web UI`：页面里直接选书、填 key、看实时进度、下载输出 EPUB。

## 安装

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

如果你暂时不用 `uv`，也可以继续用：

```bash
python3 -m pip install -r requirements.txt
```

## 环境变量

优先级从高到低如下：

1. `EPUB_TRANSLATOR_API_KEY`
2. `OPENAI_API_KEY`
3. `ALIYUN_API_KEY`
4. `DEEPSEEK_API_KEY`

可选变量：

```bash
export EPUB_TRANSLATOR_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export EPUB_TRANSLATOR_MODEL=qwen-max
```

如果你沿用旧配置，也可以直接：

```bash
export ALIYUN_API_KEY=your_aliyun_api_key
export DEEPSEEK_API_KEY=your_deepseek_api_key
```

## 快速开始

### 1. Web UI

推荐直接起页面：

```bash
uv run --python .venv/bin/python webui.py
```

默认打开本地服务：

```text
http://127.0.0.1:7860
```

页面里可以直接：

- 选择项目目录下已有的 `.epub`
- 或手动填绝对路径
- 或上传一个新的 `.epub`
- 可选再给一个“前作精翻参考 EPUB”，用于提取系列既有译名和文风
- 配置 `DeepSeek / 阿里云 / OpenAI Compatible / Mock`
- 填 `API Key`、模型、批次大小、片段上限、并发数、重试次数
- 实时查看摘要进度、翻译批次进度、活动 worker 和日志
- 任务完成后直接下载输出文件

### 2. CLI

如果你更想走命令行：

把待翻译的书放进 `testBook/`，或者直接通过 `--input` 指定路径。

最简单的运行方式：

```bash
uv run --python .venv/bin/python main.py --input testBook/yourbook.epub --source-lang 日语 --target-lang 中文
```

显式指定 OpenAI 兼容接口：

```bash
uv run --python .venv/bin/python main.py \
  --input testBook/yourbook.epub \
  --reference-epub /path/to/previous-volume.zh.epub \
  --source-lang 日语 \
  --target-lang 中文 \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen-max
```

本地联调，不调用真实模型：

```bash
uv run --python .venv/bin/python main.py \
  --input testBook/yourbook.epub \
  --source-lang 日语 \
  --target-lang 中文 \
  --provider mock \
  --title-suffix "（中文译本）"
```

默认输出到 `epubOutput/<原文件名>.<目标语言>.epub`。

## 常用参数

- `--progress-file progress.json`：指定断点续跑文件。
- `--reset-progress`：忽略已有进度，从头重新翻。
- `--reference-epub /path/to/book.epub`：可选，提供前作精翻 EPUB 作为系列软参考。
- `--max-batch-chars 3500`：控制单次发给模型的字符数。
- `--max-batch-segments 64`：控制单次发给模型的片段数，避免碎片文本一次塞太多 id。
- `--translation-workers 4`：翻译阶段的并发 worker 数。
- `--max-review-retries 2`：校对不通过时的最大重试次数。
- `--min-review-score 85`：低于该分数时触发重试。
- `--summary-model / --translation-model / --review-model`：按阶段拆分模型。
- `--title-suffix "（中文译本）"`：给输出书名加后缀。

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
│   ├── test_pipeline.py
│   ├── test_state.py
│   └── test_webapp.py
├── epubTemplates/
└── README.md
```

说明：

- `main.py`：CLI 入口。
- `webui.py`：本地 Web UI 入口。
- `translator/pipeline.py`：完整翻译流水线。
- `translator/llm.py`：OpenAI 兼容客户端和 mock 客户端。
- `translator/epub_utils.py`：EPUB 文档提取、分批、内容回写。
- `translator/state.py`：上下文状态和 progress 持久化。
- `translator/webapp.py`：Flask UI、后台任务调度与下载接口。
- `tests/`：无网络的本地回归测试。
- `epubTemplates/`：早期模板文件，当前流程默认保留原 XHTML 结构，不再依赖模板重建正文。

## 测试

```bash
uv run --python .venv/bin/python -m unittest discover -s tests -q
```

## 兼容性说明

- 推荐 Python 3.9+。
- 摘要阶段按章节顺序串行执行；翻译阶段按批次并行执行，默认 `4` 个 worker。
- 如果提供了 `--reference-epub`，会先串行执行一个 `reference phase`，提取前作的惯用译名与风格提示；更换参考书后会自动使旧 progress 失效并重跑。
- `progress.json` 会保留中间结果，便于中断恢复，也便于手工检查上下文和校对结果。

## 许可证

本项目使用 [MIT许可证](https://opensource.org/licenses/MIT)，你可以自由使用、修改和分发本项目。
