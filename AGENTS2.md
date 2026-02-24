# AGENTS2：仅基于英文 SRT 的离线 TTS 工作流

## 0) 目标
仅根据英文 SRT 的文本与时间轴，结合 `ref-audio/` 下参考音色，使用本地离线 `IndexTTS2` 合成语音，输出 `FLAC` 到根目录 `output/<name>/`，并严格对齐字幕时间轴。

## 1) 输入与输出
### 输入（必须）
- 英文字幕文件：`*.en.srt`（默认递归扫描 `project_root/output/**`）
- 参考音色：仅从 `project_root/ref-audio` 目录读取音频（支持 `wav/mp3/flac/m4a/aac/ogg`）
  - 若存在 `tiny.*`，默认优先使用 `tiny.*`
  - 若不存在 `tiny.*`，自动选择该目录下其他可用音频
- 模型目录：默认自动选择
  - 默认：`models/indexTTS2`（使用 `models/indexTTS2/checkpoints`）

### 输入（可选）
- `--srt`：指定单个 SRT 文件
- `--model_dir`：显式指定模型目录

### 多文件队列规则（强制）
- 若使用 `--srt` 指定单个文件：仅对该文件执行一次完整流程。
- 若未指定 `--srt`：递归扫描 `project_root/output/**` 下所有 `*.en.srt`。
- 当检测到多个英文 SRT（包括位于 `output/` 不同子目录）时，必须将全部文件加入队列，按修改时间从新到旧逐个执行，每个 SRT 完整执行一次本 AGENTS2.md 的全部阶段（A→B→C→D→E）后再处理下一个。
- 若未检测到任何英文 SRT，返回输入发现失败（返回码 `2`）。

### 输出（固定）
- 输出目录：`<project_root>/output/<name>/`
- 输出文件名：
  - `output/demo/demo.en.srt` -> `output/demo/demo.flac`
  - `output/demo/demo.srt` -> `output/demo/demo.flac`
- 报告：`output/reports/<name>.report.json`
- 统一规格：`22.05kHz`、`stereo`、`16-bit`、`256kbps`

## 2) 强约束
- 不读取根目录 MP3 作为文本来源。
- 文本只来自英文 SRT。
- 参考音色文件仅用于音色参考，不参与转录。
- 严格按 SRT 时间轴合成与拼接。
- SRT 自动修复只允许在“字幕格式错误”时触发；禁止在格式合法时做任何文本、语义、时间线或结构改写。
- 支持离线本地运行；优先使用 `models/` 下本地模型资产，仅本地缺失时才下载。
- 所有任务执行完毕后，必须退出模型加载并执行缓存/显存清理。

## 3) 执行流程
### 阶段 A：SRT 解析
- 解析序号、时间轴、文本。
- 校验时间合法、递增与结构完整。
- 生成 `expected_total_ms`。
- 若检测到 SRT 格式错误，必须先执行最小化自动修复再重试解析（不等待人工确认）：
  - 仅处理格式问题条目：空文本条目、异常空行、常见时间轴分隔符写法错误（如 `.` 与 `,`、`->` 与 `-->`）。
  - 仅修复可解析性所需的最小范围；不改动正常条目的文本内容、语义、翻译、时间线逻辑与顺序，不做额外重排。
  - 非“字幕格式错误”不得触发自动修复（例如术语替换、文案润色、断句风格化、时轴重算均禁止）。
  - 自动修复后仍不合法的格式条目可跳过，其余条目继续执行后续阶段。

### 阶段 B：逐条 TTS 合成（默认）
- 默认 `--batch_items 1`，逐条字幕处理。
- 每条字幕文本使用同一参考音色文件。
- 产出中间 wav（后续统一做时长对齐）。
- 后台监视窗口必须输出 `tts_to_flac progress: xx.x% stage=tts_align`。

### 阶段 C：时长对齐
- 目标时长 = `end_ms - start_ms`。
- 超长：优先 time-stretch（保音高），必要时 trim。
- 过短：补零或噪声底（按 `--gap_fill_mode`）。
- 目标误差控制：`<= 10ms`。

### 阶段 D：时间线拼接与导出
- 严格按字幕起止时间拼接。
- 插入字幕间隙（gap）。
- 导出 `FLAC` 并校验总时长误差 `<= 10ms`。
- 后台监视窗口必须继续输出百分比：
  - `tts_to_flac progress: 95.0% stage=timeline`
  - `tts_to_flac progress: 100.0% stage=flac_export`

### 阶段 E：收尾清理（强制）
- 关闭 TTS 推理 worker / 模型实例。
- 清理运行缓存并尝试释放显存。
- 写入收尾日志后再退出进程。

## 4) CLI
### 默认运行（递归扫描 `output/**/*.en.srt`）
```bash
py -3 -m app --project_root . --model_dir models/indexTTS2
```

### 指定单个 SRT
```bash
py -3 -m app \
  --project_root . \
  --srt "./output/houdini agent/houdini agent.en.srt" \
  --ref_wav "./ref-audio/TINY.wav"
```

### 强制指定模型目录
```bash
py -3 -m app \
  --project_root . \
  --srt "./output/houdini agent/houdini agent.en.srt" \
  --model_dir "./models/indexTTS2" \
  --ref_wav "./ref-audio/TINY.wav"
```

### 进度监控窗口（推荐）
```bash
py -3 -m app \
  --project_root . \
  --srt "./output/houdini agent/houdini agent.en.srt" \
  --model_dir "./models/indexTTS2" \
  --ref_wav "./ref-audio/TINY.wav" \
  --open_progress_window
```

- `--open_progress_window`：自动打开独立 CMD 窗口，实时显示执行进度。
- 单次运行有且仅有一个 CMD 监视窗口。
- `--log_file`：自定义进度日志路径。
- `--quiet`：关闭主终端进度输出（监控窗口/日志仍可用）。
- `--audio_bitrate`：导出音频码率（默认 `256k`）。
- `--mono`：切换为单声道导出（默认双声道）。

## 5) 返回码
- `0`：成功
- `2`：输入发现失败
- `3`：SRT 解析失败
- `4`：TTS 推理失败
- `5`：音频处理或导出失败
