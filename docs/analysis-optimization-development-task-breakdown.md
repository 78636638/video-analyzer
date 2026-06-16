# 视频分析系统优化开发任务拆解清单

## 1. 文档用途

本文档基于以下两份已复审文档拆解开发任务：

- [analysis-optimization-requirements.md](file:///home/waylin/project/video-analyzer/docs/analysis-optimization-requirements.md)
- [analysis-optimization-technical-design.md](file:///home/waylin/project/video-analyzer/docs/analysis-optimization-technical-design.md)

目标：

- 将需求与技术设计进一步细化到文件级别。
- 用于指导后续开发排期、评审、联调与验收。
- 保证改造对现有系统以增强为主，保留兼容与回退能力。

说明：

- 涉及核心链路改造、prompt 改造、结构化输出与剧本辅助输出，均需多人评审后实施。
- 本清单按“先稳性能，再提质量，最后做长视频强化与回归”的顺序拆分。

## 2. 复审结论

经重新审查，当前需求文档与技术设计文档在补充“剧本辅助输出”和“不降级发布策略”后，已经满足以下目标：

- 满足当前用户需求
  - 覆盖图片理解、识别与长视频分析优化
  - 覆盖音视频协同
  - 覆盖高质量分析结果输出
  - 覆盖为后续剧本/视频生成提供结构化素材
- 对当前系统能力是增强
  - 保留旧入口、旧主流程、旧输出主字段
  - 新增功能以附加字段和配置开关形式引入
- 兼容性较好
  - CLI 保持兼容
  - UI 保持基本兼容
  - JSON 输出采用增量扩展
- 不降级有保障
  - 文档已要求保留回退开关
  - 文档已要求灰度/双写验证

结论：

- 可以进入开发任务拆解与实施阶段。

## 3. 总体开发原则

### 3.1 保持统一入口

- 业务主入口只能是 `video_analyzer/cli.py`
- UI 不承载分析逻辑
- client 层不承载业务状态管理

### 3.2 保持统一收口

- 上下文状态、人物注册表、剧情记忆、块级摘要统一收口到 `VideoAnalyzer`
- 输出汇总统一在 `cli.py`
- 配置统一从 `Config` 读取

### 3.3 保持兼容和回退

- 保留旧字段：`metadata`、`transcript`、`frame_analyses`、`video_description`
- 保留旧 `response` 文本
- 新字段增量扩展
- 保留旧逻辑回退开关

## 4. 文件级开发任务拆解

## 4.1 配置与入口层

### T001 `video_analyzer/config/default_config.json`

任务目标：

- 增加 `analysis` 配置节点，承载本次优化全部新增参数。

改动内容：

- 新增：
  - `analysis.context_window`
  - `analysis.legacy_full_history`
  - `analysis.memory_update_interval`
  - `analysis.scene_change_threshold`
  - `analysis.audio_alignment_window`
  - `analysis.enable_chunk_summary`
  - `analysis.chunk_max_frames`
  - `analysis.long_video_threshold_minutes`
  - `analysis.max_image_side`
  - `analysis.image_jpeg_quality`
  - `analysis.reconstruction_frame_cap`
- 评估是否需要在 `prompts` 列表中增加：
  - `Chunk Summary`
  - `Script Guidance`

输出结果：

- 默认配置可被直接读取。

风险与注意：

- 若扩展 `prompts` 列表，需同步处理 prompt 加载逻辑，避免索引位置耦合。

需多人评审：

- 默认阈值取值。

### T002 `video_analyzer/config.py`

任务目标：

- 支持新增 `analysis` 配置读取。

改动内容：

- 增加 `analysis` 节点读取逻辑。
- 保持现有级联优先级不变。
- 评估是否为后续 CLI 参数覆盖预留映射入口。

输出结果：

- 新配置可从 `default_config.json/config.json` 正常读取。

风险与注意：

- 不要破坏已有 `clients`、`audio`、`frames` 配置读取行为。

### T003 `video_analyzer/cli.py`

任务目标：

- 扩展主编排流程，引入音频摘要、块级摘要、扩展 JSON 输出。

改动内容：

- 在现有音频提取与转写后，增加：
  - `audio_memory` 构建
  - `audio_summary` 构建
- 初始化 `VideoAnalyzer` 时注入 `analysis` 配置。
- 在逐帧分析结束过程中增加：
  - 块级摘要冲刷
  - 长视频策略日志
- 输出 `analysis.json` 时新增字段：
  - `audio_summary`
  - `chunk_summaries`
  - `story_memory`
  - `scene_cards`
  - `story_beats`
  - `character_timeline`
  - `script_guidance`

输出结果：

- CLI 仍按原命令方式运行，但结果信息更完整。

风险与注意：

- 旧字段必须完整保留。
- 即使新字段生成失败，也不能影响 `analysis.json` 产出。

## 4.2 Prompt 加载与模板层

### T004 `video_analyzer/prompt.py`

任务目标：

- 降低 prompt 通过索引读取的脆弱性。

改动内容：

- 保持 `get_by_index()` 兼容。
- 新增或强化基于名字的加载使用方式。
- 在 `VideoAnalyzer` 新逻辑中优先使用 `get_by_name()`，避免后续增加 prompt 文件时因索引变化引发破坏。

输出结果：

- prompt 可稳定按名称加载。

风险与注意：

- 当前已有逻辑仍可能依赖索引，必须保持兼容。

### T005 `video_analyzer/prompts/frame_analysis/frame_analysis.txt`

任务目标：

- 将帧级 prompt 从“全历史文本输入”改造为“最近窗口 + 全局记忆 + 音频片段”。

改动内容：

- 新增占位符：
  - `{RECENT_WINDOW}`
  - `{STORY_MEMORY}`
  - `{AUDIO_SNIPPET}`
- 保留 `{PREVIOUS_FRAMES}` 兼容占位符。
- 约束模型输出结构化字段。

输出结果：

- 帧级识别结果既有自然语言，又能提取结构化信息。

风险与注意：

- Prompt 改造会影响输出风格，需真实样本验证。

需多人评审：

- 结构化字段定义。

### T006 `video_analyzer/prompts/frame_analysis/describe.txt`

任务目标：

- 将最终总结输入改造成适配长视频的综合摘要输入。

改动内容：

- 降低对“全量逐帧原文”的依赖。
- 增加以下输入语义：
  - 块级摘要
  - 全局剧情记忆
  - 音频摘要
  - transcript 原文
- 保证最终 `video_description` 继续输出自然语言总结。

输出结果：

- 最终总结在长视频下仍然稳定。

### T007 新增 `video_analyzer/prompts/frame_analysis/chunk_summary.txt`

任务目标：

- 提供块级摘要 prompt。

改动内容：

- 输入块内结构化帧结果。
- 输出：
  - 块摘要
  - 关键人物
  - 关键事件
  - 场景推进

输出结果：

- 支持块级总结与全局记忆更新。

风险与注意：

- 该文件为新增文件，属于必要新增。

### T008 新增 `video_analyzer/prompts/frame_analysis/script_guidance.txt`

任务目标：

- 为剧本/视频生成辅助输出提供独立 prompt。

改动内容：

- 输入：
  - `scene_cards`
  - `story_beats`
  - `character_timeline`
- 输出：
  - `script_guidance`

输出结果：

- 结果可直接作为后续剧本编写或视频生成提示词素材。

风险与注意：

- 该输出只能是增强层，不能替代主视频总结。

## 4.3 分析核心层

### T009 `video_analyzer/analyzer.py` 数据结构改造

任务目标：

- 引入统一状态中心与结构化结果对象。

改动内容：

- 新增 dataclass：
  - `CharacterProfile`
  - `StructuredFrameAnalysis`
  - `ChunkSummary`
  - `AudioSnippet`
  - `StoryMemory`
- 在 `VideoAnalyzer` 中新增成员：
  - `recent_frame_window`
  - `chunk_buffer`
  - `chunk_summaries`
  - `story_memory`
  - `audio_memory`
  - `analysis_config`

输出结果：

- 状态对象统一，不再靠隐式 prompt 文本维持上下文。

### T010 `video_analyzer/analyzer.py` 帧级分析流程改造

任务目标：

- 将 `analyze_frame()` 改造成“当前帧 + 有界上下文 + 音频片段”的新逻辑。

改动内容：

- 增加：
  - `set_audio_memory()`
  - `build_frame_prompt()`
  - `parse_frame_response()`
  - `update_recent_window()`
  - `update_story_memory()`
  - `update_character_registry()`
- `analyze_frame()` 内：
  - 查询音频片段
  - 读取最近 K 帧窗口
  - 读取全局剧情记忆
  - 调用模型
  - 解析结构化结果
  - 更新状态

输出结果：

- 单帧处理耗时不再随视频长度线性膨胀。

风险与注意：

- `parse_frame_response()` 必须支持解析失败降级。

### T011 `video_analyzer/analyzer.py` 兼容与回退逻辑

任务目标：

- 保证旧逻辑可回退。

改动内容：

- 保留 `previous_analyses` 旧路径。
- 增加 `analysis.legacy_full_history` 开关。
- 当开关打开时，继续走旧逻辑。
- 当开关关闭时，走窗口化新逻辑。

输出结果：

- 新旧策略可对比回归。

风险与注意：

- 这是不降级保障的关键任务。

### T012 `video_analyzer/analyzer.py` 块级摘要机制

任务目标：

- 支持长视频分块处理。

改动内容：

- 新增：
  - `should_flush_chunk()`
  - `summarize_chunk()`
  - `flush_chunk_if_needed()`
- 触发条件：
  - 达到 `chunk_max_frames`
  - 检测到场景切换
  - 视频分析结束

输出结果：

- `chunk_summaries` 可稳定产出。

### T013 `video_analyzer/analyzer.py` 剧本辅助输出生成

任务目标：

- 基于中间产物派生：
  - `scene_cards`
  - `story_beats`
  - `character_timeline`
  - `script_guidance`

改动内容：

- 新增方法建议：
  - `build_scene_cards()`
  - `build_story_beats()`
  - `build_character_timeline()`
  - `build_script_guidance()`

输出结果：

- 分析结果可直接支持后续剧本/视频生成。

风险与注意：

- 若 `script_guidance` 生成失败，前 3 类结构仍应通过规则拼装产出。

## 4.4 音频处理层

### T014 `video_analyzer/audio_processor.py`

任务目标：

- 保持现有转写主链路，同时支持音频片段化和音频摘要准备。

改动内容：

- 保留：
  - `extract_audio()`
  - `transcribe()`
- 增加辅助方法或由 `VideoAnalyzer` 接管：
  - `build_audio_snippets()`
  - `summarize_audio_segments()`

输出结果：

- 转写结果可被帧级分析直接使用。

风险与注意：

- 音频抽取与转写仍应归 `AudioProcessor`。
- 剧情语义整理建议由 `VideoAnalyzer` 统一收口。

## 4.5 图像与抽帧层

### T015 `video_analyzer/clients/llm_client.py`

任务目标：

- 增加统一图片缩放与压缩入口。

改动内容：

- 扩展 `encode_image()`：
  - 最长边缩放
  - JPEG 质量压缩
  - 保持比例
- 默认不传参数时保持当前行为。

输出结果：

- 降低视觉请求体积，提高吞吐。

### T016 `video_analyzer/clients/ollama.py`

任务目标：

- 接入统一图片压缩参数并补充性能日志。

改动内容：

- 调用新的 `encode_image()` 参数版本。
- 增加日志：
  - prompt 长度
  - 窗口帧数
  - 图片压缩尺寸

输出结果：

- 便于评估长视频性能变化。

### T017 `video_analyzer/clients/generic_openai_api.py`

任务目标：

- 与统一图片压缩策略保持兼容。

改动内容：

- 使用新的 `encode_image()` 参数版本。
- 保持 OpenAI-compatible 请求结构不变。

输出结果：

- 不同 client 下图片压缩行为一致。

### T018 `video_analyzer/frame.py`

任务目标：

- 增加长视频自适应抽帧与场景切换候选支持。

改动内容：

- 可选扩展 `Frame` 字段：
  - `scene_hint`
- 增加长视频 bucket 采样策略。
- 保证每个时间块至少有代表帧。

输出结果：

- 长视频抽帧更均衡，减少无效推理。

风险与注意：

- 不得破坏现有 `extract_keyframes()` 的基础能力。

## 4.6 测试与验证层

### T019 `test_prompt_loading.py`

任务目标：

- 补充 prompt 新增文件与按名称加载的测试。

改动内容：

- 验证 `Frame Analysis`、`Video Reconstruction`、`Chunk Summary`、`Script Guidance` 可被正确加载。
- 验证索引与名称兼容共存。

输出结果：

- prompt 改造不破坏原有加载能力。

### T020 新增测试文件：配置与分析辅助逻辑测试

建议新增文件：

- `tests/test_analysis_config.py`
- `tests/test_audio_alignment.py`
- `tests/test_structured_parse.py`
- `tests/test_story_memory.py`

任务目标：

- 覆盖新增核心逻辑的单元测试。

风险与注意：

- 使用真实业务样本或最小真实结构数据，不使用脱离业务语义的伪造方案。

### T021 集成验证任务

任务目标：

- 基于真实视频样本验证新旧方案差异。

验证样本建议：

- 无音频短视频
- 有对白短视频
- 长视频
- 场景切换频繁视频
- 对话型静态场景视频

输出结果：

- 性能对比报告
- 输出质量对比记录
- 回退策略有效性验证

## 4.7 UI 增量适配层

### T022 `video-analyzer-ui/video_analyzer_ui/server.py`

任务目标：

- 评估是否需要把 `analysis` 配置项透出给 UI。

改动内容：

- 若后续需要可视化调参，则扩展 `/api/config` 返回 `analysis` 节点。
- 若当前阶段不做 UI 改造，可暂缓。

输出结果：

- UI 具备后续增量配置适配的接口基础。

### T023 `video-analyzer-ui/video_analyzer_ui/static/js/main.js`

任务目标：

- 可选显示新增分析参数与新增结果字段。

改动内容：

- 若进入 UI 适配阶段，再追加：
  - `context_window`
  - 长视频模式提示
  - 块级摘要与剧本辅助输出展示

输出结果：

- UI 增强但不阻塞核心后端开发。

## 5. 推荐实施顺序

### 第 1 批：必须先做

- T001
- T002
- T003
- T004
- T009
- T010
- T011
- T014
- T015
- T016
- T017

目标：

- 先把新链路跑通，并建立不降级保障。

### 第 2 批：质量增强

- T005
- T006
- T012
- T013
- T018

目标：

- 提升长视频连贯性、人物一致性、剧本辅助能力。

### 第 3 批：验证与外层适配

- T019
- T020
- T021
- T022
- T023

目标：

- 完成稳定性验证与 UI 增量增强。

## 6. 每批交付定义

### 6.1 第一批交付完成标准

- 新配置可加载
- 新旧逻辑可切换
- 单帧推理耗时不再明显线性增长
- 输出 JSON 不破坏旧字段

### 6.2 第二批交付完成标准

- 块级摘要可产出
- 长视频重建更稳定
- `scene_cards`、`story_beats`、`character_timeline` 可产出

### 6.3 第三批交付完成标准

- 有真实样本回归报告
- 有新旧方案对比结论
- UI 适配项按需完成

## 7. 关键风险提示

- Prompt 结构改造可能改变输出风格，必须灰度验证。
- 结构化解析的稳健性是核心风险点，必须允许降级。
- 新增 prompt 文件会放大现有“按索引读取”的脆弱性，因此 `prompt.py` 的兼容改造必须优先处理。
- 剧本辅助输出属于增强层，不应反向影响主分析链路。

## 8. 最终结论

当前两份文档在补充后，已满足进入开发拆解阶段的条件。

本清单可直接作为后续开发排期、评审和实施的文件级任务依据使用。
