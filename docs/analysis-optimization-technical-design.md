# 视频分析系统视觉与音频协同优化详细技术设计文档

## 1. 文档说明

本文档是 [analysis-optimization-requirements.md](file:///home/waylin/project/video-analyzer/docs/analysis-optimization-requirements.md) 的技术落地版本，用于指导后续开发实现。

目标：

- 基于当前项目真实代码与现有技术栈，给出可实现的详细技术设计。
- 支持长视频识别、分析与总结。
- 解决逐帧视觉识别随视频长度增长而越来越慢的问题。
- 在性能优化的同时，尽量保证剧情连贯性、人物一致性、场景一致性。
- 将音频处理纳入统一分析链路，而不是仅在最终总结阶段附加使用。

范围说明：

- 本文档不引入新的外部服务依赖作为一阶段前提。
- 一阶段实现基于当前已有模块：`OpenCV`、`Pillow`、`faster-whisper`、`Ollama/OpenAI-compatible API`、现有 prompt 体系。
- 需要新增的逻辑应尽量收口到现有 `cli.py`、`VideoAnalyzer`、`AudioProcessor`、`LLMClient`、`Config` 体系中。

评审说明：

- 本文档涉及核心分析链路、prompt 结构、输出结构扩展，需多人评审后实施。

## 2. 当前系统基线

### 2.1 当前入口与主流程

当前业务主入口位于 [cli.py](file:///home/waylin/project/video-analyzer/video_analyzer/cli.py)。

当前主流程：

1. 加载配置
2. 初始化音频处理器
3. 抽取音频并转写
4. 抽取关键帧
5. 逐帧调用视觉模型分析
6. 汇总帧分析与 transcript
7. 生成最终视频总结
8. 写出 `analysis.json`

### 2.2 当前性能瓶颈

当前瓶颈位于 [analyzer.py](file:///home/waylin/project/video-analyzer/video_analyzer/analyzer.py#L40-L76)：

- `previous_analyses` 保存了全部历史帧分析。
- `analyze_frame()` 每次分析时会将全量历史帧文本拼接到新的 prompt。
- 每一帧只发送 1 张当前图片，但 prompt 会随视频长度线性增长。

结果：

- 单帧耗时随总帧数上升。
- 长视频更容易触发上下文截断。
- 历史文本越多，后续帧推理越慢，且信息有效性反而下降。

### 2.3 当前音频链路限制

当前音频链路位于 [audio_processor.py](file:///home/waylin/project/video-analyzer/video_analyzer/audio_processor.py)：

- 能提取音频并完成转写。
- 能输出分段时间戳。
- 但只在 [reconstruct_video()](file:///home/waylin/project/video-analyzer/video_analyzer/analyzer.py#L84-L124) 阶段把完整 transcript 一次性注入 prompt。

问题：

- 音频信息没有参与帧级分析。
- 无法用于对白辅助、剧情推进约束、人物称谓稳定化。

## 3. 设计目标

### 3.1 功能目标

- 将长视频处理从“全历史累积式分析”改为“有界上下文 + 分层记忆 + 分块汇总”。
- 在帧级分析阶段引入音频片段辅助。
- 输出对开发和后处理更友好的结构化中间结果。
- 让最终分析结果可直接服务后续剧本整理、分镜拆解与视频生成提示词构建。

### 3.2 架构目标

- `cli.py` 继续作为唯一业务入口。
- `VideoAnalyzer` 继续作为统一分析状态中心与 prompt 构造中心。
- `Config` 继续作为唯一配置入口。
- `analysis.json` 继续作为统一结果出口。

### 3.3 非目标

- 一阶段不引入人脸识别、ReID、目标跟踪、独立向量数据库。
- 一阶段不引入新的推理框架或任务队列系统。
- 一阶段不重写 UI，只保留 UI 与 CLI 的配置兼容性。

## 4. 总体架构设计

### 4.1 新的分析架构

优化后的主链路：

1. `CLI` 加载配置并准备输出目录
2. `AudioProcessor` 提取音频并转写
3. 新增 `audio_summary` 构建
4. `VideoProcessor` 抽取关键帧并输出场景候选信息
5. `VideoAnalyzer` 初始化统一状态
6. 逐帧分析：
   - 当前图片
   - 最近 `K` 帧窗口摘要
   - 全局剧情记忆
   - 当前帧对应音频片段摘要
7. 每个分析块结束后生成 `chunk_summary`
8. 最终总结：
   - 帧级结构化结果
   - 块级摘要
   - 全局剧情记忆
   - 音频摘要
   - 原始 transcript
9. 输出扩展后的 `analysis.json`

### 4.2 状态收口原则

所有新增上下文状态统一收口在 `VideoAnalyzer` 内部，不允许在多个模块分散管理。

建议新增逻辑状态对象：

- `analysis_state`
- `recent_frame_window`
- `story_memory`
- `character_registry`
- `audio_memory`
- `chunk_buffer`

这些对象可以作为类属性或 dataclass 挂在 `VideoAnalyzer` 上，但管理入口必须唯一。

## 5. 模块改造设计

### 5.1 `video_analyzer/cli.py`

职责保持不变，但流程编排需要扩展。

#### 5.1.1 需要新增的编排步骤

在现有 Stage 1 和 Stage 2 之间增加音频摘要准备：

1. `extract_audio()`
2. `transcribe()`
3. `build_audio_memory(transcript)`
4. `extract_keyframes()`
5. `analyzer.set_audio_memory(audio_memory)`
6. `analyzer.analyze_frame(frame)`
7. `analyzer.flush_chunk_if_needed()`
8. `analyzer.reconstruct_video(...)`

#### 5.1.2 CLI 侧新增职责

- 将原始 transcript 转换为 `audio_memory` 所需结构。
- 将长视频的分析结果统一写入扩展 JSON。
- 保持旧命令行参数兼容。

#### 5.1.3 输出字段扩展

`analysis.json` 建议新增以下字段：

```json
{
  "metadata": {},
  "transcript": {},
  "audio_summary": {},
  "frame_analyses": [],
  "chunk_summaries": [],
  "story_memory": {},
  "scene_cards": [],
  "story_beats": [],
  "character_timeline": [],
  "script_guidance": {},
  "video_description": {}
}
```

兼容要求：

- 保留现有 `metadata`、`transcript`、`frame_analyses`、`video_description`。
- 新字段均为可选扩展字段。
- 与剧本/视频生成相关的新字段必须是增量追加，不允许替代现有 `video_description`。

### 5.2 `video_analyzer/analyzer.py`

该模块是本次改造的核心。

#### 5.2.1 新职责

除现有逐帧分析与最终重建外，新增以下职责：

- 上下文状态统一管理
- 结构化帧结果生成与解析
- 最近窗口维护
- 全局剧情记忆更新
- 人物注册表更新
- 音频片段按时间对齐
- 块级摘要生成
- 最终总结 prompt 构造

#### 5.2.2 建议新增数据结构

建议在 `analyzer.py` 内增加 dataclass，避免状态散落。

建议结构：

```python
@dataclass
class CharacterProfile:
    character_id: str
    aliases: list[str]
    appearance: list[str]
    roles: list[str]
    last_seen_timestamp: float | None
    last_seen_scene: str | None
    confidence: float

@dataclass
class StructuredFrameAnalysis:
    frame_number: int
    timestamp: float
    scene: str
    characters: list[dict]
    actions: list[str]
    objects: list[str]
    dialogue_hint: str
    continuity_points: list[str]
    raw_response: str
    scene_changed: bool

@dataclass
class ChunkSummary:
    chunk_id: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    summary: str
    key_characters: list[str]
    key_events: list[str]

@dataclass
class AudioSnippet:
    start: float
    end: float
    text: str
    speaker_hint: str | None
    mood_hint: str | None

@dataclass
class StoryMemory:
    scene_summary: str
    characters: dict[str, CharacterProfile]
    key_events: list[str]
    active_props: list[str]
    last_chunk_summary: str
```

#### 5.2.3 `VideoAnalyzer` 新成员

建议新增成员：

```python
self.recent_frame_window: list[StructuredFrameAnalysis]
self.chunk_buffer: list[StructuredFrameAnalysis]
self.chunk_summaries: list[ChunkSummary]
self.story_memory: StoryMemory
self.audio_memory: list[AudioSnippet]
self.analysis_config: dict[str, Any]
```

#### 5.2.4 `VideoAnalyzer` 新方法

建议新增方法：

- `set_audio_memory(audio_memory)`
- `build_frame_prompt(frame, audio_snippets)`
- `parse_frame_response(frame, response_text)`
- `update_recent_window(structured_result)`
- `update_story_memory(structured_result)`
- `update_character_registry(structured_result, audio_snippets)`
- `is_scene_change(frame, structured_result)`
- `should_flush_chunk(structured_result)`
- `summarize_chunk()`
- `build_reconstruction_prompt(...)`
- `serialize_story_memory()`

#### 5.2.5 帧级分析新流程

新的 `analyze_frame()` 建议流程：

1. 根据时间戳从 `audio_memory` 取出附近音频片段
2. 从 `recent_frame_window` 取最近 `K` 帧
3. 从 `story_memory` 取全局摘要
4. 构建帧级 prompt
5. 调用 `client.generate()`
6. 将结果解析为 `StructuredFrameAnalysis`
7. 更新窗口、人物注册表、剧情记忆
8. 根据规则决定是否触发块级摘要
9. 返回带结构化信息的分析结果

#### 5.2.6 旧逻辑兼容策略

保留旧逻辑作为回退：

- 当 `analysis.context_window_legacy = true` 时，允许继续使用旧的 `previous_analyses` 逻辑。
- 默认关闭旧逻辑，仅作为灰度与回归对比使用。

### 5.3 `video_analyzer/audio_processor.py`

该模块保留当前音频提取与转写主链路，不重写。

#### 5.3.1 保留逻辑

- `extract_audio()`
- `transcribe()`

#### 5.3.2 新增方法建议

建议新增辅助方法，但不改变 `AudioProcessor` 主职责：

- `build_audio_snippets(transcript, alignment_window)`
- `summarize_audio_segments(transcript)`

如果不希望让 `AudioProcessor` 同时承担过多语义逻辑，也可以将这两个方法放到 `analyzer.py` 或新增轻量工具模块中。

推荐原则：

- 音频抽取和转写依然归 `AudioProcessor`
- 音频与剧情语义对齐归 `VideoAnalyzer`

#### 5.3.3 音频摘要结构

建议输出：

```json
{
  "language": "en",
  "segments_count": 42,
  "key_dialogues": [],
  "ambient_sounds": [],
  "mood_keywords": [],
  "time_blocks": []
}
```

### 5.4 `video_analyzer/frame.py`

该模块保留当前关键帧抽取逻辑，并扩展“长视频自适应抽帧”。

#### 5.4.1 当前保留逻辑

- 基于帧差分筛选关键帧候选
- 通过 `frames_per_minute` 和 `max_frames` 控制总量

#### 5.4.2 新增需求

- 增加长视频模式下的自适应抽帧策略
- 增加场景切换候选信息输出

#### 5.4.3 建议新增字段

扩展 `Frame`：

```python
@dataclass
class Frame:
    number: int
    path: Path
    timestamp: float
    score: float
    scene_hint: str | None = None
```

说明：

- `scene_hint` 可先只存粗粒度判断，如 `"stable"`、`"transition"`。
- 一阶段无需复杂镜头分类模型。

#### 5.4.4 长视频抽帧策略

针对长视频，建议引入以下策略：

- 若 `video_duration > long_video_threshold_minutes`：
  - 静态区段降低采样密度
  - 差分高的区段保留更高候选密度
  - 保证每个时间块至少有代表帧

可行实现：

- 先按时间把视频切成若干 bucket
- 每个 bucket 内按差分得分保留前 N 个候选
- 最终合并并按时间排序

这样可以防止长视频中前段或后段被高分场景挤压掉。

### 5.5 `video_analyzer/clients/llm_client.py`

该模块是统一图片编码入口，适合放图片缩放逻辑。

#### 5.5.1 当前问题

当前 `encode_image()` 直接读文件并 base64 编码，没有压缩与缩放。

#### 5.5.2 设计要求

新增统一图片预处理能力：

- 最长边缩放
- JPEG 质量压缩
- 保持纵横比

建议接口：

```python
def encode_image(self, image_path: str, max_image_side: int | None = None, jpeg_quality: int = 85) -> str:
    ...
```

兼容策略：

- 若未传 `max_image_side`，则维持当前行为。

#### 5.5.3 调用方式

由各 client 继续调用 `self.encode_image(...)`，但参数由统一配置下发。

### 5.6 `video_analyzer/clients/ollama.py`

当前接口保持不变，但需要接受新的图片编码参数与输出调试日志。

建议增强项：

- 输出 prompt 长度日志
- 输出图片缩放后的尺寸日志
- 输出本次请求使用的上下文窗口长度

这些日志有助于后续评估长视频性能。

### 5.7 `video_analyzer/clients/generic_openai_api.py`

该模块也保持现有接口，但要与 `LLMClient.encode_image()` 的缩放策略兼容。

注意事项：

- OpenAI-compatible API 对 base64 图片长度更敏感，需要缩放压缩统一生效。
- 保持 `messages` 结构不变，不额外引入 provider 特化分支。

### 5.8 `video_analyzer/config.py`

该模块继续作为统一配置入口。

#### 5.8.1 新增配置读取

建议增加 `analysis` 节点读取能力，不破坏原有结构。

推荐新增配置：

```json
{
  "analysis": {
    "context_window": 5,
    "legacy_full_history": false,
    "memory_update_interval": 8,
    "scene_change_threshold": 18.0,
    "audio_alignment_window": 2.0,
    "enable_chunk_summary": true,
    "chunk_max_frames": 12,
    "long_video_threshold_minutes": 15,
    "max_image_side": 1024,
    "image_jpeg_quality": 85
  }
}
```

#### 5.8.2 参数覆盖规则

依旧保持：

1. CLI 参数
2. `config.json`
3. `default_config.json`

一阶段可暂不扩展 CLI 参数，先通过配置文件控制；如果后续需要 UI 调参，再补 CLI 参数映射。

## 6. Prompt 设计改造

### 6.1 帧级 prompt 改造

当前 `frame_analysis.txt` 假设输入是全历史文本，需要改造为多输入结构：

- 最近窗口摘要
- 全局剧情记忆
- 当前音频片段摘要
- 用户问题

建议新增占位符：

- `{RECENT_WINDOW}`
- `{STORY_MEMORY}`
- `{AUDIO_SNIPPET}`
- `{prompt}`

建议保留 `{PREVIOUS_FRAMES}` 作为兼容字段，但新逻辑默认不再使用。

### 6.2 帧级输出格式要求

建议 prompt 要求模型输出可解析结构，采用“JSON 风格 + 原文补充”的形式。

示例：

```text
{
  "scene": "...",
  "characters": [...],
  "actions": [...],
  "objects": [...],
  "dialogue_hint": "...",
  "continuity_points": [...]
}

Narrative:
...
```

解析策略：

- 优先解析 JSON 结构
- 若解析失败，降级保留 `raw_response`
- 绝不因解析失败中断整体流程

### 6.3 块级摘要 prompt

建议新增第三个 prompt 文件，例如：

- `frame_analysis/chunk_summary.txt`

职责：

- 总结一个块内的帧级结构化结果
- 提炼关键人物、事件、场景推进
- 更新全局剧情记忆

### 6.4 最终总结 prompt 改造

`describe.txt` 需要从“全量逐帧原文 + transcript”改造为：

- 关键帧结构化摘要
- 块级摘要
- 全局剧情记忆
- 音频摘要
- transcript 原文

这样最终总结将不再承担修复前序上下文膨胀的责任，而是面向高质量综合总结。

### 6.5 剧本辅助输出生成

为满足“后续生成视频使用（剧本）”需求，建议在最终总结之后增加一层派生输出生成，仍由 `VideoAnalyzer` 统一收口。

派生输出目标：

- `scene_cards`
  - 给后续分镜/场景组织使用
- `story_beats`
  - 给剧情大纲、节奏拆解使用
- `character_timeline`
  - 给角色线与关系线整理使用
- `script_guidance`
  - 给后续剧本编写或文生视频 prompt 组织使用

实现方式建议：

1. 复用已有帧级结构化结果、块级摘要、全局剧情记忆、音频摘要
2. 通过单独的派生 prompt 或代码拼装生成剧本辅助素材
3. 派生输出必须是结构化字段，不能只返回一段自由文本

推荐新增 prompt 文件：

- `frame_analysis/script_guidance.txt`

职责：

- 根据 `scene_cards`、`story_beats`、`character_timeline` 生成适合编剧/视频生成使用的文本指导
- 不替代 `video_description`

## 7. 长视频专项设计

### 7.1 长视频判定

建议以视频时长作为主判定条件：

- `video_duration_minutes >= analysis.long_video_threshold_minutes`

默认建议：

- `15` 分钟

### 7.2 长视频专用策略

长视频模式开启后：

1. 启用自适应抽帧
2. 强制启用有界上下文窗口
3. 强制启用块级摘要
4. 限制最终总结输入体积
5. 记录块级日志与中间统计

### 7.3 长视频内存控制

不保留全量原始 prompt 文本，只保留：

- 结构化帧结果
- 块级摘要
- 全局记忆摘要

原因：

- 原始 prompt 只在请求瞬间有意义，不应长期驻留内存。

### 7.4 长视频重建策略

对于长视频，不建议把每帧原文全部传给最终总结。

建议策略：

- 若 `frame_analyses` 超过阈值，仅把：
  - 每块首尾关键帧
  - 块级摘要
  - 全局剧情记忆
  - 音频摘要
  送入最终总结

可配置阈值：

- `analysis.reconstruction_frame_cap`

## 8. 音视频协同规则落地

### 8.1 音频片段对齐算法

输入：

- 当前帧时间戳 `frame.timestamp`
- transcript segments
- `analysis.audio_alignment_window`

输出：

- 与当前帧时间最接近的若干音频片段

规则：

- 优先取时间重叠片段
- 若没有重叠，取最近邻片段
- 返回数量设上限，如 `top 3`

### 8.2 音频对视觉的约束级别

约束优先级定义：

1. 视觉可见事实
2. 音频补充线索
3. 全局记忆稳定事实

解释：

- 视觉看得见的内容优先级最高
- 音频只能补充，不能覆盖明显视觉事实
- 全局记忆用于统一称谓、关系与剧情主线

### 8.3 人物一致性机制

一阶段不做强身份识别，采用“人物注册表 + 音频称谓约束”的软一致性方案。

人物注册表更新流程：

1. 从当前帧结构化结果提取角色描述
2. 从音频片段提取名字/称呼/关系词
3. 与已有角色进行软匹配
4. 更新角色档案

匹配依据：

- 外观关键词
- 服饰关键词
- 场景连续性
- 时间连续性
- 音频称谓一致性

### 8.4 场景连续性机制

输入：

- 帧差分分数
- 当前帧与上一帧结构化场景字段
- 音频是否连续

规则：

- 视觉变化小且音频连续 -> 同场景概率高
- 视觉变化大且音频切换 -> 新场景概率高
- 若判断不确定，优先维持当前块，避免过度切块

## 9. 输出结构设计

### 9.1 帧级结果结构

建议单个 `frame_analyses[]` 元素输出：

```json
{
  "frame_number": 12,
  "timestamp": 15.5,
  "response": "原始自然语言描述",
  "structured": {
    "scene": "living room",
    "characters": [],
    "actions": [],
    "objects": [],
    "dialogue_hint": "",
    "continuity_points": [],
    "scene_changed": false
  },
  "audio_context": {
    "matched_segments": [],
    "summary": ""
  }
}
```

### 9.2 块级结果结构

```json
{
  "chunk_id": 2,
  "start_frame": 10,
  "end_frame": 18,
  "start_timestamp": 14.0,
  "end_timestamp": 26.0,
  "summary": "",
  "key_characters": [],
  "key_events": []
}
```

### 9.3 全局剧情记忆结构

```json
{
  "scene_summary": "",
  "characters": {},
  "key_events": [],
  "active_props": [],
  "last_chunk_summary": ""
}
```

### 9.4 音频摘要结构

```json
{
  "language": "en",
  "segments_count": 0,
  "key_dialogues": [],
  "ambient_sounds": [],
  "mood_keywords": [],
  "time_blocks": []
}
```

### 9.5 剧本辅助输出结构

`scene_cards` 建议结构：

```json
[
  {
    "scene_id": "scene-001",
    "start_timestamp": 0.0,
    "end_timestamp": 18.5,
    "location": "",
    "characters": [],
    "key_actions": [],
    "key_props": [],
    "dialogue_summary": "",
    "visual_style_hint": ""
  }
]
```

`story_beats` 建议结构：

```json
[
  {
    "beat_id": "beat-001",
    "order": 1,
    "summary": "",
    "related_scene_ids": [],
    "conflict": "",
    "transition_type": ""
  }
]
```

`character_timeline` 建议结构：

```json
[
  {
    "character_id": "char-001",
    "display_name": "",
    "appearances": [],
    "relations": [],
    "key_actions": [],
    "dialogue_hints": []
  }
]
```

`script_guidance` 建议结构：

```json
{
  "theme": "",
  "tone": "",
  "scene_order": [],
  "dialogue_candidates": [],
  "visual_prompt_hints": [],
  "screenplay_notes": []
}
```

## 10. 失败处理与回退设计

### 10.1 帧级分析失败

策略：

- 记录错误
- 生成最小结构化占位结果
- 继续后续帧分析

### 10.2 结构化解析失败

策略：

- 保留 `raw_response`
- `structured` 使用默认空结构
- 打日志，不中断流程

### 10.3 音频转写失败

策略：

- `transcript = None`
- `audio_summary = None`
- 继续纯视觉分析

### 10.4 块级摘要失败

策略：

- 写入降级块记录
- 最终总结阶段直接使用该块内关键帧结构化结果

### 10.5 最终总结失败

策略：

- 输出前序中间结果
- `video_description.response` 写错误信息
- 保证 `analysis.json` 仍然可生成

### 10.6 派生剧本输出失败

策略：

- 保留 `scene_cards`、`story_beats`、`character_timeline` 中已能通过代码拼装得到的基础结构
- `script_guidance` 可为空或写入错误说明
- 不能影响 `video_description` 与主分析链路成功写出

## 11. 兼容性设计

### 11.1 对现有 CLI 的兼容

- 保持命令不变
- 保持三阶段语义不变
- 默认可通过配置启用新特性

### 11.2 对现有 UI 的兼容

- UI 当前主要提交参数与查看输出
- 只要核心字段保留，现有 UI 基本可继续工作
- 若后续需要展示块级摘要、音频摘要，再做增量前端适配

### 11.3 对现有输出的兼容

- 旧字段保留
- 新字段追加
- 不删除现有 `response` 文本字段
- 剧本辅助字段仅为增量增强，不改变现有结果消费者读取老字段的方式

### 11.4 不降级发布策略

为满足“只有增强，没有破坏”的目标，设计实现时必须采用保守发布策略：

- 保留旧逻辑回退开关
- 保留老输出字段与老 JSON 路径
- 新增结构化输出与剧本辅助输出采用附加字段
- 首次上线建议使用灰度配置或双写对比
- 未通过回归验证前，不应移除旧 prompt 占位符与旧输出字段

## 12. 实施计划

### 12.1 第一阶段

实现范围：

- `analysis` 配置项
- `LLMClient.encode_image()` 缩放压缩
- `VideoAnalyzer` 状态中心
- 有界上下文窗口
- 结构化帧结果
- 音频片段对齐

目标：

- 先把单帧耗时稳定下来

### 12.2 第二阶段

实现范围：

- 块级摘要
- 全局剧情记忆
- 人物注册表
- 长视频重建裁剪策略

目标：

- 提升长视频连贯性与人物一致性

### 12.3 第三阶段

实现范围：

- 自适应抽帧优化
- 回归测试
- 不同模型与不同视频时长的性能对比

目标：

- 完成全链路优化闭环

## 13. 测试设计

### 13.1 单元测试建议

建议新增或补充以下测试：

- `Config` 新配置项加载
- 音频片段对齐函数
- 结构化解析函数
- 人物注册表更新函数
- 块级摘要触发条件

### 13.2 集成测试建议

准备真实业务样本，至少覆盖：

- 无音频短视频
- 有对白短视频
- 长视频
- 场景切换频繁视频
- 对话为主的静态场景视频

### 13.3 验收指标

性能指标：

- 长视频逐帧平均耗时趋于稳定
- prompt 截断日志显著减少

质量指标：

- 人物称谓一致性更稳定
- 场景切换误污染减少
- 音频对白与画面描述关联增强
- `scene_cards`、`story_beats`、`character_timeline`、`script_guidance` 能稳定产出，且能支撑后续剧本/视频生成使用
- 派生剧本辅助输出与 `video_description` 在主剧情、角色、场景顺序上无明显冲突

### 13.4 回归对比方法

建议同一视频同时跑两套策略：

- 旧版：全历史拼接
- 新版：窗口 + 记忆 + 音频联动

对比：

- 总耗时
- 每帧平均耗时
- 输出长度
- 主观质量

## 14. 开发注意事项

### 14.1 必须保持统一入口

- 不要在 UI 中实现独立分析逻辑
- 不要让 client 层处理业务上下文
- 不要在 prompt 文件中硬编码业务状态

### 14.2 必须保持统一收口

- 状态更新只允许通过 `VideoAnalyzer` 进行
- JSON 汇总只允许在 `cli.py` 完成
- 配置读取只允许通过 `Config`

### 14.3 必须保留回退能力

- 对于核心改造，必须有配置开关可退回旧逻辑
- 便于线上灰度与质量比对

## 15. 自审结论

### 15.1 是否可落地

可落地。

原因：

- 所有设计都建立在当前已有模块能力之上
- 一阶段不依赖新服务和新基础设施
- 改造点清晰，职责边界明确

### 15.2 是否适合指导开发

适合。

原因：

- 已细化到模块、方法、数据结构、配置、prompt、输出结构、失败处理、测试策略
- 开发可据此拆分任务与排期

### 15.3 是否覆盖长视频

已覆盖。

体现在：

- 长视频判定
- 自适应抽帧
- 块级摘要
- 有界上下文
- 重建阶段输入裁剪
- 内存控制

### 15.4 是否符合统一入口、统一收口、闭环原则

符合。

原因：

- 入口仍为 `cli.py`
- 分析状态统一收口到 `VideoAnalyzer`
- 输出仍统一写入 `analysis.json`
- 音频、视觉、块级摘要、最终总结形成完整闭环

### 15.5 是否存在高风险未决项

存在，需多人评审：

- 结构化帧结果字段最终定义
- 块级摘要 prompt 设计
- 人物注册表匹配策略
- 长视频总结裁剪阈值
- 新旧逻辑灰度开关命名与默认值
