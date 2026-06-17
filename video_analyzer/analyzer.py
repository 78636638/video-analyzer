from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional

from .audio_processor import AudioSnippet, AudioTranscript
from .clients.llm_client import LLMClient
from .frame import Frame
from .prompt import PromptLoader

logger = logging.getLogger(__name__)

GENERIC_CHARACTER_LABELS = {
    "brother",
    "sister",
    "mother",
    "mom",
    "father",
    "dad",
    "teacher",
    "student",
    "worker",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "person",
    "character",
}

GENERIC_SCENE_LABELS_ZH = {
    "场景",
    "阶段",
    "延续场景",
    "过渡场景",
    "地下室场景",
    "室内场景",
    "户外场景",
    "视频内容展示",
}

SAFE_GENERIC_CHARACTER_LABELS_ZH = {
    "人物",
    "某人物",
    "主角",
    "女性",
    "男性",
    "少女",
    "男孩",
    "女孩",
    "男人",
    "女人",
    "年轻女性",
    "年轻男性",
    "工人",
    "打工者",
    "打工人",
    "学生",
    "老师",
    "同伴",
    "陪同人员",
    "路人",
    "对方",
}

SUPPORTED_RELATION_ALIASES_ZH = {
    "哥哥": ["哥哥", "哥"],
    "姐姐": ["姐姐", "姐"],
    "弟弟": ["弟弟", "弟"],
    "妹妹": ["妹妹", "妹"],
    "父亲": ["父亲", "爸爸", "爸"],
    "母亲": ["母亲", "妈妈", "妈"],
}

UNCERTAIN_CHARACTER_TOKENS_ZH = {
    "可能",
    "似乎",
    "疑似",
    "未明确",
    "未知",
    "推测",
    "像是",
    "仿佛",
}

STAGE_PROMPT_ECHO_TOKENS_ZH = {
    "按画面先后分段概括剧情",
    "标注每段对应的帧区间",
    "每个人物单独一条",
    "人物外貌特征+出场帧范围",
    "区分不同角色",
    "按时间顺序列出所有场景",
    "标注场景类型",
    "出现帧区间",
    "室内/室外/街道/办公室/卧室等",
    "提取视频里重要情节",
    "对应发生在哪一段帧",
    "画面里的道具、字幕、标识、环境特征全部记录",
    "严格按照下面固定结构输出",
    "要求：中文输出",
    "补充上下文",
    "当前阶段帧清单",
    "当前阶段时间范围",
    "上一阶段摘要",
    "邻近音频线索",
    "【视频内容】",
}

OVERLAY_TEXT_MARKERS_ZH = {
    "字幕",
    "画面文字",
    "文字",
    "弹幕",
    "话题标签",
    "平台标题",
    "营销文案",
    "封面文案",
    "点赞",
    "评论",
    "分享",
    "按钮",
}

PLATFORM_OVERLAY_NOISE_TOKENS_ZH = {
    "关注",
    "推荐",
    "朋友",
    "点赞",
    "评论",
    "分享",
    "话题",
    "标签",
    "底部文案",
    "平台标题",
    "营销文案",
    "封面文案",
    "主页",
}

# #region debug-point B:analysis-metrics
def _debug_emit(hypothesis_id: str, location: str, msg: str, data: Optional[Dict[str, Any]] = None, run_id: str = "pre") -> None:
    server_url = "http://127.0.0.1:7777/event"
    session_id = "long-video-performance"
    try:
        env_text = ""
        env_candidates: List[Path] = []
        env_override = os.environ.get("TRAE_DEBUG_ENV_FILE", "").strip()
        if env_override:
            env_candidates.append(Path(env_override))
        dbg_dir = Path(".dbg")
        if dbg_dir.exists():
            env_candidates.extend(sorted(dbg_dir.glob("*.env"), key=lambda item: item.stat().st_mtime, reverse=True))
        env_candidates.append(Path(".dbg/long-video-performance.env"))
        for candidate in env_candidates:
            if not candidate.exists():
                continue
            env_text = candidate.read_text(encoding="utf-8")
            if env_text:
                break
        for line in env_text.splitlines():
            if line.startswith("DEBUG_SERVER_URL="):
                server_url = line.split("=", 1)[1].strip()
            elif line.startswith("DEBUG_SESSION_ID="):
                session_id = line.split("=", 1)[1].strip()
    except Exception:
        pass

    payload = {
        "sessionId": session_id,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": f"[DEBUG] {msg}",
        "data": data or {},
        "ts": int(time.time() * 1000),
    }
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                server_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=1,
        ).read()
    except Exception:
        pass
# #endregion

FALLBACK_CHUNK_PROMPT = """Summarize the following video chunk. Focus on scene continuity, key characters, and the most important actions.

Chunk Frame Notes:
{CHUNK_FRAME_NOTES}

Output:
- A compact summary paragraph
- Key characters
- Key events
"""

FALLBACK_SCRIPT_PROMPT = """You are preparing story guidance for downstream screenplay or video generation.

Scene Cards:
{SCENE_CARDS}

Story Beats:
{STORY_BEATS}

Character Timeline:
{CHARACTER_TIMELINE}

Produce practical notes about theme, tone, scene order, dialogue candidates, and visual prompt hints.
"""

FALLBACK_STAGE_VERIFIER_PROMPT = """Stage Batch Verification Instructions

Output Language
{OUTPUT_LANGUAGE}

Stage Metadata
- stage_id: {STAGE_ID}/{TOTAL_STAGES}
- time_range: {STAGE_TIME_RANGE}
- frame_numbers: {FRAME_NUMBERS}

Previous Stage Summary
{PREVIOUS_STAGE_SUMMARY}

Current Verification Batch
- batch_id: {VERIFIER_BATCH_ID}/{VERIFIER_BATCH_TOTAL}
- batch_time_range: {VERIFIER_BATCH_TIME_RANGE}
- batch_frame_numbers: {VERIFIER_BATCH_FRAME_NUMBERS}

Audio Evidence
{AUDIO_EVIDENCE}

Candidate Structured Result
{CANDIDATE_JSON}

Candidate Raw Response
{RAW_RESPONSE}

Your Task
请结合当前阶段图片、音频线索和候选结果，对当前阶段做二次核验。

Rules
- 只保留能被当前阶段连续画面或音频支持的事实。
- 优先保证故事连续性、人物一致性、场景一致性。
- 人物身份不确定时使用保守通用称谓，不要脑补姓名、职业、亲属关系。
- 画面文字、弹幕、平台标题、营销文案只能放在 overlay_text_lines，不能进入 key_events、scene_label、key_characters。
- 如果候选结果存在漂移、跳场景、错人物或明显无证据剧情，必须剔除。
- 只校验当前阶段，不要重写前面所有阶段。
- 严格输出 JSON，对应字段必须存在。

Output JSON Schema
{
  "scene_label": "",
  "scene_lines": [],
  "key_characters": [],
  "key_events": [],
  "detail_lines": [],
  "overlay_text_lines": [],
  "timeline_summary": ""
}
"""


@dataclass
class CharacterProfile:
    character_id: str
    aliases: List[str] = field(default_factory=list)
    appearance: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    last_seen_timestamp: Optional[float] = None
    last_seen_scene: Optional[str] = None
    confidence: float = 0.5
    appearances: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    key_actions: List[str] = field(default_factory=list)
    dialogue_hints: List[str] = field(default_factory=list)


@dataclass
class StructuredFrameAnalysis:
    frame_number: int
    timestamp: float
    scene: str = ""
    characters: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    overlay_text_lines: List[str] = field(default_factory=list)
    dialogue_hint: str = ""
    continuity_points: List[str] = field(default_factory=list)
    raw_response: str = ""
    scene_changed: bool = False


@dataclass
class ChunkSummary:
    chunk_id: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    summary: str
    scene_label: str = ""
    key_characters: List[str] = field(default_factory=list)
    key_events: List[str] = field(default_factory=list)


@dataclass
class BatchStageAnalysis:
    stage_id: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    frame_numbers: List[int] = field(default_factory=list)
    prompt_path: str = ""
    raw_response: str = ""
    timeline_summary: str = ""
    character_lines: List[str] = field(default_factory=list)
    scene_lines: List[str] = field(default_factory=list)
    event_lines: List[str] = field(default_factory=list)
    detail_lines: List[str] = field(default_factory=list)
    overlay_text_lines: List[str] = field(default_factory=list)
    key_characters: List[str] = field(default_factory=list)
    key_events: List[str] = field(default_factory=list)
    scene_label: str = ""
    dialogue_summary: str = ""


@dataclass
class StoryMemory:
    scene_summary: str = ""
    characters: Dict[str, CharacterProfile] = field(default_factory=dict)
    key_events: List[str] = field(default_factory=list)
    active_props: List[str] = field(default_factory=list)
    overlay_text_lines: List[str] = field(default_factory=list)
    last_chunk_summary: str = ""


class VideoAnalyzer:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        prompt_loader: PromptLoader,
        temperature: float,
        user_prompt: str = "",
        analysis_config: Optional[Dict[str, Any]] = None,
        verifier_client: Optional[LLMClient] = None,
        verifier_model: str = "",
    ):
        self.client = client
        self.model = model
        self.verifier_client = verifier_client
        self.verifier_model = verifier_model.strip()
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.user_prompt = user_prompt
        self.analysis_config = analysis_config or {}

        self.context_window = int(self.analysis_config.get("context_window", 5))
        self.legacy_full_history = bool(self.analysis_config.get("legacy_full_history", False))
        self.scene_change_threshold = float(self.analysis_config.get("scene_change_threshold", 18.0))
        self.audio_alignment_window = float(self.analysis_config.get("audio_alignment_window", 2.0))
        self.enable_chunk_summary = bool(self.analysis_config.get("enable_chunk_summary", True))
        self.chunk_max_frames = max(1, int(self.analysis_config.get("chunk_max_frames", 12)))
        self.chunk_min_frames_before_summary = max(
            2,
            int(self.analysis_config.get("chunk_min_frames_before_summary", 6)),
        )
        self.chunk_min_duration_seconds = max(
            0.0,
            float(self.analysis_config.get("chunk_min_duration_seconds", 8.0)),
        )
        self.reconstruction_frame_cap = max(1, int(self.analysis_config.get("reconstruction_frame_cap", 40)))
        self.frame_prompt_story_memory_char_cap = max(
            400,
            int(self.analysis_config.get("frame_prompt_story_memory_char_cap", 1800)),
        )
        self.frame_prompt_recent_window_char_cap = max(
            200,
            int(self.analysis_config.get("frame_prompt_recent_window_char_cap", 1200)),
        )
        self.reconstruction_chunk_cap = max(
            4,
            int(self.analysis_config.get("reconstruction_chunk_cap", 18)),
        )
        self.reconstruction_transcript_block_cap = max(
            4,
            int(self.analysis_config.get("reconstruction_transcript_block_cap", 18)),
        )
        self.script_scene_cap = max(4, int(self.analysis_config.get("script_scene_cap", 18)))
        self.script_character_cap = max(2, int(self.analysis_config.get("script_character_cap", 12)))
        self.enable_stage_batch_analysis = bool(self.analysis_config.get("enable_stage_batch_analysis", False))
        self.stage_batch_size = max(2, int(self.analysis_config.get("stage_batch_size", 10)))
        self.stage_batch_max_images = max(2, int(self.analysis_config.get("stage_batch_max_images", self.stage_batch_size)))
        self.stage_batch_num_predict = max(256, int(self.analysis_config.get("stage_batch_num_predict", 1200)))
        self.stage_batch_num_ctx = max(4096, int(self.analysis_config.get("stage_batch_num_ctx", 16384)))
        self.stage_batch_prompt_path = str(self.analysis_config.get("stage_batch_prompt_path", "")).strip()
        self.stage_batch_context_summary_count = max(
            1,
            int(self.analysis_config.get("stage_batch_context_summary_count", 3)),
        )
        self.stage_verifier_enabled = bool(self.analysis_config.get("stage_verifier_enabled", False) and self.verifier_client)
        self.stage_verifier_num_predict = max(
            256,
            int(self.analysis_config.get("stage_verifier_num_predict", 800)),
        )
        self.stage_verifier_num_ctx = max(
            4096,
            int(self.analysis_config.get("stage_verifier_num_ctx", 16384)),
        )
        self.stage_verifier_max_images = max(
            1,
            int(self.analysis_config.get("stage_verifier_max_images", 6)),
        )
        self.stage_verifier_retry_enabled = bool(
            self.analysis_config.get("stage_verifier_retry_enabled", True)
        )
        self.stage_verifier_retry_num_predict = max(
            self.stage_verifier_num_predict,
            int(self.analysis_config.get("stage_verifier_retry_num_predict", max(1400, self.stage_verifier_num_predict * 2))),
        )
        self.stage_verifier_attempts = max(
            1,
            int(self.analysis_config.get("stage_verifier_attempts", 3)),
        )
        self.stage_verifier_retry_max_images = max(
            1,
            min(
                self.stage_verifier_max_images,
                int(self.analysis_config.get("stage_verifier_retry_max_images", min(4, self.stage_verifier_max_images))),
            ),
        )
        self.text_refiner_enabled = bool(self.analysis_config.get("text_refiner_enabled", False) and self.verifier_client)
        self.output_language = str(self.analysis_config.get("output_language", "zh")).strip() or "zh"

        self.previous_analyses: List[Dict[str, Any]] = []
        self.recent_frame_window: List[StructuredFrameAnalysis] = []
        self.chunk_buffer: List[StructuredFrameAnalysis] = []
        self.chunk_summaries: List[ChunkSummary] = []
        self.batch_stage_analyses: List[BatchStageAnalysis] = []
        self.story_memory = StoryMemory()
        self.audio_memory: List[AudioSnippet] = []
        self.audio_summary: Optional[Dict[str, Any]] = None
        self._load_prompts()

    def _format_user_prompt(self) -> str:
        if self.user_prompt:
            if self.output_language.lower().startswith("zh"):
                return f"用户关注点：{self.user_prompt}"
            return f"I want to know {self.user_prompt}"
        return ""

    def _get_output_language_name(self) -> str:
        normalized = self.output_language.lower()
        if normalized in {"zh", "zh-cn", "zh-hans", "cn"}:
            return "简体中文"
        if normalized in {"en", "en-us", "en-gb"}:
            return "English"
        return self.output_language

    def _load_prompts(self) -> None:
        self.frame_prompt = self.prompt_loader.get_by_name("Frame Analysis")
        self.video_prompt = self.prompt_loader.get_by_name("Video Reconstruction")
        self.chunk_prompt = self.prompt_loader.get_optional_by_name("Chunk Summary", FALLBACK_CHUNK_PROMPT)
        self.script_prompt = self.prompt_loader.get_optional_by_name("Script Guidance", FALLBACK_SCRIPT_PROMPT)
        self.stage_verifier_prompt = self.prompt_loader.get_optional_by_name(
            "Stage Batch Verifier",
            FALLBACK_STAGE_VERIFIER_PROMPT,
        )
        self.stage_batch_prompt = self._load_stage_batch_prompt()
        self.stage_prompt_reference_lines = self._build_stage_prompt_reference_lines(self.stage_batch_prompt)

    def _load_stage_batch_prompt(self) -> str:
        if not self.stage_batch_prompt_path:
            return ""
        prompt_path = Path(self.stage_batch_prompt_path).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = Path.cwd() / prompt_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Stage batch prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8").strip()

    def _build_stage_prompt_reference_lines(self, prompt_text: str) -> List[str]:
        if not prompt_text:
            return []
        references = []
        for line in prompt_text.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip(" -*")
            if len(cleaned) >= 8:
                references.append(cleaned)
        return list(dict.fromkeys(references))

    def set_audio_memory(self, audio_memory: Optional[List[AudioSnippet]]) -> None:
        self.audio_memory = audio_memory or []

    def set_audio_summary(self, audio_summary: Optional[Dict[str, Any]]) -> None:
        self.audio_summary = audio_summary

    def _format_previous_analyses(self) -> str:
        if not self.previous_analyses:
            return ""

        formatted_analyses = []
        for i, analysis in enumerate(self.previous_analyses):
            formatted_analysis = (
                f"Frame {i}\n"
                f"{analysis.get('response', 'No analysis available')}\n"
            )
            formatted_analyses.append(formatted_analysis)

        return "\n".join(formatted_analyses)

    def _format_recent_window(self) -> str:
        if not self.recent_frame_window:
            return "No recent frames available."

        snippets = []
        for item in self.recent_frame_window[-self.context_window:]:
            snippets.append(
                f"Frame {item.frame_number} @ {item.timestamp:.2f}s | scene={item.scene or 'unknown'} | "
                f"actions={', '.join(item.actions[:3]) or 'n/a'} | continuity={'; '.join(item.continuity_points[:2]) or 'n/a'}"
            )
        return self._truncate_text("\n".join(snippets), self.frame_prompt_recent_window_char_cap)

    def _format_story_memory(self) -> str:
        characters = []
        sorted_profiles = sorted(
            self.story_memory.characters.values(),
            key=lambda item: (
                len(item.appearances),
                item.last_seen_timestamp or 0.0,
            ),
            reverse=True,
        )
        for profile in sorted_profiles[:8]:
            label = profile.aliases[0] if profile.aliases else profile.character_id
            characters.append(
                f"{label} | appearance={', '.join(profile.appearance[:3]) or 'n/a'} | "
                f"roles={', '.join(profile.roles[:2]) or 'n/a'} | last_scene={profile.last_seen_scene or 'n/a'}"
            )

        text = "\n".join(
            [
                f"Scene summary: {self.story_memory.scene_summary or 'n/a'}",
                f"Active props: {', '.join(self.story_memory.active_props[:5]) or 'n/a'}",
                f"Key events: {'; '.join(self.story_memory.key_events[-5:]) or 'n/a'}",
                f"Characters:\n" + ("\n".join(characters) if characters else "n/a"),
                f"Last chunk summary: {self.story_memory.last_chunk_summary or 'n/a'}",
            ]
        )
        return self._truncate_text(text, self.frame_prompt_story_memory_char_cap, keep_tail=True)

    def _truncate_text(self, text: str, max_chars: int, *, keep_tail: bool = False) -> str:
        if max_chars <= 0:
            return ""
        normalized = (text or "").strip()
        if len(normalized) <= max_chars:
            return normalized
        if max_chars <= 24:
            return normalized[:max_chars]
        if keep_tail:
            head = max_chars // 2 - 2
            tail = max_chars - head - 5
            return f"{normalized[:head]} ... {normalized[-tail:]}"
        return normalized[: max_chars - 3].rstrip() + "..."

    def _sample_ordered_items(self, items: List[Any], limit: int) -> List[Any]:
        if limit <= 0 or not items:
            return []
        if len(items) <= limit:
            return list(items)
        if limit == 1:
            return [items[0]]
        indices = sorted({round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)})
        return [items[index] for index in indices]

    def _get_audio_snippets_for_timestamp(self, timestamp: float, limit: int = 3) -> List[AudioSnippet]:
        if not self.audio_memory:
            return []

        overlapping = [
            snippet
            for snippet in self.audio_memory
            if snippet.start <= timestamp <= snippet.end
            or abs(snippet.start - timestamp) <= self.audio_alignment_window
            or abs(snippet.end - timestamp) <= self.audio_alignment_window
        ]
        if overlapping:
            return sorted(overlapping, key=lambda item: abs(((item.start + item.end) / 2.0) - timestamp))[:limit]

        nearest = sorted(
            self.audio_memory,
            key=lambda item: min(abs(item.start - timestamp), abs(item.end - timestamp)),
        )
        return nearest[:limit]

    def _format_audio_snippets(self, snippets: List[AudioSnippet]) -> str:
        if not snippets:
            return "No nearby audio context."

        return "\n".join(
            f"{snippet.start:.2f}-{snippet.end:.2f}s | {snippet.text} | speaker={snippet.speaker_hint or 'unknown'} | mood={snippet.mood_hint or 'neutral'}"
            for snippet in snippets
        )

    def build_frame_prompt(self, frame: Frame, audio_snippets: List[AudioSnippet]) -> str:
        prompt = self.frame_prompt
        if self.legacy_full_history:
            prompt = prompt.replace("{PREVIOUS_FRAMES}", self._format_previous_analyses())
        else:
            prompt = prompt.replace("{PREVIOUS_FRAMES}", "")
        prompt = prompt.replace("{RECENT_WINDOW}", self._format_recent_window())
        prompt = prompt.replace("{STORY_MEMORY}", self._format_story_memory())
        prompt = prompt.replace("{AUDIO_SNIPPET}", self._format_audio_snippets(audio_snippets))
        prompt = prompt.replace("{prompt}", self._format_user_prompt())
        prompt = prompt.replace("{OUTPUT_LANGUAGE}", self._get_output_language_name())
        prompt = (
            f"{prompt}\n\nCurrent frame metadata:\n"
            f"- frame_number: {frame.number}\n"
            f"- timestamp: {frame.timestamp:.2f}s\n"
            f"- scene_hint: {frame.scene_hint or 'unknown'}\n"
            f"- difference_score: {frame.score:.2f}\n"
        )
        return prompt

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        for block in fenced_blocks:
            candidate = block.strip()
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(text[start:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            start = text.find("{", start + 1)
        return None

    def _strip_code_fences(self, text: str) -> str:
        cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "")
        return cleaned.strip()

    def _strip_english_parenthetical_gloss(self, text: str) -> str:
        if not text or not self.output_language.lower().startswith("zh"):
            return text
        return re.sub(r"\s*\((?=[^)]*[A-Za-z])[^)]*\)", "", text).strip()

    def _looks_like_payload_fragment(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        if normalized.startswith("{") or normalized.startswith("["):
            return True
        return any(
            token in normalized
            for token in ['"scene"', '"characters"', '"actions"', '"dialogue_hint"', 'narrative:']
        )

    def _clean_text_field(self, value: Any, *, allow_json_like: bool = False) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        text = self._strip_code_fences(text)
        text = re.sub(r"^\s*(narrative|dialogue_hint|dialogue|scene)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" -")
        text = self._strip_english_parenthetical_gloss(text)
        if not allow_json_like and self._looks_like_payload_fragment(text):
            return ""
        return text

    def _strip_markdown_emphasis(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"^\s*#+\s*", "", text)
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("__", "")
        return cleaned.strip()

    def _strip_timestamp_prefix(self, text: str) -> str:
        if not text:
            return ""
        cleaned = self._strip_markdown_emphasis(text)
        cleaned = re.sub(
            r"^\s*[A-Za-z]?\s*-\s*\d+(?:\.\d+)?s\s*[:：\-]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*\d+(?:\.\d+)?s(?:\s*-\s*\d+(?:\.\d+)?s)?\s*[:：\-]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?s?\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def _remove_uncertain_tone(self, text: str) -> str:
        cleaned = text or ""
        if not cleaned or not self.output_language.lower().startswith("zh"):
            return cleaned
        cleaned = re.sub(r"(?:似乎|可能|疑似|仿佛|像是|推测)(?:有些|有点)?", "", cleaned)
        cleaned = re.sub(r"[，,]{2,}", "，", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip("，,；;。 ")

    def _looks_like_stage_prompt_echo(self, text: str) -> bool:
        normalized = self._strip_markdown_emphasis(self._clean_text_field(text, allow_json_like=False))
        if not normalized:
            return False
        lower_text = normalized.lower()
        if "frame=" in lower_text or "scene_hint=" in lower_text or "difference_score=" in lower_text:
            return True
        for token in STAGE_PROMPT_ECHO_TOKENS_ZH:
            if token in normalized:
                return True
        for reference in getattr(self, "stage_prompt_reference_lines", []):
            if reference in normalized or normalized in reference:
                return True
        return False

    def _sanitize_stage_text(self, text: str, *, keep_sentence: bool = True) -> str:
        cleaned = self._clean_text_field(text, allow_json_like=False)
        cleaned = self._strip_timestamp_prefix(cleaned)
        cleaned = self._remove_uncertain_tone(cleaned)
        cleaned = re.sub(r"([男女]性|人物|角色|主角)\s*[A-ZＡ-Ｚ]\b", r"\1", cleaned)
        cleaned = re.sub(r"（\s*\d+\s*-\s*\d+\s*帧\s*）", "", cleaned)
        cleaned = re.sub(r"\(\s*\d+\s*-\s*\d+\s*帧\s*\)", "", cleaned)
        cleaned = re.sub(r"（\s*\d+(?:\.\d+)?s\s*）", "", cleaned)
        cleaned = re.sub(r"\(\s*\d+(?:\.\d+)?s\s*\)", "", cleaned)
        cleaned = re.sub(r"^[：:;；、，,\-\s]+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
        cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
        if not cleaned or self._looks_like_stage_prompt_echo(cleaned):
            return ""
        if keep_sentence:
            return cleaned.strip()
        return re.split(r"[，。；;]", cleaned, maxsplit=1)[0].strip()

    def _sanitize_stage_lines(self, lines: List[str], *, kind: str) -> List[str]:
        sanitized = []
        for line in lines:
            cleaned = self._sanitize_stage_text(line, keep_sentence=kind != "scene")
            if not cleaned:
                continue
            if kind == "detail" and len(cleaned) > 120:
                cleaned = self._truncate_text(cleaned, 120)
            if cleaned not in sanitized:
                sanitized.append(cleaned)
        return sanitized

    def _is_generic_stage_event_text(self, text: str) -> bool:
        normalized = self._normalize_summary_phrase(text)
        if not normalized:
            return True
        generic_tokens = {
            "无新增或延续的关键事件节点",
            "无明确关键事件",
            "无关键事件",
            "暂无关键事件",
            "当前阶段没有新增或延续的剧情",
        }
        return normalized in generic_tokens

    def _normalize_match_text(self, text: str) -> str:
        normalized = self._clean_text_field(text, allow_json_like=False).lower()
        normalized = re.sub(r"[\s|:：;；,，。、“”\"'`()（）\-\[\]]+", "", normalized)
        return normalized

    def _extract_quoted_fragments(self, text: str) -> List[str]:
        return [
            self._clean_text_field(item, allow_json_like=False)
            for item in re.findall(r"[“\"']([^”\"']+)[”\"']", text or "")
            if self._clean_text_field(item, allow_json_like=False)
        ]

    def _contains_overlay_text_marker(self, text: str) -> bool:
        normalized = self._clean_text_field(text, allow_json_like=False)
        if not normalized:
            return False
        return any(marker in normalized for marker in OVERLAY_TEXT_MARKERS_ZH)

    def _extract_overlay_text_lines(self, lines: List[str]) -> tuple[List[str], List[str]]:
        detail_lines: List[str] = []
        overlay_lines: List[str] = []
        for line in lines:
            cleaned = self._sanitize_stage_text(line)
            if not cleaned:
                continue
            if self._is_stage_descriptor_text(cleaned):
                continue
            if self._contains_overlay_text_marker(cleaned):
                overlay_lines.append(self._normalize_overlay_text_line(cleaned))
            else:
                detail_lines.append(cleaned)
        return list(dict.fromkeys(detail_lines)), list(dict.fromkeys(overlay_lines))

    def _normalize_overlay_text_line(self, text: str) -> str:
        cleaned = self._sanitize_stage_text(text)
        if not cleaned:
            return ""
        marker_match = re.match(
            r"^(字幕|画面文字|弹幕|话题标签|平台标题|营销文案|封面文案|点赞|评论|分享|按钮)\s*[:：]\s*(.*)$",
            cleaned,
        )
        if not marker_match:
            return cleaned
        marker = marker_match.group(1)
        payload = marker_match.group(2).strip()
        quoted_fragments = self._extract_quoted_fragments(payload)
        if quoted_fragments:
            return f"{marker}：" + "；".join(quoted_fragments[:8])
        payload = re.sub(r"^(?:视频中出现的文字包括|画面中出现的文字包括)\s*[:：]?\s*", "", payload)
        payload = re.split(r"(?:这些文字|以上文字|用于提示|反映了|表示了|说明了)", payload, maxsplit=1)[0]
        payload = re.sub(r"[，,]\s*(?:并|以及)\s*", "；", payload)
        payload = re.sub(r"\s+", " ", payload).strip("；;，,。 ")
        if not payload:
            return f"{marker}："
        return f"{marker}：{payload}"

    def _extract_overlay_text_payload(self, text: str) -> str:
        cleaned = self._normalize_overlay_text_line(text)
        if not cleaned:
            return ""
        cleaned = re.sub(
            r"^(?:字幕|画面文字|弹幕|话题标签|平台标题|营销文案|封面文案|点赞|评论|分享|按钮)\s*[:：]\s*",
            "",
            cleaned,
        )
        return self._normalize_summary_phrase(cleaned)

    def _strip_platform_overlay_noise(self, text: str) -> str:
        cleaned = self._sanitize_stage_text(text)
        if not cleaned:
            return ""
        cleaned = re.sub(r"^\s*(?:底部文案|标题文案|顶部文案)\s*[:：]\s*", "", cleaned)
        cleaned = re.sub(r"#[^\s#]+", "", cleaned)
        cleaned = re.sub(r"(?:点赞|评论|分享|关注|推荐)\s*(?:按钮|入口|提示)?", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip("；;，,。# ")
        return cleaned

    def _looks_like_platform_overlay_noise(self, text: str) -> bool:
        cleaned = self._sanitize_stage_text(text)
        if not cleaned:
            return False
        lowered = cleaned.lower()
        if "#" in cleaned:
            return True
        if re.search(r"(?:点赞|评论|分享)\s*\d*", cleaned):
            return True
        return any(token in cleaned or token.lower() in lowered for token in PLATFORM_OVERLAY_NOISE_TOKENS_ZH)

    def _collect_overlay_text_references(self, overlay_text_lines: Optional[List[str]]) -> List[str]:
        references: List[str] = []
        for line in overlay_text_lines or []:
            cleaned = self._sanitize_stage_text(line)
            if not cleaned:
                continue
            references.append(cleaned)
            payload = self._extract_overlay_text_payload(cleaned)
            if payload:
                references.append(payload)
            references.extend(self._extract_quoted_fragments(cleaned))
        return list(dict.fromkeys([item for item in references if item]))

    def _extract_stage_descriptor_value(self, text: str, labels: List[str]) -> str:
        if not text:
            return ""
        escaped_labels = "|".join(re.escape(label) for label in labels)
        pattern = rf"(?:{escaped_labels})\s*[:=：]\s*(.+?)(?=\s*\|\s*(?:场景|关键人物|关键事件|阶段)\s*[:=：]|\Z)"
        match = re.search(pattern, text)
        if match:
            return self._clean_text_field(match.group(1), allow_json_like=False)
        return ""

    def _is_stage_descriptor_text(self, text: str) -> bool:
        normalized = self._clean_text_field(text, allow_json_like=False)
        if not normalized:
            return False
        if "关键事件=" in normalized or "关键人物=" in normalized or "场景=" in normalized:
            return True
        if re.match(r"^\s*(?:\d+[.、]\s*)?阶段\s*\d+\s*[|｜]", normalized):
            return True
        return bool(re.search(r"阶段\s*\d+\s*[|｜].*场景\s*[:=：]", normalized))

    def _normalize_scene_line_value(self, text: str) -> str:
        candidate = self._extract_stage_descriptor_value(text, ["场景", "scene"]) or text
        candidate = candidate.split("|")[0].strip()
        candidate = re.sub(r"^\s*阶段\s*\d+\s*[|｜]\s*", "", candidate)
        candidate = re.sub(r"^\s*\d+(?:\.\d+)?s?\s*-\s*\d+(?:\.\d+)?s?\s*[|｜]\s*", "", candidate)
        candidate = re.sub(r"[：:]\s*\d+\s*-\s*\d+\s*帧\s*$", "", candidate)
        candidate = re.sub(r"[：:]\s*延续上一阶段场景\s*$", "", candidate)
        candidate = re.sub(r"[：:]\s*延续前一阶段场景\s*$", "", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip(" -")
        candidate = self._normalize_scene_location(candidate)
        if self._contains_overlay_text_marker(candidate):
            return ""
        return candidate

    def _matches_overlay_text(self, text: str, overlay_text_lines: Optional[List[str]]) -> bool:
        normalized_text = self._normalize_match_text(text)
        if not normalized_text:
            return False
        for reference in self._collect_overlay_text_references(overlay_text_lines):
            normalized_reference = self._normalize_match_text(reference)
            if not normalized_reference:
                continue
            if min(len(normalized_text), len(normalized_reference)) < 4:
                continue
            if normalized_text in normalized_reference or normalized_reference in normalized_text:
                return True
        return False

    def _filter_story_detail_lines(
        self,
        lines: List[str],
        overlay_text_lines: Optional[List[str]] = None,
        limit: int = 8,
    ) -> List[str]:
        filtered: List[str] = []
        for line in lines:
            cleaned = self._sanitize_stage_text(line)
            if not cleaned:
                continue
            if self._is_generic_stage_event_text(cleaned):
                continue
            if self._is_stage_descriptor_text(cleaned) or self._contains_overlay_text_marker(cleaned):
                continue
            if self._matches_overlay_text(cleaned, overlay_text_lines):
                continue
            if cleaned not in filtered:
                filtered.append(cleaned)
        return filtered[:limit]

    def _filter_story_event_candidates(
        self,
        lines: List[str],
        overlay_text_lines: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[str]:
        filtered: List[str] = []
        for line in lines:
            cleaned = self._sanitize_stage_text(line)
            if not cleaned:
                continue
            if self._is_generic_stage_event_text(cleaned):
                continue
            if self._is_stage_descriptor_text(cleaned) or self._contains_overlay_text_marker(cleaned):
                continue
            if self._matches_overlay_text(cleaned, overlay_text_lines):
                continue
            if cleaned not in filtered:
                filtered.append(cleaned)
        return filtered[:limit]

    def _sanitize_stage_raw_response(self, raw_response: str) -> str:
        if not raw_response:
            return ""
        cleaned_lines: List[str] = []
        for line in raw_response.splitlines():
            if self._is_stage_descriptor_text(line):
                continue
            cleaned_lines.append(line.rstrip())
        cleaned_response = "\n".join(cleaned_lines).strip()
        return cleaned_response or raw_response.strip()

    def _has_audio_support_for_phrase(self, phrase: str, audio_reference_text: str) -> bool:
        normalized_phrase = self._normalize_match_text(phrase)
        if not normalized_phrase or not audio_reference_text:
            return False
        return normalized_phrase in audio_reference_text

    def _filter_stage_event_lines(
        self,
        lines: List[str],
        audio_snippets: Optional[List[AudioSnippet]] = None,
        overlay_text_lines: Optional[List[str]] = None,
    ) -> List[str]:
        audio_reference_text = self._normalize_match_text(
            " ".join(snippet.text for snippet in (audio_snippets or []) if getattr(snippet, "text", ""))
        )
        overlay_reference_text = self._normalize_match_text(" ".join(overlay_text_lines or []))
        filtered: List[str] = []
        for line in lines:
            cleaned = self._sanitize_stage_text(
                self._extract_stage_descriptor_value(line, ["关键事件", "事件"]) or line
            )
            if not cleaned:
                continue
            if self._is_generic_stage_event_text(cleaned):
                continue
            if self._contains_overlay_text_marker(cleaned):
                continue
            quoted_fragments = self._extract_quoted_fragments(cleaned)
            if quoted_fragments and not any(
                self._has_audio_support_for_phrase(fragment, audio_reference_text)
                for fragment in quoted_fragments
            ):
                continue
            if self._matches_overlay_text(cleaned, overlay_text_lines) and not self._has_audio_support_for_phrase(
                cleaned,
                audio_reference_text,
            ):
                continue
            if overlay_reference_text and self._normalize_match_text(cleaned) in overlay_reference_text:
                continue
            if cleaned not in filtered:
                filtered.append(cleaned)
        return filtered[:8]

    def _normalize_stage_summary_text(
        self,
        summary: str,
        scene_label: str,
        key_characters: List[str],
        key_events: List[str],
        stage_id: int,
    ) -> str:
        cleaned = self._sanitize_stage_text(summary)
        if not cleaned:
            cleaned = ""
        if self._is_stage_descriptor_text(cleaned):
            if self.output_language.lower().startswith("zh"):
                parts = [f"阶段{stage_id}在{scene_label or '当前场景'}中继续推进。"]
                if key_events:
                    parts.append(f"本阶段重点为{self._join_readable_items(key_events[:2], '、')}。")
                elif key_characters:
                    parts.append(f"画面主要围绕{self._join_readable_items(key_characters[:2], '、')}展开。")
                return "".join(parts)
            return cleaned
        return cleaned

    def _post_process_stage_batch_parsed(
        self,
        parsed: Dict[str, Any],
        audio_snippets: Optional[List[AudioSnippet]],
        stage_id: int,
    ) -> Dict[str, Any]:
        detail_lines, extracted_overlay_lines = self._extract_overlay_text_lines(parsed.get("detail_lines", []))
        overlay_text_lines = [
            normalized
            for normalized in (
                self._normalize_overlay_text_line(item)
                for item in (parsed.get("overlay_text_lines", []) or [])
            )
            if normalized
        ]
        overlay_text_lines = list(dict.fromkeys(overlay_text_lines + extracted_overlay_lines))
        detail_lines = self._filter_story_detail_lines(detail_lines, overlay_text_lines=overlay_text_lines)
        scene_lines = [
            normalized
            for normalized in (
                self._normalize_scene_line_value(line)
                for line in parsed.get("scene_lines", [])
            )
            if normalized
        ]
        scene_lines = list(dict.fromkeys(scene_lines))
        scene_label = self._normalize_scene_line_value(parsed.get("scene_label", ""))
        if not scene_label and scene_lines:
            scene_label = scene_lines[0]
        key_events = self._filter_stage_event_lines(
            parsed.get("event_lines", []),
            audio_snippets=audio_snippets,
            overlay_text_lines=overlay_text_lines,
        )
        key_characters = list(dict.fromkeys(parsed.get("key_characters", [])[:6]))
        summary = self._normalize_stage_summary_text(
            parsed.get("timeline_summary", ""),
            scene_label,
            key_characters,
            key_events,
            stage_id,
        )
        parsed.update(
            {
                "timeline_summary": summary,
                "scene_lines": scene_lines,
                "scene_label": scene_label,
                "detail_lines": detail_lines,
                "overlay_text_lines": overlay_text_lines,
                "event_lines": key_events,
                "key_events": key_events,
                "key_characters": key_characters,
            }
        )
        return parsed

    def _format_stage_candidate_for_verifier(self, parsed: Dict[str, Any]) -> str:
        payload = {
            "scene_label": parsed.get("scene_label", ""),
            "scene_lines": parsed.get("scene_lines", [])[:4],
            "key_characters": parsed.get("key_characters", [])[:6],
            "key_events": parsed.get("key_events", [])[:6],
            "detail_lines": parsed.get("detail_lines", [])[:6],
            "overlay_text_lines": parsed.get("overlay_text_lines", [])[:6],
            "timeline_summary": parsed.get("timeline_summary", ""),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_stage_verifier_prompt(
        self,
        frames: List[Frame],
        parsed: Dict[str, Any],
        raw_response: str,
        stage_id: int,
        total_stages: int,
        audio_snippets: Optional[List[AudioSnippet]] = None,
        batch_id: int = 1,
        batch_total: int = 1,
    ) -> str:
        audio_snippets = audio_snippets or []
        audio_lines = self._build_dialogue_lines_from_snippets(audio_snippets, limit=4)
        if not audio_lines:
            audio_lines = self._get_audio_lines_for_range(frames[0].timestamp, frames[-1].timestamp, limit=4)
        previous_stage_summary = (
            self.batch_stage_analyses[-1].timeline_summary
            if self.batch_stage_analyses
            else ("无上一阶段摘要" if self.output_language.lower().startswith("zh") else "No previous stage summary")
        )
        prompt = self.stage_verifier_prompt
        prompt = prompt.replace("{OUTPUT_LANGUAGE}", self._get_output_language_name())
        prompt = prompt.replace("{STAGE_ID}", str(stage_id))
        prompt = prompt.replace("{TOTAL_STAGES}", str(total_stages))
        prompt = prompt.replace("{STAGE_TIME_RANGE}", f"{frames[0].timestamp:.2f}s-{frames[-1].timestamp:.2f}s")
        prompt = prompt.replace("{FRAME_NUMBERS}", ", ".join(str(frame.number) for frame in frames))
        prompt = prompt.replace("{PREVIOUS_STAGE_SUMMARY}", previous_stage_summary or "无")
        prompt = prompt.replace("{VERIFIER_BATCH_ID}", str(batch_id))
        prompt = prompt.replace("{VERIFIER_BATCH_TOTAL}", str(batch_total))
        prompt = prompt.replace("{VERIFIER_BATCH_TIME_RANGE}", f"{frames[0].timestamp:.2f}s-{frames[-1].timestamp:.2f}s")
        prompt = prompt.replace("{VERIFIER_BATCH_FRAME_NUMBERS}", ", ".join(str(frame.number) for frame in frames))
        prompt = prompt.replace("{AUDIO_EVIDENCE}", "\n".join(f"- {line}" for line in audio_lines) or "- 无明确音频线索")
        prompt = prompt.replace("{CANDIDATE_JSON}", self._format_stage_candidate_for_verifier(parsed))
        prompt = prompt.replace("{RAW_RESPONSE}", self._truncate_text(self._sanitize_stage_raw_response(raw_response), 1800))
        return prompt

    def _build_compact_stage_verifier_prompt(
        self,
        frames: List[Frame],
        parsed: Dict[str, Any],
        stage_id: int,
        total_stages: int,
        audio_snippets: Optional[List[AudioSnippet]] = None,
        batch_id: int = 1,
        batch_total: int = 1,
        attempt_id: int = 1,
        verified_progress_summary: str = "",
    ) -> str:
        audio_lines = self._build_dialogue_lines_from_snippets(audio_snippets or [], limit=3)
        if not audio_lines:
            audio_lines = self._get_audio_lines_for_range(frames[0].timestamp, frames[-1].timestamp, limit=3)
        previous_stage_summary = (
            self.batch_stage_analyses[-1].timeline_summary
            if self.batch_stage_analyses
            else ("无上一阶段摘要" if self.output_language.lower().startswith("zh") else "No previous stage summary")
        )
        compact_payload = {
            "scene_label": parsed.get("scene_label", ""),
            "key_characters": parsed.get("key_characters", [])[:3],
            "key_events": parsed.get("key_events", [])[:3],
            "overlay_text_lines": parsed.get("overlay_text_lines", [])[:3],
        }
        verified_progress_summary = self._truncate_text(verified_progress_summary or "无", 120)
        if self.output_language.lower().startswith("zh"):
            return (
                "你是视频阶段复核器。请只根据当前阶段图片、音频线索和候选结果，输出一个可解析的纯 JSON。"
                "不要解释，不要思考过程，不要 markdown 代码块。\n"
                f"阶段: {stage_id}/{total_stages}\n"
                f"当前复核批次: {batch_id}/{batch_total}\n"
                f"当前复核尝试: {attempt_id}/{self.stage_verifier_attempts}\n"
                f"时间: {frames[0].timestamp:.2f}s-{frames[-1].timestamp:.2f}s\n"
                f"帧: {', '.join(str(frame.number) for frame in frames)}\n"
                f"上一阶段摘要: {self._truncate_text(previous_stage_summary or '无', 120)}\n"
                f"当前阶段已确认进度: {verified_progress_summary}\n"
                f"音频线索: {self._join_readable_items(audio_lines[:3], '；') or '无'}\n"
                f"候选结果: {json.dumps(compact_payload, ensure_ascii=False)}\n"
                "规则:\n"
                "- 输出必须首字符就是 {，最后一个字符就是 }。\n"
                "- 不要输出任何前言、解释、项目符号、标题、代码块、注释或额外换行说明。\n"
                "- 所有字段都必须保留；无信息时返回空字符串或空数组，不得省略字段。\n"
                "- 只保留当前阶段能被连续画面或音频支持的事实。\n"
                "- 只核验当前批次图片，不要把前一批未出现的画面或动作写进当前批次结论。\n"
                "- 若当前批次与候选结果冲突，以当前批次图片和音频共同支持的事实为准。\n"
                "- 当前批次图片必须全部参与核验，不得只根据部分图片下结论。\n"
                "- 人物身份不确定时用“人物”“女性”“男性”等保守通用称谓。\n"
                "- 字幕、弹幕、按钮、标题等只能放入 overlay_text_lines。\n"
                "- scene_lines、key_events、detail_lines、overlay_text_lines 每个数组最多 3 项。\n"
                "- timeline_summary 控制在 80 字以内。\n"
                '只输出以下 JSON: {"scene_label":"","scene_lines":[],"key_characters":[],"key_events":[],"detail_lines":[],"overlay_text_lines":[],"timeline_summary":""}'
            )
        return (
            "You are a video stage verifier. Return plain JSON only with no explanation, no reasoning, and no markdown.\n"
            f"Stage: {stage_id}/{total_stages}\n"
            f"Current verification batch: {batch_id}/{batch_total}\n"
            f"Current attempt: {attempt_id}/{self.stage_verifier_attempts}\n"
            f"Time: {frames[0].timestamp:.2f}s-{frames[-1].timestamp:.2f}s\n"
            f"Frames: {', '.join(str(frame.number) for frame in frames)}\n"
            f"Previous summary: {self._truncate_text(previous_stage_summary or 'none', 120)}\n"
            f"Confirmed progress in current stage: {verified_progress_summary}\n"
            f"Audio evidence: {self._join_readable_items(audio_lines[:3], '; ') or 'none'}\n"
            f"Candidate: {json.dumps(compact_payload, ensure_ascii=False)}\n"
            "Rules:\n"
            "- The first character must be { and the last character must be }.\n"
            "- Do not output any preface, explanation, bullets, headings, code fences, comments, or extra notes.\n"
            "- Keep every field in the schema; use empty string or empty arrays when unsure.\n"
            "- Keep only facts supported by current images or audio.\n"
            "- Verify only this batch of images and do not import unseen actions from previous batches.\n"
            "- If this batch conflicts with the candidate, trust facts jointly supported by this batch and audio.\n"
            "- All images in this batch must be considered before answering.\n"
            "- Use conservative generic labels for uncertain identities.\n"
            "- Put on-screen text only in overlay_text_lines.\n"
            "- Limit each array field to 3 items and timeline_summary to 80 characters.\n"
            'Return only this JSON object: {"scene_label":"","scene_lines":[],"key_characters":[],"key_events":[],"detail_lines":[],"overlay_text_lines":[],"timeline_summary":""}'
        )

    def _build_stage_verifier_progress_summary(self, aggregated: Dict[str, Any]) -> str:
        parts: List[str] = []
        scene_label = self._normalize_scene_location(aggregated.get("scene_label", ""))
        if scene_label:
            parts.append(f"已确认场景：{scene_label}")
        key_events = [
            self._clean_text_field(item, allow_json_like=False)
            for item in (aggregated.get("key_events", []) or [])[:2]
            if self._clean_text_field(item, allow_json_like=False)
        ]
        if key_events:
            parts.append(f"已确认事件：{'；'.join(key_events)}")
        timeline_summary = self._clean_text_field(aggregated.get("timeline_summary", ""), allow_json_like=False)
        if timeline_summary:
            parts.append(f"已确认摘要：{self._truncate_text(timeline_summary, 80)}")
        return self._truncate_text(" | ".join(parts), 120)

    def _split_stage_verifier_frame_batches(self, frames: List[Frame]) -> List[List[Frame]]:
        if not frames:
            return []
        batch_size = max(1, self.stage_verifier_max_images)
        return [frames[index:index + batch_size] for index in range(0, len(frames), batch_size)]

    def _merge_stage_verifier_result(
        self,
        aggregated: Dict[str, Any],
        verified: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(aggregated)
        current_scene_label = self._normalize_scene_location(merged.get("scene_label", ""))
        verified_scene_label = self._normalize_scene_location(verified.get("scene_label", ""))
        if not current_scene_label or self._is_generic_scene_label(current_scene_label):
            if verified_scene_label:
                merged["scene_label"] = verified_scene_label
        elif verified_scene_label and (
            current_scene_label == verified_scene_label
            or current_scene_label in verified_scene_label
            or verified_scene_label in current_scene_label
        ):
            merged["scene_label"] = verified_scene_label if len(verified_scene_label) >= len(current_scene_label) else current_scene_label
        for field_name in ["scene_lines", "key_characters", "key_events", "detail_lines", "overlay_text_lines"]:
            combined: List[str] = []
            for item in (merged.get(field_name, []) or []) + (verified.get(field_name, []) or []):
                cleaned = self._clean_text_field(item, allow_json_like=False) if isinstance(item, str) else ""
                if cleaned and cleaned not in combined:
                    combined.append(cleaned)
            merged[field_name] = combined
        summary_parts: List[str] = []
        for item in [merged.get("timeline_summary", ""), verified.get("timeline_summary", "")]:
            cleaned = self._clean_text_field(item, allow_json_like=False)
            if cleaned and cleaned not in summary_parts:
                summary_parts.append(cleaned)
        merged["timeline_summary"] = self._truncate_text("；".join(summary_parts), 220)
        merged["event_lines"] = merged.get("key_events", [])[:]
        if not merged.get("scene_lines") and merged.get("scene_label"):
            merged["scene_lines"] = [merged["scene_label"]]
        return merged

    def _finalize_stage_verifier_result(
        self,
        parsed: Dict[str, Any],
        aggregated: Dict[str, Any],
    ) -> Dict[str, Any]:
        finalized = dict(parsed)
        for field_name in ["scene_lines", "key_characters", "key_events", "detail_lines", "overlay_text_lines"]:
            aggregated_items = aggregated.get(field_name, []) or []
            if aggregated_items:
                finalized[field_name] = aggregated_items
        aggregated_scene_label = self._normalize_scene_location(aggregated.get("scene_label", ""))
        if aggregated_scene_label:
            finalized["scene_label"] = aggregated_scene_label
        aggregated_summary = self._clean_text_field(aggregated.get("timeline_summary", ""), allow_json_like=False)
        if aggregated_summary:
            finalized["timeline_summary"] = aggregated_summary
        finalized["event_lines"] = finalized.get("key_events", [])[:]
        if not finalized.get("scene_lines") and finalized.get("scene_label"):
            finalized["scene_lines"] = [finalized["scene_label"]]
        return finalized

    def _run_stage_verifier_batch(
        self,
        frames: List[Frame],
        parsed: Dict[str, Any],
        raw_response: str,
        stage_id: int,
        total_stages: int,
        audio_snippets: Optional[List[AudioSnippet]] = None,
        batch_id: int = 1,
        batch_total: int = 1,
    ) -> Optional[Dict[str, Any]]:
        last_finish_reason = ""
        last_response_text = ""
        last_raw_response_text = ""
        verified_progress_summary = self._build_stage_verifier_progress_summary(parsed)
        for attempt_index in range(self.stage_verifier_attempts):
            attempt_id = attempt_index + 1
            prompt = self._build_compact_stage_verifier_prompt(
                frames,
                parsed,
                stage_id,
                total_stages,
                audio_snippets=audio_snippets,
                batch_id=batch_id,
                batch_total=batch_total,
                attempt_id=attempt_id,
                verified_progress_summary=verified_progress_summary,
            )
            num_predict = self.stage_verifier_retry_num_predict if attempt_id > 1 else max(
                self.stage_verifier_num_predict,
                self.stage_verifier_retry_num_predict,
            )
            # #region debug-point A:minimax-request
            _debug_emit(
                "A",
                "analyzer.py:_run_stage_verifier_batch",
                "MiniMax verifier request prepared",
                {
                    "stage_id": stage_id,
                    "batch_id": batch_id,
                    "batch_total": batch_total,
                    "attempt_id": attempt_id,
                    "attempt_total": self.stage_verifier_attempts,
                    "image_count": len(frames),
                    "frame_numbers": [frame.number for frame in frames],
                    "prompt_length": len(prompt),
                    "candidate_json_length": len(self._format_stage_candidate_for_verifier(parsed)),
                    "raw_response_length": len(self._sanitize_stage_raw_response(raw_response)),
                    "audio_snippet_count": len(audio_snippets or []),
                    "num_predict": num_predict,
                    "num_ctx": self.stage_verifier_num_ctx,
                },
                run_id="pre-fix",
            )
            # #endregion
            response: Dict[str, Any] = {}
            try:
                response = self.verifier_client.generate(
                    prompt=prompt,
                    image_paths=[str(frame.path) for frame in frames],
                    model=self.verifier_model,
                    temperature=0.0,
                    num_predict=num_predict,
                    num_ctx=self.stage_verifier_num_ctx,
                )
            except Exception as exc:
                logger.warning(
                    "Stage verifier failed on stage %s batch %s/%s attempt %s/%s: %s",
                    stage_id,
                    batch_id,
                    batch_total,
                    attempt_id,
                    self.stage_verifier_attempts,
                    exc,
                )
                response = {}
            response_text = (response or {}).get("response", "")
            raw_response_text = str((response or {}).get("raw_response", "") or "")
            finish_reason = str((response or {}).get("finish_reason", "") or "")
            # #region debug-point B:minimax-response
            _debug_emit(
                "B",
                "analyzer.py:_run_stage_verifier_batch",
                "MiniMax verifier response received",
                {
                    "stage_id": stage_id,
                    "batch_id": batch_id,
                    "batch_total": batch_total,
                    "attempt_id": attempt_id,
                    "attempt_total": self.stage_verifier_attempts,
                    "finish_reason": finish_reason,
                    "response_length": len(response_text or ""),
                    "raw_response_length": len(raw_response_text or ""),
                    "has_json_in_response": bool(self._extract_json_object(response_text or "")),
                    "has_json_in_raw_response": bool(
                        raw_response_text and self._extract_json_object(self._strip_code_fences(str(raw_response_text or "")))
                    ),
                },
                run_id="pre-fix",
            )
            # #endregion
            last_response_text = response_text
            last_raw_response_text = raw_response_text
            last_finish_reason = finish_reason
            verified = self._parse_stage_verifier_response(
                response_text,
                raw_response_text=raw_response_text,
            )
            if verified:
                return verified
            if attempt_id < self.stage_verifier_attempts:
                logger.warning(
                    "Stage verifier returned invalid payload on stage %s batch %s/%s attempt %s/%s%s",
                    stage_id,
                    batch_id,
                    batch_total,
                    attempt_id,
                    self.stage_verifier_attempts,
                    f" (finish_reason={finish_reason})" if finish_reason else "",
                )
        if last_finish_reason:
            logger.warning(
                "Stage verifier returned invalid payload on stage %s batch %s/%s after %s attempts (finish_reason=%s)",
                stage_id,
                batch_id,
                batch_total,
                self.stage_verifier_attempts,
                last_finish_reason,
            )
        else:
            logger.warning(
                "Stage verifier returned invalid payload on stage %s batch %s/%s after %s attempts",
                stage_id,
                batch_id,
                batch_total,
                self.stage_verifier_attempts,
            )
        if self._should_retry_stage_verifier(last_response_text, last_raw_response_text, last_finish_reason):
            return None
        return None

    def _parse_stage_verifier_response(
        self,
        response_text: str,
        raw_response_text: str = "",
    ) -> Optional[Dict[str, Any]]:
        payload = self._extract_json_object(response_text or "")
        if not payload and raw_response_text:
            payload = self._extract_json_object(self._strip_code_fences(str(raw_response_text or "")))
        if not payload:
            payload = self._extract_stage_verifier_key_value_payload(response_text or "")
        if not payload and raw_response_text:
            payload = self._extract_stage_verifier_key_value_payload(self._strip_code_fences(str(raw_response_text or "")))
        if not payload:
            return None
        scene_label = self._clean_text_field(payload.get("scene_label") or "", allow_json_like=False)
        scene_lines = self._clean_list_field(payload.get("scene_lines") or [])
        key_characters = self._clean_list_field(payload.get("key_characters") or [])
        key_events = self._clean_list_field(payload.get("key_events") or [])
        detail_lines = self._clean_list_field(payload.get("detail_lines") or [])
        overlay_text_lines = self._clean_list_field(payload.get("overlay_text_lines") or [])
        timeline_summary = self._clean_text_field(payload.get("timeline_summary") or "", allow_json_like=False)
        if not scene_lines and scene_label:
            scene_lines = [scene_label]
        return {
            "scene_label": scene_label,
            "scene_lines": scene_lines,
            "key_characters": key_characters[:6],
            "key_events": key_events[:8],
            "event_lines": key_events[:8],
            "detail_lines": detail_lines[:8],
            "overlay_text_lines": overlay_text_lines[:8],
            "timeline_summary": timeline_summary,
        }

    def _should_retry_stage_verifier(
        self,
        response_text: str,
        raw_response_text: str,
        finish_reason: str,
    ) -> bool:
        normalized_finish_reason = str(finish_reason or "").strip().lower()
        if normalized_finish_reason == "length":
            return True
        if self._extract_json_object(response_text or ""):
            return False
        if raw_response_text and self._extract_json_object(self._strip_code_fences(str(raw_response_text or ""))):
            return False
        return True

    def _extract_stage_verifier_key_value_payload(self, text: str) -> Optional[Dict[str, Any]]:
        normalized = self._strip_code_fences(text or "")
        normalized = re.sub(r"<think>[\s\S]*?</think>", "", normalized, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"<thinking>[\s\S]*?</thinking>", "", normalized, flags=re.IGNORECASE).strip()
        if not normalized:
            return None
        payload: Dict[str, Any] = {}
        field_names = [
            "scene_label",
            "scene_lines",
            "key_characters",
            "key_events",
            "detail_lines",
            "overlay_text_lines",
            "timeline_summary",
        ]
        for field_name in field_names:
            pattern = rf"(?im)^\s*{re.escape(field_name)}\s*[:：]\s*(.+?)(?=^\s*(?:{'|'.join(re.escape(item) for item in field_names)})\s*[:：]|\Z)"
            match = re.search(pattern, normalized, flags=re.MULTILINE | re.DOTALL)
            if not match:
                continue
            value_text = match.group(1).strip().strip(",")
            if field_name.endswith("_lines") or field_name in {"key_characters", "key_events"}:
                parsed_list = self._parse_stage_verifier_list_value(value_text)
                if parsed_list:
                    payload[field_name] = parsed_list
            else:
                payload[field_name] = self._clean_text_field(value_text, allow_json_like=False)
        if any(payload.get(name) for name in field_names):
            return payload
        return None

    def _parse_stage_verifier_list_value(self, value_text: str) -> List[str]:
        normalized = self._strip_code_fences(value_text or "").strip().strip(",")
        if not normalized:
            return []
        if normalized.startswith("[") and normalized.endswith("]"):
            try:
                parsed = json.loads(normalized)
                if isinstance(parsed, list):
                    return self._clean_list_field(parsed)
            except json.JSONDecodeError:
                pass
        lines = []
        for line in normalized.splitlines():
            cleaned = line.strip().strip(",")
            cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned)
            if cleaned:
                lines.append(cleaned)
        if len(lines) > 1:
            return self._clean_list_field(lines)
        separators = ["；", ";", "，", ","]
        for separator in separators:
            if separator in normalized:
                return self._clean_list_field([part.strip() for part in normalized.split(separator)])
        return self._clean_list_field([normalized])

    def _verify_stage_batch_parsed(
        self,
        frames: List[Frame],
        parsed: Dict[str, Any],
        raw_response: str,
        stage_id: int,
        total_stages: int,
        audio_snippets: Optional[List[AudioSnippet]] = None,
    ) -> Dict[str, Any]:
        if not self.stage_verifier_enabled or not self.verifier_client or not self.verifier_model:
            return parsed
        frame_batches = self._split_stage_verifier_frame_batches(frames)
        aggregated = {
            "scene_label": "",
            "scene_lines": [],
            "key_characters": [],
            "key_events": [],
            "event_lines": [],
            "detail_lines": [],
            "overlay_text_lines": [],
            "timeline_summary": "",
        }
        for batch_index, batch_frames in enumerate(frame_batches, 1):
            verified = self._run_stage_verifier_batch(
                batch_frames,
                parsed if batch_index == 1 else {
                    **parsed,
                    "scene_label": aggregated.get("scene_label") or parsed.get("scene_label", ""),
                    "timeline_summary": self._build_stage_verifier_progress_summary(aggregated) or parsed.get("timeline_summary", ""),
                },
                raw_response,
                stage_id,
                total_stages,
                audio_snippets=audio_snippets,
                batch_id=batch_index,
                batch_total=len(frame_batches),
            )
            if not verified:
                return parsed
            aggregated = self._merge_stage_verifier_result(aggregated, verified)
        finalized = self._finalize_stage_verifier_result(parsed, aggregated)
        if self._is_tail_single_frame_stage(frames, stage_id, total_stages):
            tail_timestamps = [f"{frames[0].timestamp:.2f}s"]
            filtered_events = self._filter_events_within_timestamp_range(
                finalized.get("key_events", []),
                frames[0].timestamp - 1.0,
                frames[-1].timestamp + 1.0,
                fallback_events=self._build_dialogue_lines_from_snippets(audio_snippets or [], limit=2),
                timestamp_hints=tail_timestamps,
            )
            finalized["key_events"] = filtered_events
            finalized["event_lines"] = filtered_events
            if not filtered_events:
                previous_stage = self.batch_stage_analyses[-1] if self.batch_stage_analyses else None
                scene_anchor = finalized.get("scene_label") or (previous_stage.scene_label if previous_stage else "")
                finalized["key_events"] = [self._build_tail_closing_event_text(scene_anchor, frames[-1].timestamp)]
                finalized["event_lines"] = finalized["key_events"]
        return self._post_process_stage_batch_parsed(finalized, audio_snippets, stage_id)

    def _get_refinement_client_and_model(self) -> tuple[LLMClient, str]:
        if self.text_refiner_enabled and self.verifier_client and self.verifier_model:
            return self.verifier_client, self.verifier_model
        return self.client, self.model

    def _build_dialogue_lines_from_snippets(self, snippets: List[AudioSnippet], limit: int = 3) -> List[str]:
        collected: List[str] = []
        for snippet in snippets:
            cleaned = self._clean_text_field(getattr(snippet, "text", "") or "")
            if cleaned:
                collected.append(self._truncate_text(cleaned, 80))
        unique_lines = list(dict.fromkeys([item for item in collected if item]))
        return self._sample_ordered_items(unique_lines, limit)

    def _build_dialogue_summary_from_snippets(self, snippets: List[AudioSnippet], limit: int = 3) -> str:
        return self._join_readable_items(self._build_dialogue_lines_from_snippets(snippets, limit=limit), "；")

    def _is_tail_single_frame_stage(self, frames: List[Frame], stage_id: int, total_stages: int) -> bool:
        return bool(frames) and stage_id == total_stages and len(frames) <= 2

    def _looks_like_tail_stage_recap(
        self,
        raw_response: str,
        current_stage_id: int,
        start_timestamp: float,
    ) -> bool:
        if not raw_response:
            return False
        stage_numbers = {
            int(match)
            for match in re.findall(r"阶段\s*(\d+)", raw_response)
            if str(match).isdigit()
        }
        if len(stage_numbers) >= 2:
            return True
        if stage_numbers and any(number != current_stage_id for number in stage_numbers):
            return True
        for start_text, _ in re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)s", raw_response):
            try:
                if float(start_text) + 3 < start_timestamp:
                    return True
            except ValueError:
                continue
        return False

    def _looks_like_stage_fallback_needed(
        self,
        parsed: Dict[str, Any],
        raw_response: str,
        frames: Optional[List[Frame]] = None,
        stage_id: Optional[int] = None,
        total_stages: Optional[int] = None,
    ) -> bool:
        summary = parsed.get("timeline_summary", "")
        scene_label = parsed.get("scene_label", "")
        key_events = parsed.get("key_events", []) or []
        has_prompt_echo = self._looks_like_stage_prompt_echo(summary) or self._looks_like_stage_prompt_echo(raw_response)
        if has_prompt_echo:
            return True
        if not summary and not key_events:
            return True
        if scene_label and self._looks_like_stage_prompt_echo(scene_label):
            return True
        if frames and stage_id and total_stages and self._is_tail_single_frame_stage(frames, stage_id, total_stages):
            if self._looks_like_tail_stage_recap(raw_response, stage_id, frames[0].timestamp):
                return True
        return False

    def _looks_like_qwen_stage_output_anomaly(
        self,
        parsed: Dict[str, Any],
        raw_response: str,
        frames: List[Frame],
    ) -> bool:
        if not raw_response:
            return True
        if self._looks_like_stage_prompt_echo(raw_response):
            return True
        summary = (parsed.get("timeline_summary") or "").strip()
        scene_label = (parsed.get("scene_label") or "").strip()
        key_events = [item for item in (parsed.get("key_events") or []) if isinstance(item, str) and item.strip()]
        if not summary and not key_events and not scene_label:
            return True
        if summary and self._looks_like_stage_prompt_echo(summary):
            return True
        if frames:
            start_ts = float(frames[0].timestamp)
            end_ts = float(frames[-1].timestamp)
            out_of_range_events: List[str] = []
            timestamped_events: List[str] = []
            for event in key_events:
                ts = self._extract_event_timestamp(event)
                if ts is not None:
                    timestamped_events.append(event)
                    if ts + 0.5 < start_ts or ts - 0.5 > end_ts:
                        out_of_range_events.append(event)
            if timestamped_events and len(out_of_range_events) == len(timestamped_events):
                return True
        return False

    def _build_stage_batch_fallback(
        self,
        frames: List[Frame],
        parsed: Dict[str, Any],
        stage_id: int,
        total_stages: int,
    ) -> Dict[str, Any]:
        previous_stage = self.batch_stage_analyses[-1] if self.batch_stage_analyses else None
        audio_lines = self._get_audio_lines_for_range(frames[0].timestamp, frames[-1].timestamp, limit=3)
        is_tail_single_frame = self._is_tail_single_frame_stage(frames, stage_id, total_stages)
        tail_timestamps = [f"{frames[0].timestamp:.2f}s"] if is_tail_single_frame else []
        if is_tail_single_frame:
            candidate_events = self._filter_events_within_timestamp_range(
                parsed.get("key_events", []),
                frames[0].timestamp - 1.0,
                frames[-1].timestamp + 1.0,
                fallback_events=audio_lines,
                timestamp_hints=tail_timestamps,
            )
            key_events = self._sanitize_stage_lines(candidate_events, kind="event")
        else:
            key_events = self._sanitize_stage_lines(parsed.get("key_events", []), kind="event")
            if not key_events:
                key_events = self._sanitize_stage_lines(audio_lines, kind="event")
        if is_tail_single_frame and not key_events and previous_stage:
            scene_anchor = previous_stage.scene_label or self._normalize_scene_location(parsed.get("scene_label", ""))
            key_events = [self._build_tail_closing_event_text(scene_anchor, frames[-1].timestamp)]
        elif not key_events and previous_stage:
            key_events = previous_stage.key_events[:2]
        key_characters = parsed.get("key_characters", []) or []
        if (not key_characters or is_tail_single_frame) and previous_stage:
            key_characters = previous_stage.key_characters[:]
        scene_label = self._normalize_scene_location(parsed.get("scene_label", "")) or ""
        if scene_label and self._is_generic_scene_label(scene_label):
            scene_label = ""
        if (not scene_label or is_tail_single_frame) and previous_stage:
            scene_label = previous_stage.scene_label
        if not scene_label:
            scene_label = "延续场景" if self.output_language.lower().startswith("zh") else "continued scene"
        if self.output_language.lower().startswith("zh"):
            if is_tail_single_frame and previous_stage:
                summary_parts = [f"阶段{stage_id}是结尾画面，延续上一阶段的场景和人物关系。"]
                if key_events:
                    summary_parts.append(f"本阶段仅确认{self._join_readable_items(key_events[:2], '、')}。")
                elif audio_lines:
                    summary_parts.append(f"本阶段可确认对白为{self._join_readable_items(audio_lines[:2], '、')}。")
                else:
                    summary_parts.append("画面未出现新的独立剧情转折。")
                timeline_summary = "".join(summary_parts)
            else:
                summary_parts = [f"阶段{stage_id}延续前一阶段的场景和人物关系。"] if previous_stage else []
                if key_events:
                    summary_parts.append(f"本阶段重点为{self._join_readable_items(key_events[:3], '、')}。")
                elif audio_lines:
                    summary_parts.append(f"音频提到{self._join_readable_items(audio_lines[:2], '、')}。")
                timeline_summary = "".join(summary_parts) or f"阶段{stage_id}继续推进当前剧情。"
        else:
            if is_tail_single_frame and previous_stage:
                timeline_summary = f"Stage {stage_id} is a closing shot that continues the previous scene."
            else:
                timeline_summary = f"Stage {stage_id} continues the current scene progression."
        return {
            "timeline_summary": timeline_summary,
            "character_lines": parsed.get("character_lines", []),
            "scene_lines": [scene_label],
            "event_lines": key_events,
            "detail_lines": [],
            "key_characters": list(dict.fromkeys(key_characters))[:6],
            "key_events": list(dict.fromkeys(key_events))[:8],
            "scene_label": scene_label,
        }

    def _extract_event_timestamp(self, text: str) -> Optional[float]:
        if not text:
            return None
        match = re.search(r"(\d+(?:\.\d+)?)\s*s", text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _filter_events_within_timestamp_range(
        self,
        events: Any,
        start_timestamp: float,
        end_timestamp: float,
        fallback_events: Optional[List[str]] = None,
        timestamp_hints: Optional[List[str]] = None,
    ) -> List[str]:
        cleaned_events: List[str] = []
        if isinstance(events, list):
            for event in events:
                if isinstance(event, str) and event.strip():
                    cleaned_events.append(event.strip())
        in_range_events: List[str] = []
        for event in cleaned_events:
            event_ts = self._extract_event_timestamp(event)
            if event_ts is None:
                in_range_events.append(event)
            elif start_timestamp <= event_ts <= end_timestamp:
                in_range_events.append(event)
        if in_range_events:
            return in_range_events
        if cleaned_events and not any(self._extract_event_timestamp(item) is not None for item in cleaned_events):
            return cleaned_events
        for hint in timestamp_hints or []:
            in_range_events.append(hint)
        for item in fallback_events or []:
            if isinstance(item, str) and item.strip():
                in_range_events.append(item.strip())
        return in_range_events

    def _build_tail_closing_event_text(self, scene_anchor: str, timestamp: float) -> str:
        timestamp_text = f"{timestamp:.2f}s"
        if self.output_language.lower().startswith("zh"):
            anchor = self._normalize_scene_location(scene_anchor) or "当前场景"
            return f"{timestamp_text} | {anchor}结尾画面定格，留给观众对人物关系与场景氛围的回味"
        return f"{timestamp_text} | closing shot in {scene_anchor or 'current scene'}, leaving room for character and atmosphere reflection"

    def _clean_list_field(self, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        cleaned_items = []
        for item in value:
            cleaned = self._clean_text_field(item)
            if cleaned:
                cleaned_items.append(cleaned)
        return cleaned_items

    def _extract_narrative_text(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(r"Narrative\s*:\s*([\s\S]+)$", text, flags=re.IGNORECASE)
        if match:
            return self._strip_code_fences(match.group(1).strip())
        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        cleaned = text
        for block in fenced_blocks:
            cleaned = cleaned.replace(f"```json\n{block}```", "")
            cleaned = cleaned.replace(f"```\n{block}```", "")
            cleaned = cleaned.replace(f"```{block}```", "")
        if self._extract_json_object(cleaned):
            return ""
        return self._strip_code_fences(cleaned)

    def _normalize_character(self, character: Any) -> Dict[str, Any]:
        if isinstance(character, str):
            return {"name": self._clean_text_field(character), "appearance": [], "role": ""}
        if isinstance(character, dict):
            appearance = character.get("appearance") or []
            if isinstance(appearance, str):
                appearance = [appearance]
            return {
                "name": self._clean_text_field(character.get("name") or character.get("label") or ""),
                "appearance": [item for item in self._clean_list_field(appearance) if item],
                "role": self._clean_text_field(character.get("role") or character.get("identity") or ""),
            }
        return {"name": "", "appearance": [], "role": ""}

    def _build_audio_reference_text(self) -> str:
        if not self.audio_summary:
            return ""
        parts = []
        for dialogue in self.audio_summary.get("key_dialogues") or []:
            parts.append(str(dialogue))
        for block in self.audio_summary.get("time_blocks") or []:
            summary = block.get("summary")
            if summary:
                parts.append(str(summary))
        return " ".join(parts).lower()

    def _strip_character_label_qualifiers(self, label: str) -> str:
        cleaned = self._clean_text_field(label)
        if not cleaned:
            return ""
        cleaned = re.sub(r"[（(][^）)]*[）)]", "", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        return cleaned.strip("，,；;。 ")

    def _is_safe_generic_character_label(self, label: str) -> bool:
        normalized = self._strip_character_label_qualifiers(label)
        if not normalized:
            return False
        if normalized.lower() in GENERIC_CHARACTER_LABELS:
            return True
        return normalized in SAFE_GENERIC_CHARACTER_LABELS_ZH

    def _is_audio_supported_character_label(self, label: str) -> bool:
        normalized = self._strip_character_label_qualifiers(label).lower()
        if not normalized:
            return False
        audio_text = self._build_audio_reference_text()
        if not audio_text:
            return False
        if normalized in audio_text:
            return True
        for canonical, aliases in SUPPORTED_RELATION_ALIASES_ZH.items():
            variants = [canonical, *aliases]
            if normalized in {item.lower() for item in variants}:
                return any(alias.lower() in audio_text for alias in variants)
        return False

    def _to_conservative_character_label(self, label: str, fallback: str = "") -> str:
        cleaned = self._clean_text_field(label)
        normalized = self._strip_character_label_qualifiers(label)
        if not cleaned or not normalized:
            return fallback
        if self.output_language.lower().startswith("zh"):
            if any(token in cleaned for token in UNCERTAIN_CHARACTER_TOKENS_ZH):
                return fallback or "人物"
            if self._is_safe_generic_character_label(normalized):
                return normalized
            if self._is_audio_supported_character_label(normalized):
                return normalized
            return fallback or "人物"
        if self._is_safe_generic_character_label(normalized) or self._is_audio_supported_character_label(normalized):
            return normalized
        return fallback or "person"

    def _get_character_output_label(self, character: Dict[str, Any]) -> str:
        name = self._clean_text_field(character.get("name") or "")
        if name and self._is_verified_character_name(name):
            return self._strip_character_label_qualifiers(name) or name

        role = self._to_conservative_character_label(character.get("role") or "", fallback="")
        if role:
            return role

        appearance = [item for item in self._clean_list_field(character.get("appearance") or []) if item]
        if appearance:
            return "人物" if self.output_language.lower().startswith("zh") else "person"
        return ""

    def _is_informative_scene_label(self, scene: str) -> bool:
        normalized = self._clean_text_field(scene).lower()
        if not normalized:
            return False
        generic_tokens = {
            "场景过渡",
            "过渡",
            "scene transition",
            "scene change",
            "transition",
            "unknown",
            "same scene",
            "continued scene",
        }
        return normalized not in generic_tokens

    def _normalize_summary_phrase(self, text: str) -> str:
        cleaned = self._clean_text_field(text, allow_json_like=False)
        if not cleaned:
            return ""
        return re.sub(r"[；;、，,。\s]+$", "", cleaned).strip()

    def _join_readable_items(self, items: List[str], separator: str = "；") -> str:
        cleaned_items = []
        for item in items:
            cleaned = self._normalize_summary_phrase(item)
            if cleaned and cleaned not in cleaned_items:
                cleaned_items.append(cleaned)
        return separator.join(cleaned_items)

    def _is_generic_scene_label(self, label: str) -> bool:
        normalized = self._normalize_summary_phrase(label)
        if not normalized:
            return True
        return normalized in GENERIC_SCENE_LABELS_ZH

    def _normalize_scene_location(self, label: str) -> str:
        normalized = self._normalize_summary_phrase(label)
        if not normalized:
            return ""
        if normalized in {"延续场景", "过渡场景"}:
            return normalized
        normalized = normalized.split("|")[0].strip()
        normalized = re.sub(r"[：:]\s*\d+\s*-\s*\d+\s*帧\s*$", "", normalized)
        normalized = re.sub(r"[：:]\s*延续上一阶段场景\s*$", "", normalized)
        normalized = re.sub(r"[：:]\s*延续前一阶段场景\s*$", "", normalized)
        normalized = normalized.replace("室内场景", "室内").replace("室外场景", "室外")
        if "/" in normalized:
            parts = []
            for part in normalized.split("/"):
                cleaned_part = part.strip()
                if cleaned_part.endswith("场景"):
                    cleaned_part = cleaned_part[:-2].strip()
                if cleaned_part and cleaned_part not in parts:
                    parts.append(cleaned_part)
            normalized = "/".join(parts)
        composite_scene_parts: List[str] = []
        if "室内" in normalized and "室内" not in composite_scene_parts:
            composite_scene_parts.append("室内")
        elif any(token in normalized for token in ["室外", "户外"]) and "室外" not in composite_scene_parts:
            composite_scene_parts.append("室外")
        for anchor in ["街道", "办公室", "卧室", "宿舍", "工地", "仓库", "楼道", "门口", "车内"]:
            if anchor in normalized and anchor not in composite_scene_parts:
                composite_scene_parts.append(anchor)
        if len(composite_scene_parts) >= 2:
            normalized = "/".join(composite_scene_parts)
        if normalized.endswith("场景") and len(normalized) <= 6:
            return normalized[:-2] or normalized
        return normalized

    def _compact_action_phrase(self, action: str, max_chars: int = 24) -> str:
        cleaned = self._sanitize_stage_text(action)
        cleaned = re.sub(r"^\d+(?:\.\d+)?\s*s\s*[\|｜]\s*", "", cleaned).strip()
        cleaned = re.sub(r"^(人物|主角|男子|女子|黑发男子|黑发人物|一位人物|一名人物)", "", cleaned).strip()
        return self._truncate_text(cleaned, max_chars)

    def _is_transition_action_text(self, text: str) -> bool:
        cleaned = self._normalize_summary_phrase(text)
        if not cleaned:
            return False
        transition_tokens = (
            "镜头切换",
            "切换到",
            "切换至",
            "转到",
            "转入",
            "场景切换",
            "画面转到",
            "画面切到",
            "转场",
            "模糊转场",
        )
        return any(token in cleaned for token in transition_tokens)

    def _is_opening_description_action_text(self, text: str) -> bool:
        cleaned = self._normalize_summary_phrase(text)
        if not cleaned:
            return False
        if self._is_transition_action_text(cleaned):
            return False
        if len(cleaned) > 80:
            return False
        description_patterns = (
            re.compile(r"在.{0,30}?(表达|倾诉|阐述|陈述|讲述).{0,30}?(感受|想法|心声|意见)"),
            re.compile(r"在.{0,30}?(表示|提到|说).{0,30}?(感受|想法|心声|意见|看法)"),
            re.compile(r"(身穿|身着|穿着).{0,30}?(服装|外套|衣服|套装)"),
            re.compile(r"处于.{0,20}?(状态|氛围|环境)"),
            re.compile(r"画面以.{0,30}?(开场|开始|呈现)"),
            re.compile(r"^一位?人物.{0,30}?(出现|登场|入镜)"),
        )
        return any(pattern.search(cleaned) for pattern in description_patterns)

    def _extract_scene_tokens(self, text: str) -> List[str]:
        cleaned = self._normalize_scene_location(text)
        if not cleaned:
            return []
        token_patterns = [
            "室内",
            "室外",
            "户外",
            "街道",
            "办公室",
            "卧室",
            "宿舍",
            "工地",
            "仓库",
            "楼道",
            "门口",
            "车内",
            "车里",
            "车上",
            "夜晚",
            "白天",
        ]
        return [token for token in token_patterns if token in cleaned]

    def _extract_scene_phrase_from_text(self, text: str) -> str:
        cleaned = self._clean_text_field(text, allow_json_like=False)
        if not cleaned:
            return ""
        typed_scene_match = re.search(r"场景(?:类型)?[:=：]\s*([^|；，。]+)", cleaned)
        if typed_scene_match:
            return self._normalize_scene_location(typed_scene_match.group(1))
        combined_match = re.search(
            r"(?:室内|室外|户外|街道|办公室|卧室|宿舍|工地|仓库|楼道|门口|夜晚|白天)"
            r"(?:内景|外景|内|外)?"
            r"(?:[/、\s]*(?:室内|室外|户外|街道|办公室|卧室|宿舍|工地|仓库|楼道|门口|夜晚|白天)"
            r"(?:内景|外景|内|外)?)*",
            cleaned,
        )
        if not combined_match or not combined_match.group(0):
            return ""
        return self._normalize_scene_location(combined_match.group(0))

    def _extract_scene_candidates_from_text(self, text: str) -> List[str]:
        cleaned = self._clean_text_field(text, allow_json_like=False)
        if not cleaned:
            return []
        candidates: List[str] = []
        typed_scene_match = re.search(r"场景(?:类型)?[:=：]\s*([^|；，。]+)", cleaned)
        if typed_scene_match:
            candidate = self._normalize_scene_location(typed_scene_match.group(1))
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        combined_pattern = re.compile(
            r"(?:室内|室外|户外|街道|办公室|卧室|宿舍|工地|仓库|楼道|门口|车内|车里|车上|夜晚|白天)"
            r"(?:内景|外景|内|外)?"
            r"(?:[/、\s]*(?:室内|室外|户外|街道|办公室|卧室|宿舍|工地|仓库|楼道|门口|车内|车里|车上|夜晚|白天)"
            r"(?:内景|外景|内|外)?)*"
        )
        for match in combined_pattern.finditer(cleaned):
            candidate = self._normalize_scene_location(match.group(0))
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _extract_scene_candidates_from_lines(self, lines: Optional[List[str]]) -> List[str]:
        candidates: List[str] = []
        for line in lines or []:
            for candidate in self._extract_scene_candidates_from_text(line):
                if candidate and not self._is_generic_scene_label(candidate) and candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def _scene_label_specificity_score(self, label: str) -> int:
        normalized = self._normalize_scene_location(label)
        if not normalized:
            return 0
        tokens = self._extract_scene_tokens(normalized)
        anchor_tokens = {
            "街道",
            "办公室",
            "卧室",
            "宿舍",
            "工地",
            "仓库",
            "楼道",
            "门口",
            "车内",
            "车里",
            "车上",
        }
        score = 0
        for token in tokens:
            if token in anchor_tokens:
                score += 4
            elif token in {"室内", "室外", "户外"}:
                score += 2
            elif token in {"夜晚", "白天"}:
                score += 1
        score += min(len(normalized), 16) // 4
        return score

    def _is_low_specificity_scene_label(self, label: str) -> bool:
        normalized = self._normalize_scene_location(label)
        if not normalized:
            return True
        tokens = set(self._extract_scene_tokens(normalized))
        if not tokens:
            return True
        anchor_tokens = {
            "街道",
            "办公室",
            "卧室",
            "宿舍",
            "工地",
            "仓库",
            "楼道",
            "门口",
            "车内",
            "车里",
            "车上",
        }
        if tokens & anchor_tokens:
            return False
        return tokens <= {"室内", "室外", "户外", "夜晚", "白天"}

    def _collect_stage_scene_candidates(self, parsed: Dict[str, Any], raw_response: str = "") -> List[str]:
        current_label = self._normalize_scene_location(parsed.get("scene_label", ""))
        candidate_sources: List[str] = []
        candidate_sources.extend(parsed.get("scene_lines", []) or [])
        candidate_sources.extend(parsed.get("key_events", []) or [])
        if parsed.get("timeline_summary"):
            candidate_sources.append(parsed.get("timeline_summary", ""))
        scene_support_exists = bool((parsed.get("scene_lines", []) or []) or parsed.get("timeline_summary"))
        if parsed.get("detail_lines") and (
            not current_label
            or self._is_generic_scene_label(current_label)
            or not scene_support_exists
        ):
            candidate_sources.extend(parsed.get("detail_lines", []) or [])
        if raw_response and (
            not current_label
            or self._is_generic_scene_label(current_label)
            or not candidate_sources
        ) and not self._looks_like_stage_prompt_echo(raw_response):
            candidate_sources.append(raw_response)
        candidates: List[str] = []
        for source in candidate_sources:
            for candidate in self._extract_scene_candidates_from_text(source):
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def _detect_transition_scene_composite(self, parsed: Dict[str, Any], raw_response: str) -> str:
        sources: List[str] = []
        if parsed.get("timeline_summary"):
            sources.append(parsed["timeline_summary"])
        if parsed.get("detail_lines"):
            sources.extend(parsed["detail_lines"])
        if raw_response:
            sources.append(raw_response)
        transition_pattern = re.compile(
            r"(?:从|由|自|在)([^，,。；;]{1,12}?)(?:切换至|切换到|转到|转入|转换到|切到)([^，,。；;]{1,12}?)(?=[\s，,。；;、]|$)"
        )
        for source in sources:
            for match in transition_pattern.finditer(source or ""):
                from_scene = self._normalize_scene_location(match.group(1))
                to_scene = self._normalize_scene_location(match.group(2))
                if from_scene and to_scene and from_scene != to_scene and "/" not in from_scene and "/" not in to_scene:
                    composite = self._normalize_scene_location(f"{from_scene}/{to_scene}")
                    if composite:
                        return composite
        return ""

    def _refine_stage_scene_label(
        self,
        parsed: Dict[str, Any],
        raw_response: str,
        previous_stage: Optional[BatchStageAnalysis],
        frames: List[Frame],
        stage_id: int,
        total_stages: int,
    ) -> Dict[str, Any]:
        refined = dict(parsed)
        composite_label = self._detect_transition_scene_composite(refined, raw_response)
        if composite_label:
            refined["scene_label"] = composite_label
            if not refined.get("scene_lines") or len(refined["scene_lines"]) < 2:
                existing = list(refined.get("scene_lines", []) or [])
                parts = composite_label.split("/")
                refined["scene_lines"] = list(dict.fromkeys([*existing, *parts]))[:3]
        current_label = self._normalize_scene_location(refined.get("scene_label", ""))
        current_score = self._scene_label_specificity_score(current_label)
        current_tokens = set(self._extract_scene_tokens(current_label))
        current_is_weak = (
            not current_label
            or self._is_generic_scene_label(current_label)
            or self._is_low_specificity_scene_label(current_label)
        )
        candidates = self._collect_stage_scene_candidates(refined, raw_response=raw_response)
        best_candidate = current_label
        best_score = current_score
        for candidate in candidates:
            candidate_score = self._scene_label_specificity_score(candidate)
            candidate_tokens = set(self._extract_scene_tokens(candidate))
            can_replace_current = (
                current_is_weak
                or not current_label
                or current_label in candidate
                or candidate in current_label
                or bool(current_tokens & candidate_tokens)
            )
            if not can_replace_current:
                continue
            if candidate_score > best_score:
                best_candidate = candidate
                best_score = candidate_score
            elif candidate_score == best_score and candidate_score > 0 and len(candidate) > len(best_candidate):
                best_candidate = candidate
                best_score = candidate_score
        if best_candidate and best_candidate != current_label and not composite_label:
            refined["scene_label"] = best_candidate
            if not refined.get("scene_lines"):
                refined["scene_lines"] = [best_candidate]
            elif refined.get("scene_lines"):
                existing = self._extract_scene_candidates_from_lines(refined.get("scene_lines", []))
                if best_candidate not in existing:
                    refined["scene_lines"] = [best_candidate] + [line for line in refined.get("scene_lines", []) if line != best_candidate]
        elif best_candidate:
            if not refined.get("scene_lines"):
                refined["scene_lines"] = [best_candidate]
            elif best_candidate not in self._extract_scene_candidates_from_lines(refined.get("scene_lines", [])):
                existing = self._extract_scene_candidates_from_lines(refined.get("scene_lines", []))
                if best_candidate not in existing:
                    refined["scene_lines"] = [best_candidate] + [line for line in refined.get("scene_lines", []) if line != best_candidate]
        if self._is_tail_single_frame_stage(frames, stage_id, total_stages) and previous_stage:
            previous_label = self._normalize_scene_location(previous_stage.scene_label)
            previous_score = self._scene_label_specificity_score(previous_label)
            if previous_score > self._scene_label_specificity_score(refined.get("scene_label", "")):
                refined["scene_label"] = previous_label
                refined["scene_lines"] = [previous_label]
        return refined

    def _action_conflicts_with_scene_location(self, action: str, location: str) -> bool:
        normalized_action = self._sanitize_stage_text(action)
        location_tokens = self._extract_scene_tokens(location)
        action_tokens = self._extract_scene_tokens(normalized_action)
        if not normalized_action or not location_tokens or not action_tokens:
            return False
        if any(token in location_tokens for token in action_tokens):
            return False
        indoor_outdoor_pairs = {("室内", "室外"), ("室外", "室内")}
        for location_token in location_tokens:
            for action_token in action_tokens:
                if (location_token, action_token) in indoor_outdoor_pairs:
                    return True
        anchor_tokens = {"宿舍", "仓库", "街道", "工地", "办公室", "卧室", "楼道", "门口"}
        if any(token in anchor_tokens for token in location_tokens) and any(token in anchor_tokens for token in action_tokens):
            return True
        if any(token in {"夜晚", "白天"} for token in location_tokens) and any(token in {"夜晚", "白天"} for token in action_tokens):
            return True
        return False

    def _filter_scene_card_actions(
        self,
        actions: List[str],
        location: str,
        scene_lines: Optional[List[str]] = None,
        limit: int = 8,
    ) -> List[str]:
        filtered: List[str] = []
        allow_mixed_scene = len(self._extract_scene_candidates_from_lines(scene_lines)) >= 2
        for action in actions:
            cleaned = self._sanitize_stage_text(action)
            if not cleaned or self._is_generic_stage_event_text(cleaned):
                continue
            if not allow_mixed_scene and self._action_conflicts_with_scene_location(cleaned, location):
                continue
            if cleaned not in filtered:
                filtered.append(cleaned)
        return filtered[:limit]

    def _is_explicit_short_scene_continuation(self, card: Dict[str, Any]) -> bool:
        current_duration = float(card.get("end_timestamp", 0.0)) - float(card.get("start_timestamp", 0.0))
        if current_duration <= 3.0:
            return True
        summary = self._clean_text_field(card.get("summary") or "", allow_json_like=False)
        continuation_markers = (
            "结尾画面延续上一阶段",
            "延续上一阶段动作",
            "延续上一阶段画面",
            "延续前一阶段动作",
            "收尾镜头延续",
        )
        return any(marker in summary for marker in continuation_markers)

    def _pick_primary_scene_action(self, actions: List[str]) -> str:
        if not actions:
            return ""
        preferred = []
        fallback = []
        for action in actions:
            compact = self._compact_action_phrase(action, 24)
            if not compact or self._is_generic_stage_event_text(compact):
                continue
            if self._is_opening_description_action_text(compact):
                continue
            if self._is_transition_action_text(compact):
                fallback.append(compact)
                continue
            preferred.append(compact)
        ordered = preferred or fallback
        return ordered[0] if ordered else ""

    def _build_scene_title(
        self,
        chunk: ChunkSummary,
        stage_analysis: Optional[BatchStageAnalysis] = None,
    ) -> str:
        base = self._normalize_scene_location(self._sanitize_stage_text(chunk.scene_label, keep_sentence=False))
        distinct_scene_lines = self._extract_scene_candidates_from_lines(stage_analysis.scene_lines if stage_analysis else [])
        if base and len(distinct_scene_lines) >= 2:
            return base
        first_event = self._pick_primary_scene_action(chunk.key_events)
        if first_event and (
            not base
            or self._action_conflicts_with_scene_location(first_event, base)
            or self._is_transition_action_text(first_event)
            or self._is_opening_description_action_text(first_event)
        ):
            first_event = ""
        if base and first_event:
            return f"{base}：{first_event}"
        if base:
            return base
        if first_event:
            return first_event
        summary_text = self._sanitize_stage_text(chunk.summary)
        if summary_text and not self._is_fallback_stage_template(summary_text):
            return self._truncate_text(summary_text, 24)
        return base or f"场景{chunk.chunk_id:03d}"

    def _build_scene_title_from_chunk(self, chunk: ChunkSummary) -> str:
        return self._build_scene_title(chunk, stage_analysis=None)

    def _is_fallback_stage_template(self, text: str) -> bool:
        cleaned = self._normalize_summary_phrase(text)
        if not cleaned:
            return True
        markers = (
            re.compile(r"^阶段\d+延续"),
            re.compile(r"^阶段\d+是结尾画面"),
            re.compile(r"^Stage \d+ continues"),
            re.compile(r"^Stage \d+ is a closing"),
            re.compile(r"^continued scene"),
            re.compile(r"^延续场景"),
        )
        return any(pattern.match(cleaned) for pattern in markers)

    def _get_audio_lines_for_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
        transcript: Optional[AudioTranscript] = None,
        limit: int = 2,
    ) -> List[str]:
        collected: List[str] = []
        if transcript and transcript.segments:
            for segment in transcript.segments:
                start = float(segment.get("start", 0.0))
                end = float(segment.get("end", start))
                if end < start_timestamp or start > end_timestamp:
                    continue
                text = self._clean_text_field(segment.get("text") or "")
                if text:
                    collected.append(self._truncate_text(text, 80))
        if not collected and self.audio_summary:
            for block in self.audio_summary.get("time_blocks") or []:
                start = float(block.get("start", 0.0))
                end = float(block.get("end", start))
                if end < start_timestamp or start > end_timestamp:
                    continue
                summary = self._clean_text_field(block.get("summary") or "")
                if summary:
                    collected.append(self._truncate_text(summary, 100))
        unique_lines = list(dict.fromkeys([item for item in collected if item]))
        return self._sample_ordered_items(unique_lines, limit)

    def _build_visual_style_hint(self, scene_title: str, characters: List[str], key_actions: List[str]) -> str:
        parts = []
        if scene_title:
            parts.append(scene_title)
        if characters:
            parts.append(f"人物：{'、'.join(characters[:3])}")
        if key_actions:
            parts.append(f"动作：{self._join_readable_items(key_actions[:2], '、')}")
        return self._truncate_text("；".join(parts), 120)

    def _get_script_location_text(self, card: Dict[str, Any]) -> str:
        location = self._normalize_summary_phrase(card.get("location", ""))
        if not location or location in {"过渡", "场景", "视频内容展示"}:
            return card.get("title") or location or "未明确场景"
        return location

    def _get_script_character_label(self, entry: Dict[str, Any]) -> str:
        display_name = self._clean_text_field(entry.get("display_name") or "")
        if display_name and display_name not in {"人物", "person"}:
            return display_name
        character_id = self._clean_text_field(entry.get("character_id") or "")
        if character_id.startswith("appearance:"):
            appearance_label = character_id.split(":", 1)[1].split("，")[0].strip()
            return appearance_label or display_name or "人物"
        return display_name or "人物"

    def _build_final_audio_reference_text(self, transcript: Optional[AudioTranscript] = None) -> str:
        fragments: List[str] = []
        if transcript and getattr(transcript, "text", ""):
            fragments.append(str(transcript.text))
        if self.audio_summary:
            fragments.extend(
                str(item)
                for item in (self.audio_summary.get("key_dialogues") or [])[:12]
                if str(item).strip()
            )
            for block in (self.audio_summary.get("time_blocks") or [])[:12]:
                summary = self._clean_text_field(block.get("summary") or "")
                if summary:
                    fragments.append(summary)
        return self._normalize_match_text(" ".join(fragments))

    def _filter_final_overlay_text_lines(
        self,
        lines: List[str],
        audio_reference_text: str,
        limit: int = 3,
    ) -> List[str]:
        filtered: List[str] = []
        for line in lines:
            normalized = self._normalize_overlay_text_line(line)
            if not normalized:
                continue
            payload = self._strip_platform_overlay_noise(self._extract_overlay_text_payload(normalized))
            if not payload:
                continue
            marker = normalized.split("：", 1)[0].strip() if "：" in normalized else ""
            if self._looks_like_platform_overlay_noise(normalized) and not self._has_audio_support_for_phrase(
                payload,
                audio_reference_text,
            ):
                continue
            if marker not in {"字幕", "画面文字"} and not self._has_audio_support_for_phrase(
                payload,
                audio_reference_text,
            ):
                continue
            cleaned_line = f"{marker or '字幕'}：{payload}"
            if cleaned_line not in filtered:
                filtered.append(cleaned_line)
        return filtered[:limit]

    def _filter_final_story_actions(
        self,
        lines: List[str],
        overlay_text_lines: Optional[List[str]],
        audio_reference_text: str,
        limit: int = 6,
    ) -> List[str]:
        filtered: List[str] = []
        for line in self._filter_story_event_candidates(lines, overlay_text_lines=overlay_text_lines, limit=limit * 2):
            cleaned = self._sanitize_stage_text(line)
            if not cleaned:
                continue
            if self._is_opening_description_action_text(cleaned):
                continue
            if self._looks_like_platform_overlay_noise(cleaned) and not self._has_audio_support_for_phrase(
                cleaned,
                audio_reference_text,
            ):
                continue
            stripped = self._strip_platform_overlay_noise(cleaned)
            if not stripped:
                continue
            if stripped not in filtered:
                filtered.append(stripped)
        return filtered[:limit]

    def _sanitize_scene_cards_for_final_output(
        self,
        scene_cards: List[Dict[str, Any]],
        transcript: Optional[AudioTranscript] = None,
    ) -> List[Dict[str, Any]]:
        sanitized_cards: List[Dict[str, Any]] = []
        audio_reference_text = self._build_final_audio_reference_text(transcript)
        for card in scene_cards:
            normalized_card = dict(card)
            source_overlay_lines = card.get("overlay_text_lines", []) or []
            overlay_reference_matches = [
                self._normalize_match_text(self._strip_platform_overlay_noise(item) or item)
                for item in self._collect_overlay_text_references(source_overlay_lines)
            ]
            overlay_text_lines = self._filter_final_overlay_text_lines(
                source_overlay_lines,
                audio_reference_text,
                limit=3,
            )
            key_actions = self._filter_final_story_actions(
                card.get("key_actions", []) or [],
                source_overlay_lines,
                audio_reference_text,
                limit=6,
            )
            summary = self._clean_text_field(card.get("summary") or "", allow_json_like=False)
            if summary and self._looks_like_platform_overlay_noise(summary) and not self._has_audio_support_for_phrase(
                summary,
                audio_reference_text,
            ):
                summary = ""
            title = self._clean_text_field(card.get("title") or "", allow_json_like=False)
            stripped_title = self._strip_platform_overlay_noise(title)
            normalized_title_reference = self._normalize_match_text(stripped_title or title)
            title_contains_overlay_phrase = any(
                reference and len(reference) >= 4 and reference in normalized_title_reference
                for reference in overlay_reference_matches
            )
            if (
                (
                    self._looks_like_platform_overlay_noise(title)
                    or self._matches_overlay_text(title, source_overlay_lines)
                    or self._matches_overlay_text(stripped_title, source_overlay_lines)
                    or title_contains_overlay_phrase
                )
                and not self._has_audio_support_for_phrase(stripped_title or title, audio_reference_text)
            ):
                title = ""
            normalized_card["key_actions"] = key_actions
            normalized_card["overlay_text_lines"] = overlay_text_lines
            normalized_card["summary"] = summary
            if title:
                normalized_card["title"] = title
            elif normalized_card.get("location"):
                normalized_card["title"] = normalized_card.get("location")
            elif key_actions:
                normalized_card["title"] = key_actions[0]
            sanitized_cards.append(normalized_card)
        return sanitized_cards

    def _sanitize_final_video_description(
        self,
        response_text: str,
        scene_cards: List[Dict[str, Any]],
        transcript: Optional[AudioTranscript] = None,
    ) -> str:
        cleaned_lines: List[str] = []
        audio_reference_text = self._build_final_audio_reference_text(transcript)
        overlay_references: List[str] = []
        for card in scene_cards:
            overlay_references.extend(card.get("overlay_text_lines", []) or [])
        for line in (response_text or "").splitlines():
            cleaned = self._clean_text_field(line, allow_json_like=False)
            if not cleaned:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            stripped = self._strip_platform_overlay_noise(cleaned)
            if not stripped:
                continue
            if self._looks_like_platform_overlay_noise(cleaned) and not self._has_audio_support_for_phrase(
                stripped,
                audio_reference_text,
            ):
                continue
            if self._matches_overlay_text(cleaned, overlay_references) and not self._has_audio_support_for_phrase(
                stripped,
                audio_reference_text,
            ):
                continue
            cleaned_lines.append(stripped)
        sanitized = "\n".join(cleaned_lines).strip()
        return sanitized or self._clean_text_field(response_text or "", allow_json_like=False)

    def _build_scene_plot_for_script(self, card: Dict[str, Any], beat: Dict[str, Any]) -> str:
        location = self._get_script_location_text(card)
        characters = card.get("characters", []) or ["人物"]
        actions_text = self._join_readable_items(card.get("key_actions", [])[:3], "、")
        dialogue_text = card.get("dialogue_summary") or self._join_readable_items(card.get("dialogue", [])[:2], "；")
        dialogue_text = self._truncate_text(self._sanitize_stage_text(dialogue_text), 120)
        summary_text = self._clean_text_field(card.get("summary") or beat.get("summary") or "")
        if self.output_language.lower().startswith("zh"):
            parts = []
            if actions_text:
                parts.append(f"在{location}，{'、'.join(characters[:4])}围绕{actions_text}展开。")
            elif summary_text:
                parts.append(summary_text)
            if dialogue_text:
                parts.append(f"旁白/对白提及：{self._normalize_summary_phrase(dialogue_text)}。")
            return "".join(parts) or f"在{location}，画面继续推进当前剧情。"
        parts = []
        if actions_text:
            parts.append(f"In {location}, {', '.join(characters[:4])} carry out {actions_text}.")
        elif summary_text:
            parts.append(summary_text)
        if dialogue_text:
            parts.append(f"Dialogue/Narration: {dialogue_text}.")
        return " ".join(parts) or f"The story continues in {location}."

    def _build_transition_note(self, previous_title: str, current_title: str, index: int) -> str:
        if index == 1:
            return "开场建立当前人物状态与核心环境。"
        if not previous_title or previous_title == current_title:
            return "延续上一阶段，保持动作与情绪承接。"
        return f"从“{previous_title}”转入“{current_title}”，保持时间顺序连续。"

    def _format_scene_cards_for_prompt(self, scene_cards: List[Dict[str, Any]], limit: Optional[int] = None) -> str:
        selected = self._sample_ordered_items(scene_cards, limit or len(scene_cards))
        lines = []
        for card in selected:
            lines.append(
                f"{card.get('scene_id')} | {card.get('start_timestamp', 0):.2f}-{card.get('end_timestamp', 0):.2f}s | "
                f"title={card.get('title', '')} | location={card.get('location', '')} | "
                f"characters={', '.join(card.get('characters', [])[:4]) or 'n/a'} | "
                f"actions={self._join_readable_items(card.get('key_actions', [])[:3], '、') or 'n/a'} | "
                f"dialogue={card.get('dialogue_summary', '') or 'n/a'}"
            )
        return "\n".join(lines)

    def _format_story_beats_for_prompt(self, story_beats: List[Dict[str, Any]], limit: Optional[int] = None) -> str:
        selected = self._sample_ordered_items(story_beats, limit or len(story_beats))
        lines = []
        for beat in selected:
            lines.append(
                f"{beat.get('beat_id')} | order={beat.get('order')} | "
                f"scene={','.join(beat.get('related_scene_ids', []))} | "
                f"summary={beat.get('summary', '')} | transition={beat.get('transition_type', '')}"
            )
        return "\n".join(lines)

    def _format_character_timeline_for_prompt(
        self,
        character_timeline: List[Dict[str, Any]],
        limit: Optional[int] = None,
    ) -> str:
        sorted_timeline = sorted(
            character_timeline,
            key=lambda item: len(item.get("appearances", [])),
            reverse=True,
        )
        selected = self._sample_ordered_items(sorted_timeline, limit or len(sorted_timeline))
        lines = []
        for entry in selected:
            lines.append(
                f"{entry.get('display_name')} | appearances={len(entry.get('appearances', []))} | "
                f"actions={self._join_readable_items(entry.get('key_actions', [])[:4], '、') or 'n/a'} | "
                f"dialogue={self._join_readable_items(entry.get('dialogue_hints', [])[:3], '、') or 'n/a'}"
            )
        return "\n".join(lines)

    def _format_chunk_summaries_for_prompt(self, limit: Optional[int] = None) -> str:
        selected = self._sample_ordered_items(self.chunk_summaries, limit or len(self.chunk_summaries))
        lines = []
        for chunk in selected:
            lines.append(
                f"Chunk {chunk.chunk_id} | {chunk.start_timestamp:.2f}-{chunk.end_timestamp:.2f}s | "
                f"scene={chunk.scene_label or 'n/a'} | summary={self._truncate_text(chunk.summary, 180)} | "
                f"events={self._join_readable_items(chunk.key_events[:3], '、') or 'n/a'}"
            )
        return "\n".join(lines)

    def _build_transcript_digest(self, transcript: Optional[AudioTranscript]) -> str:
        if self.audio_summary and self.audio_summary.get("time_blocks"):
            blocks = self._sample_ordered_items(
                self.audio_summary.get("time_blocks") or [],
                self.reconstruction_transcript_block_cap,
            )
            return "\n".join(
                f"{float(block.get('start', 0.0)):.2f}-{float(block.get('end', 0.0)):.2f}s | "
                f"{self._truncate_text(self._clean_text_field(block.get('summary') or ''), 160)}"
                for block in blocks
                if self._clean_text_field(block.get("summary") or "")
            )
        if transcript and transcript.segments:
            segments = self._sample_ordered_items(transcript.segments, self.reconstruction_transcript_block_cap)
            return "\n".join(
                f"{float(segment.get('start', 0.0)):.2f}-{float(segment.get('end', 0.0)):.2f}s | "
                f"{self._truncate_text(self._clean_text_field(segment.get('text') or ''), 160)}"
                for segment in segments
                if self._clean_text_field(segment.get("text") or "")
            )
        return ""

    def _build_storyline_text(self, scene_cards: List[Dict[str, Any]]) -> str:
        scene_labels = []
        action_fragments = []
        transition_fragments = []
        for card in scene_cards:
            label = self._normalize_summary_phrase(card.get("location", ""))
            if label and self._is_informative_scene_label(label) and label not in scene_labels:
                scene_labels.append(label)
            for action in card.get("key_actions", [])[:2]:
                cleaned_action = self._sanitize_stage_text(action)
                if not cleaned_action:
                    continue
                if self._is_transition_action_text(cleaned_action):
                    if cleaned_action not in transition_fragments:
                        transition_fragments.append(cleaned_action)
                    continue
                if cleaned_action not in action_fragments:
                    action_fragments.append(cleaned_action)
        selected_actions = action_fragments or transition_fragments

        if self.output_language.lower().startswith("zh"):
            if scene_labels and selected_actions:
                return f"视频按时间顺序围绕{'、'.join(scene_labels[:4])}等阶段展开，重点呈现{self._join_readable_items(selected_actions[:4], '、')}。"
            if selected_actions:
                return f"视频按时间顺序推进，重点呈现{self._join_readable_items(selected_actions[:4], '、')}。"
            return "根据已识别画面与音频，故事围绕人物关系、劳动报酬与去留选择逐步展开。"

        if scene_labels and selected_actions:
            return f"The video moves through {' / '.join(scene_labels[:4])} and focuses on {self._join_readable_items(selected_actions[:4], ', ')}."
        if selected_actions:
            return f"The video develops chronologically around {self._join_readable_items(selected_actions[:4], ', ')}."
        return "The story advances through consecutive scenes and character interactions."

    def _extract_section_lines(self, text: str, headers: List[str]) -> List[str]:
        if not text:
            return []
        lines = [line.strip(" -\t") for line in self._strip_code_fences(text).splitlines()]
        target_headers = {header.lower() for header in headers}
        capture = False
        collected: List[str] = []
        for line in lines:
            if not line:
                if capture and collected:
                    break
                continue
            line_lower = line.lower().rstrip(":")
            if line_lower in target_headers:
                capture = True
                continue
            if capture and line.endswith(":") and line_lower.rstrip(":") not in target_headers:
                break
            if capture:
                collected.append(line.strip())
        return collected

    def _extract_section_text(self, text: str, headers: List[str], stop_headers: List[str]) -> str:
        lines = self._extract_section_lines(text, headers)
        if lines:
            return " ".join(lines).strip()

        escaped_headers = "|".join(re.escape(header) for header in headers)
        escaped_stops = "|".join(re.escape(header) for header in stop_headers)
        if escaped_stops:
            stop_pattern = rf"(?=\s*(?:{escaped_stops})\s*:|$)"
        else:
            stop_pattern = r"(?=$)"
        pattern = rf"(?:^|\s)(?:{escaped_headers})\s*:\s*(.+?){stop_pattern}"
        match = re.search(pattern, self._strip_code_fences(text), flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _parse_section_items(self, text: str) -> List[str]:
        if not text:
            return []
        normalized = re.sub(r"\s+", " ", text).strip()
        bullet_items = [
            item.strip()
            for item in re.split(r"(?:^|\s)-\s+", normalized)
            if item.strip()
        ]
        if len(bullet_items) > 1:
            return self._clean_list_field(bullet_items)
        return self._clean_list_field([normalized])

    def _parse_chunk_model_response(self, response_text: str) -> Dict[str, List[str] | str]:
        summary = self._clean_text_field(
            self._extract_section_text(
                response_text,
                ["Summary", "摘要"],
                ["Key Characters", "关键人物", "Key Events", "关键事件"],
            )
        )
        return {
            "summary": summary,
            "key_characters": self._parse_section_items(
                self._extract_section_text(
                    response_text,
                    ["Key Characters", "关键人物"],
                    ["Key Events", "关键事件"],
                )
            ),
            "key_events": self._parse_section_items(
                self._extract_section_text(
                    response_text,
                    ["Key Events", "关键事件"],
                    [],
                )
            ),
        }

    def _format_character_display_name(self, key: str, profile: CharacterProfile) -> str:
        for alias in profile.aliases:
            if self._is_verified_character_name(alias):
                return self._strip_character_label_qualifiers(alias) or alias
        if profile.roles:
            for role in profile.roles:
                conservative_role = self._to_conservative_character_label(role, fallback="")
                if conservative_role:
                    return conservative_role
        if profile.appearance:
            return "人物" if self.output_language.lower().startswith("zh") else "person"
        if key.startswith("appearance:"):
            return "人物" if self.output_language.lower().startswith("zh") else "person"
        return self._to_conservative_character_label(key, fallback="人物" if self.output_language.lower().startswith("zh") else "person")

    def _is_verified_character_name(self, name: str) -> bool:
        normalized = (name or "").strip().lower()
        if not normalized:
            return False
        if self._is_safe_generic_character_label(name):
            return True
        if normalized in GENERIC_CHARACTER_LABELS:
            return True
        if any(label in normalized.split() for label in GENERIC_CHARACTER_LABELS):
            return True
        if normalized.startswith("character_") or normalized.startswith("person_"):
            return True

        for profile in self.story_memory.characters.values():
            aliases = [alias.strip().lower() for alias in profile.aliases if alias]
            if normalized == profile.character_id or normalized in aliases:
                return True

        return self._is_audio_supported_character_label(name)

    def _sanitize_character_names(self, structured: StructuredFrameAnalysis) -> None:
        for character in structured.characters:
            name = (character.get("name") or "").strip()
            if not name:
                character["role"] = self._to_conservative_character_label(character.get("role") or "", fallback="")
                continue
            if not self._is_verified_character_name(name):
                character["name"] = ""
            character["role"] = self._to_conservative_character_label(character.get("role") or "", fallback="")

    def parse_frame_response(self, frame: Frame, response_text: str) -> StructuredFrameAnalysis:
        payload = self._extract_json_object(response_text) or {}

        scene = self._clean_text_field(payload.get("scene") or "")
        actions = self._clean_list_field(payload.get("actions") or [])
        objects = self._clean_list_field(payload.get("objects") or [])
        continuity_points = self._clean_list_field(payload.get("continuity_points") or payload.get("key_continuity") or [])
        dialogue_hint = self._clean_text_field(payload.get("dialogue_hint") or payload.get("dialogue") or "")
        characters = [self._normalize_character(item) for item in (payload.get("characters") or [])]

        model_scene_changed = bool(payload.get("scene_changed", False))
        structured = StructuredFrameAnalysis(
            frame_number=frame.number,
            timestamp=frame.timestamp,
            scene=scene,
            characters=characters,
            actions=actions,
            objects=objects,
            dialogue_hint=dialogue_hint,
            continuity_points=continuity_points,
            raw_response=response_text.strip(),
            scene_changed=model_scene_changed,
        )
        self._apply_fallback_structure(structured)
        self._sanitize_character_names(structured)
        structured.scene_changed = self.is_scene_change(frame, structured, model_scene_changed=model_scene_changed)
        return structured

    def _apply_fallback_structure(self, structured: StructuredFrameAnalysis) -> None:
        text = self._extract_narrative_text(structured.raw_response) or structured.raw_response
        if not text:
            return

        sentences = [
            sentence.strip(" -")
            for sentence in self._strip_code_fences(text).replace("\n", " ").split(".")
            if sentence.strip()
        ]
        lower_text = text.lower()

        if not structured.scene and sentences:
            structured.scene = sentences[0][:160]

        if not structured.actions and sentences:
            structured.actions = sentences[:2]

        if not structured.continuity_points and len(sentences) > 1:
            structured.continuity_points = sentences[1:3]

        if not structured.objects:
            keyword_candidates = [
                "book", "cash", "phone", "camera", "bed", "table", "chair",
                "screen", "game interface", "door", "bag", "food",
            ]
            matched = [keyword for keyword in keyword_candidates if keyword in lower_text]
            structured.objects = matched[:5]

        if not structured.characters:
            if "character" in lower_text or "person" in lower_text or "people" in lower_text:
                structured.characters = [{"name": "", "appearance": [], "role": ""}]

        if not structured.dialogue_hint and any(token in lower_text for token in ["say", "says", "dialogue", "ask", "asks"]):
            structured.dialogue_hint = sentences[0][:160] if sentences else ""

    def _build_compact_frame_response(self, structured: StructuredFrameAnalysis) -> str:
        character_labels = []
        for character in structured.characters:
            label = character.get("name") or character.get("role") or ", ".join(character.get("appearance") or [])
            if label:
                character_labels.append(label)
        character_text = "、".join(list(dict.fromkeys(character_labels))[:3]) or "人物未明确"
        action_text = "；".join(structured.actions[:2]) or "暂无明确动作"
        scene_text = structured.scene or "场景未明确"
        dialogue_text = structured.dialogue_hint or "无明确对白"
        if self.output_language.lower().startswith("zh"):
            return f"场景：{scene_text}；人物：{character_text}；动作：{action_text}；对白线索：{dialogue_text}。"
        return f"Scene: {scene_text}; Characters: {character_text}; Actions: {action_text}; Dialogue: {dialogue_text}."

    def _append_unique(self, collection: List[str], values: List[str], limit: int = 20) -> None:
        for value in values:
            if value and value not in collection:
                collection.append(value)
        if len(collection) > limit:
            del collection[:-limit]

    def _character_key(self, character: Dict[str, Any]) -> str:
        name = (character.get("name") or "").strip().lower()
        if name:
            return name
        appearance = ",".join(sorted(character.get("appearance") or [])).strip().lower()
        if appearance:
            return f"appearance:{appearance}"
        return "unknown"

    def _extract_audio_labels(self, audio_snippets: List[AudioSnippet]) -> List[str]:
        labels = []
        for snippet in audio_snippets:
            if snippet.speaker_hint:
                labels.append(snippet.speaker_hint)
        return labels

    def update_character_registry(self, structured_result: StructuredFrameAnalysis, audio_snippets: List[AudioSnippet]) -> None:
        audio_labels = self._extract_audio_labels(audio_snippets)
        for index, character in enumerate(structured_result.characters):
            key = self._character_key(character)
            if key == "unknown" and audio_labels:
                key = audio_labels[0]
                character["name"] = audio_labels[0]
            elif key == "unknown":
                key = f"character_{index + 1}"

            profile = self.story_memory.characters.get(key)
            if profile is None:
                profile = CharacterProfile(character_id=key, confidence=0.6 if character.get("name") else 0.4)
                self.story_memory.characters[key] = profile

            if character.get("name"):
                self._append_unique(profile.aliases, [character["name"]], limit=5)
            self._append_unique(profile.appearance, character.get("appearance") or [], limit=8)
            if character.get("role"):
                self._append_unique(profile.roles, [character["role"]], limit=5)
            self._append_unique(profile.key_actions, structured_result.actions[:3], limit=10)
            if structured_result.dialogue_hint:
                self._append_unique(profile.dialogue_hints, [structured_result.dialogue_hint], limit=10)
            profile.last_seen_timestamp = structured_result.timestamp
            profile.last_seen_scene = structured_result.scene or profile.last_seen_scene
            profile.appearances.append(
                {
                    "frame_number": structured_result.frame_number,
                    "timestamp": structured_result.timestamp,
                    "scene": structured_result.scene,
                }
            )

    def update_story_memory(self, structured_result: StructuredFrameAnalysis) -> None:
        if structured_result.scene and self._is_informative_scene_label(structured_result.scene):
            self.story_memory.scene_summary = structured_result.scene
        overlay_text_lines = [
            line
            for line in (
                self._normalize_summary_phrase(item)
                for item in structured_result.overlay_text_lines[:5]
            )
            if line
        ]
        self._append_unique(self.story_memory.overlay_text_lines, overlay_text_lines, limit=20)
        filtered_props = self._filter_story_detail_lines(
            structured_result.objects[:5],
            overlay_text_lines=structured_result.overlay_text_lines,
            limit=12,
        )
        self._append_unique(self.story_memory.active_props, filtered_props, limit=12)
        event_candidates = self._filter_story_event_candidates(
            structured_result.actions[:3] + structured_result.continuity_points[:2],
            overlay_text_lines=structured_result.overlay_text_lines,
            limit=20,
        )
        self._append_unique(self.story_memory.key_events, event_candidates, limit=20)

    def update_recent_window(self, structured_result: StructuredFrameAnalysis) -> None:
        self.recent_frame_window.append(structured_result)
        if len(self.recent_frame_window) > max(self.context_window, 1):
            self.recent_frame_window = self.recent_frame_window[-max(self.context_window, 1):]

    def is_scene_change(
        self,
        frame: Frame,
        _structured_result: StructuredFrameAnalysis,
        model_scene_changed: bool = False,
    ) -> bool:
        if not self.recent_frame_window:
            return False
        if model_scene_changed:
            return True
        if frame.score >= self.scene_change_threshold:
            return True
        return False

    def should_flush_chunk(self, structured_result: StructuredFrameAnalysis) -> bool:
        if not self.enable_chunk_summary:
            return False
        if len(self.chunk_buffer) >= self.chunk_max_frames:
            return True
        if not structured_result.scene_changed:
            return False
        if len(self.chunk_buffer) < self.chunk_min_frames_before_summary:
            return False
        chunk_duration = self.chunk_buffer[-1].timestamp - self.chunk_buffer[0].timestamp
        return chunk_duration >= self.chunk_min_duration_seconds

    def _build_deterministic_chunk_summary(self) -> str:
        if not self.chunk_buffer:
            return ""
        scenes = [item.scene for item in self.chunk_buffer if item.scene]
        actions = []
        characters = []
        for item in self.chunk_buffer:
            actions.extend(item.actions[:2])
            for character in item.characters:
                label = character.get("name") or ", ".join(character.get("appearance") or [])
                if label:
                    characters.append(label)
        if self.output_language.lower().startswith("zh"):
            return (
                f"阶段场景推进：{' -> '.join(scenes[:3]) or '未明确'}。"
                f" 关键人物：{'、'.join(list(dict.fromkeys(characters))[:4]) or '未明确'}。"
                f" 关键动作：{'、'.join(list(dict.fromkeys(actions))[:5]) or '未明确'}。"
            )
        return (
            f"Scene progression: {' -> '.join(scenes[:3]) or 'n/a'}. "
            f"Key characters: {', '.join(list(dict.fromkeys(characters))[:4]) or 'n/a'}. "
            f"Key actions: {', '.join(list(dict.fromkeys(actions))[:5]) or 'n/a'}."
        )

    def summarize_chunk(self) -> Optional[ChunkSummary]:
        if not self.chunk_buffer:
            return None

        summarize_started_at = time.perf_counter()
        chunk_id = len(self.chunk_summaries) + 1
        chunk_notes = []
        for item in self.chunk_buffer:
            chunk_notes.append(
                f"Frame {item.frame_number} ({item.timestamp:.2f}s) | scene={item.scene or 'unknown'} | "
                f"actions={', '.join(item.actions) or 'n/a'} | dialogue={item.dialogue_hint or 'n/a'}"
            )
        chunk_prompt = self.chunk_prompt.replace("{CHUNK_FRAME_NOTES}", "\n".join(chunk_notes))
        chunk_prompt = chunk_prompt.replace("{OUTPUT_LANGUAGE}", self._get_output_language_name())

        summary_text = self._build_deterministic_chunk_summary()
        try:
            llm_started_at = time.perf_counter()
            response = self.client.generate(
                prompt=chunk_prompt,
                model=self.model,
                temperature=self.temperature,
                num_predict=220,
            )
            llm_elapsed = time.perf_counter() - llm_started_at
            model_summary = self._clean_text_field((response or {}).get("response", ""), allow_json_like=False)
            summary_text = model_summary or summary_text
        except Exception as exc:
            llm_elapsed = None
            logger.warning("Chunk summary fallback triggered: %s", exc)

        key_characters = []
        key_events = []
        dominant_scene = ""
        for item in self.chunk_buffer:
            key_events.extend(item.actions[:2])
            for character in item.characters:
                label = self._get_character_output_label(character)
                if label:
                    key_characters.append(label)
        scene_counts = Counter(item.scene for item in self.chunk_buffer if self._is_informative_scene_label(item.scene))
        if scene_counts:
            dominant_scene = scene_counts.most_common(1)[0][0]

        chunk = ChunkSummary(
            chunk_id=chunk_id,
            start_frame=self.chunk_buffer[0].frame_number,
            end_frame=self.chunk_buffer[-1].frame_number,
            start_timestamp=self.chunk_buffer[0].timestamp,
            end_timestamp=self.chunk_buffer[-1].timestamp,
            summary=summary_text,
            scene_label=dominant_scene,
            key_characters=list(dict.fromkeys(key_characters))[:6],
            key_events=list(dict.fromkeys(key_events))[:8],
        )
        parsed_sections = self._parse_chunk_model_response(summary_text)
        parsed_summary = self._clean_text_field(parsed_sections.get("summary", ""))
        if parsed_summary:
            chunk.summary = parsed_summary
        parsed_characters = [
            self._to_conservative_character_label(item, fallback="人物" if self.output_language.lower().startswith("zh") else "person")
            for item in parsed_sections.get("key_characters", [])
            if item
        ]
        if parsed_characters:
            chunk.key_characters = list(dict.fromkeys(parsed_characters))[:6]
        parsed_events = [item for item in parsed_sections.get("key_events", []) if item]
        if parsed_events:
            chunk.key_events = list(dict.fromkeys(parsed_events))[:8]
        self.chunk_summaries.append(chunk)
        self.story_memory.last_chunk_summary = chunk.summary
        self.chunk_buffer = []
        _debug_emit(
            "C",
            "analyzer.py:summarize_chunk",
            "Chunk summary completed",
            {
                "chunk_id": chunk_id,
                "frame_span": [chunk.start_frame, chunk.end_frame],
                "key_characters": len(chunk.key_characters),
                "key_events": len(chunk.key_events),
                "llm_elapsed_seconds": round(llm_elapsed, 3) if llm_elapsed is not None else None,
                "total_elapsed_seconds": round(time.perf_counter() - summarize_started_at, 3),
            },
        )
        return chunk

    def flush_chunk_if_needed(self, structured_result: Optional[StructuredFrameAnalysis] = None, force: bool = False) -> Optional[ChunkSummary]:
        if force:
            return self.summarize_chunk()
        if structured_result is not None and self.should_flush_chunk(structured_result):
            return self.summarize_chunk()
        return None

    def _get_audio_snippets_for_range(self, start_timestamp: float, end_timestamp: float) -> List[AudioSnippet]:
        if not self.audio_memory:
            return []
        matched = []
        for snippet in self.audio_memory:
            if snippet.end < start_timestamp or snippet.start > end_timestamp:
                continue
            matched.append(snippet)
        return matched

    def _format_stage_context(self) -> str:
        if not self.batch_stage_analyses:
            return "无上一阶段摘要。"
        selected = self.batch_stage_analyses[-self.stage_batch_context_summary_count :]
        lines = []
        for item in selected:
            lines.append(
                f"阶段{item.stage_id} | {item.start_timestamp:.2f}-{item.end_timestamp:.2f}s | "
                f"场景={item.scene_label or '未明确'} | 关键人物={self._join_readable_items(item.key_characters[:4], '、') or '未明确'} | "
                f"关键事件={self._join_readable_items(item.key_events[:4], '、') or '未明确'}"
            )
        return "\n".join(lines)

    def _build_stage_batch_prompt(self, frames: List[Frame], stage_id: int, total_stages: int) -> str:
        if not self.stage_batch_prompt:
            raise ValueError("Stage batch prompt is not configured")
        stage_start = frames[0]
        stage_end = frames[-1]
        stage_audio_snippets = self._get_audio_snippets_for_range(stage_start.timestamp, stage_end.timestamp)
        audio_lines = self._build_dialogue_lines_from_snippets(stage_audio_snippets, limit=6)
        if not audio_lines:
            audio_lines = self._get_audio_lines_for_range(stage_start.timestamp, stage_end.timestamp, limit=6)
        frame_lines = [
            f"- frame={frame.number} timestamp={frame.timestamp:.2f}s scene_hint={frame.scene_hint or 'unknown'} difference_score={frame.score:.2f}"
            for frame in frames
        ]
        extras = [
            "",
            "补充上下文：",
            f"- 当前阶段：{stage_id}/{total_stages}",
            f"- 当前阶段帧范围：{stage_start.number}-{stage_end.number}",
            f"- 当前阶段时间范围：{stage_start.timestamp:.2f}s-{stage_end.timestamp:.2f}s",
            f"- 上一阶段摘要：\n{self._format_stage_context()}",
            f"- 邻近音频线索：\n{self._join_readable_items(audio_lines, '；') or '无明确音频线索'}",
            "- 当前阶段帧清单：",
            "\n".join(frame_lines),
        ]
        return self.stage_batch_prompt + "\n" + "\n".join(extras)

    def _extract_numbered_section(self, text: str, title: str, next_titles: List[str]) -> str:
        escaped_title = re.escape(title)
        heading_prefix = r"(?:^|\n)\s*(?:#+\s*)?(?:\d+\.\s*)?【"
        if next_titles:
            next_pattern = "|".join(re.escape(item) for item in next_titles)
            lookahead = rf"(?=\n\s*(?:#+\s*)?(?:\d+\.\s*)?【(?:{next_pattern})】|\Z)"
        else:
            lookahead = r"(?=\Z)"
        pattern = rf"{heading_prefix}{escaped_title}】\s*([\s\S]*?){lookahead}"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    def _split_stage_lines(self, text: str) -> List[str]:
        if not text:
            return []
        lines = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*[-*]\s*", "", line.strip())
            cleaned = re.sub(r"^\s*\d+[.、]\s+", "", cleaned)
            cleaned = self._clean_text_field(cleaned, allow_json_like=False)
            if cleaned:
                lines.append(cleaned)
        return list(dict.fromkeys(lines))

    def _extract_character_labels_from_stage_lines(self, lines: List[str]) -> List[str]:
        labels: List[str] = []
        for line in lines:
            candidate = ""
            numbered_character_match = re.search(r"人物\d+\s*[:：]\s*([^|；，。]+)", line)
            if numbered_character_match:
                candidate = numbered_character_match.group(1).strip()
            if not candidate:
                bold_name_match = re.search(r"\*\*([^*]+)\*\*\s*[:：]", line)
                if bold_name_match:
                    candidate = bold_name_match.group(1).strip()
            role_match = re.search(r"角色[:：]\s*([^|；，。]+)", line)
            if not candidate and role_match:
                candidate = role_match.group(1).strip()
            if not candidate:
                generic_name_match = re.search(r"^([^:：]+)\s*[:：]", line)
                if generic_name_match:
                    candidate = generic_name_match.group(1).strip(" -*")
            if not candidate:
                for token in [
                    "哥哥",
                    "姐姐",
                    "弟弟",
                    "妹妹",
                    "父亲",
                    "母亲",
                    "少女",
                    "女孩",
                    "男孩",
                    "工人",
                    "人物",
                    "同伴",
                    "老师",
                ]:
                    if token in line:
                        candidate = token
                        break
            normalized_candidate = self._strip_character_label_qualifiers(candidate or "")
            if normalized_candidate in SUPPORTED_RELATION_ALIASES_ZH:
                candidate = normalized_candidate
            else:
                candidate = self._to_conservative_character_label(
                    candidate or line,
                    fallback="人物" if self.output_language.lower().startswith("zh") else "person",
                )
            if candidate and candidate not in labels:
                labels.append(candidate)
        return labels[:6]

    def _extract_scene_label_from_stage_lines(self, lines: List[str], fallback_summary: str = "") -> str:
        for line in lines:
            scene_phrase = self._extract_scene_phrase_from_text(line)
            if scene_phrase:
                return scene_phrase
            cleaned = self._normalize_scene_location(line.split("|")[0].split("帧")[0].strip())
            if cleaned and not self._is_generic_scene_label(cleaned):
                return cleaned
        if fallback_summary:
            summary_phrase = self._extract_scene_phrase_from_text(fallback_summary)
            if summary_phrase:
                return summary_phrase
        return "阶段场景"

    def _parse_stage_batch_response(self, response_text: str) -> Dict[str, Any]:
        section_titles = [
            "完整时间线剧情总结",
            "全部出场人物清单",
            "场景切换记录",
            "关键事件节点",
            "画面细节补充",
        ]
        timeline_text = self._extract_numbered_section(response_text, section_titles[0], section_titles[1:])
        character_text = self._extract_numbered_section(response_text, section_titles[1], section_titles[2:])
        scene_text = self._extract_numbered_section(response_text, section_titles[2], section_titles[3:])
        event_text = self._extract_numbered_section(response_text, section_titles[3], section_titles[4:])
        detail_text = self._extract_numbered_section(response_text, section_titles[4], [])
        character_lines = self._split_stage_lines(character_text)
        scene_lines = self._sanitize_stage_lines(self._split_stage_lines(scene_text), kind="scene")
        event_lines = self._sanitize_stage_lines(self._split_stage_lines(event_text), kind="event")
        detail_lines = self._sanitize_stage_lines(self._split_stage_lines(detail_text), kind="detail")
        summary = self._sanitize_stage_text(timeline_text)
        return {
            "timeline_summary": summary,
            "character_lines": character_lines,
            "scene_lines": scene_lines,
            "event_lines": event_lines,
            "detail_lines": detail_lines,
            "overlay_text_lines": [],
            "key_characters": self._extract_character_labels_from_stage_lines(character_lines),
            "key_events": event_lines[:8],
            "scene_label": self._extract_scene_label_from_stage_lines(scene_lines, fallback_summary=summary),
        }

    def _build_stage_structured_result(
        self,
        frames: List[Frame],
        parsed: Dict[str, Any],
        raw_response: str,
        stage_audio_snippets: Optional[List[AudioSnippet]] = None,
    ) -> StructuredFrameAnalysis:
        mid_frame = frames[len(frames) // 2]
        characters = [{"name": "", "appearance": [], "role": label} for label in parsed.get("key_characters", [])]
        continuity_points = parsed.get("scene_lines", [])[:2]
        detail_objects = self._filter_story_detail_lines(
            parsed.get("detail_lines", []),
            overlay_text_lines=parsed.get("overlay_text_lines", []),
            limit=5,
        )
        overlay_text_lines = parsed.get("overlay_text_lines", [])[:5]
        sanitized_raw_response = self._sanitize_stage_raw_response(raw_response)
        stage_audio_snippets = stage_audio_snippets or []
        dialogue_summary = self._build_dialogue_summary_from_snippets(stage_audio_snippets, limit=3)
        if not dialogue_summary:
            dialogue_summary = self._join_readable_items(
                self._get_audio_lines_for_range(frames[0].timestamp, frames[-1].timestamp, limit=3),
                "；",
            )
        dialogue_summary = self._truncate_text(dialogue_summary, 220)
        return StructuredFrameAnalysis(
            frame_number=mid_frame.number,
            timestamp=mid_frame.timestamp,
            scene=parsed.get("scene_label", ""),
            characters=characters,
            actions=parsed.get("key_events", [])[:5],
            objects=detail_objects,
            overlay_text_lines=overlay_text_lines,
            dialogue_hint=dialogue_summary,
            continuity_points=continuity_points,
            raw_response=sanitized_raw_response,
            scene_changed=bool(self.batch_stage_analyses),
        )

    def _append_stage_chunk_summary(self, frames: List[Frame], parsed: Dict[str, Any]) -> ChunkSummary:
        chunk = ChunkSummary(
            chunk_id=len(self.chunk_summaries) + 1,
            start_frame=frames[0].number,
            end_frame=frames[-1].number,
            start_timestamp=frames[0].timestamp,
            end_timestamp=frames[-1].timestamp,
            summary=parsed.get("timeline_summary", ""),
            scene_label=parsed.get("scene_label", ""),
            key_characters=parsed.get("key_characters", [])[:6],
            key_events=parsed.get("key_events", [])[:8],
        )
        self.chunk_summaries.append(chunk)
        self.story_memory.last_chunk_summary = chunk.summary
        return chunk

    def analyze_frame_batch(self, frames: List[Frame], stage_id: int, total_stages: int) -> List[Dict[str, Any]]:
        if not frames:
            return []
        batch_started_at = time.perf_counter()
        stage_audio_snippets = self._get_audio_snippets_for_range(frames[0].timestamp, frames[-1].timestamp)
        prompt = self._build_stage_batch_prompt(frames, stage_id, total_stages)
        _debug_emit(
            "B",
            "analyzer.py:analyze_frame_batch",
            "Batch stage request prepared",
            {
                "stage_id": stage_id,
                "image_count": len(frames),
                "frame_range": [frames[0].number, frames[-1].number],
                "prompt_length": len(prompt),
            },
        )
        response = self.client.generate(
            prompt=prompt,
            image_paths=[str(frame.path) for frame in frames[: self.stage_batch_max_images]],
            model=self.model,
            temperature=self.temperature,
            num_predict=self.stage_batch_num_predict,
            num_ctx=self.stage_batch_num_ctx,
        )
        response_text = (response or {}).get("response", "").strip()
        parsed = self._parse_stage_batch_response(response_text)
        parsed = self._post_process_stage_batch_parsed(parsed, stage_audio_snippets, stage_id)
        skip_verifier = self._looks_like_qwen_stage_output_anomaly(parsed, response_text, frames)
        if skip_verifier:
            _debug_emit(
                "C",
                "analyzer.py:analyze_frame_batch",
                "Skip stage verifier due to qwen anomaly",
                {
                    "stage_id": stage_id,
                    "response_length": len(response_text),
                    "timeline_summary_length": len(parsed.get("timeline_summary", "")),
                    "scene_label": parsed.get("scene_label", ""),
                    "key_event_count": len(parsed.get("key_events", []) or []),
                    "finish_reason": (response or {}).get("finish_reason"),
                },
            )
        else:
            parsed = self._verify_stage_batch_parsed(
                frames,
                parsed,
                response_text,
                stage_id,
                total_stages,
                audio_snippets=stage_audio_snippets,
            )
        if self._looks_like_stage_fallback_needed(parsed, response_text, frames=frames, stage_id=stage_id, total_stages=total_stages):
            parsed = self._build_stage_batch_fallback(frames, parsed, stage_id, total_stages)
        elif not parsed.get("timeline_summary"):
            parsed["timeline_summary"] = self._truncate_text(self._sanitize_stage_text(response_text), 800)
        parsed = self._refine_stage_scene_label(
            parsed,
            raw_response=response_text,
            previous_stage=self.batch_stage_analyses[-1] if self.batch_stage_analyses else None,
            frames=frames,
            stage_id=stage_id,
            total_stages=total_stages,
        )
        sanitized_stage_raw_response = self._sanitize_stage_raw_response(response_text)
        structured_result = self._build_stage_structured_result(
            frames,
            parsed,
            sanitized_stage_raw_response,
            stage_audio_snippets=stage_audio_snippets,
        )
        self.update_recent_window(structured_result)
        self.update_story_memory(structured_result)
        self.update_character_registry(structured_result, stage_audio_snippets)
        dialogue_summary = structured_result.dialogue_hint
        chunk = self._append_stage_chunk_summary(frames, parsed)
        stage_analysis = BatchStageAnalysis(
            stage_id=stage_id,
            start_frame=frames[0].number,
            end_frame=frames[-1].number,
            start_timestamp=frames[0].timestamp,
            end_timestamp=frames[-1].timestamp,
            frame_numbers=[frame.number for frame in frames],
            prompt_path=self.stage_batch_prompt_path,
            raw_response=sanitized_stage_raw_response,
            timeline_summary=parsed.get("timeline_summary", ""),
            character_lines=parsed.get("character_lines", []),
            scene_lines=parsed.get("scene_lines", []),
            event_lines=parsed.get("event_lines", []),
            detail_lines=parsed.get("detail_lines", []),
            overlay_text_lines=parsed.get("overlay_text_lines", []),
            key_characters=chunk.key_characters,
            key_events=chunk.key_events,
            scene_label=chunk.scene_label,
            dialogue_summary=dialogue_summary,
        )
        self.batch_stage_analyses.append(stage_analysis)
        refined_after_chunk = self._refine_stage_scene_label(
            {
                "scene_label": chunk.scene_label,
                "scene_lines": stage_analysis.scene_lines,
                "timeline_summary": stage_analysis.timeline_summary,
                "key_events": chunk.key_events,
                "detail_lines": stage_analysis.detail_lines,
                "overlay_text_lines": stage_analysis.overlay_text_lines,
                "key_characters": chunk.key_characters,
            },
            raw_response=response_text,
            previous_stage=self.batch_stage_analyses[-2] if len(self.batch_stage_analyses) >= 2 else None,
            frames=frames,
            stage_id=stage_id,
            total_stages=total_stages,
        )
        refined_chunk_label = self._normalize_scene_location(refined_after_chunk.get("scene_label", ""))
        if refined_chunk_label and refined_chunk_label != chunk.scene_label:
            chunk.scene_label = refined_chunk_label
            stage_analysis.scene_label = refined_chunk_label
            if refined_chunk_label not in stage_analysis.scene_lines:
                stage_analysis.scene_lines = [refined_chunk_label] + [
                    line for line in stage_analysis.scene_lines if line != refined_chunk_label
                ]

        compact_response = self._truncate_text(
            f"阶段{stage_id}：{stage_analysis.scene_label or '阶段场景'}；"
            f"人物：{self._join_readable_items(stage_analysis.key_characters[:4], '、') or '人物'}；"
            f"事件：{self._join_readable_items(stage_analysis.key_events[:4], '、') or '无明确关键事件'}；"
            f"总结：{stage_analysis.timeline_summary}",
            600,
        )
        results = []
        for frame in frames:
            frame_result = {
                "frame_number": frame.number,
                "timestamp": frame.timestamp,
                "response": compact_response,
                "structured": {
                    "frame_number": frame.number,
                    "timestamp": frame.timestamp,
                    "scene": stage_analysis.scene_label,
                    "characters": [{"name": "", "appearance": [], "role": label} for label in stage_analysis.key_characters],
                    "actions": stage_analysis.key_events[:5],
                    "objects": stage_analysis.detail_lines[:5],
                    "overlay_text_lines": stage_analysis.overlay_text_lines[:5],
                    "dialogue_hint": dialogue_summary,
                    "continuity_points": stage_analysis.scene_lines[:2],
                    "raw_response": sanitized_stage_raw_response,
                    "scene_changed": frame.number == frames[0].number and stage_id > 1,
                },
                "audio_context": {
                    "matched_segments": [asdict(snippet) for snippet in stage_audio_snippets],
                    "summary": self._format_audio_snippets(stage_audio_snippets),
                },
                "stage_batch": {
                    "stage_id": stage_id,
                    "frame_range": [frames[0].number, frames[-1].number],
                    "time_range": [frames[0].timestamp, frames[-1].timestamp],
                },
            }
            results.append(frame_result)
            self.previous_analyses.append(frame_result)

        _debug_emit(
            "B",
            "analyzer.py:analyze_frame_batch",
            "Batch stage completed",
            {
                "stage_id": stage_id,
                "image_count": len(frames),
                "response_length": len(response_text),
                "scene_label": stage_analysis.scene_label,
                "key_characters": len(stage_analysis.key_characters),
                "key_events": len(stage_analysis.key_events),
                "elapsed_seconds": round(time.perf_counter() - batch_started_at, 3),
            },
        )
        return results

    def analyze_frame(self, frame: Frame) -> Dict[str, Any]:
        analyze_started_at = time.perf_counter()
        audio_snippets = self._get_audio_snippets_for_timestamp(frame.timestamp)
        prompt = self.build_frame_prompt(frame, audio_snippets)
        _debug_emit(
            "B",
            "analyzer.py:analyze_frame",
            "Frame analysis request prepared",
            {
                "frame_number": frame.number,
                "timestamp": round(frame.timestamp, 3),
                "prompt_length": len(prompt),
                "audio_snippet_count": len(audio_snippets),
                "recent_window_size": len(self.recent_frame_window),
                "chunk_buffer_size": len(self.chunk_buffer),
            },
        )

        try:
            llm_started_at = time.perf_counter()
            response = self.client.generate(
                prompt=prompt,
                image_path=str(frame.path),
                model=self.model,
                temperature=self.temperature,
                num_predict=300,
            )
            llm_elapsed = time.perf_counter() - llm_started_at
            response_text = (response or {}).get("response", "").strip()
            structured_result = self.parse_frame_response(frame, response_text)
            self.update_recent_window(structured_result)
            self.update_story_memory(structured_result)
            self.update_character_registry(structured_result, audio_snippets)
            self.chunk_buffer.append(structured_result)
            self.flush_chunk_if_needed(structured_result=structured_result)

            analysis_result = {
                "frame_number": frame.number,
                "timestamp": frame.timestamp,
                "response": response_text,
                "structured": asdict(structured_result),
                "audio_context": {
                    "matched_segments": [asdict(snippet) for snippet in audio_snippets],
                    "summary": self._format_audio_snippets(audio_snippets),
                },
            }
            self.previous_analyses.append(analysis_result)
            _debug_emit(
                "A",
                "analyzer.py:analyze_frame",
                "Frame analysis completed",
                {
                    "frame_number": frame.number,
                    "llm_elapsed_seconds": round(llm_elapsed, 3),
                    "total_elapsed_seconds": round(time.perf_counter() - analyze_started_at, 3),
                    "response_length": len(response_text),
                    "scene_changed": structured_result.scene_changed,
                },
            )
            logger.debug("Successfully analyzed frame %s", frame.number)
            return analysis_result
        except Exception as exc:
            _debug_emit(
                "A",
                "analyzer.py:analyze_frame",
                "Frame analysis failed",
                {
                    "frame_number": frame.number,
                    "elapsed_seconds": round(time.perf_counter() - analyze_started_at, 3),
                    "error": str(exc),
                },
            )
            logger.error("Error analyzing frame %s: %s", frame.number, exc)
            structured_result = StructuredFrameAnalysis(
                frame_number=frame.number,
                timestamp=frame.timestamp,
                raw_response=f"Error analyzing frame {frame.number}: {exc}",
            )
            self.chunk_buffer.append(structured_result)
            error_result = {
                "frame_number": frame.number,
                "timestamp": frame.timestamp,
                "response": structured_result.raw_response,
                "structured": asdict(structured_result),
                "audio_context": {
                    "matched_segments": [],
                    "summary": "",
                },
            }
            self.previous_analyses.append(error_result)
            return error_result

    def _select_reconstruction_notes(self, frame_analyses: List[Dict[str, Any]], frames: List[Frame]) -> str:
        if self.batch_stage_analyses:
            lines = []
            for stage in self.batch_stage_analyses[: self.reconstruction_chunk_cap]:
                lines.append(
                    f"Stage {stage.stage_id} ({stage.start_timestamp:.2f}s-{stage.end_timestamp:.2f}s):\n"
                    f"scene={stage.scene_label or 'unknown'}\n"
                    f"characters={stage.key_characters[:4]}\n"
                    f"events={stage.key_events[:5]}\n"
                    f"timeline={stage.timeline_summary}\n"
                    f"dialogue_hint={stage.dialogue_summary or 'n/a'}"
                )
            if lines:
                return "\n\n".join(lines)

        selected_indices = list(range(min(len(frame_analyses), self.reconstruction_frame_cap)))
        if len(frame_analyses) > self.reconstruction_frame_cap and self.chunk_summaries:
            candidate_indices = []
            for chunk in self.chunk_summaries:
                mid_frame = (chunk.start_frame + chunk.end_frame) // 2
                candidate_indices.extend([chunk.start_frame, mid_frame, chunk.end_frame])
            selected_indices = sorted({idx for idx in candidate_indices if idx < len(frame_analyses)})
            selected_indices = self._sample_ordered_items(selected_indices, self.reconstruction_frame_cap)

        frame_notes = []
        for index in selected_indices:
            frame = frames[index]
            analysis = frame_analyses[index]
            structured = analysis.get("structured") or {}
            character_labels = []
            for character in structured.get("characters", [])[:4]:
                label = self._get_character_output_label(character)
                if label and label not in character_labels:
                    character_labels.append(label)
            frame_note = (
                f"Frame {index} ({frame.timestamp:.2f}s):\n"
                f"scene={structured.get('scene', '')}\n"
                f"characters={character_labels}\n"
                f"actions={structured.get('actions', [])[:3]}\n"
                f"dialogue_hint={structured.get('dialogue_hint', '')}\n"
                f"continuity={structured.get('continuity_points', [])[:2]}"
            )
            frame_notes.append(frame_note)
        return "\n\n".join(frame_notes)

    def _build_audio_summary_text(self, max_blocks: Optional[int] = None) -> str:
        if not self.audio_summary:
            return ""
        blocks = self.audio_summary.get("time_blocks") or []
        selected_blocks = self._sample_ordered_items(
            blocks,
            max_blocks or self.reconstruction_transcript_block_cap,
        )
        block_text = "\n".join(
            f"{block.get('start', 0):.2f}-{block.get('end', 0):.2f}s | "
            f"{self._truncate_text(self._clean_text_field(block.get('summary', '')), 160)}"
            for block in selected_blocks
        )
        return (
            f"language={self.audio_summary.get('language')}\n"
            f"key_dialogues={self.audio_summary.get('key_dialogues', [])}\n"
            f"mood_keywords={self.audio_summary.get('mood_keywords', [])}\n"
            f"time_blocks=\n{block_text}"
        )

    def _is_effective_text(self, text: str, minimum_length: int = 24) -> bool:
        normalized = self._strip_code_fences(text or "")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if len(normalized) < minimum_length:
            return False
        if normalized.strip("#*`:- ").lower() in {"", "video summary", "narrative"}:
            return False
        if self._looks_like_payload_fragment(normalized):
            return False
        return True

    def _build_verified_character_text(self) -> str:
        entries = []
        for key, profile in self.story_memory.characters.items():
            display_name = self._format_character_display_name(key, profile)
            entries.append(
                f"{display_name} | roles={', '.join(profile.roles[:2]) or 'n/a'} | "
                f"appearance={', '.join(profile.appearance[:3]) or 'n/a'}"
            )
        return "\n".join(entries) or "No verified character names. Use generic role labels."

    def serialize_story_memory(self) -> Dict[str, Any]:
        return {
            "scene_summary": self.story_memory.scene_summary,
            "characters": {
                key: asdict(value)
                for key, value in self.story_memory.characters.items()
            },
            "key_events": self.story_memory.key_events,
            "active_props": self.story_memory.active_props,
            "overlay_text_lines": self.story_memory.overlay_text_lines,
            "last_chunk_summary": self.story_memory.last_chunk_summary,
        }

    def _should_merge_scene_cards(self, previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
        previous_location = self._normalize_scene_location(previous.get("location", ""))
        current_location = self._normalize_scene_location(current.get("location", ""))
        if not previous_location or previous_location != current_location:
            return False
        previous_duration = float(previous.get("end_timestamp", 0.0)) - float(previous.get("start_timestamp", 0.0))
        current_duration = float(current.get("end_timestamp", 0.0)) - float(current.get("start_timestamp", 0.0))
        previous_frames = previous.get("end_frame", 0) - previous.get("start_frame", 0) + 1
        current_frames = current.get("end_frame", 0) - current.get("start_frame", 0) + 1
        previous_actions = set(item for item in (previous.get("key_actions", []) or []) if isinstance(item, str) and item.strip())
        current_actions = set(item for item in (current.get("key_actions", []) or []) if isinstance(item, str) and item.strip())
        new_actions = current_actions - previous_actions
        if current_frames <= 1 and previous_frames > 1 and new_actions:
            return False
        if previous_frames <= 1 and current_frames > 1 and previous_actions - current_actions:
            return False
        if previous_duration > 0 and current_duration > 0 and min(previous_duration, current_duration) <= 1.5 and not new_actions:
            return True
        if previous_duration > 0 and current_duration > 0 and min(previous_duration, current_duration) <= 1.5:
            return False
        previous_actions = [self._normalize_summary_phrase(item) for item in previous.get("key_actions", []) if item]
        current_actions = [self._normalize_summary_phrase(item) for item in current.get("key_actions", []) if item]
        is_short_continuation = self._is_explicit_short_scene_continuation(current)
        if previous.get("title") == current.get("title") and is_short_continuation:
            return True
        if not current_actions:
            return False
        shared_actions = set(previous_actions) & set(current_actions)
        if shared_actions and is_short_continuation:
            return True
        return False

    def _merge_scene_card_pair(self, previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(previous)
        merged["end_timestamp"] = current.get("end_timestamp", previous.get("end_timestamp"))
        for field_name in ["characters", "key_actions", "key_props", "overlay_text_lines"]:
            merged[field_name] = list(
                dict.fromkeys((previous.get(field_name) or []) + (current.get(field_name) or []))
            )
        for field_name in ["dialogue_summary", "summary"]:
            merged[field_name] = self._join_readable_items(
                [previous.get(field_name, ""), current.get(field_name, "")],
                "；",
            )
        merged["visual_style_hint"] = self._join_readable_items(
            [previous.get("visual_style_hint", ""), current.get("visual_style_hint", "")],
            "；",
        )
        merged["transition_note"] = current.get("transition_note") or previous.get("transition_note", "")
        return merged

    def _merge_adjacent_scene_cards(self, cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not cards:
            return []
        merged_cards: List[Dict[str, Any]] = [cards[0]]
        for card in cards[1:]:
            previous = merged_cards[-1]
            if self._should_merge_scene_cards(previous, card):
                merged_cards[-1] = self._merge_scene_card_pair(previous, card)
            else:
                merged_cards.append(card)
        for index, card in enumerate(merged_cards, 1):
            card["scene_id"] = f"scene-{index:03d}"
        return merged_cards

    def build_scene_cards(self, transcript: Optional[AudioTranscript] = None) -> List[Dict[str, Any]]:
        cards = []
        previous_title = ""
        for index, chunk in enumerate(self.chunk_summaries, 1):
            stage_analysis = self.batch_stage_analyses[index - 1] if index - 1 < len(self.batch_stage_analyses) else None
            scene_title = self._build_scene_title(chunk, stage_analysis=stage_analysis)
            scene_location = self._normalize_scene_location(chunk.scene_label) or scene_title
            filtered_actions = self._filter_scene_card_actions(
                chunk.key_events,
                scene_location,
                scene_lines=stage_analysis.scene_lines if stage_analysis else None,
                limit=8,
            )
            if not filtered_actions:
                filtered_actions = [item for item in chunk.key_events[:8] if item]
            scene_title_action = self._pick_primary_scene_action(filtered_actions)
            multi_scene_stage = len(
                self._extract_scene_candidates_from_lines(stage_analysis.scene_lines if stage_analysis else [])
            ) >= 2
            title_action_fragment = scene_title.split("：", 1)[1] if "：" in scene_title else scene_title
            title_needs_rebuild = (
                scene_title == scene_location
                or self._action_conflicts_with_scene_location(title_action_fragment, scene_location)
                or self._is_transition_action_text(title_action_fragment)
            )
            if scene_title_action and self._is_transition_action_text(scene_title_action):
                scene_title_action = ""
            if (
                scene_location
                and scene_title_action
                and title_needs_rebuild
                and not multi_scene_stage
            ):
                scene_title = f"{scene_location}：{scene_title_action}"
            elif scene_location and title_needs_rebuild and not scene_title_action:
                scene_title = scene_location
            dialogue_summary = self._join_readable_items(
                self._get_audio_lines_for_range(chunk.start_timestamp, chunk.end_timestamp, transcript=transcript, limit=2),
                "；",
            )
            if stage_analysis and stage_analysis.dialogue_summary:
                dialogue_summary = stage_analysis.dialogue_summary
            overlay_text_lines = stage_analysis.overlay_text_lines[:5] if stage_analysis else []
            key_props = self._filter_story_detail_lines(
                stage_analysis.detail_lines if stage_analysis and stage_analysis.detail_lines else self.story_memory.active_props[:5],
                overlay_text_lines=overlay_text_lines,
                limit=5,
            )
            cards.append(
                {
                    "scene_id": f"scene-{chunk.chunk_id:03d}",
                    "start_frame": chunk.start_frame,
                    "end_frame": chunk.end_frame,
                    "start_timestamp": chunk.start_timestamp,
                    "end_timestamp": chunk.end_timestamp,
                    "title": scene_title,
                    "scene_type": chunk.scene_label,
                    "location": scene_location,
                    "characters": chunk.key_characters,
                    "key_actions": filtered_actions,
                    "key_props": key_props,
                    "overlay_text_lines": overlay_text_lines,
                    "dialogue_summary": dialogue_summary,
                    "visual_style_hint": self._build_visual_style_hint(scene_title, chunk.key_characters, chunk.key_events),
                    "summary": stage_analysis.timeline_summary if stage_analysis else chunk.summary,
                    "transition_note": self._build_transition_note(previous_title, scene_title, index),
                }
            )
            previous_title = scene_title
        return self._merge_adjacent_scene_cards(cards)

    def build_story_beats(self, scene_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        beats = []
        for index, card in enumerate(scene_cards, 1):
            card_title = card.get("title") or card.get("location") or f"场景{index}"
            card_summary = card.get("summary") or ", ".join(card.get("key_actions", [])[:3]) or card_title
            beats.append(
                {
                    "beat_id": f"beat-{index:03d}",
                    "order": index,
                    "scene_id": card.get("scene_id"),
                    "title": card_title,
                    "start_timestamp": card.get("start_timestamp", 0.0),
                    "end_timestamp": card.get("end_timestamp", 0.0),
                    "location": card.get("location", ""),
                    "summary": card_summary,
                    "related_scene_ids": [card.get("scene_id")],
                    "conflict": "",
                    "transition_type": "scene_change" if index > 1 and card.get("transition_note") else "opening",
                }
            )
        return beats

    def build_character_timeline(self) -> List[Dict[str, Any]]:
        merged_timeline: Dict[str, Dict[str, Any]] = {}
        for profile in self.story_memory.characters.values():
            display_name = self._format_character_display_name(profile.character_id, profile)
            entry = merged_timeline.setdefault(
                display_name,
                {
                    "character_id": profile.character_id,
                    "display_name": display_name,
                    "appearances": [],
                    "relations": [],
                    "key_actions": [],
                    "dialogue_hints": [],
                },
            )
            entry["appearances"].extend(profile.appearances)
            entry["relations"].extend(profile.relations)
            entry["key_actions"].extend(profile.key_actions)
            entry["dialogue_hints"].extend(profile.dialogue_hints)

        timeline = []
        for entry in merged_timeline.values():
            deduped_appearances = []
            seen_appearances = set()
            for appearance in sorted(
                entry["appearances"],
                key=lambda item: (item.get("timestamp", 0), item.get("frame_number", 0)),
            ):
                key = (
                    appearance.get("frame_number"),
                    appearance.get("timestamp"),
                    appearance.get("scene"),
                )
                if key in seen_appearances:
                    continue
                seen_appearances.add(key)
                deduped_appearances.append(appearance)
            entry["appearances"] = deduped_appearances
            entry["relations"] = list(dict.fromkeys(entry["relations"]))
            entry["key_actions"] = list(dict.fromkeys(entry["key_actions"]))
            entry["dialogue_hints"] = list(dict.fromkeys(entry["dialogue_hints"]))
            timeline.append(entry)
        return timeline

    def build_script_guidance(
        self,
        scene_cards: List[Dict[str, Any]],
        story_beats: List[Dict[str, Any]],
        character_timeline: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        script_started_at = time.perf_counter()
        guidance = {
            "theme": self._build_storyline_text(scene_cards),
            "tone": ", ".join(self.audio_summary.get("mood_keywords", [])[:3]) if self.audio_summary else "",
            "scene_order": [card.get("scene_id") for card in scene_cards],
            "dialogue_candidates": self.audio_summary.get("key_dialogues", [])[:8] if self.audio_summary else [],
            "visual_prompt_hints": [card.get("location", "") for card in scene_cards[:5] if card.get("location")],
            "screenplay_notes": [beat.get("summary", "") for beat in story_beats[:5] if beat.get("summary")],
        }

        try:
            prompt = self.script_prompt
            prompt = prompt.replace(
                "{SCENE_CARDS}",
                self._format_scene_cards_for_prompt(scene_cards, limit=self.script_scene_cap),
            )
            prompt = prompt.replace(
                "{STORY_BEATS}",
                self._format_story_beats_for_prompt(story_beats, limit=self.script_scene_cap),
            )
            prompt = prompt.replace(
                "{CHARACTER_TIMELINE}",
                self._format_character_timeline_for_prompt(character_timeline, limit=self.script_character_cap),
            )
            prompt = prompt.replace("{OUTPUT_LANGUAGE}", self._get_output_language_name())
            guidance_client, guidance_model = self._get_refinement_client_and_model()
            llm_started_at = time.perf_counter()
            response = guidance_client.generate(
                prompt=prompt,
                model=guidance_model,
                temperature=0.0 if guidance_client is not self.client else self.temperature,
                num_predict=320,
            )
            llm_elapsed = time.perf_counter() - llm_started_at
            if response and response.get("response"):
                note = self._clean_text_field(response["response"])
                if self._is_effective_text(note):
                    guidance["screenplay_notes"].append(note)
        except Exception as exc:
            llm_elapsed = None
            logger.warning("Script guidance fallback triggered: %s", exc)

        if not guidance["screenplay_notes"]:
            if self.output_language.lower().startswith("zh"):
                guidance["screenplay_notes"] = [
                    "按时间顺序整理每个场景，优先保留关键转场、人物关系和动作承接。",
                    "对白以真实转写为准，人物身份不明时使用通用称谓，避免误写专有姓名。",
                ]
            else:
                guidance["screenplay_notes"] = [
                    "Keep scene order chronological and preserve major transitions and callbacks.",
                    "Use transcript-backed dialogue first, and prefer generic labels when identity is uncertain.",
                ]

        _debug_emit(
            "C",
            "analyzer.py:build_script_guidance",
            "Script guidance completed",
            {
                "scene_cards": len(scene_cards),
                "story_beats": len(story_beats),
                "character_timeline": len(character_timeline),
                "screenplay_notes": len(guidance.get("screenplay_notes", [])),
                "llm_elapsed_seconds": round(llm_elapsed, 3) if llm_elapsed is not None else None,
                "total_elapsed_seconds": round(time.perf_counter() - script_started_at, 3),
            },
        )
        return guidance

    def reconstruct_video(
        self,
        frame_analyses: List[Dict[str, Any]],
        frames: List[Frame],
        transcript: Optional[AudioTranscript] = None,
    ) -> Dict[str, Any]:
        reconstruct_started_at = time.perf_counter()
        self.flush_chunk_if_needed(force=True)

        analysis_text = self._select_reconstruction_notes(frame_analyses, frames)
        first_frame_text = frame_analyses[0].get("response", "") if frame_analyses else ""
        transcript_text = transcript.text if transcript and transcript.text.strip() else ""
        transcript_digest = self._build_transcript_digest(transcript)
        chunk_summary_text = self._format_chunk_summaries_for_prompt(limit=self.reconstruction_chunk_cap)
        actual_duration_seconds = max((frame.timestamp for frame in frames), default=0.0)

        prompt = self.video_prompt.replace("{prompt}", self._format_user_prompt())
        prompt = prompt.replace("{FRAME_NOTES}", analysis_text)
        prompt = prompt.replace("{FIRST_FRAME}", first_frame_text)
        prompt = prompt.replace("{TRANSCRIPT}", transcript_digest or transcript_text[:1200])
        prompt = prompt.replace("{CHUNK_SUMMARIES}", chunk_summary_text)
        prompt = prompt.replace("{STORY_MEMORY}", self._format_story_memory())
        prompt = prompt.replace("{AUDIO_SUMMARY}", self._build_audio_summary_text(max_blocks=self.reconstruction_transcript_block_cap))
        prompt = prompt.replace("{VIDEO_DURATION}", f"{actual_duration_seconds:.2f}s")
        prompt = prompt.replace("{VERIFIED_CHARACTERS}", self._build_verified_character_text())
        prompt = prompt.replace("{OUTPUT_LANGUAGE}", self._get_output_language_name())
        _debug_emit(
            "C",
            "analyzer.py:reconstruct_video",
            "Video reconstruction request prepared",
            {
                "frame_analysis_count": len(frame_analyses),
                "chunk_summary_count": len(self.chunk_summaries),
                "prompt_length": len(prompt),
                "transcript_length": len(transcript_text),
                "transcript_digest_length": len(transcript_digest),
                "analysis_text_length": len(analysis_text),
            },
        )

        try:
            refinement_client, refinement_model = self._get_refinement_client_and_model()
            llm_started_at = time.perf_counter()
            response = refinement_client.generate(
                prompt=prompt,
                model=refinement_model,
                temperature=0.0 if refinement_client is not self.client else self.temperature,
                num_predict=1000,
            )
            llm_elapsed = time.perf_counter() - llm_started_at
            response_text = self._clean_text_field((response or {}).get("response", ""), allow_json_like=False)
            logger.info("Successfully reconstructed video description")
        except Exception as exc:
            llm_elapsed = None
            logger.error("Error reconstructing video: %s", exc)
            response_text = f"Error reconstructing video: {exc}"

        scene_cards = self._sanitize_scene_cards_for_final_output(
            self.build_scene_cards(transcript=transcript),
            transcript=transcript,
        )
        story_beats = self.build_story_beats(scene_cards)
        character_timeline = self.build_character_timeline()
        script_guidance = self.build_script_guidance(scene_cards, story_beats, character_timeline)
        video_script = self.build_video_script(
            scene_cards=scene_cards,
            story_beats=story_beats,
            character_timeline=character_timeline,
            actual_duration_seconds=actual_duration_seconds,
            transcript=transcript,
        )
        if self._is_effective_text(response_text):
            response_text = self._sanitize_final_video_description(
                response_text,
                scene_cards=scene_cards,
                transcript=transcript,
            )
        if not self._is_effective_text(response_text):
            response_text = self._build_fallback_video_summary(
                actual_duration_seconds=actual_duration_seconds,
                scene_cards=scene_cards,
                _story_beats=story_beats,
                script_guidance=script_guidance,
                transcript_text=transcript_text,
            )
        _debug_emit(
            "C",
            "analyzer.py:reconstruct_video",
            "Video reconstruction completed",
            {
                "llm_elapsed_seconds": round(llm_elapsed, 3) if llm_elapsed is not None else None,
                "total_elapsed_seconds": round(time.perf_counter() - reconstruct_started_at, 3),
                "scene_cards": len(scene_cards),
                "story_beats": len(story_beats),
                "character_timeline": len(character_timeline),
                "script_guidance_notes": len((script_guidance or {}).get("screenplay_notes", [])),
            },
        )

        return {
            "response": response_text,
            "scene_cards": scene_cards,
            "story_beats": story_beats,
            "character_timeline": character_timeline,
            "script_guidance": script_guidance,
            "video_script": video_script,
            "stage_batch_analyses": [asdict(item) for item in self.batch_stage_analyses],
            "chunk_summaries": [asdict(chunk) for chunk in self.chunk_summaries],
            "story_memory": self.serialize_story_memory(),
        }

    def build_video_script(
        self,
        scene_cards: List[Dict[str, Any]],
        story_beats: List[Dict[str, Any]],
        character_timeline: List[Dict[str, Any]],
        actual_duration_seconds: float,
        transcript: Optional[AudioTranscript] = None,
    ) -> Dict[str, Any]:
        main_characters = []
        for entry in character_timeline[:8]:
            appearances = entry.get("appearances", [])
            main_characters.append(
                {
                    "name": self._get_script_character_label(entry),
                    "appearance_count": len(appearances),
                    "core_actions": entry.get("key_actions", [])[:5],
                    "dialogue_hints": entry.get("dialogue_hints", [])[:3],
                }
            )

        scenes = []
        full_script_lines = [
            "视频生成脚本",
            f"总时长：{actual_duration_seconds:.2f}s",
            f"剧情主线：{self._build_storyline_text(scene_cards)}",
            f"主要人物：{'、'.join([item.get('name', '') for item in main_characters if item.get('name')]) or '人物'}",
            "",
        ]
        for index, card in enumerate(scene_cards[: max(len(scene_cards), 1)], 1):
            beat = story_beats[index - 1] if index - 1 < len(story_beats) else {}
            dialogue_lines = self._get_audio_lines_for_range(
                card.get("start_timestamp", 0.0),
                card.get("end_timestamp", 0.0),
                transcript=transcript,
                limit=3,
            )
            plot_text = self._build_scene_plot_for_script(card, beat)
            scene_entry = {
                "scene_id": card.get("scene_id"),
                "order": index,
                "time_range": f"{card.get('start_timestamp', 0.0):.2f}s-{card.get('end_timestamp', 0.0):.2f}s",
                "title": card.get("title") or card.get("location"),
                "location": self._get_script_location_text(card),
                "characters": card.get("characters", []),
                "plot": plot_text,
                "key_actions": card.get("key_actions", [])[:6],
                "key_props": card.get("key_props", [])[:6],
                "scene_change_note": card.get("transition_note", ""),
                "visuals": [
                    item
                    for item in [
                        card.get("visual_style_hint", ""),
                        self._join_readable_items(card.get("key_props", [])[:3], "、"),
                    ]
                    if item
                ],
                "overlay_text_lines": card.get("overlay_text_lines", [])[:5],
                "dialogue": dialogue_lines,
                "transition": card.get("transition_note", ""),
            }
            scenes.append(scene_entry)
            full_script_lines.extend(
                [
                    f"场景{index}：{scene_entry['title']}",
                    f"时间：{scene_entry['time_range']}",
                    f"地点：{scene_entry['location'] or scene_entry['title']}",
                    f"人物：{'、'.join(scene_entry['characters']) or '人物'}",
                    f"剧情：{scene_entry['plot'] or '根据当前阶段画面延续推进。'}",
                    f"画面提示：{'；'.join(scene_entry['visuals']) or '保持与上一场景的视觉连续性。'}",
                    f"画面文字参考：{'；'.join(scene_entry['overlay_text_lines']) or '无明确画面文字。'}",
                    f"对白/旁白：{'；'.join(scene_entry['dialogue']) or '无明确对白，保留环境或旁白空位。'}",
                    f"转场：{scene_entry['transition'] or '按时间顺序自然衔接下一场景。'}",
                    "",
                ]
            )

        return {
            "title": "视频生成脚本",
            "duration_seconds": round(actual_duration_seconds, 2),
            "storyline": self._build_storyline_text(scene_cards),
            "main_characters": main_characters,
            "scene_scripts": scenes,
            "full_script": "\n".join(full_script_lines).strip(),
        }

    def _build_fallback_video_summary(
        self,
        actual_duration_seconds: float,
        scene_cards: List[Dict[str, Any]],
        _story_beats: List[Dict[str, Any]],
        script_guidance: Dict[str, Any],
        transcript_text: str = "",
    ) -> str:
        duration_text = f"{actual_duration_seconds:.2f}s"
        character_timeline = self.build_character_timeline()
        key_characters = [
            entry.get("display_name", "")
            for entry in character_timeline
            if entry.get("display_name")
        ]
        events = []
        for card in scene_cards:
            events.extend(card.get("key_actions", [])[:2])
        events_text = self._join_readable_items(events[:8])
        dialogue = (self.audio_summary or {}).get("key_dialogues", [])[:3] if self.audio_summary else []
        if not dialogue and transcript_text:
            dialogue = [segment.strip() for segment in re.split(r"[。！？!?]", transcript_text) if segment.strip()][:3]
        dialogue_text = self._join_readable_items(dialogue[:3])
        scene_lines = []
        for index, card in enumerate(scene_cards[:6], 1):
            actions = self._join_readable_items(card.get("key_actions", [])[:3], "；") or "阶段动作未明确"
            scene_label = self._normalize_summary_phrase(card.get("location") or card.get("scene_id")) or card.get("scene_id")
            scene_lines.append(f"{index}. {scene_label}: {actions}")
        storyline = self._build_storyline_text(scene_cards)

        if self.output_language.lower().startswith("zh"):
            sections = [
                "视频总结",
                f"时长：{duration_text}",
                f"主要人物：{'、'.join(key_characters[:6]) or '人物身份未完全确认'}",
                f"故事主线：{storyline}",
                "阶段推进：",
                "\n".join(scene_lines) or "暂无可用阶段摘要。",
                f"关键事件：{events_text or '暂无明确关键事件。'}",
                f"对白线索：{dialogue_text or '暂无明确对白。'}",
            ]
            if script_guidance.get("theme") and self._is_informative_scene_label(script_guidance.get("theme", "")):
                sections.append(f"创作提示：{script_guidance.get('theme')}")
            return "\n\n".join(sections)

        sections = [
            "Video Summary",
            f"Duration: {duration_text}",
            f"Main characters: {', '.join(key_characters[:6]) or 'uncertain'}",
            f"Story line: {storyline}",
            "Scene progression:",
            "\n".join(scene_lines) or "No scene summary available.",
            f"Key events: {events_text or 'n/a'}",
            f"Dialogue hints: {dialogue_text or 'n/a'}",
        ]
        return "\n\n".join(sections)
