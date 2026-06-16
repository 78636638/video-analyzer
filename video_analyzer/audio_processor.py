import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import subprocess
import json
import time
import urllib.request
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# #region debug-point A:audio-init-metrics
def _debug_emit(hypothesis_id: str, location: str, msg: str, data: Optional[Dict[str, Any]] = None, run_id: str = "pre") -> None:
    server_url = "http://127.0.0.1:7777/event"
    session_id = "long-video-performance"
    try:
        env_text = (Path(".dbg") / "long-video-performance.env").read_text(encoding="utf-8")
        for line in env_text.splitlines():
            if line.startswith("DEBUG_SERVER_URL="):
                server_url = line.split("=", 1)[1].strip()
            elif line.startswith("DEBUG_SESSION_ID="):
                session_id = line.split("=", 1)[1].strip()
    except Exception:
        pass

    try:
        urllib.request.urlopen(
            urllib.request.Request(
                server_url,
                data=json.dumps(
                    {
                        "sessionId": session_id,
                        "runId": run_id,
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": f"[DEBUG] {msg}",
                        "data": data or {},
                        "ts": int(time.time() * 1000),
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=1,
        ).read()
    except Exception:
        pass
# #endregion

@dataclass
class AudioTranscript:
    text: str
    segments: List[Dict[str, Any]]
    language: str


@dataclass
class AudioSnippet:
    start: float
    end: float
    text: str
    speaker_hint: Optional[str] = None
    mood_hint: Optional[str] = None

class AudioProcessor:
    def __init__(self, 
                 language: str | None = None,
                 model_size_or_path: str = "medium",
                 device: str = "cpu"):
        """Initialize audio processor with specified Whisper model size or model path. By default, the medium model is used."""
        init_started_at = time.perf_counter()
        _debug_emit(
            "A",
            "audio_processor.py:__init__",
            "AudioProcessor initialization started",
            {
                "language": language,
                "model_size_or_path": model_size_or_path,
                "device": device,
            },
        )
        try:
            from faster_whisper import WhisperModel
            
            # Log cache directory
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            logger.debug(f"Using HuggingFace cache directory: {cache_dir}")
            
            # Force CPU usage for now faster whisper having issues with cudas
            self.device = device
            compute_type = "float32"
            logger.debug(f"Using device: {self.device}")

            self.language = language if language else None

            self.model = WhisperModel(
                model_size_or_path,
                device=device,
                compute_type=compute_type
            )
            _debug_emit(
                "A",
                "audio_processor.py:__init__",
                "WhisperModel initialized",
                {
                    "elapsed_seconds": round(time.perf_counter() - init_started_at, 3),
                    "device": device,
                    "compute_type": compute_type,
                },
            )
            logger.info(f"Initiation Input: Model size or path: {model_size_or_path}, Device: {device}, Compute type: {compute_type}, Language: {self.language if self.language else 'auto detected'}")
            logger.debug(f"Successfully loaded Whisper model: {model_size_or_path}")
            
            # Check for ffmpeg installation
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                self.has_ffmpeg = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.has_ffmpeg = False
                logger.warning("FFmpeg not found. Please install ffmpeg for better audio extraction.")
            _debug_emit(
                "A",
                "audio_processor.py:__init__",
                "AudioProcessor initialization completed",
                {
                    "elapsed_seconds": round(time.perf_counter() - init_started_at, 3),
                    "has_ffmpeg": self.has_ffmpeg,
                },
            )
                
        except Exception as e:
            _debug_emit(
                "A",
                "audio_processor.py:__init__",
                "AudioProcessor initialization failed",
                {
                    "elapsed_seconds": round(time.perf_counter() - init_started_at, 3),
                    "error": str(e),
                },
            )
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def extract_audio(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        """Extract audio from video file and convert to format suitable for Whisper.
        Returns None if video has no audio streams."""
        audio_path = output_dir / "audio.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract audio using ffmpeg
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
                "-ar", "16000",  # 16kHz sampling rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                str(audio_path)
            ], check=True, capture_output=True)
            
            logger.debug("Successfully extracted audio using ffmpeg")
            return audio_path
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode()
            logger.error(f"FFmpeg error: {error_output}")
            
            # Check if error indicates no audio streams
            if "Output file does not contain any stream" in error_output:
                logger.debug("No audio streams found in video - skipping audio extraction")
                return None
                
            # If error is not about missing audio, try pydub as fallback
            logger.info("Falling back to pydub for audio extraction...")
            try:
                from pydub import AudioSegment
                video = AudioSegment.from_file(str(video_path))
                audio = video.set_channels(1).set_frame_rate(16000)
                audio.export(str(audio_path), format="wav")
                logger.debug("Successfully extracted audio using pydub")
                return audio_path
            except Exception as e2:
                logger.error(f"Error extracting audio using pydub: {e2}")
                # If both methods fail, raise error
                raise RuntimeError(
                    "Failed to extract audio. Please install ffmpeg using:\n"
                    "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
                    "MacOS: brew install ffmpeg\n"
                    "Windows: choco install ffmpeg"
                )

    def transcribe(self, audio_path: Path) -> Optional[AudioTranscript]:
        """Transcribe audio file using Whisper with quality checks."""
        accepted_languages = {
                "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue"
        }
        if self.language and self.language not in accepted_languages:
            logger.warning(f"Invalid language code: {self.language}, will detect language automatically")
        try:
            # Initial transcription with VAD filtering
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                language = self.language if self.language in accepted_languages else None
            )
            
            segments_list = list(segments)
            if not segments_list:
                logger.warning("No speech detected in audio")
                return None
            
            # Convert segments to the expected format
            segment_data = [
                {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in (segment.words or [])
                    ]
                }
                for segment in segments_list
            ]
            
            return AudioTranscript(
                text=" ".join(segment.text for segment in segments_list),
                segments=segment_data,
                language=info.language
            )
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            logger.exception(e)
            return None

    def build_audio_snippets(self, transcript: Optional[AudioTranscript], alignment_window: float = 2.0) -> List[AudioSnippet]:
        """Convert transcript segments into lightweight snippets that can be aligned to frames."""
        if transcript is None:
            return []

        snippets: List[AudioSnippet] = []
        for segment in transcript.segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            snippets.append(
                AudioSnippet(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    text=text,
                    speaker_hint=self._guess_speaker_hint(text),
                    mood_hint=self._guess_mood_hint(text),
                )
            )

        logger.debug(
            "Built %s audio snippets using alignment_window=%s",
            len(snippets),
            alignment_window,
        )
        return snippets

    def summarize_audio_segments(
        self,
        transcript: Optional[AudioTranscript],
        chunk_length: int = 8,
    ) -> Optional[Dict[str, Any]]:
        """Create a compact audio summary for downstream prompt construction."""
        if transcript is None:
            return None

        segments = transcript.segments or []
        if not segments:
            return {
                "language": transcript.language,
                "segments_count": 0,
                "key_dialogues": [],
                "ambient_sounds": [],
                "mood_keywords": [],
                "time_blocks": [],
            }

        time_blocks: List[Dict[str, Any]] = []
        key_dialogues: List[str] = []
        mood_keywords: List[str] = []

        current_block: Dict[str, Any] | None = None
        for segment in segments:
            text = self._normalize_dialogue_text(segment.get("text") or "")
            if not text:
                continue

            start = float(segment.get("start", 0.0))
            block_index = int(start // max(chunk_length, 1))
            block_start = block_index * max(chunk_length, 1)
            block_end = block_start + max(chunk_length, 1)

            if current_block is None or current_block["index"] != block_index:
                current_block = {
                    "index": block_index,
                    "start": block_start,
                    "end": block_end,
                    "lines": [],
                }
                time_blocks.append(current_block)

            current_block["lines"].append(text)

            if self._is_key_dialogue_candidate(text):
                key_dialogues.append(text)

            mood = self._guess_mood_hint(text)
            if mood:
                mood_keywords.append(mood)

        return {
            "language": transcript.language,
            "segments_count": len(segments),
            "key_dialogues": key_dialogues[:10],
            "ambient_sounds": [],
            "mood_keywords": list(dict.fromkeys(mood_keywords)),
            "time_blocks": [
                {
                    "start": block["start"],
                    "end": block["end"],
                    "summary": self._summarize_audio_block(block["lines"]),
                }
                for block in time_blocks
                if block["lines"]
            ],
        }

    def _normalize_dialogue_text(self, text: str) -> str:
        normalized = " ".join((text or "").strip().split())
        normalized = normalized.strip(" -")
        return normalized

    def _is_key_dialogue_candidate(self, text: str) -> bool:
        normalized = self._normalize_dialogue_text(text)
        if not normalized:
            return False
        if len(normalized) >= 16:
            return True
        if any(token in normalized for token in ["哥", "姐", "工资", "上学", "回去", "拿着", "孩子", "工作", "打工"]):
            return True
        if "?" in normalized or "？" in normalized:
            return True
        return len(normalized.split()) >= 4

    def _summarize_audio_block(self, lines: List[str], max_items: int = 3, max_chars: int = 140) -> str:
        unique_lines: List[str] = []
        for line in lines:
            normalized = self._normalize_dialogue_text(line)
            if normalized and normalized not in unique_lines:
                unique_lines.append(normalized)
        summary = "；".join(unique_lines[:max_items])
        if len(summary) <= max_chars:
            return summary
        return summary[: max_chars - 3].rstrip() + "..."

    def _guess_speaker_hint(self, text: str) -> Optional[str]:
        normalized = text.lower()
        if "brother" in normalized:
            return "brother"
        if "sister" in normalized:
            return "sister"
        if "mom" in normalized or "mother" in normalized:
            return "mother"
        if "dad" in normalized or "father" in normalized:
            return "father"
        return None

    def _guess_mood_hint(self, text: str) -> Optional[str]:
        normalized = text.lower()
        if "tired" in normalized:
            return "fatigue"
        if "why" in normalized or "?" in normalized:
            return "questioning"
        if "take it" in normalized or "salary" in normalized:
            return "support"
        return None
