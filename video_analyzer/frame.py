from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging
import json
import time
import urllib.request

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime when frame extraction is used
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime when frame extraction is used
    np = None

logger = logging.getLogger(__name__)

# #region debug-point D:frame-extraction-metrics
def _debug_emit(hypothesis_id: str, location: str, msg: str, data: Optional[dict] = None, run_id: str = "pre") -> None:
    env_path = Path(".dbg/long-video-performance.env")
    server_url = "http://127.0.0.1:7777/event"
    session_id = "long-video-performance"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    server_url = line.split("=", 1)[1].strip()
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip()
        except Exception:
            return

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


def resolve_effective_frames_per_minute(
    requested_frames_per_minute: int,
    video_duration_minutes: float,
    analysis_config: Optional[dict] = None,
) -> int:
    """Resolve a continuity-friendly frame budget for medium and long videos."""
    analysis_config = analysis_config or {}
    effective = max(1, int(requested_frames_per_minute))

    medium_threshold = float(analysis_config.get("adaptive_frame_budget_threshold_minutes", 1.0))
    medium_budget = analysis_config.get("adaptive_frames_per_minute")
    extended_threshold = float(analysis_config.get("extended_frame_budget_threshold_minutes", 3.0))
    extended_budget = analysis_config.get("extended_frames_per_minute")

    if extended_budget is not None and video_duration_minutes >= extended_threshold:
        effective = min(effective, max(1, int(extended_budget)))
    elif medium_budget is not None and video_duration_minutes >= medium_threshold:
        effective = min(effective, max(1, int(medium_budget)))

    return effective



@dataclass
class Frame:
    number: int
    path: Path
    timestamp: float
    score: float
    scene_hint: Optional[str] = None

class VideoProcessor:
    # Class constants
    FRAME_DIFFERENCE_THRESHOLD = 10.0
    
    def __init__(self, video_path: Path, output_dir: Path, model: str):
        if cv2 is None or np is None:
            raise ImportError("OpenCV and numpy are required for video frame extraction")
        self.video_path = video_path
        self.output_dir = output_dir
        self.model = model
        self.frames: List[Frame] = []
        
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate the difference between two frames using absolute difference."""
        if frame1 is None or frame2 is None:
            return 0.0
        
        # Convert to grayscale for simpler comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference and mean
        diff = cv2.absdiff(gray1, gray2)
        score = np.mean(diff)
        
        return float(score)

    def _is_keyframe(self, current_frame: np.ndarray, prev_frame: np.ndarray, threshold: float = FRAME_DIFFERENCE_THRESHOLD) -> bool:
        """Determine if frame is significantly different from previous frame."""
        if prev_frame is None:
            return True
            
        score = self._calculate_frame_difference(current_frame, prev_frame)
        return score > threshold

    def extract_keyframes(
        self,
        frames_per_minute: int = 10,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
        analysis_config: Optional[dict] = None,
    ) -> List[Frame]:
        """Extract keyframes from video targeting a specific number of frames per minute."""
        started_at = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        analysis_config = analysis_config or {}
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        if duration:
            video_duration = min(duration, video_duration)
            total_frames = int(min(total_frames, duration * fps))
        
        effective_frames_per_minute = resolve_effective_frames_per_minute(
            requested_frames_per_minute=frames_per_minute,
            video_duration_minutes=video_duration / 60.0,
            analysis_config=analysis_config,
        )

        # Calculate target number of frames
        target_frames = max(1, min(
            int((video_duration / 60) * effective_frames_per_minute),
            total_frames,
            max_frames if max_frames is not None else float('inf')
        ))
        
        # Calculate adaptive sampling interval
        sample_interval = max(1, total_frames // (target_frames * 2))
        _debug_emit(
            "D",
            "frame.py:extract_keyframes",
            "Keyframe extraction started",
            {
                "video_duration_seconds": round(video_duration, 3),
                "fps": round(fps, 3),
                "total_frames": total_frames,
                "target_frames": target_frames,
                "requested_frames_per_minute": frames_per_minute,
                "effective_frames_per_minute": effective_frames_per_minute,
                "sample_interval": sample_interval,
                "long_video_threshold_minutes": analysis_config.get("long_video_threshold_minutes", 15),
            },
        )
        
        frame_candidates = []
        prev_frame = None
        frame_count = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                score = self._calculate_frame_difference(frame, prev_frame)
                if score > self.FRAME_DIFFERENCE_THRESHOLD:
                    scene_hint = "transition" if score >= analysis_config.get("scene_change_threshold", self.FRAME_DIFFERENCE_THRESHOLD * 1.5) else "stable"
                    frame_candidates.append((frame_count, frame, score, scene_hint))
                prev_frame = frame.copy()
                
            frame_count += 1
            
        cap.release()
        
        # Select candidates. Long videos keep representatives per time bucket so early/late scenes are not starved.
        long_video_threshold_minutes = analysis_config.get("long_video_threshold_minutes", 15)
        is_long_video = (video_duration / 60.0) >= long_video_threshold_minutes
        if is_long_video and frame_candidates:
            bucket_count = max(4, min(target_frames, int(video_duration // 120) + 1))
            bucket_size = max(1, total_frames // bucket_count)
            selected_candidates = []
            for bucket_index in range(bucket_count):
                bucket_start = bucket_index * bucket_size
                bucket_end = total_frames if bucket_index == bucket_count - 1 else (bucket_index + 1) * bucket_size
                bucket_candidates = [
                    candidate for candidate in frame_candidates
                    if bucket_start <= candidate[0] < bucket_end
                ]
                if not bucket_candidates:
                    continue
                per_bucket_target = max(1, target_frames // bucket_count)
                selected_candidates.extend(
                    sorted(bucket_candidates, key=lambda x: x[2], reverse=True)[:per_bucket_target]
                )

            if len(selected_candidates) < target_frames:
                used_numbers = {candidate[0] for candidate in selected_candidates}
                remaining = [
                    candidate for candidate in sorted(frame_candidates, key=lambda x: x[2], reverse=True)
                    if candidate[0] not in used_numbers
                ]
                selected_candidates.extend(remaining[: max(0, target_frames - len(selected_candidates))])
        else:
            selected_candidates = sorted(frame_candidates, key=lambda x: x[2], reverse=True)[:target_frames]

        # If max_frames is specified, sample evenly across the candidates
        if max_frames is not None and max_frames < len(selected_candidates):
            step = len(selected_candidates) / max_frames
            selected_frames = [selected_candidates[int(i * step)] for i in range(max_frames)]
        else:
            selected_frames = selected_candidates

        # Re-sort by frame number so frames on disk and in the JSON are chronological
        selected_frames = sorted(selected_frames, key=lambda x: x[0])

        self.frames = []
        for idx, (frame_num, frame, score, scene_hint) in enumerate(selected_frames):
            frame_path = self.output_dir / f"frame_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            timestamp = frame_num / fps
            self.frames.append(Frame(idx, frame_path, timestamp, score, scene_hint=scene_hint))
        
        _debug_emit(
            "D",
            "frame.py:extract_keyframes",
            "Keyframe extraction completed",
            {
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "candidate_count": len(frame_candidates),
                "selected_count": len(selected_frames),
                "is_long_video": is_long_video,
            },
        )
        logger.info(f"Extracted {len(self.frames)} frames from video (target was {target_frames})")
        return self.frames
