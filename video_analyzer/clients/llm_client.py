from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional, Dict, Any, List
import base64
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image


class LLMClient(ABC):
    max_image_side: Optional[int] = None
    jpeg_quality: int = 85

    def configure_image_encoding(self, max_image_side: Optional[int] = None, jpeg_quality: int = 85) -> None:
        """Configure shared image preprocessing for all client implementations."""
        self.max_image_side = max_image_side
        self.jpeg_quality = jpeg_quality

    def encode_image(
        self,
        image_path: str,
        max_image_side: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
    ) -> str:
        """Encode an image as base64, optionally resizing and recompressing it first."""
        # #region debug-point E:image-encoding-metrics
        started_at = time.perf_counter()
        def _debug_emit(data: Dict[str, Any]) -> None:
            server_url = "http://127.0.0.1:7777/event"
            session_id = "long-video-performance"
            try:
                env_text = Path(".dbg/long-video-performance.env").read_text(encoding="utf-8")
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
                                "runId": "pre",
                                "hypothesisId": "E",
                                "location": "llm_client.py:encode_image",
                                "msg": "[DEBUG] Image encoded",
                                "data": data,
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
        target_side = self.max_image_side if max_image_side is None else max_image_side
        target_quality = self.jpeg_quality if jpeg_quality is None else jpeg_quality

        with Image.open(image_path) as image:
            original_size = image.size
            processed = image.convert("RGB")
            if target_side and target_side > 0:
                processed.thumbnail((target_side, target_side))

            buffer = BytesIO()
            processed.save(buffer, format="JPEG", quality=target_quality, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            _debug_emit(
                {
                    "image_path": image_path,
                    "original_size": list(original_size),
                    "processed_size": list(processed.size),
                    "max_image_side": target_side,
                    "jpeg_quality": target_quality,
                    "encoded_length": len(encoded),
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                }
            )
            return encoded

    @abstractmethod
    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256,
        num_ctx: Optional[int] = None) -> Dict[Any, Any]:
        pass
