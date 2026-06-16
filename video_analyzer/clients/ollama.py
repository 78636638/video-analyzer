import json
import logging
from typing import Optional, Dict, Any, List

import requests

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"

    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256,
        num_ctx: Optional[int] = None) -> Dict[Any, Any]:
        try:
            # Build the request data
            data = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict
                }
            }
            if num_ctx:
                data["options"]["num_ctx"] = num_ctx
            
            resolved_image_paths = list(image_paths or [])
            if image_path:
                resolved_image_paths.insert(0, image_path)

            if resolved_image_paths:
                data["images"] = [self.encode_image(path) for path in resolved_image_paths]
                logger.debug(
                    "Sending %s images to Ollama with max_image_side=%s jpeg_quality=%s",
                    len(resolved_image_paths),
                    getattr(self, "max_image_side", None),
                    getattr(self, "jpeg_quality", 85),
                )

            logger.debug(
                "Ollama request prompt_length=%s image_count=%s",
                len(prompt),
                len(data.get("images", [])),
            )
                    
            response = requests.post(self.generate_url, json=data)
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            response_text = ""
            try:
                response_text = f" | body={e.response.text[:1000]}" if getattr(e, "response", None) is not None else ""
            except Exception:
                response_text = ""
            raise Exception(f"API request failed: {str(e)}{response_text}")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")
            
    def _handle_streaming_response(self, response: requests.Response) -> Dict[Any, Any]:
        accumulated_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'response' in json_response:
                        accumulated_response += json_response['response']
                except json.JSONDecodeError:
                    continue
                    
        return {"response": accumulated_response}
