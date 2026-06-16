import requests
import json
import time
import re
from typing import Optional, Dict, Any, List
from .llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RETRIES = 3
RATE_LIMIT_WAIT_TIME = 25  # seconds
DEFAULT_WAIT_TIME = 25  # seconds
OVERLOAD_WAIT_TIME = 45  # seconds
DEFAULT_REQUEST_TIMEOUT = 180  # seconds
MAX_RETRY_WAIT_TIME = 120  # seconds
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}

class GenericOpenAIAPIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        api_url: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ):
        self.api_key = api_key
        self.base_url = api_url.rstrip('/')  # Remove trailing slash if present
        self.generate_url = f"{self.base_url}/chat/completions"
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    def _sanitize_message_content(self, content: Any) -> str:
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            normalized = "\n".join(parts)
        else:
            normalized = str(content or "")
        normalized = re.sub(r"<think>[\s\S]*?</think>", "", normalized, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"<thinking>[\s\S]*?</thinking>", "", normalized, flags=re.IGNORECASE).strip()
        return normalized

    def _compute_retry_wait_time(
        self,
        error: Exception,
        attempt: int,
    ) -> int:
        if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
            retry_after = error.response.headers.get("Retry-After")
            if retry_after:
                try:
                    parsed_wait = int(retry_after)
                    logger.info("Using Retry-After header value: %s seconds", parsed_wait)
                    return max(1, min(parsed_wait, MAX_RETRY_WAIT_TIME))
                except (ValueError, TypeError):
                    logger.warning("Invalid Retry-After header value, using computed wait time")
            status_code = int(error.response.status_code)
            if status_code == 429:
                return min(RATE_LIMIT_WAIT_TIME * (attempt + 1), MAX_RETRY_WAIT_TIME)
            if status_code in {500, 502, 503, 504, 529}:
                return min(OVERLOAD_WAIT_TIME * (attempt + 1), MAX_RETRY_WAIT_TIME)
        if isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return min(DEFAULT_WAIT_TIME * (attempt + 1), MAX_RETRY_WAIT_TIME)
        return min(DEFAULT_WAIT_TIME * (attempt + 1), MAX_RETRY_WAIT_TIME)

    def _is_retryable_error(self, error: Exception) -> bool:
        if isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
        if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
            return int(error.response.status_code) in RETRYABLE_HTTP_STATUS_CODES
        return False

    def _build_error_log_suffix(self, error: Exception) -> str:
        if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
            response_text = (error.response.text or "").strip()
            if response_text:
                return f" body={response_text[:240]}"
        return ""

    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256,
        num_ctx: Optional[int] = None) -> Dict[Any, Any]:
        """Generate response from OpenAI-compatible API."""
        # Prepare request content
        resolved_image_paths = list(image_paths or [])
        if image_path:
            resolved_image_paths.insert(0, image_path)

        if resolved_image_paths:
            logger.debug(
                "Sending %s images to OpenAI-compatible API with max_image_side=%s jpeg_quality=%s",
                len(resolved_image_paths),
                getattr(self, "max_image_side", None),
                getattr(self, "jpeg_quality", 85),
            )
            content = [{"type": "text", "text": prompt}]
            for path in resolved_image_paths:
                base64_image = self.encode_image(path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
        else:
            content = prompt

        # Prepare request data
        data = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
            "temperature": temperature,
            "max_tokens": num_predict
        }
        logger.debug(
            "OpenAI-compatible request prompt_length=%s image_count=%s",
            len(prompt),
            len(resolved_image_paths),
        )

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/byjlw/video-analyzer",
            "X-Title": "Video Analyzer",
            "Content-Type": "application/json"
        }

        # Try request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.generate_url,
                    headers=headers,
                    json=data,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                
                # Parse successful response
                try:
                    json_response = response.json()
                    if 'error' in json_response:
                        raise Exception(f"API error: {json_response['error']}")
                    
                    if stream:
                        return self._handle_streaming_response(response)
                    
                    if 'choices' not in json_response or not json_response['choices']:
                        raise Exception("No choices in response")
                        
                    choice = json_response['choices'][0]
                    message = choice.get('message', {})
                    if not message or 'content' not in message:
                        raise Exception("No content in response message")
                    raw_content = message.get("content", "")
                    return {
                        "response": self._sanitize_message_content(raw_content),
                        "raw_response": raw_content,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": json_response.get("usage"),
                    }
                    
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response: {response.text}")
                    
            except Exception as e:
                if attempt == self.max_retries - 1:  # Last attempt
                    raise Exception(f"An error occurred: {str(e)}")
                if not self._is_retryable_error(e):
                    raise Exception(f"An error occurred: {str(e)}")
                wait_time = self._compute_retry_wait_time(e, attempt)
                logger.warning(
                    "Request failed (attempt %s/%s): %s%s",
                    attempt + 1,
                    self.max_retries,
                    str(e),
                    self._build_error_log_suffix(e),
                )
                logger.warning("Waiting %s seconds before retry", wait_time)
                time.sleep(wait_time)

    def _handle_streaming_response(self, response: requests.Response) -> Dict[Any, Any]:
        """Handle streaming response from API.
        
        Args:
            response: Streaming response from API
            
        Returns:
            Dict containing accumulated response
        """
        accumulated_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'choices' in json_response and len(json_response['choices']) > 0:
                        delta = json_response['choices'][0].get('delta', {})
                        if 'content' in delta:
                            accumulated_response += delta['content']
                except json.JSONDecodeError:
                    continue

        return {"response": accumulated_response}
