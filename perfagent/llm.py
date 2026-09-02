import httpx, os, json
from dotenv import load_dotenv
from perfagent.contracts import LLMResponse, Usage
import time
load_dotenv()

STOP_MAP = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}

class GroqClient:

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b") -> None:
        self.api_key = api_key
        self.model = model
    
    def complete(self, *, messages: list[dict], tools: list[dict] | None = None,
                 max_tokens: int = 4096) -> LLMResponse:
        start = time.perf_counter()
        r = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model,
                    "max_tokens": 256,
                    "messages": messages,
                    "tools": tools},
                timeout=60,
            )        
        latency_ms = (time.perf_counter() - start) * 1000
        return self._to_response(r.json(), latency_ms)
    

    def _to_response(self, raw: dict, latency_ms: float) -> LLMResponse:
        msg = raw["choices"][0]["message"]
        content = []

        if msg.get("content"):
            content.append({"type": "text", "text": msg["content"]})

        for tc in msg.get("tool_calls") or []:
            content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"]),
            })

        usage = Usage(input_tokens= raw["usage"]["prompt_tokens"],
              output_tokens=raw["usage"]["completion_tokens"])

        finish = raw["choices"][0]["finish_reason"]
        stop_reason = STOP_MAP.get(finish, finish)

        return LLMResponse(
            content=content, stop_reason=stop_reason,
            latency_ms=latency_ms, usage=usage
        )