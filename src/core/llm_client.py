import os
import json
import time
from typing import Generator, Optional, Any
from google import genai as google_genai

from src.core.prompt_engine import PromptRegistry, PromptTemplate


class LLMResponse:
    def __init__(self, content: str, input_tokens: int, output_tokens: int):
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.parsed: Optional[dict] = None

    def parse_json(self) -> dict:
        if self.parsed is not None:
            return self.parsed

        text = self.content.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        try:
            self.parsed = json.loads(text)
        except json.JSONDecodeError:
            self.parsed = {}

        return self.parsed


class LLMClient:
    def __init__(
        self,
        registry: PromptRegistry,
        config: dict,
        api_key: Optional[str] = None
    ):
        self.registry = registry
        self.config = config
        api_key = api_key or os.getenv(config.get("llm", {}).get("api_key_env", "GEMINI_API_KEY"))
        self.client = google_genai.Client(api_key=api_key)

    def call(
        self,
        prompt_name: str,
        variables: dict[str, Any]
    ) -> LLMResponse:
        template = self.registry.get(prompt_name)
        system_prompt, user_prompt = template.render(variables)

        temp_config = self.config.get("llm", {}).get("temperature", {})
        default_temp = temp_config.get(prompt_name, temp_config.get("conversational", 0.7))
        temperature = template.get_temperature(default_temp)

        model = template.get_model(self.config["llm"]["primary_model"])

        max_retries = self.config["llm"].get("max_retries", 5)
        backoff_base = self.config["llm"].get("retry_backoff_base", 2)

        for attempt in range(max_retries):
            try:
                combined_contents = f"{system_prompt}\n\n---\n\n{user_prompt}"
                
                response = self.client.models.generate_content(
                    model=model,
                    contents=combined_contents,
                    config={
                        "temperature": temperature
                    }
                )

                content = response.text or ""

                input_tokens = len(system_prompt.split()) + len(user_prompt.split())
                output_tokens = len(content.split())

                return LLMResponse(content, input_tokens, output_tokens)

            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "429" in error_msg or "resource_exhausted" in error_msg:
                    if attempt < max_retries - 1:
                        wait = backoff_base ** attempt
                        print(f"[Retry {attempt + 1}/{max_retries}] Quota hit, waiting {wait}s before retry...")
                        time.sleep(wait)
                        continue
                if attempt < max_retries - 1:
                    wait = backoff_base ** attempt
                    print(f"[Retry {attempt + 1}/{max_retries}] Error: {e}, waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                raise

    def stream(
        self,
        prompt_name: str,
        variables: dict[str, Any]
    ) -> Generator[str, None, None]:
        template = self.registry.get(prompt_name)
        system_prompt, user_prompt = template.render(variables)

        temp_config = self.config.get("llm", {}).get("temperature", {})
        default_temp = temp_config.get(prompt_name, temp_config.get("conversational", 0.7))
        temperature = template.get_temperature(default_temp)

        model = template.get_model(self.config["llm"]["primary_model"])

        stream = self.client.models.generate_content(
            model=model,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": temperature,
                "response_modalities": ["TEXT"]
            }
        )

        total_tokens = 0
        for chunk in stream:
            delta = chunk.text
            if delta:
                total_tokens += len(delta.split())
                yield delta