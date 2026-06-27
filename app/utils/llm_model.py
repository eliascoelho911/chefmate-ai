import logging
import os
import re
import time
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMRunner:
    """
    Adapter that satisfies the LLMRunner protocol using OpenRouter.

    Interface:
        stream_response(messages) -> Iterator[str]
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-5.4-mini",
        context_length: int = 4096,
    ):
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        self.model = model
        self.context_length = context_length
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        logger.info("Configured OpenRouter client for model %s", self.model)

    def truncate_prompt(self, prompt: str) -> str:
        tokens = prompt.split()
        if len(tokens) > self.context_length - 512:
            tokens = tokens[-(self.context_length - 512) :]
        return " ".join(tokens)

    def _truncate_messages(self, messages: list) -> list:
        if not messages:
            return messages
        system_msg = messages[0]
        if system_msg.get("role") == "system":
            system_msg["content"] = self.truncate_prompt(system_msg["content"])
        return messages

    @staticmethod
    def _clean_streamed_text(text: str) -> str:
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n\s+", "\n", text)
        return text

    def generate_response(self, messages: list) -> str:
        try:
            logger.info("Generating full response via OpenRouter...")
            messages = self._truncate_messages(messages)
            t_llm_start = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )
            t_llm_elapsed = time.perf_counter() - t_llm_start
            logger.debug("generate_response llm_time_ms=%.2f", t_llm_elapsed * 1000)
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def stream_response(self, messages: list):
        try:
            logger.info("Streaming response via OpenRouter...")
            messages = self._truncate_messages(messages)
            buffer = ""
            t_llm_start = time.perf_counter()
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token is None:
                    continue
                buffer += token

                if re.search(r"[ \n]$", buffer) or re.search(r"\]\([^)]+?\)$", buffer):
                    cleaned = self._clean_streamed_text(buffer)
                    if cleaned:
                        yield cleaned
                    buffer = ""

            if buffer.strip():
                yield self._clean_streamed_text(buffer)

            t_llm_elapsed = time.perf_counter() - t_llm_start
            logger.debug("stream_response llm_time_ms=%.2f", t_llm_elapsed * 1000)

        except Exception as e:
            yield f"\n[Error generating response: {str(e)}]"
