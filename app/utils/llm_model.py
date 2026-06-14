import os
import re
from openai import OpenAI
from app.utils.config_loader import load_config

class LLMRunner:
    def __init__(self):
        config = load_config()
        self.context_length = 4096

        api_key = config.get("openrouter", {}).get("api_key")
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")

        self.model = config.get("openrouter", {}).get("model", "openai/gpt-5.4-mini")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        print(f"[INFO] Configured OpenRouter client for model {self.model}")

    def truncate_prompt(self, prompt: str) -> str:
        tokens = prompt.split()
        if len(tokens) > self.context_length - 512:
            tokens = tokens[-(self.context_length - 512):]
        return " ".join(tokens)

    def _truncate_messages(self, messages: list) -> list:
        if not messages:
            return messages
        system_msg = messages[0]
        if system_msg.get("role") == "system":
            system_msg["content"] = self.truncate_prompt(system_msg["content"])
        return messages

    def _clean_streamed_text(self, text: str) -> str:
        text = re.sub(r'[ ]{2,}', ' ', text)
        text = re.sub(r'\s+\n', '\n', text)
        text = re.sub(r'\n\s+', '\n', text)
        return text

    def generate_response(self, messages: list) -> str:
        try:
            print("[INFO] Generating full response via OpenRouter...")
            messages = self._truncate_messages(messages)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def stream_response(self, messages: list):
        try:
            print("[INFO] Streaming response via OpenRouter...")
            messages = self._truncate_messages(messages)
            buffer = ""
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token is None:
                    continue
                buffer += token

                if re.search(r"[ \n]$", buffer) or re.search(r'\]\([^)]+?\)$', buffer):
                    cleaned = self._clean_streamed_text(buffer)
                    if cleaned:
                        yield cleaned
                    buffer = ""

            if buffer.strip():
                yield self._clean_streamed_text(buffer)

        except Exception as e:
            yield f"\n[Error generating response: {str(e)}]"
