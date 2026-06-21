"""Chat engine: loads character card, manages conversation memory."""

import json
import os
from pathlib import Path

import requests


class ChatEngine:
    def __init__(self, character_card=None, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-placeholder")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-chat")
        self.card = character_card or {}
        self.history: list[dict] = []  # short-term memory
        self.long_term: list[str] = []  # key events

    def load_character(self, path_or_dict):
        if isinstance(path_or_dict, dict):
            self.card = path_or_dict
        else:
            p = Path(path_or_dict)
            self.card = json.loads(p.read_text(encoding="utf-8"))
        self.history = []
        self.long_term = []
        return self

    def chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        system_prompt = self._build_system()
        messages = [{"role": "system", "content": system_prompt}]
        # Add recent history (last 20 turns)
        messages.extend(self.history[-20:])
        response = self._call_llm(messages)
        self.history.append({"role": "assistant", "content": response})
        # Store key events
        if len(self.history) % 6 == 0:
            self._summarize_memory()
        return response

    def chat_stream(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        system_prompt = self._build_system()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-20:])

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 512,
                "stream": True,
            },
            timeout=60,
            stream=True,
        )
        resp.raise_for_status()
        full = ""
        for line in resp.iter_lines():
            if not line or line.startswith(b": "):
                continue
            data = line.decode("utf-8").lstrip("data: ").strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full += content
                    yield content
            except Exception:
                continue
        self.history.append({"role": "assistant", "content": full})

    def _build_system(self):
        c = self.card
        parts = []
        parts.append(f"你是{c.get('name', '角色')}。你必须始终保持这个身份，不能说自己是AI。")
        if c.get("age"):
            parts.append(f"年龄: {c['age']}")
        if c.get("gender"):
            parts.append(f"性别: {c['gender']}")
        if c.get("personality"):
            parts.append(f"性格: {'、'.join(c['personality'])}")
        if c.get("speaking_style"):
            parts.append(f"说话风格: {c['speaking_style']}")
        if c.get("background"):
            parts.append(f"背景: {c['background']}")
        if c.get("taboos"):
            parts.append(f"绝对不能做的事: {'; '.join(c['taboos'])}")
        if c.get("likes"):
            parts.append(f"喜欢: {'、'.join(c['likes'])}")
        if c.get("dislikes"):
            parts.append(f"讨厌: {'、'.join(c['dislikes'])}")
        # Inject long-term memory
        if self.long_term:
            parts.append(f"\n过去的重要记忆:\n" + "\n".join(f"- {m}" for m in self.long_term[-5:]))
        # Emotion hint
        if c.get("emotion_range"):
            parts.append(f"\n当前可以表达的情绪范围: {', '.join(c['emotion_range'])}")
        parts.append("\n回复要简短自然，像真人聊天。每个回复2-4句话。")
        return "\n".join(parts)

    def _call_llm(self, messages):
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _summarize_memory(self):
        """Extract key events from recent history."""
        recent = self.history[-12:]
        if len(recent) < 4:
            return
        dialogue = "\n".join(
            f"{'用户' if m['role']=='user' else '角色'}: {m['content'][:100]}"
            for m in recent
        )
        prompt = f"从以下对话中提取1-2个关键事件或情感转折点，用一句话概括:\n{dialogue}"
        try:
            summary = self._call_llm([
                {"role": "system", "content": "提取对话中的关键事件，一句话概括。"},
                {"role": "user", "content": prompt},
            ])
            self.long_term.append(summary.strip())
        except Exception:
            pass
