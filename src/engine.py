"""Chat engine: loads character card, manages conversation memory."""

import json
import os
from pathlib import Path

import requests


"""Chat engine: loads character card, manages conversation memory with long-term support."""

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
        self.history: list[dict] = []       # Full conversation
        self.mid_term: list[str] = []       # Mid-term: compressed blocks
        self.long_term: list[str] = []       # Long-term: consolidated memories
        self.memory_counter = 0

    def load_character(self, path_or_dict):
        if isinstance(path_or_dict, dict):
            self.card = path_or_dict
        else:
            p = Path(path_or_dict)
            self.card = json.loads(p.read_text(encoding="utf-8"))
        self.history = []
        self.mid_term = []
        self.long_term = []
        self.memory_counter = 0
        return self

    def chat_stream(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        self.memory_counter += 1
        system_prompt = self._build_system()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._get_context_window())

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
            if not line:
                continue
            text = line.decode("utf-8").strip()
            if not text or text.startswith(":"):
                continue
            if not text.startswith("data: "):
                continue
            text = text[6:]
            if text == "[DONE]":
                break
            try:
                chunk = json.loads(text)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full += content
                    yield content
            except Exception:
                continue
        self.history.append({"role": "assistant", "content": full})

        # Memory management: compress old history periodically
        if self.memory_counter % 6 == 0:
            self._compress_to_mid_term()
        if self.memory_counter % 18 == 0:
            self._consolidate_long_term()

    def _get_context_window(self):
        """Build the context window: recent turns + mid-term summaries."""
        # Always include last 10 turns (20 messages)
        recent = self.history[-20:] if len(self.history) >= 20 else self.history

        # If we have mid-term memories, inject them before recent history
        if self.mid_term:
            mids = [{"role": "system", "content": "[Earlier conversation summary]\n" + "\n".join(f"- {m}" for m in self.mid_term[-3:])}]
            return mids + recent
        return recent

    def _compress_to_mid_term(self):
        """Compress oldest history into a mid-term summary block."""
        if len(self.history) < 12:
            return
        # Take the oldest 12 turns not yet compressed
        cutoff = len(self.history) - 20
        if cutoff <= 0:
            return
        old_block = self.history[:cutoff]
        if len(old_block) < 4:
            return

        dialogue = "\n".join(
            f"{'User' if m['role']=='user' else 'Character'}: {m['content'][:150]}"
            for m in old_block
        )
        prompt = f"Summarize this conversation segment in 2-3 key sentences. Focus on: emotional developments, important revelations, relationship changes, and plot-relevant events.\n\n{dialogue}"
        try:
            summary = self._call_llm([
                {"role": "system", "content": "Summarize conversation segments concisely. Focus on emotional and relational developments."},
                {"role": "user", "content": prompt},
            ])
            self.mid_term.append(summary.strip())
            # Keep only last 20 turns in history
            self.history = self.history[-20:]
        except Exception:
            pass

    def _consolidate_long_term(self):
        """Merge mid-term memories into long-term consolidated memories."""
        if len(self.mid_term) < 2:
            return
        recent_mids = self.mid_term[-4:]
        prompt = "Merge these conversation summaries into 1-2 consolidated memory entries. Focus on the overall narrative arc and relationship evolution:\n\n" + "\n".join(f"- {m}" for m in recent_mids)
        try:
            merged = self._call_llm([
                {"role": "system", "content": "Consolidate conversation memories into concise narrative arcs."},
                {"role": "user", "content": prompt},
            ])
            self.long_term.append(merged.strip())
            # Keep last 10 mid-term entries
            self.mid_term = self.mid_term[-10:]
        except Exception:
            pass

    def _build_system(self):
        c = self.card
        parts = []
        parts.append(f"You are {c.get('name', 'character')}. You must ALWAYS stay in character. Never say you are AI.")
        if c.get("age"):
            parts.append(f"Age: {c['age']}")
        if c.get("gender"):
            parts.append(f"Gender: {c['gender']}")
        if c.get("personality"):
            parts.append(f"Personality: {', '.join(c['personality'])}")
        if c.get("speaking_style"):
            parts.append(f"Speaking style: {c['speaking_style']}")
        if c.get("background"):
            parts.append(f"Background: {c['background']}")
        if c.get("taboos"):
            parts.append(f"Never do: {'; '.join(c['taboos'])}")
        if c.get("likes"):
            parts.append(f"Likes: {', '.join(c['likes'])}")
        if c.get("dislikes"):
            parts.append(f"Dislikes: {', '.join(c['dislikes'])}")
        # Long-term consolidated memories
        if self.long_term:
            parts.append(f"\n[Long-term relationship memories]\n" + "\n".join(f"- {m}" for m in self.long_term[-3:]))
        if c.get("emotion_range"):
            parts.append(f"\nAllowed emotions: {', '.join(c['emotion_range'])}")
        parts.append("\nKeep replies short and natural, like real chat. 2-4 sentences per reply.")
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
