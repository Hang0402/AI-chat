"""Character distiller: web scrape + LLM extraction -> structured character card."""

import json
import os
import re
from pathlib import Path

import requests

DISTILL_PROMPT = """\
You are a character analyst. Extract a detailed roleplay character card from the provided text.
Output ONLY valid JSON, no markdown fences. Use this exact structure:

{
  "name": "character name in Chinese",
  "age": number or "unknown",
  "gender": "male/female/other",
  "personality": ["trait1", "trait2", "trait3", "trait4", "trait5"],
  "speaking_style": "describe how they talk: sentence patterns, catchphrases, emoji usage",
  "background": "brief backstory in 2-3 sentences",
  "likes": ["thing1", "thing2"],
  "dislikes": ["thing1"],
  "taboos": ["never do this", "never say that"],
  "greeting": "what they say when first meeting someone",
  "sample_dialogue": [
    {"user": "example user message", "char": "how they would respond"}
  ],
  "emotion_range": ["happy", "sad", "angry", "shy", "playful"],
  "relationship_goal": "friend/lover/mentor/etc"
}

The text to analyze:
__TEXT__
"""


class CharacterDistiller:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-placeholder")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-chat")

    def distill_from_url(self, url):
        """Scrape a webpage and distill character from its content."""
        text = self._scrape(url)
        if not text:
            return {"error": "Failed to scrape content"}
        return self.distill_from_text(text, source_url=url)

    def distill_from_text(self, text, source_url=""):
        """Distill character card from raw text."""
        # Truncate if too long
        if len(text) > 8000:
            text = text[:8000] + "..."
        prompt = DISTILL_PROMPT.replace("__TEXT__", text)
        raw = self._call_llm(prompt)
        card = self._parse(raw)
        if source_url:
            card["source_url"] = source_url
        return card

    def distill_from_search(self, query):
        """Search the web and distill from top results. Uses DuckDuckGo."""
        try:
            results = self._search_ddg(query, max_results=3)
        except Exception as e:
            return {"error": f"Search failed: {e}"}
        if not results:
            return {"error": "No search results found"}
        # Combine snippets
        combined = "\n\n".join(
            f"Source: {r['title']}\n{r['snippet']}" for r in results
        )
        return self.distill_from_text(combined)

    def _scrape(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            # Simple HTML text extraction
            html = resp.text
            # Remove scripts and styles
            html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
            # Remove tags
            text = re.sub(r"<[^>]+>", " ", html)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text[:10000]
        except Exception as e:
            print(f"Scrape error: {e}")
            return ""

    def _search_ddg(self, query, max_results=3):
        """Search DuckDuckGo HTML (no API key needed)."""
        url = "https://html.duckduckgo.com/html/"
        resp = requests.post(
            url,
            data={"q": query, "b": ""},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        # Parse results
        results = []
        # Simple regex extraction
        snippets = re.findall(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            resp.text, re.DOTALL
        )
        titles = re.findall(
            r'<a[^>]*class="result__a"[^>]*>(.*?)</a>',
            resp.text, re.DOTALL
        )
        for i in range(min(len(titles), max_results)):
            title = re.sub(r"<[^>]+>", "", titles[i]).strip()
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            results.append({"title": title, "snippet": snippet})
        return results

    def _call_llm(self, prompt):
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a character analyst. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse(self, raw):
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_output": raw, "error": "JSON parse failed"}

    def save_character(self, card, output_dir="characters"):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        name = card.get("name", "unknown").replace(" ", "_")
        filepath = path / f"{name}.json"
        filepath.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        return filepath
