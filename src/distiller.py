"""Character distiller: web scrape + LLM extraction -> structured character card."""

import json
import os
import re
from pathlib import Path

import requests
import hashlib
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import colorsys

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
        result = self.distill_from_text(text, source_url=url)
        if "error" not in result:
            try:
                avatar_path = self._generate_avatar(result.get("name", "unknown"))
                result["avatar"] = avatar_path.replace("\\", "/")
            except:
                pass
        return result

    def distill_from_text(self, text, source_url=""):
        """Distill character card from raw text."""
        if len(text) > 8000:
            text = text[:8000] + "..."
        prompt = DISTILL_PROMPT.replace("__TEXT__", text)
        raw = self._call_llm(prompt)
        card = self._parse(raw)
        if source_url:
            card["source_url"] = source_url
        if "error" not in card:
            try:
                avatar_path = self._generate_avatar(card.get("name", "unknown"))
                card["avatar"] = avatar_path.replace("\\", "/")
            except:
                pass
        return card

    def distill_from_search(self, query):
        """Search the web and distill from top results. Falls back to LLM knowledge."""
        text = ""
        try:
            results = self._search_ddg(query, max_results=3)
            if results:
                text = "\n\n".join(
                    f"Source: {r['title']}\n{r['snippet']}" for r in results
                )
        except Exception:
            pass

        if not text or len(text) < 50:
            text = self._llm_knowledge(query)

        result = self.distill_from_text(text)
        if "error" not in result:
            try:
                avatar_path = self._generate_avatar(result.get("name", "unknown"))
                result["avatar"] = avatar_path.replace("\\", "/")
            except Exception as e:
                print(f"Avatar generation skipped: {e}")
        return result

    def _llm_knowledge(self, query):
        """Ask the LLM to describe a known character/persona."""
        prompt = f"Describe the character '{query}' in detail. Include: personality, appearance, speaking style, background, likes/dislikes. Write 3-5 paragraphs as if writing a character wiki entry. Output plain text only."
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a character encyclopedia. Describe characters in detail."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _scrape(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            html = resp.text
            html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", html)
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
        results = []
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

    def _generate_avatar(self, name, output_dir="avatars"):
        """Generate a styled initial avatar with Chinese character and gradient circle."""
        h = int(hashlib.md5(name.encode()).hexdigest(), 16)
        hue1 = h % 360
        hue2 = (hue1 + 60) % 360

        size = 256
        grad = Image.new("RGBA", (size, size))
        for y in range(size):
            for x in range(size):
                ratio = y / size
                h_mix = (hue1 + (hue2 - hue1) * ratio) % 360
                r, g, b = colorsys.hls_to_rgb(h_mix / 360, 0.55, 0.70)
                grad.putpixel((x, y), (int(r * 255), int(g * 255), int(b * 255), 255))

        mask = Image.new("L", (size, size), 0)
        mask_draw = ImageDraw.Draw(mask)
        padding = 4
        mask_draw.ellipse((padding, padding, size - padding, size - padding), fill=255)

        result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        result.paste(grad, (0, 0), mask)

        char = name[0] if name else "?"
        text_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)

        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        font = None
        for fp in font_paths:
            try:
                font = ImageFont.truetype(fp, 130)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()

        bbox = text_draw.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (size - tw) / 2 - bbox[0]
        y = (size - th) / 2 - bbox[1]

        text_draw.text((x + 3, y + 3), char, fill=(0, 0, 0, 50), font=font)
        text_draw.text((x, y), char, fill=(255, 255, 255, 255), font=font)

        result = Image.alpha_composite(result, text_layer)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        safe_id = hashlib.md5(name.encode()).hexdigest()[:8]
        fname = f"auto_{safe_id}.png"
        fpath = out_path / fname
        result.save(fpath, "PNG")
        return str(fpath)

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
        """Save character card. If avatar path is in card, copy/reference it."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        name = card.get("name", "unknown").replace(" ", "_")
        safe_name = re.sub(r"[^\w\-]", "_", name)
        filepath = path / f"{safe_name}.json"
        filepath.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        return filepath
