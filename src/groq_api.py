import os
import re
import time
import json
import logging
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("groq_artists")

@dataclass(frozen=True)
class GroqArtistConfig:
    api_key: str
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 256
    retries: int = 3
    backoff_initial: float = 0.6
    backoff_max: float = 6.0
    enforce_count: int = 10

class GroqMusicArtistService:
    def __init__(self, config: GroqArtistConfig):
        self.config = config
        self.client = Groq(api_key=config.api_key)

    def get_artists(self, mood: str, language: Optional[str] = None, count: Optional[int] = None) -> List[str]:
        if not isinstance(mood, str) or not mood.strip():
            return []
        mood = mood.strip()
        if language is not None and (not isinstance(language, str) or not language.strip()):
            language = None
        target_count = count or self.config.enforce_count
        attempts = max(1, self.config.retries)
        system_base = (
            f"You are a music expert. Output exactly {target_count} distinct, well-known, popular recording artists who match the user's mood and optional language. "
            "Artists must be primary performers (solo singers, bands, DJs). Use official spellings as on Spotify. "
            "Output format: one artist name per line, no numbers, no bullets, no punctuation, no explanations, no extra text. "
            "Exclude actors or non-performing composers. If a language is provided, primarily choose artists who release music in that language."
        )

        for attempt in range(1, attempts + 1):
            system_prompt = system_base
            if attempt == 2:
                system_prompt += " Return ONLY the names. Do not include any headings or counts."
            if attempt >= 3:
                system_prompt += f" Return EXACTLY {target_count} lines with ONLY the artist name on each line."

            user_parts = [f"Mood: {mood}"]
            if language:
                user_parts.append(f"Language: {language}")
            user_prompt = " | ".join(user_parts)

            try:
                logger.info(f"Requesting artists (attempt {attempt}/{attempts})")
                completion = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw = completion.choices[0].message.content if completion and completion.choices else ""
                artists = self._parse_artists(raw, target_count)
                if len(artists) == target_count:
                    return artists
                if attempt < attempts:
                    delay = min(self.config.backoff_initial * (2 ** (attempt - 1)), self.config.backoff_max)
                    time.sleep(delay)
            except Exception as e:
                logger.warning(f"Groq request failed on attempt {attempt}/{attempts}: {e}")
                if attempt < attempts:
                    delay = min(self.config.backoff_initial * (2 ** (attempt - 1)), self.config.backoff_max)
                    time.sleep(delay)
                else:
                    return []
        return []

    def _parse_artists(self, raw: str, target_count: int) -> List[str]:
        if not raw or not isinstance(raw, str):
            return []
        text = raw.strip()
        text = re.sub(r"^```[a-zA-Z0-9]*\s*|\s*```$", "", text, flags=re.MULTILINE)
        lines = [l for l in text.replace("\r", "").split("\n") if l.strip()]
        if len(lines) == 1 and ("," in lines[0] or " • " in lines[0] or " | " in lines[0]):
            sep = ","
            if " • " in lines[0]:
                sep = " • "
            if " | " in lines[0]:
                sep = " | "
            lines = [p for p in [x.strip() for x in lines[0].split(sep)] if p]

        cleaned = []
        seen = set()
        for line in lines:
            item = re.sub(r"^\s*(?:[\-\*\u2022•]|[0-9]{1,2}[.)])\s*", "", line).strip()
            item = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", item).strip()
            if any(prefix in item.lower() for prefix in ["artists:", "here are", "list:", "output", "names:"]):
                continue
            if not re.search(r"[A-Za-z\u00C0-\u024F\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]", item):
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(item)
            if len(cleaned) >= target_count:
                break

        if len(cleaned) > target_count:
            cleaned = cleaned[:target_count]
        return cleaned

def _load_config() -> Optional[GroqArtistConfig]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.error("GROQ_API_KEY not set")
        return None
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
    try:
        temperature = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    except ValueError:
        temperature = 0.1
    try:
        max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "256"))
    except ValueError:
        max_tokens = 256
    try:
        retries = int(os.getenv("GROQ_RETRIES", "3"))
    except ValueError:
        retries = 3
    return GroqArtistConfig(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
    )

def get_artists_from_groq(mood: str, language: Optional[str] = None) -> List[str]:
    cfg = _load_config()
    if not cfg:
        return []
    service = GroqMusicArtistService(cfg)
    return service.get_artists(mood=mood, language=language, count=10)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="groq_artists", add_help=True)
    parser.add_argument("mood", type=str)
    parser.add_argument("-l", "--language", type=str, default=None)
    parser.add_argument("-n", "--count", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    cfg = _load_config()
    if not cfg:
        print("[]")
        raise SystemExit(1)
    service = GroqMusicArtistService(cfg)
    artists = service.get_artists(args.mood, args.language, count=args.count)
    if args.json:
        print(json.dumps(artists, ensure_ascii=False))
    else:
        print("\n".join(artists))