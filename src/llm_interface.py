from __future__ import annotations
import os, re, json
from typing import Dict, Any, Optional, List
from .config import OPENAI_API_KEY, OPENAI_MODEL

def _fallback_parse(query: str) -> Dict[str, Any]:
    # naive heuristics: look for 'like <Title>' and genres keywords
    genres = ["action","comedy","drama","thriller","romance","sci-fi","horror","animation","adventure","fantasy","crime","mystery"]
    lower = query.lower()
    picked = [g for g in genres if g in lower]
    seed = None
    m = re.search(r"like\s+([\w\s:'\-\(\)]+)", query, re.IGNORECASE)
    if m:
        seed = m.group(1).strip().rstrip(".!?")
    k = 10
    mk = re.search(r"k\s*=\s*(\d+)", query, re.IGNORECASE)
    if mk:
        k = int(mk.group(1))
    intent = "similar" if "like " in lower or "similar" in lower else "recommend"
    return {"intent": intent, "seed_movie": seed, "genres": picked, "k": k}

def parse_with_openai(query: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return _fallback_parse(query)
    try:
        # Lazy import to avoid hard dep if key not set
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        system = (
            "You translate movie queries into JSON with fields: "
            "{intent: 'recommend'|'similar'|'filter', seed_movie: string|null, genres: string[], k: number}. "
            "Be terse; respond with ONLY JSON."
        )
        user = f"Query: {query}"
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            response_format={"type":"json_object"},
            temperature=0
        )
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        # light validation
        data.setdefault("intent", "recommend")
        data.setdefault("genres", [])
        data.setdefault("seed_movie", None)
        data.setdefault("k", 10)
        return data
    except Exception:
        return _fallback_parse(query)
