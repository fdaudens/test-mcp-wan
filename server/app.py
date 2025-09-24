# server.py
from __future__ import annotations

import os
from typing import Any
import re
from urllib.parse import urlparse, parse_qs

import datetime as dt
import httpx
import feedparser

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import AnyHttpUrl

from mcp.server.fastmcp import FastMCP
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

load_dotenv()

# ---- Configuration ----
PORT = int(os.getenv("PORT", "8788"))
RESOURCE_SERVER_URL = os.getenv("RESOURCE_SERVER_URL", f"http://localhost:{PORT}/")
AUTH_ISSUER = os.getenv("AUTH0_ISSUER", "https://dev-65wmmp5d56ev40iy.us.auth0.com/")
REQUIRED_SCOPES = os.getenv("REQUIRED_SCOPES", "user").split(",")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Not required for transcript tool

# ---- Simple token verifier (replace with real validation in prod) ----
class SimpleTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        # TODO: verify signature, issuer, expiry, audience/resource, scopes, etc.
        return AccessToken(
            token=token or "dev_token",
            client_id="dev_client",
            subject="dev",
            scopes=REQUIRED_SCOPES,
            claims={"debug": True},
        )

# ---- FastMCP server (no FastAPI/Uvicorn needed) ----
mcp = FastMCP(
    name="python-authenticated-mcp",
    instructions="Authenticated MCP server in Python. Implements `search` and `fetch` with OpenAI Vector Stores.",
    # Authentication disabled for now
        # If you prefer stateless requests, set stateless_http=True
    # token_verifier=SimpleTokenVerifier(),
    # auth=AuthSettings(
    #     issuer_url=AnyHttpUrl(AUTH_ISSUER),
    #     resource_server_url=AnyHttpUrl(RESOURCE_SERVER_URL),
    #     required_scopes=REQUIRED_SCOPES,
    # ),
)

# Mount Streamable HTTP at a dedicated path to avoid conflicts with root
mcp.settings.streamable_http_path = "/sse"

# ---- Tools ----
def _openai_client() -> OpenAI:
    # If OPENAI_API_KEY is unset, OpenAI() will use env/ambient config if available.
    return OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()


### YOUTUBE TRANSCRIPT (yt-dlp) ###
try:
    from yt_dlp import YoutubeDL  # type: ignore
except Exception:
    YoutubeDL = None  # type: ignore


def _extract_youtube_video_id(video: str) -> str | None:
    """
    Extract a YouTube video ID from a URL or return the ID if already provided.
    Supports standard, short, embed, and shorts URLs.
    """
    if not video:
        return None
    # Already an ID?
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", video):
        return video
    try:
        parsed = urlparse(video)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        if host in {"youtu.be", "www.youtu.be"}:
            candidate = path.lstrip("/").split("/")[0]
            return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None
        if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if path == "/watch":
                v = parse_qs(parsed.query or "").get("v", [None])[0]
                if v and re.fullmatch(r"[A-Za-z0-9_-]{11}", v):
                    return v
            for prefix in ("/embed/", "/shorts/"):
                if path.startswith(prefix):
                    candidate = path[len(prefix):].split("/")[0]
                    return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None
    except Exception:
        return None
    return None


@mcp.tool()
async def youtube_transcript(
    video: str,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """
    Fetch the transcript for a YouTube video using yt-dlp.

    Args:
    - video: A YouTube URL or 11-char video ID.
    - languages: Preferred languages (e.g., ["en","en-US"]). Fallbacks applied.

    Returns: {
      "videoId",
      "url",
      "language",  # language code if available
      "segments": [{"text","start","duration"}...],
      "text": joined string of transcript,
      "availableLanguages": [language codes]
    }

    Notes:
    - Does not require a YouTube API key.
    - Translation is not supported in this implementation.
    """
    if YoutubeDL is None:
        return {"error": "Missing dependency: yt-dlp. Ask the server admin to install it."}

    video_id = _extract_youtube_video_id(video)
    if not video_id:
        return {"error": "Invalid YouTube URL or video ID."}

    preferred_langs = list(languages or ["en", "en-US", "en-GB"])  # copy
    url = f"https://www.youtube.com/watch?v={video_id}"

    # yt-dlp options to extract subtitles/captions without downloading media
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "json3",  # request json3 if available
        "subtitleslangs": preferred_langs + ["en"],
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "dump_single_json": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:  # type: ignore
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        return {"videoId": video_id, "url": url, "segments": [], "text": "", "error": f"Failed to extract info: {e}"}

    # Captions are in info.get('automatic_captions') and/or info.get('subtitles') keyed by lang
    captions = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    # Build available languages list
    available_langs_set = set(captions.keys()) | set(auto.keys())
    available_languages = sorted(list(available_langs_set))

    # Pick a language per preference order
    chosen_lang = None
    for lang in preferred_langs:
        if lang in captions:
            chosen_lang = (lang, captions[lang])
            break
        if lang in auto:
            chosen_lang = (lang, auto[lang])
            break
    if chosen_lang is None:
        # fallback to any
        if captions:
            lang = next(iter(captions.keys()))
            chosen_lang = (lang, captions[lang])
        elif auto:
            lang = next(iter(auto.keys()))
            chosen_lang = (lang, auto[lang])
        else:
            return {"videoId": video_id, "url": url, "segments": [], "text": "", "availableLanguages": available_languages, "error": "No captions available."}

    language_code, variants = chosen_lang

    # yt-dlp provides variants with different formats; prefer json3 or ttml/srt fallback
    def _pick_variant(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not isinstance(items, list):
            return None
        # Prefer json3
        for it in items:
            if isinstance(it, dict) and (it.get("ext") == "json3" or "json3" in (it.get("format") or "")):
                return it
        # Then vtt
        for it in items:
            if isinstance(it, dict) and (it.get("ext") == "vtt"):
                return it
        # Then srt
        for it in items:
            if isinstance(it, dict) and (it.get("ext") == "srt"):
                return it
        # Finally any
        return items[0] if items else None

    variant = _pick_variant(variants)
    if not variant or not isinstance(variant, dict):
        return {"videoId": video_id, "url": url, "segments": [], "text": "", "availableLanguages": available_languages, "error": "No usable caption variant found."}

    # Download caption text via HTTP
    caption_url = variant.get("url")
    if not caption_url:
        return {"videoId": video_id, "url": url, "segments": [], "text": "", "availableLanguages": available_languages, "error": "Caption URL missing."}

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(caption_url)
            resp.raise_for_status()
            caption_body = resp.text or ""
    except httpx.HTTPStatusError as e:
        return {"videoId": video_id, "url": url, "segments": [], "text": "", "availableLanguages": available_languages, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"videoId": video_id, "url": url, "segments": [], "text": "", "availableLanguages": available_languages, "error": f"Failed to download captions: {e}"}

    # Parse json3 (if ext=json3) else do a best-effort parse for VTT/SRT
    segments: list[dict[str, Any]] = []
    try:
        if variant.get("ext") == "json3":
            import json
            data = json.loads(caption_body)
            # json3 has events with segments
            for ev in data.get("events", []) or []:
                if not isinstance(ev, dict):
                    continue
                if not ev.get("segs"):
                    continue
                # tStartMs and dDurationMs are in ms
                start = (ev.get("tStartMs") or 0) / 1000.0
                duration = (ev.get("dDurationMs") or 0) / 1000.0
                text_parts = []
                for seg in ev.get("segs", []) or []:
                    if isinstance(seg, dict) and seg.get("utf8"):
                        text_parts.append(seg.get("utf8"))
                text = ("".join(text_parts or [])).strip()
                if text:
                    segments.append({"text": text, "start": start, "duration": duration})
        else:
            # Minimal VTT/SRT parser to extract time and text blocks
            block_texts: list[str] = []
            current_lines: list[str] = []
            for line in caption_body.splitlines():
                if line.strip() == "":
                    if current_lines:
                        block_texts.append("\n".join(current_lines))
                        current_lines = []
                else:
                    current_lines.append(line.rstrip("\n"))
            if current_lines:
                block_texts.append("\n".join(current_lines))

            def _parse_time(ts: str) -> float:
                # Formats: HH:MM:SS.mmm or MM:SS.mmm
                ts = ts.strip()
                parts = ts.split(":")
                parts = [p for p in parts if p != ""]
                if len(parts) == 3:
                    h, m, s = parts
                elif len(parts) == 2:
                    h, m, s = "0", parts[0], parts[1]
                else:
                    return 0.0
                if "," in s:
                    s = s.replace(",", ".")
                return int(h) * 3600 + int(m) * 60 + float(s)

            for block in block_texts:
                lines = [l for l in block.splitlines() if l.strip()]
                if not lines:
                    continue
                # VTT may have a cue id in first line; time in second; SRT often has index line then time line
                time_idx = 0
                if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
                    time_idx = 1
                if "-->" not in lines[time_idx]:
                    # not a cue block
                    continue
                times = lines[time_idx]
                text_lines = lines[time_idx + 1 :]
                try:
                    start_str = times.split("-->")[0].strip()
                    end_str = times.split("-->")[1].strip().split(" ")[0]
                    start = _parse_time(start_str)
                    end = _parse_time(end_str)
                    duration = max(0.0, end - start)
                except Exception:
                    start = 0.0
                    duration = 0.0
                text = " ".join([t.strip() for t in text_lines if t.strip()])
                if text:
                    segments.append({"text": text, "start": start, "duration": duration})
    except Exception as e:
        return {"videoId": video_id, "url": url, "segments": [], "text": "", "availableLanguages": available_languages, "error": f"Failed to parse captions: {e}"}

    joined_text = " ".join([s.get("text") or "" for s in segments]).strip()

    return {
        "videoId": video_id,
        "url": url,
        "language": language_code,
        "segments": segments,
        "text": joined_text,
        "availableLanguages": available_languages,
    }

@mcp.tool()
async def search(query: str) -> dict[str, Any]:
    """
    Search for documents in the configured OpenAI Vector Store.
    Returns: {"results": [{"id","title","text","url"}...]}
    """
    results: list[dict[str, str]] = []
    if not query or not query.strip() or not VECTOR_STORE_ID:
        return {"results": results}

    client = _openai_client()

    try:
        # Prefer the current signature
        resp = client.vector_stores.search(
            VECTOR_STORE_ID,
            {"query": query, "ranking_options": {"score_threshold": 0.5}, "rewrite_query": True},
        )
        data = getattr(resp, "data", None) or []
    except Exception:
        # Fallback to keyword args in case of SDK shape differences
        try:
            resp = client.vector_stores.search(
                vector_store_id=VECTOR_STORE_ID,
                query=query,
                ranking_options={"score_threshold": 0.5},
                rewrite_query=True,
            )
            data = getattr(resp, "data", None) or []
        except Exception:
            data = []

    for i, item in enumerate(data):
        file_id = getattr(item, "file_id", None) or getattr(item, "id", None) or f"vs_{i}"
        filename = getattr(item, "filename", None) or f"Document {i+1}"
        content_list = getattr(item, "content", None) or []
        text_content = ""
        if content_list:
            first = content_list[0]
            if isinstance(first, dict) and "text" in first:
                text_content = first.get("text") or ""
            elif isinstance(first, str):
                text_content = first
        text_snippet = (text_content[:200] + "...") if len(text_content) > 200 else (text_content or "No content available")
        results.append(
            {
                "id": str(file_id),
                "title": str(filename),
                "text": text_snippet,
                "url": f"https://platform.openai.com/storage/files/{file_id}",
            }
        )

    return {"results": results}

@mcp.tool()
async def fetch(id: str) -> dict[str, Any]:
    """
    Fetch full content of a document by file ID from the OpenAI Vector Store.
    Returns: {"id","title","text","url","metadata":optional}
    """
    client = _openai_client()
    title = f"Document {id}"
    metadata: Any = None
    full_text = "No content available."

    if not id or not VECTOR_STORE_ID:
        return {"id": id, "title": title, "text": full_text, "url": f"https://platform.openai.com/storage/files/{id}", "metadata": metadata}

    try:
        # Retrieve content chunks
        content_resp = client.vector_stores.files.content(id, {"vector_store_id": VECTOR_STORE_ID})
        parts: list[str] = []
        for item in getattr(content_resp, "data", None) or []:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
        if parts:
            full_text = "\n".join(parts)

        # Optionally improve title/metadata
        try:
            file_info = client.vector_stores.files.retrieve(vector_store_id=VECTOR_STORE_ID, file_id=id)
            filename = getattr(file_info, "filename", None)
            if filename:
                title = filename
            attrs = getattr(file_info, "attributes", None)
            if attrs:
                metadata = attrs
        except Exception:
            pass

    except Exception:
        pass

    return {"id": id, "title": title, "text": full_text, "url": f"https://platform.openai.com/storage/files/{id}", "metadata": metadata}


### FACT CHECK ###

@mcp.tool()
async def fact_check_search(
    query: str,
    language_code: str | None = "en",
    page_size: int = 10,
    page_token: str | None = None,
) -> dict[str, Any]:
    """
    Search for fact checks using Google's Fact Check Tools API.
    Returns: {"results": [{"text","claimant","claimDate","reviews":[{"publisher","url","title","reviewDate","textualRating","languageCode"}]}], "nextPageToken": optional}

    Notes:
    - Requires the env var GOOGLE_FACT_CHECK_API_KEY to be set.
    - Accepts optional language code and pagination.
    """
    results: list[dict[str, Any]] = []

    if not query or not query.strip():
        return {"results": results}

    api_key = GOOGLE_FACT_CHECK_API_KEY or os.getenv("GOOGLE_FACT_CHECK_API_KEY")
    if not api_key:
        return {
            "results": results,
            "error": "Missing GOOGLE_FACT_CHECK_API_KEY. Set it in your environment to use this tool.",
        }

    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params: dict[str, Any] = {
        "query": query,
        "pageSize": max(1, min(int(page_size or 10), 50)),
        "key": api_key,
    }
    if language_code:
        params["languageCode"] = language_code
    if page_token:
        params["pageToken"] = page_token

    next_page_token: str | None = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json() or {}
            next_page_token = data.get("nextPageToken")
            for claim in data.get("claims", []) or []:
                reviews = []
                for r in claim.get("claimReview", []) or []:
                    publisher = None
                    pub = r.get("publisher") or {}
                    if isinstance(pub, dict):
                        publisher = pub.get("name")
                    reviews.append(
                        {
                            "publisher": publisher,
                            "url": r.get("url"),
                            "title": r.get("title"),
                            "reviewDate": r.get("reviewDate"),
                            "textualRating": r.get("textualRating"),
                            "languageCode": r.get("languageCode"),
                        }
                    )
                results.append(
                    {
                        "text": claim.get("text"),
                        "claimant": claim.get("claimant"),
                        "claimDate": claim.get("claimDate"),
                        "reviews": reviews,
                    }
                )
    except httpx.HTTPStatusError as e:
        return {"results": results, "error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"results": results, "error": f"Request failed: {e}"}

    out: dict[str, Any] = {"results": results}
    if next_page_token:
        out["nextPageToken"] = next_page_token
    return out






### RSS FEED ###

DEFAULT_FEED = "https://feeds.bbci.co.uk/news/technology/rss.xml"


def _norm_entry(entry: Any) -> dict[str, Any]:
    title = getattr(entry, "title", None) or entry.get("title")
    link = getattr(entry, "link", None) or entry.get("link")
    summary = getattr(entry, "summary", None) or entry.get("summary")
    published = getattr(entry, "published", None) or entry.get("published")
    published_iso = None
    try:
        ts = entry.get("published_parsed")
        if ts:
            published_iso = dt.datetime(*ts[:6]).isoformat()
    except Exception:
        pass
    return {
        "title": title,
        "link": link,
        "summary": summary,
        "published": published or published_iso,
    }


@mcp.tool()
async def rss_fetch(limit: int = 10) -> dict[str, Any]:
    """
    Fetch and return recent items from the BBC Technology RSS feed.

    Args:
    - limit: Max number of items to return (1..50). Defaults to 10.

    Returns: {"feed": {title}, "items": [{title, link, summary, published}...]}
    """
    url = DEFAULT_FEED

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; mcp-rss-tool/1.0; +https://modelcontextprotocol.io)",
            "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.9, */*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        }
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return {"items": [], "error": f"HTTP {resp.status_code}: unable to fetch feed"}
            text = resp.text

        parsed = feedparser.parse(text)
        bozo = bool(getattr(parsed, "bozo", False))
        bozo_err = getattr(parsed, "bozo_exception", None)

        items: list[dict[str, Any]] = []
        count = max(1, min(int(limit or 10), 50))
        for entry in (parsed.entries or [])[:count]:
            items.append(_norm_entry(entry))
        feed_title = None
        try:
            feed_title = getattr(parsed.feed, "title", None)
        except Exception:
            pass
        out: dict[str, Any] = {"feed": {"title": feed_title, "url": url}, "items": items}
        if bozo and not items:
            out["error"] = f"Invalid feed: {bozo_err}"
        elif bozo and items:
            out["warning"] = f"Parse warning: {bozo_err}"
        return out
    except Exception as e:
        return {"items": [], "error": f"Failed to fetch feed: {e}"}


### GUARDIAN ###

@mcp.tool()
async def guardian_search(
    query: str,
    section: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    page: int = 1,
    page_size: int = 10,
    order_by: str | None = None,
    show_fields: str | None = None,
) -> dict[str, Any]:
    """
    Search the Guardian Content API for articles.

    Args:
    - query: Search query string. Required.
    - section: Optional Guardian section ID (e.g., "technology").
    - from_date: Optional start date (YYYY-MM-DD).
    - to_date: Optional end date (YYYY-MM-DD).
    - page: Page number for pagination (>=1). Defaults to 1.
    - page_size: Results per page (1..50). Defaults to 10.
    - order_by: One of {"newest","oldest","relevance"}.
    - show_fields: Optional comma-separated fields to include (e.g., "byline,trailText").

    Returns: {"results": [{"id","title","url","section","published","type","pillar"}...],
              "pagination": {"currentPage","pages","pageSize","total","orderBy"},
              "nextPage": optional}

    Notes:
    - Requires the env var GUARDIAN_API_KEY to be set.
    """

    results: list[dict[str, Any]] = []

    if not query or not query.strip():
        return {"results": results}

    api_key = GUARDIAN_API_KEY or os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        return {
            "results": results,
            "error": "Missing GUARDIAN_API_KEY. Set it in your environment to use this tool.",
        }

    base_url = "https://content.guardianapis.com/search"
    params: dict[str, Any] = {
        "api-key": api_key,
        "q": query,
        "page": max(1, int(page or 1)),
        "page-size": max(1, min(int(page_size or 10), 50)),
    }
    if section:
        params["section"] = section
    if from_date:
        params["from-date"] = from_date
    if to_date:
        params["to-date"] = to_date
    if order_by:
        params["order-by"] = order_by
    if show_fields:
        params["show-fields"] = show_fields

    headers = {
        "User-Agent": "mcp-guardian-tool/1.0 (+https://modelcontextprotocol.io)",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers, follow_redirects=True) as client:
            resp = await client.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json() or {}
            response = data.get("response", {})
            status = response.get("status")
            if status != "ok":
                message = response.get("message") or status or "unknown error"
                return {"results": results, "error": f"Guardian API error: {message}"}

            for item in response.get("results", []) or []:
                results.append(
                    {
                        "id": item.get("id"),
                        "title": item.get("webTitle"),
                        "url": item.get("webUrl"),
                        "section": item.get("sectionName"),
                        "published": item.get("webPublicationDate"),
                        "type": item.get("type"),
                        "pillar": item.get("pillarName"),
                    }
                )

            pagination = {
                "currentPage": response.get("currentPage"),
                "pages": response.get("pages"),
                "pageSize": response.get("pageSize"),
                "total": response.get("total"),
                "orderBy": response.get("orderBy"),
            }

            out: dict[str, Any] = {"results": results, "pagination": pagination}
            try:
                cur = int(response.get("currentPage") or 0)
                pages = int(response.get("pages") or 0)
                if cur and pages and cur < pages:
                    out["nextPage"] = cur + 1
            except Exception:
                pass

            return out
    except httpx.HTTPStatusError as e:
        return {"results": results, "error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"results": results, "error": f"Request failed: {e}"}


app = mcp.streamable_http_app()

# Add health and test streaming directly to the MCP app to preserve its lifespan
try:
    import asyncio
    from starlette.responses import StreamingResponse, JSONResponse

    if hasattr(app, "include_router"):
        # FastAPI path
        from fastapi import APIRouter

        router = APIRouter()

        @router.get("/healthz")
        async def _healthz() -> dict[str, bool]:
            return {"ok": True}

        async def _event_generator():
            while True:
                yield "event: ping\ndata: ok\n\n"
                await asyncio.sleep(1)

        @router.get("/test-sse")
        async def _test_sse() -> StreamingResponse:
            return StreamingResponse(_event_generator(), media_type="text/event-stream")

        app.include_router(router)
    else:
        # Starlette path
        async def _healthz(request):
            return JSONResponse({"ok": True})

        async def _event_generator():
            while True:
                yield "event: ping\ndata: ok\n\n"
                await asyncio.sleep(1)

        async def _test_sse(request):
            return StreamingResponse(_event_generator(), media_type="text/event-stream")

        app.add_route("/healthz", _healthz)
        app.add_route("/test-sse", _test_sse)
except Exception:
    pass

if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8788")))
