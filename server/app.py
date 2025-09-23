# server.py
from __future__ import annotations

import os
from typing import Any

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
    # If you prefer stateless requests, set stateless_http=True
    token_verifier=SimpleTokenVerifier(),
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(AUTH_ISSUER),
        resource_server_url=AnyHttpUrl(RESOURCE_SERVER_URL),
        required_scopes=REQUIRED_SCOPES,
    ),
)

# Mount Streamable HTTP at the root (i.e., the MCP endpoint is "/")
mcp.settings.streamable_http_path = "/"

# ---- Tools ----
def _openai_client() -> OpenAI:
    # If OPENAI_API_KEY is unset, OpenAI() will use env/ambient config if available.
    return OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

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


app = mcp.streamable_http_app()

# Add a simple health check endpoint for Render
try:
    from fastapi import FastAPI

    _outer_app = FastAPI()

    @_outer_app.get("/healthz")
    async def _healthz() -> dict[str, bool]:
        return {"ok": True}

    _outer_app.mount("/", app)
    app = _outer_app
except Exception:
    # If FastAPI is unavailable for any reason, continue without health route
    pass

if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8788")))
