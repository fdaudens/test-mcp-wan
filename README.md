# Authenticated MCP Server Scaffold (Python)

This directory contains a production‑ready scaffold for building an authenticated [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) server in Python that you can register as a ChatGPT connector.

It mirrors the TypeScript version in `authenticated-mcp-server-scaffold`, but uses the official Python MCP SDK.

- Implements the required MCP tools: **`search`** and **`fetch`**; plus **`fact_check_search`** using Google's Fact Check Tools API, **`rss_fetch`** for RSS feeds, and **`youtube_transcript`** for YouTube captions (via yt-dlp).
- Secures access using **OAuth 2.1** with an **Auth0** authorization server.
- Runs locally with FastAPI and the Python MCP SDK’s HTTP streaming transport.

---

## 1. What you are building

Your server:

- Exposes a `search` tool that queries an OpenAI Vector Store and returns results.
- Exposes a `fetch` tool that returns full document content by ID.
- Exposes a `fact_check_search` tool that queries Google Fact Check Tools for claim reviews.
- Requires a valid access token issued by your Auth0 tenant before executing tools.

---

## 2. Clone & install

```bash
git clone org-14957082@github.com:openai/mcp-in-a-box.git
cd mcp-in-a-box/python-authenticated-mcp-server-scaffold
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Auth0 setup (one-time)

1) In Auth0 Dashboard, create an **API** (APIs → Create API). Note the **Identifier**; use it as `AUTH0_AUDIENCE`.

2) Enable RBAC and “Add Permissions in the Access Token” in your API → Settings. Add a permission such as `user`.

3) Ensure your tenant exposes discovery and JWKS (default). The issuer is your tenant domain URL, e.g.:  
`https://dev-65wmmp5d56ev40iy.us.auth0.com/`

4) Ensure at least one Connection (Database/Social/Enterprise) is enabled for any client that will authenticate.

---

## 4. Configure environment

We support a `.env` file (loaded automatically).

- Start with the sample file and copy it:
```bash
cp env.example .env
```
- Then edit `.env` and fill in your values.

`.env` is already ignored by `.gitignore` in this folder.

Required keys:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key
VECTOR_STORE_ID=your_vector_store_id

# Auth0 (Authorization Server)
AUTH0_ISSUER=https://YOUR_AUTH0_DOMAIN/
AUTH0_AUDIENCE=https://YOUR_API_IDENTIFIER

# Local server
PORT=8788
RESOURCE_SERVER_URL=http://localhost:8788 # update this do your MCP server deployed domain when you are deploying in production.
```

Optional keys (for additional tools):

```bash
# Google Fact Check Tools API (for `fact_check_search`)
GOOGLE_FACT_CHECK_API_KEY=your_google_api_key

# YouTube API (optional; transcript tool does not require it)
# YOUTUBE_API_KEY=your_youtube_api_key
```

---

## 5. Run locally

```bash
uv run server/app.py
```

The server exposes MCP over HTTP streaming at:

```
http://localhost:8788
```

---

## 6. Test with MCP Inspector

1. Keep the server running.  
2. In a new terminal:

```bash
npx @modelcontextprotocol/inspector@latest
```

3. In the Inspector UI:  
- Transport Type: **HTTP streaming**  
- Server URL: `http://localhost:8788`
- Open Auth Settings: enter your Auth0 tenant issuer if prompted.
- Connect → should turn green.
- List Tools → should show `search`, `fetch`, `fact_check_search`, and `rss_fetch`.
- Call `search` with a test query, then `fetch` using a returned `id`.
 - Call `fact_check_search` with a query like: `"vaccine microchips"` (requires `GOOGLE_FACT_CHECK_API_KEY`).

---

## 7. What the tools do

- `search(query: string)` → queries your OpenAI Vector Store and returns a list of `{ id, title, text, url }`. Keep these IDs stable so `fetch` can resolve them.
- `fetch(id: string)` → returns the full text for a given file ID using the OpenAI Vector Store Files API.
- `fact_check_search(query: string, language_code?: string = "en", page_size?: number = 10, page_token?: string)` → searches Google Fact Check Tools for claims and returns `{ text, claimant, claimDate, reviews: [{ publisher, url, title, reviewDate, textualRating, languageCode }] }` and an optional `nextPageToken`.
 - `rss_fetch(limit?: number = 10)` → returns recent items from the BBC Technology RSS feed: `https://feeds.bbci.co.uk/news/technology/rss.xml`.
 - `youtube_transcript(video: string, languages?: string[], cookies_from_browser?: string, browser_profile?: string, use_keyring?: boolean, cookie_file_path?: string, cookie_string?: string)` → fetches captions via yt-dlp. Returns `{ videoId, url, language, segments, text, availableLanguages, needsCookies?, tips? }`. No translation; no API key required. When YouTube requires login/bot check, pass cookies using one of:
   - `cookies_from_browser`: e.g., `"chrome"`, `"firefox"`, `"safari"` (optionally `browser_profile`, `use_keyring`).
   - `cookie_file_path`: path to a Netscape `cookies.txt` file.
   - `cookie_string`: raw `Cookie` header string as a last resort.

Both tools require a valid access token. The server enforces token verification with Auth0’s JWKS and checks the `user` permission (scope) if present.

---

## 8. Code tour

- `server/app.py`
  - Configures the MCP server via `FastMCP`.
  - Declares `search`, `fetch`, and `fact_check_search` tools and routes tool calls.

- `server/tools/fact_check.py`: Tool module.

---

## 9. Production notes

- Host behind HTTPS and set `RESOURCE_SERVER_URL` to your public URL.  
- Consider caching JWKS and adding retry/backoff logic.  
- Add fine‑grained authorization checks (e.g., entitlements) in tool handlers.

---

## 10. Troubleshooting

- `401 Unauthorized`: Ensure the client is sending `Authorization: Bearer <token>` and that `iss` and `aud` match your Auth0 tenant and API Identifier.
- Empty `search` results: Verify `VECTOR_STORE_ID` and that your vector store contains files (see `python-rss-data-pipeline`).

---

## 11. License

MIT
