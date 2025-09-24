# server.py
from __future__ import annotations

import os
from typing import Any

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


# =====================
# Google Analytics MCP
# =====================

# Notes:
# - These tools use Application Default Credentials (ADC). Ensure one of the following:
#   - Set GOOGLE_APPLICATION_CREDENTIALS to a JSON credentials file path, OR
#   - Run `gcloud auth application-default login` and enable the Analytics APIs.
# - Required APIs: Google Analytics Admin API, Google Analytics Data API


def _format_ga_dimension_or_metric(value: str) -> str:
    return value.strip()


@mcp.tool()
async def analytics_get_account_summaries() -> dict[str, Any]:
    """
    List Google Analytics account and property summaries for the authenticated user.

    Returns: {"accounts": [{"account": str, "displayName": str, "properties": [{"property": str, "displayName": str}] }]}
    """
    try:
        # Import inside function to avoid hard dependency at import time
        from google.analytics.admin_v1beta import AnalyticsAdminServiceClient
    except Exception as e:
        return {"accounts": [], "error": f"Missing Google Analytics Admin client: {e}"}

    try:
        client = AnalyticsAdminServiceClient()
        results: list[dict[str, Any]] = []
        for summary in client.list_account_summaries():
            props = []
            for p in summary.property_summaries:
                props.append({
                    "property": getattr(p, "property", None),
                    "displayName": getattr(p, "display_name", None),
                })
            results.append({
                "account": getattr(summary, "account", None),
                "displayName": getattr(summary, "display_name", None),
                "properties": props,
            })
        return {"accounts": results}
    except Exception as e:
        return {"accounts": [], "error": f"Failed to list account summaries: {e}"}


@mcp.tool()
async def analytics_get_property_details(property_id: str) -> dict[str, Any]:
    """
    Get details for a GA4 property.

    Args:
    - property_id: Numeric property ID (e.g., "123456789") or full name ("properties/123456789").
    """
    try:
        from google.analytics.admin_v1beta import AnalyticsAdminServiceClient
    except Exception as e:
        return {"error": f"Missing Google Analytics Admin client: {e}"}

    if not property_id:
        return {"error": "property_id is required"}
    name = property_id if str(property_id).startswith("properties/") else f"properties/{property_id}"

    try:
        client = AnalyticsAdminServiceClient()
        p = client.get_property(name=name)
        return {
            "name": getattr(p, "name", None),
            "propertyId": getattr(p, "name", "").split("/")[-1] if getattr(p, "name", None) else None,
            "displayName": getattr(p, "display_name", None),
            "currencyCode": getattr(p, "currency_code", None),
            "timeZone": getattr(p, "time_zone", None),
            "industryCategory": getattr(p, "industry_category", None),
            "serviceLevel": getattr(p, "service_level", None),
            "createTime": getattr(p, "create_time", None).isoformat() if getattr(p, "create_time", None) else None,
            "updateTime": getattr(p, "update_time", None).isoformat() if getattr(p, "update_time", None) else None,
        }
    except Exception as e:
        return {"error": f"Failed to get property: {e}"}


@mcp.tool()
async def analytics_list_google_ads_links(property_id: str) -> dict[str, Any]:
    """
    List Google Ads links for a given GA4 property.

    Args:
    - property_id: Numeric ID or "properties/{id}".
    """
    try:
        from google.analytics.admin_v1beta import AnalyticsAdminServiceClient
    except Exception as e:
        return {"links": [], "error": f"Missing Google Analytics Admin client: {e}"}

    if not property_id:
        return {"links": [], "error": "property_id is required"}
    parent = property_id if str(property_id).startswith("properties/") else f"properties/{property_id}"

    try:
        client = AnalyticsAdminServiceClient()
        links: list[dict[str, Any]] = []
        for link in client.list_google_ads_links(parent=parent):
            links.append({
                "name": getattr(link, "name", None),
                "customerId": getattr(link, "customer_id", None),
                "canManageClients": getattr(link, "can_manage_clients", None),
                "adsPersonalizationEnabled": getattr(link, "ads_personalization_enabled", None),
                "emailAddress": getattr(link, "email_address", None),
            })
        return {"links": links}
    except Exception as e:
        return {"links": [], "error": f"Failed to list Google Ads links: {e}"}


@mcp.tool()
async def analytics_get_custom_dimensions_and_metrics(property_id: str) -> dict[str, Any]:
    """
    Retrieve custom dimensions and custom metrics for a GA4 property.

    Args:
    - property_id: Numeric ID or "properties/{id}".
    """
    try:
        from google.analytics.admin_v1beta import AnalyticsAdminServiceClient
    except Exception as e:
        return {"customDimensions": [], "customMetrics": [], "error": f"Missing Google Analytics Admin client: {e}"}

    if not property_id:
        return {"customDimensions": [], "customMetrics": [], "error": "property_id is required"}
    parent = property_id if str(property_id).startswith("properties/") else f"properties/{property_id}"

    try:
        client = AnalyticsAdminServiceClient()
        dims = []
        mets = []
        for d in client.list_custom_dimensions(parent=parent):
            dims.append({
                "name": getattr(d, "name", None),
                "parameterName": getattr(d, "parameter_name", None),
                "displayName": getattr(d, "display_name", None),
                "scope": getattr(d, "scope", None),
                "disallowAdsPersonalization": getattr(d, "disallow_ads_personalization", None),
            })
        for m in client.list_custom_metrics(parent=parent):
            mets.append({
                "name": getattr(m, "name", None),
                "parameterName": getattr(m, "parameter_name", None),
                "displayName": getattr(m, "display_name", None),
                "measurementUnit": getattr(m, "measurement_unit", None),
                "scope": getattr(m, "scope", None),
            })
        return {"customDimensions": dims, "customMetrics": mets}
    except Exception as e:
        return {"customDimensions": [], "customMetrics": [], "error": f"Failed to list custom resources: {e}"}


@mcp.tool()
async def analytics_run_report(
    property_id: str,
    dimensions: list[str],
    metrics: list[str],
    start_date: str,
    end_date: str,
    limit: int | None = 100,
) -> dict[str, Any]:
    """
    Run a GA4 core report.

    Args:
    - property_id: Numeric ID or "properties/{id}".
    - dimensions: List of dimension names (e.g., ["country", "city"]).
    - metrics: List of metric names (e.g., ["activeUsers"]).
    - start_date, end_date: Date strings like "2024-01-01" or relative like "7daysAgo".
    - limit: Max rows to return (default 100).
    """
    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Dimension, Metric
    except Exception as e:
        return {"rows": [], "error": f"Missing Google Analytics Data client: {e}"}

    if not property_id:
        return {"rows": [], "error": "property_id is required"}
    if not dimensions or not metrics:
        return {"rows": [], "error": "dimensions and metrics are required"}

    name = property_id if str(property_id).startswith("properties/") else f"properties/{property_id}"

    try:
        client = BetaAnalyticsDataClient()
        req = RunReportRequest(
            property=name,
            dimensions=[Dimension(name=_format_ga_dimension_or_metric(d)) for d in dimensions],
            metrics=[Metric(name=_format_ga_dimension_or_metric(m)) for m in metrics],
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            limit=max(1, int(limit or 100)),
        )
        resp = client.run_report(req)

        dim_headers = [h.name for h in resp.dimension_headers]
        met_headers = [h.name for h in resp.metric_headers]
        rows: list[dict[str, Any]] = []
        for r in resp.rows:
            row: dict[str, Any] = {}
            for i, dv in enumerate(r.dimension_values):
                if i < len(dim_headers):
                    row[dim_headers[i]] = dv.value
            for j, mv in enumerate(r.metric_values):
                if j < len(met_headers):
                    row[met_headers[j]] = mv.value
            rows.append(row)

        return {
            "rowCount": getattr(resp, "row_count", len(rows)),
            "rows": rows,
            "totals": [
                {met_headers[i]: tv.value for i, tv in enumerate(t.totals)}
                for t in getattr(resp, "totals", [])
            ] if getattr(resp, "totals", None) else None,
            "metadata": {
                "dimensionHeaders": dim_headers,
                "metricHeaders": met_headers,
            },
        }
    except Exception as e:
        return {"rows": [], "error": f"Failed to run report: {e}"}


@mcp.tool()
async def analytics_run_realtime_report(
    property_id: str,
    dimensions: list[str],
    metrics: list[str],
    limit: int | None = 100,
) -> dict[str, Any]:
    """
    Run a GA4 realtime report.

    Args:
    - property_id: Numeric ID or "properties/{id}".
    - dimensions, metrics: Lists of field names.
    - limit: Max rows to return (default 100).
    """
    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import RunRealtimeReportRequest, Dimension, Metric
    except Exception as e:
        return {"rows": [], "error": f"Missing Google Analytics Data client: {e}"}

    if not property_id:
        return {"rows": [], "error": "property_id is required"}
    if not dimensions or not metrics:
        return {"rows": [], "error": "dimensions and metrics are required"}

    name = property_id if str(property_id).startswith("properties/") else f"properties/{property_id}"

    try:
        client = BetaAnalyticsDataClient()
        req = RunRealtimeReportRequest(
            property=name,
            dimensions=[Dimension(name=_format_ga_dimension_or_metric(d)) for d in dimensions],
            metrics=[Metric(name=_format_ga_dimension_or_metric(m)) for m in metrics],
            limit=max(1, int(limit or 100)),
        )
        resp = client.run_realtime_report(req)

        dim_headers = [h.name for h in resp.dimension_headers]
        met_headers = [h.name for h in resp.metric_headers]
        rows: list[dict[str, Any]] = []
        for r in resp.rows:
            row: dict[str, Any] = {}
            for i, dv in enumerate(r.dimension_values):
                if i < len(dim_headers):
                    row[dim_headers[i]] = dv.value
            for j, mv in enumerate(r.metric_values):
                if j < len(met_headers):
                    row[met_headers[j]] = mv.value
            rows.append(row)

        return {
            "rowCount": getattr(resp, "row_count", len(rows)),
            "rows": rows,
            "metadata": {
                "dimensionHeaders": dim_headers,
                "metricHeaders": met_headers,
            },
        }
    except Exception as e:
        return {"rows": [], "error": f"Failed to run realtime report: {e}"}


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
