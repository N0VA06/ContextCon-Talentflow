"""
CrustData API Client — Correct per official documentation (2025-11-01)

Verified endpoints and bodies:
─────────────────────────────────────────────────────────────────────────────
COMPANY
  POST /company/identify    {"names":["Stripe"]} or {"domains":["stripe.com"]}
                            → bare list of company objects
  POST /company/enrich      {"company_id":[123]}
  GET  /screener/company    ?company_domain=... or ?company_id=...

PERSON SEARCH  (POST /person/search)
  body: {
    "filters": {
      "op": "and",
      "conditions": [
        {"field": "experience.employment_details.current.company_name", "type": "in",  "value": ["Stripe"]},
        {"field": "experience.employment_details.current.title",        "type": "(.)", "value": "founder|CEO|CTO"}
      ]
    },
    "limit": 10
  }
  response: {"profiles": [...], "total_count": N, "next_cursor": "..."}

PERSON ENRICH  (POST /person/enrich)
  body: {"professional_network_profile_urls": ["https://linkedin.com/in/..."]}
     or {"business_emails": ["user@company.com"], "min_similarity_score": 0.8}
  response: [{matched_on, match_type, matches:[{confidence_score, person_data:{...}}]}]

JOB SEARCH  (POST /job/search)
  body: {
    "filters": {"op":"and","conditions":[
      {"field":"company.basic_info.company_id","type":"=","value":631394},
      {"field":"job_details.category","type":"=","value":"Engineering"}
    ]},
    "limit": 10,
    "sorts": [{"column":"metadata.date_added","order":"desc"}],
    "fields": ["job_details.title","job_details.url","metadata.date_added","location.country"]
  }
  response: {"job_listings":[...],"total_count":N,"next_cursor":"..."}

WEB SEARCH  (POST /screener/web-search)
  body: {"query":"...","geolocation":"US","sources":["web","news"]}
  response: {"success":true,"results":[{"title","url","snippet","position"},...]}

DATA LAB
  GET /data_lab/decision_makers                   ?company_id=...
  GET /data_lab/headcount_timeseries              ?company_id=...
  GET /data_lab/funding_milestone_timeseries      ?company_id=...
  GET /data_lab/job_openings_by_facet_timeseries  ?company_id=...&facet=title
  GET /employee_review/enrich                     ?company_id=...
─────────────────────────────────────────────────────────────────────────────
All requests send:
  authorization: Bearer <API_KEY>
  x-api-version: 2025-11-01
  content-type: application/json
"""

import logging
import time
from typing import Any, Optional
import httpx

from config import settings

logger = logging.getLogger(__name__)

BASE    = settings.CRUSTDATA_BASE_URL          # "https://api.crustdata.com"
VERSION = settings.CRUSTDATA_API_VERSION       # "2025-11-01"


class CrustDataClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or settings.CRUSTDATA_API_KEY
        if not self.api_key:
            raise ValueError("CRUSTDATA_API_KEY is missing — set it in .env")
        self.timeout = timeout
        self._headers = {
            "authorization": f"Bearer {self.api_key}",
            "x-api-version":  VERSION,
            "content-type":   "application/json",
            "accept":         "application/json",
        }

    # ── internal helpers ───────────────────────────────────────────────────────

    async def _post(self, path: str, body: dict) -> Any:
        url = f"{BASE}{path}"
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                resp = await c.post(url, headers=self._headers, json=body)
            logger.info("POST %-50s %d  (%.2fs)", path, resp.status_code, time.monotonic() - t0)
            if resp.status_code not in (200, 201):
                raise CrustDataError(f"{path} returned {resp.status_code}: {resp.text[:500]}")
            return resp.json()
        except httpx.RequestError as e:
            raise CrustDataError(f"Network error on POST {path}: {e}") from e

    async def _get(self, path: str, params: dict) -> Any:
        url = f"{BASE}{path}"
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                resp = await c.get(url, headers=self._headers, params=params)
            logger.info("GET  %-50s %d  (%.2fs)", path, resp.status_code, time.monotonic() - t0)
            if resp.status_code not in (200, 201):
                raise CrustDataError(f"{path} returned {resp.status_code}: {resp.text[:500]}")
            return resp.json()
        except httpx.RequestError as e:
            raise CrustDataError(f"Network error on GET {path}: {e}") from e

    # ── Company ────────────────────────────────────────────────────────────────

    async def company_identify(
        self,
        *,
        name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Any:
        """
        POST /company/identify
        Returns a bare list: [{company_id, company_name, ...}, ...]
        """
        body: dict[str, Any] = {}
        if domain:
            body["domains"] = [domain]
        elif name:
            body["names"] = [name]
        else:
            raise ValueError("Provide name or domain")
        return await self._post("/company/identify", body)

    async def company_enrich(self, company_id: int) -> Any:
        """
        POST /company/enrich — full company profile.
        CORRECT body key: crustdata_company_ids (not company_id)
        Confirmed from 400 error: 'crustdata_company_ids, names, domains, or professional_network_profile_urls'
        """
        return await self._post("/company/enrich", {"crustdata_company_ids": [company_id]})

    async def company_enrich_with_fields(self, company_id: int, fields: list[str]) -> Any:
        """
        POST /company/enrich with explicit fields.
        Object fields (founders, decision_makers) require explicit field request.
        """
        return await self._post("/company/enrich", {
            "crustdata_company_ids": [company_id],
            "fields": fields,
        })

    async def screener_company(
        self,
        domain: Optional[str] = None,
        company_id: Optional[int] = None,
    ) -> Any:
        """GET /screener/company — CONFIRMED working (200 OK)."""
        params: dict[str, Any] = {}
        if domain:       params["company_domain"] = domain
        elif company_id: params["company_id"] = company_id
        return await self._get("/screener/company", params)

    async def person_search_autocomplete(
        self,
        field: str,
        query: str = "",
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[str]:
        """
        POST /person/search/autocomplete
        Discovers valid indexed field values for Person Search filters.
        Use BEFORE building search filters to avoid 400 errors from invalid values.

        Common autocomplete-enabled fields:
          experience.employment_details.current.title
          experience.employment_details.current.company_name (or .name)
          experience.employment_details.current.seniority_level
          experience.employment_details.current.function_category
          experience.employment_details.current.company_industries
          education.schools.school
          skills.professional_network_skills
          basic_profile.location.country

        Returns list of exact string values to use in person_search filters.
        """
        body: dict[str, Any] = {"field": field, "query": query, "limit": limit}
        if filters:
            body["filters"] = filters
        try:
            data = await self._post("/person/search/autocomplete", body)
            suggestions = data.get("suggestions") or []
            return [s["value"] for s in suggestions if isinstance(s, dict) and s.get("value")]
        except CrustDataError as e:
            logger.warning("person_search_autocomplete failed for field=%s: %s", field, e)
            return []

    # ── Person Search (POST /person/search) ────────────────────────────────────
    # Filter format: {"op":"and","conditions":[{"field":"...","type":"...","value":"..."}]}
    # Field reference (key searchable fields):
    #   experience.employment_details.current.company_name  → current employer
    #   experience.employment_details.past.company_name     → past employer
    #   experience.employment_details.company_name          → current + past
    #   experience.employment_details.current.title         → current title (regex with "(.)")
    #   experience.employment_details.past.title            → past title
    #   education.schools.school                            → school name
    #   basic_profile.name                                  → full name
    #   basic_profile.location.country                      → country
    #   experience.employment_details.current.seniority_level → seniority

    async def person_search(
        self, filters: dict, limit: int = 10,
        cursor: Optional[str] = None,
        sorts: Optional[list] = None,
    ) -> Any:
        """
        POST /person/search
        Response: {"profiles":[...], "total_count":N, "next_cursor":"..."}
        sorts example: [{"field": "professional_network.connections", "order": "desc"}]
        """
        body: dict[str, Any] = {"filters": filters, "limit": limit}
        if cursor: body["cursor"] = cursor
        if sorts:  body["sorts"] = sorts
        return await self._post("/person/search", body)

    async def person_search_current_company(
        self,
        company_name: str,
        # TIGHTENED: only true founder-level titles; Director/VP/Partner too broad
        title_regex: str = "Founder|Co-Founder|CEO|CTO|CPO|Chief Executive|Chief Technology|President",
        limit: int = 10,
    ) -> Any:
        """
        Find current founders/execs at a company.
        Sorts by connections DESC — founders have the most connections,
        so Patrick Collison appears before random VPs.
        """
        return await self.person_search(
            filters={
                "op": "and",
                "conditions": [
                    {"field": "experience.employment_details.current.company_name",
                     "type": "in", "value": [company_name]},
                    {"field": "experience.employment_details.current.title",
                     "type": "(.)", "value": title_regex},
                ]
            },
            limit=limit,
            sorts=[{"field": "professional_network.connections", "order": "desc"}],
        )

    async def person_search_past_company(
        self, company_name: str, title_regex: str = "founder|CEO|CTO|CPO", limit: int = 10
    ) -> Any:
        """Find alumni (past employees) now in founder/exec roles."""
        return await self.person_search({
            "op": "and",
            "conditions": [
                {"field": "experience.employment_details.past.company_name", "type": "in",  "value": [company_name]},
                {"field": "experience.employment_details.current.title",     "type": "(.)", "value": title_regex},
            ]
        }, limit=limit)

    async def person_search_school(
        self, school_name: str, title_regex: str = "founder|CEO|CTO|director|VP", limit: int = 10
    ) -> Any:
        """Find alumni of a school now in senior roles."""
        return await self.person_search({
            "op": "and",
            "conditions": [
                {"field": "education.schools.school",                    "type": "(.)", "value": school_name},
                {"field": "experience.employment_details.current.title", "type": "(.)", "value": title_regex},
            ]
        }, limit=limit)

    # ── Person Enrich (POST /person/enrich) ────────────────────────────────────
    # CRITICAL: parameter name is "professional_network_profile_urls" (array)
    # Response: [{matched_on, match_type, matches:[{confidence_score, person_data:{...}}]}]

    async def person_enrich_by_urls(self, linkedin_urls: list[str]) -> Any:
        """
        POST /person/enrich with professional_network_profile_urls.
        Returns array of enrich results.  Max 25 URLs per call.
        """
        if not linkedin_urls:
            return []
        return await self._post("/person/enrich", {
            "professional_network_profile_urls": linkedin_urls[:25]
        })

    async def person_enrich_by_emails(
        self, emails: list[str], min_similarity: float = 0.8
    ) -> Any:
        """POST /person/enrich with business_emails reverse lookup."""
        if not emails:
            return []
        return await self._post("/person/enrich", {
            "business_emails": emails[:25],
            "min_similarity_score": min_similarity,
        })

    # ── Job Search (POST /job/search) ─────────────────────────────────────────
    # Requires x-api-version header (same as others).
    # Filter format: {"op":"and","conditions":[{"field":"...","type":"=","value":"..."}]}
    # Key filter fields:
    #   company.basic_info.company_id   (int)
    #   company.basic_info.name         (string)
    #   job_details.title               (string, supports "(.)" regex)
    #   job_details.category            (Engineering, Sales, Marketing, etc.)
    #   job_details.description         (string, supports "(.)" regex)
    #   location.country                (string)
    #   metadata.date_added             (date string "YYYY-MM-DD")
    # Response: {"job_listings":[{job_details:{title,url,...},location:{...},metadata:{...}}],
    #            "total_count":N, "next_cursor":"..."}

    async def job_search(
        self,
        filters: dict,
        limit: int = 10,
        sorts: Optional[list[dict]] = None,
        fields: Optional[list[str]] = None,
        cursor: Optional[str] = None,
    ) -> Any:
        """
        POST /job/search — find active job listings.
        Default sorts by date_added desc (most recent first).
        """
        body: dict[str, Any] = {
            "filters": filters,
            "limit":   limit,
            "sorts":   sorts or [{"column": "metadata.date_added", "order": "desc"}],
        }
        if fields:
            body["fields"] = fields
        if cursor:
            body["cursor"] = cursor
        return await self._post("/job/search", body)

    async def job_search_by_company(
        self,
        company_id: int,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> Any:
        """Find jobs at a specific company, optionally filtered by category."""
        conditions: list[dict] = [
            {"field": "company.basic_info.company_id", "type": "=", "value": company_id}
        ]
        if category:
            conditions.append({"field": "job_details.category", "type": "=", "value": category})
        return await self.job_search(
            filters={"op": "and", "conditions": conditions},
            limit=limit,
            fields=["job_details.title", "job_details.url", "job_details.category",
                    "metadata.date_added", "location.country"],
        )

    async def job_search_by_company_name(
        self,
        company_name: str,
        title_keywords: Optional[str] = None,
        limit: int = 10,
    ) -> Any:
        """Find jobs at a company by name (slower than ID-based search)."""
        conditions: list[dict] = [
            {"field": "company.basic_info.name", "type": "(.)", "value": company_name}
        ]
        if title_keywords:
            conditions.append(
                {"field": "job_details.title", "type": "(.)", "value": title_keywords}
            )
        return await self.job_search(
            filters={"op": "and", "conditions": conditions},
            limit=limit,
        )

    # ── Web Search (POST /screener/web-search) ────────────────────────────────

    async def web_search(
        self,
        query: str,
        sources: Optional[list[str]] = None,
        geolocation: str = "US",
    ) -> Any:
        """
        POST /screener/web-search
        Response: {"success":true,"results":[{"title","url","snippet","position"},...]}
        Credits: 1 per query. Free tier: 5 total.
        """
        return await self._post("/screener/web-search", {
            "query":       query,
            "geolocation": geolocation,
            "sources":     sources or ["web", "news"],
        })

    # ── Data Lab ──────────────────────────────────────────────────────────────
    # NOTE: /data_lab/* endpoints return 404/301/405 on this plan — disabled.
    # All callers guard with try/except CrustDataError so these raise safely.

    async def screener_decision_makers(self, company_id: int) -> Any:
        """Disabled — /data_lab/decision_makers returns 404 on this plan."""
        raise CrustDataError("/data_lab/decision_makers not available on this plan (404)")

    async def screener_headcount_timeseries(self, company_id: int) -> Any:
        """Disabled — /data_lab/headcount_timeseries returns 404 on this plan."""
        raise CrustDataError("/data_lab/headcount_timeseries not available (404)")

    async def screener_funding_timeseries(self, company_id: int) -> Any:
        """Disabled — /data_lab/funding_milestone_timeseries returns 301 on this plan."""
        raise CrustDataError("/data_lab/funding_milestone_timeseries not available (301)")

    async def screener_job_openings_timeseries(self, company_id: int, facet: str = "title") -> Any:
        """Disabled — returns 404 on this plan."""
        raise CrustDataError("/data_lab/job_openings_by_facet_timeseries not available (404)")

    async def employee_reviews(self, company_id: int) -> Any:
        """Disabled — /employee_review/enrich returns 405 on this plan."""
        raise CrustDataError("/employee_review/enrich not available on this plan (405)")


class CrustDataError(Exception):
    """Raised on any CrustData API failure."""
