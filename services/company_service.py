"""
Company Service — fixed per CrustData 2025-11-01 docs.

ROOT CAUSE of "Fetching leaders for: Unknown":
  _normalize_company was checking raw.get("company_name") only.
  The NEW 2025-11-01 /company/identify response uses NESTED structure:
    {"basic_info": {"name": "Stripe", "primary_domain": "stripe.com"}, "company_id": 12345}
  The OLD screener response uses FLAT structure:
    {"company_name": "Stripe", "company_website_domain": "stripe.com", "company_id": 12345}
  Fix: check BOTH shapes, with fallback to the original input name.

Additional fixes:
  - company_enrich uses fields=["basic_info","headcount","funding","locations","people"]
    to get people.founders directly (avoids separate person search)
  - keep original_name from request as final fallback
  - rich debug logging to surface raw response shape
"""

from __future__ import annotations
import json
import logging
from typing import Optional

from crustdata_client import CrustDataClient, CrustDataError

logger = logging.getLogger(__name__)


async def resolve_company(
    client: CrustDataClient,
    name: Optional[str] = None,
    domain: Optional[str] = None,
) -> dict:
    """
    Resolve a company name/domain to a CrustData entity.

    CONFIRMED response shape from /company/identify (2025-11-01):
      {
        "matched_on": "OpenAI",
        "match_type": "name",
        "matches": [
          {
            "confidence_score": 1.0,
            "company_data": {
              "crustdata_company_id": 631466,
              "basic_info": {"crustdata_company_id": 631466, "name": "OpenAI", ...}
            }
          }
        ]
      }

    This is the same envelope pattern as /person/enrich.
    """
    original_name = name or domain or "Unknown"

    # ── /company/identify (new versioned API) ─────────────────────────────────
    try:
        data = await client.company_identify(name=name, domain=domain)
        logger.info("[Company] identify response keys: %s | preview: %s",
                    list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                    str(data)[:300])

        company_raw = _extract_company_data(data)
        if company_raw:
            normalized = _normalize_company(company_raw, fallback_name=original_name)
            logger.info("[Company] Resolved '%s' → id=%s, name='%s', domain='%s'",
                        original_name, normalized.get("id"), normalized.get("name"), normalized.get("domain"))
            return normalized
    except CrustDataError as e:
        logger.warning("[Company] company_identify failed: %s — trying screener", e)

    # ── Screener fallback (old API) ───────────────────────────────────────────
    if domain:
        try:
            data = await client.screener_company(domain=domain)
            if isinstance(data, list): data = data[0] if data else {}
            normalized = _normalize_company(data, fallback_name=original_name)
            logger.info("[Company] Screener resolved '%s' → id=%s", domain, normalized.get("id"))
            return normalized
        except CrustDataError as e:
            logger.warning("[Company] screener fallback failed: %s", e)

    # Also try screener by name
    if name:
        try:
            data = await client.screener_company(domain=name.lower().replace(" ", "") + ".com")
            if isinstance(data, list): data = data[0] if data else {}
            if data and isinstance(data, dict) and data.get("company_id"):
                normalized = _normalize_company(data, fallback_name=original_name)
                if normalized.get("id"):
                    return normalized
        except CrustDataError:
            pass

    logger.warning("[Company] Could not resolve '%s' — using stub", original_name)
    return {"id": None, "name": original_name, "domain": domain, "raw": {}}


async def enrich_company(client: CrustDataClient, company: dict) -> dict:
    """
    Enrich company with full profile.
    /company/enrich now uses correct key: crustdata_company_ids
    /data_lab/* endpoints return 404/301/405 — skip them entirely.
    Only reliable endpoints: /company/enrich, /screener/company.
    """
    company_id = company.get("id")
    domain     = company.get("domain")
    enriched   = dict(company)

    # ── /company/enrich with people fields ────────────────────────────────────
    if company_id:
        try:
            data = await client.company_enrich_with_fields(
                company_id,
                fields=["basic_info", "headcount", "funding", "locations", "people"],
            )
            # company/enrich returns same envelope: [{matched_on, matches:[{company_data}]}]
            # OR may return {"companies": [...]} — try both
            company_raw = _extract_company_data(data)
            if not company_raw:
                # fallback: bare list of company objects
                companies = _extract_list(data, keys=["companies", "results", "data"])
                company_raw = companies[0] if companies else (data if isinstance(data, dict) else None)
            if company_raw and isinstance(company_raw, dict):
                enriched.update(_normalize_company(company_raw, fallback_name=company.get("name")))
                founders_count = len(enriched.get("people_founders") or [])
                logger.info("[Company] Enriched '%s': founders=%d, headcount=%s",
                            enriched.get("name"), founders_count, enriched.get("headcount"))
        except CrustDataError as e:
            logger.warning("[Company] company_enrich failed: %s", e)

    # ── /screener/company — rich flat data (CONFIRMED working) ────────────────
    if domain or company_id:
        try:
            sc = await client.screener_company(domain=domain, company_id=company_id if not domain else None)
            if isinstance(sc, list): sc = sc[0] if sc else {}
            if isinstance(sc, dict):
                enriched["screener"] = sc
                # Screener has headcount, funding, industry at top level
                if not enriched.get("headcount"):
                    enriched["headcount"] = sc.get("headcount") or sc.get("employee_count")
                if not enriched.get("total_funding_usd"):
                    enriched["total_funding_usd"] = sc.get("total_funding_raised_usd")
        except CrustDataError:
            pass

    return enriched


async def get_company_leaders(client: CrustDataClient, company: dict, max_founders: int = 5) -> list[dict]:
    """
    Fetch founders/leaders — 5-stage fallback, never raises.

    Stage 1: people.founders from company enrich data (free, zero API calls)
    Stage 2: /data_lab/decision_makers (plan-dependent, fast)
    Stage 3: Autocomplete exact name → person search with exact `in` match
    Stage 4: Regex `(.)` person search on company name (handles "Stripe, Inc." etc.)
    Stage 5: ALL employers title-regex search (widest net)
    """
    from services.person_service import (
        _extract_profiles,
        _normalize_person_from_search,
        FOUNDER_TITLE_REGEX,
    )

    company_id   = company.get("id")
    company_name = company.get("name") or ""

    if not company_name or company_name.lower() in ("unknown", ""):
        logger.warning("[Company] Skipping leader search — company name is '%s'", company_name)
        return []

    # ── Stage 1: people.founders from company enrich ──────────────────────────
    raw_people = (
        (company.get("people_founders") or []) +
        (company.get("people_decision_makers") or []) +
        (company.get("people_cxos") or [])
    )
    if raw_people:
        logger.debug("[Company] Stage-1 raw_people count: %d", len(raw_people))
        logger.debug("[Company] Stage-1 first person sample: %s", str(raw_people[0])[:200] if raw_people else "empty")
        normalized = [_normalize_person_from_company_people(p) for p in raw_people]
        logger.debug("[Company] Stage-1 normalized founders: %s", 
                    [{"name": n.get("name"), "title": n.get("title")} for n in normalized])
        leaders = _deduplicate_people(normalized)
        logger.info("[Company] Stage-1 company.people: %d leaders for '%s'", len(leaders), company_name)
        if leaders:
            return leaders[:max_founders]

    # ── Stage 2: DataLab decision_makers ──────────────────────────────────────
    if company_id:
        try:
            data = await client.screener_decision_makers(company_id)
            raw = _extract_list(data, keys=["decision_makers", "results", "profiles", "people"])
            if raw:
                leaders = _deduplicate_people(raw)
                logger.info("[Company] Stage-2 decision_makers: %d found", len(leaders))
                return leaders[:max_founders]
        except CrustDataError as e:
            logger.warning("[Company] Stage-2 failed: %s", e)

    # ── Stage 3: Autocomplete → filtered exact names → person search ─────────
    try:
        exact_names = await client.person_search_autocomplete(
            "experience.employment_details.current.company_name",
            query=company_name, limit=8,
        )
        # CRITICAL: filter to only names that are close matches
        # Prevents "OpenAirlines" contaminating an "OpenAI" search
        close = _filter_close_names(exact_names, company_name)
        logger.info("[Company] Stage-3 autocomplete '%s' → all=%s filtered=%s",
                    company_name, exact_names[:3], close[:3])
        search_names = close[:3] if close else [company_name]

        data = await client.person_search(
            filters={"op": "and", "conditions": [
                {"field": "experience.employment_details.current.company_name",
                 "type": "in", "value": search_names},
                {"field": "experience.employment_details.current.title",
                 "type": "(.)", "value": FOUNDER_TITLE_REGEX},
            ]},
            limit=10,
            sorts=[{"field": "professional_network.connections", "order": "desc"}],
        )
        profiles = _extract_profiles(data)
        leaders = [_normalize_person_from_search(p) for p in profiles]
        if leaders:
            leaders = _deduplicate_people(leaders)
            logger.info("[Company] Stage-3 exact match: %d leaders for '%s'", len(leaders), company_name)
            return leaders[:max_founders]
    except CrustDataError as e:
        logger.warning("[Company] Stage-3 failed: %s", e)

    # ── Stage 4: Use exact names from autocomplete, tight founder titles ──────
    try:
        ac_names = close if ("close" in dir() and close) else [company_name]
        safe_names = _filter_close_names(ac_names, company_name)[:3]

        # TIGHT founder titles only — Director/VP/Partner too broad, returns VCs
        FOUNDER_ONLY = "Founder|Co-Founder|CEO|CTO|CPO|Chief Executive|Chief Technology|President"
        data = await client.person_search(
            filters={"op": "and", "conditions": [
                {"field": "experience.employment_details.current.company_name",
                 "type": "in", "value": safe_names},
                {"field": "experience.employment_details.current.title",
                 "type": "(.)", "value": FOUNDER_ONLY},
            ]},
            limit=15,
            sorts=[{"field": "professional_network.connections", "order": "desc"}],
        )
        profiles = _extract_profiles(data)
        leaders = [_normalize_person_from_search(p) for p in profiles]
        leaders = _deduplicate_people(leaders)
        if leaders:
            logger.info("[Company] Stage-4 title-filtered: %d people for '%s'", len(leaders), company_name)
            return leaders[:max_founders]
    except CrustDataError as e:
        logger.warning("[Company] Stage-4 failed: %s", e)

    # ── Stage 5: Web search → extract LinkedIn URLs → person enrich ───────────
    try:
        leaders = await _founders_via_web_search(client, company_name)
        leaders = _deduplicate_people(leaders)
        if leaders:
            logger.info("[Company] Stage-5 web+enrich: %d leaders for '%s'", len(leaders), company_name)
            return leaders[:max_founders]
    except Exception as e:
        logger.warning("[Company] Stage-5 failed: %s", e)

    logger.warning("[Company] All 5 stages returned 0 leaders for '%s'", company_name)
    return []


async def _founders_via_web_search(client, company_name: str) -> list[dict]:
    """
    Fallback: web search for founders → extract LinkedIn URLs → enrich via /person/enrich.
    Used when person search returns 0 due to plan limits or indexing gaps.
    """
    import re as _re
    from services.person_service import _normalize_person_from_enrich, _extract_person_data_from_enrich

    try:
        web_data = await client.web_search(
            f"{company_name} founder CEO CTO LinkedIn",
            sources=["web"],
        )
    except CrustDataError:
        return []

    results = web_data.get("results") or [] if isinstance(web_data, dict) else []
    linkedin_urls = []
    seen = set()
    for item in results:
        text = (item.get("url") or "") + " " + (item.get("snippet") or "") + " " + (item.get("title") or "")
        for url in _re.findall(r"https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+", text):
            if url not in seen:
                seen.add(url)
                linkedin_urls.append(url)

    logger.info("[Company] Stage-5 found %d LinkedIn URLs for '%s'", len(linkedin_urls), company_name)

    if not linkedin_urls:
        return []

    try:
        enrich_data = await client.person_enrich_by_urls(linkedin_urls[:5])
        people = []
        for item in (enrich_data if isinstance(enrich_data, list) else [enrich_data]):
            if not isinstance(item, dict): continue
            pd = _extract_person_data_from_enrich([item], item.get("matched_on",""))
            if pd:
                from services.person_service import _normalize_person_minimal
                norm = _normalize_person_from_enrich(pd, {})
                if norm.get("name") and norm["name"] != "Unknown":
                    people.append(norm)
        return people
    except CrustDataError as e:
        logger.warning("[Company] person_enrich in Stage-5 failed: %s", e)
        return []


def _filter_close_names(names: list, query: str) -> list:
    """
    Filter autocomplete results to only names genuinely matching the query company.

    Strict rules (word-boundary only):
    1. Exact match: "OpenAI" == "OpenAI" ✅
    2. Name STARTS WITH query + word boundary:
       "Crustdata (YC F24)" starts with "Crustdata " ✅
       "OpenAI ChatGPT" starts with "OpenAI " ✅
       "OpenAirlines" starts with "OpenAI" but next char is "r" ❌
    3. Query starts with name (name is shorter abbreviation):
       query="openai inc", name="OpenAI" ✅
    4. REMOVED: word-in-middle matching caused false positives like
       "Generative AI Tutorials...OpenAI" matching "OpenAI" query
    """
    if not names:
        return []
    q = query.lower().strip()
    result = []
    for n in names:
        nl = n.lower().strip()
        # Rule 1: exact match
        if nl == q:
            result.append(n)
            continue
        # Rule 2: name starts with query (word boundary check)
        if nl.startswith(q):
            rest = nl[len(q):]
            if not rest or rest[0] in " ,(.!:;-_/":
                result.append(n)
                continue
        # Rule 3: query starts with name (name is a prefix/abbreviation of query)
        if q.startswith(nl) and len(nl) >= 4 and (len(q) == len(nl) or q[len(nl)] in " ,(.!"):
            result.append(n)
    return result if result else [query]


def _is_current_at(profile: dict, company_name: str) -> bool:
    """Check if a person is currently at the given company."""
    lower = company_name.lower()
    # New /person/search API shape
    exp = (profile.get("experience") or {}).get("employment_details") or {}
    for job in (exp.get("current") or []):
        if isinstance(job, dict):
            # NEW API: company is in "name" field; OLD: "company_name"
            cn = (job.get("name") or job.get("company_name") or job.get("company") or "").lower()
            if lower in cn or cn in lower:
                return True
    # Old screener shape — employer[]
    for job in (profile.get("employer") or []):
        if isinstance(job, dict) and not job.get("end_date"):
            cn = (job.get("company_name") or job.get("name") or "").lower()
            if lower in cn or cn in lower:
                return True
    return False


# ── Normaliser — handles BOTH new nested (basic_info) and old flat structures ──

def _extract_company_data(data) -> Optional[dict]:
    """
    Extract the raw company dict from any /company/identify response shape.

    CONFIRMED shape from logs (2025-11-01 API) — a BARE LIST of envelopes:
      [{"matched_on":"Crustdata","match_type":"name","matches":[{"confidence_score":1.0,"company_data":{...}}]}]

    Also handles:
      plain dict envelope: {"matched_on":"...","matches":[...]}
      dict with "companies" key: {"companies":[...]}
    """
    # Shape 1 (CONFIRMED): bare list of envelope objects
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            # Each list item IS the envelope
            return _extract_from_matches_envelope(item)

    # Shape 2: single dict envelope
    if isinstance(data, dict):
        if data.get("matches"):
            return _extract_from_matches_envelope(data)
        # Dict with companies/results wrapper
        for key in ("companies", "results", "data"):
            val = data.get(key)
            if isinstance(val, list) and val:
                item = val[0]
                if isinstance(item, dict):
                    return _extract_from_matches_envelope(item) or item.get("company_data") or item

    return None


def _extract_from_matches_envelope(item: dict) -> Optional[dict]:
    """
    Extract company_data from envelope: {matched_on, match_type, matches:[{confidence_score, company_data}]}
    """
    matches = item.get("matches")
    if not isinstance(matches, list) or not matches:
        return None
    best = max(matches, key=lambda m: m.get("confidence_score", 0) if isinstance(m, dict) else 0)
    if isinstance(best, dict):
        cd = best.get("company_data")
        if isinstance(cd, dict):
            return cd
    return None


def _normalize_company(raw: dict, fallback_name: str = "Unknown") -> dict:
    """
    Map CrustData company data to internal schema.

    CONFIRMED field names from logs:
      raw.crustdata_company_id = 631466
      raw.basic_info.crustdata_company_id = 631466
      raw.basic_info.name = "OpenAI"
      raw.basic_info.primary_domain = "openai.com"
    """
    if not isinstance(raw, dict):
        return {"id": None, "name": fallback_name, "domain": None, "raw": {}}

    bi   = raw.get("basic_info") or {}
    hc   = raw.get("headcount") or {}
    fund = raw.get("funding") or {}
    locs = raw.get("locations") or {}
    ppl  = raw.get("people") or {}

    # Company ID — CONFIRMED field name is crustdata_company_id
    raw_id = (
        raw.get("crustdata_company_id")     # CONFIRMED: top-level in company_data
        or bi.get("crustdata_company_id")   # CONFIRMED: also inside basic_info
        or bi.get("company_id")
        or raw.get("company_id")
        or raw.get("id")
    )
    try:
        company_id = int(raw_id) if raw_id is not None else None
    except (TypeError, ValueError):
        company_id = None

    name = (
        bi.get("name")
        or raw.get("company_name")
        or raw.get("name")
        or fallback_name
    )

    domain = (
        bi.get("primary_domain")
        or bi.get("website")
        or raw.get("company_website_domain")
        or raw.get("domain")
    )

    if company_id:
        logger.debug("[Company] Normalised id=%s name='%s' domain='%s'", company_id, name, domain)

    return {
        "id":                   company_id,
        "name":                 name,
        "domain":               domain,
        "linkedin_url":         bi.get("professional_network_url") or raw.get("linkedin_profile_url"),
        "industry":             bi.get("industries") or raw.get("industry"),
        "description":          bi.get("description") or raw.get("short_description"),
        "employee_count_range": bi.get("employee_count_range"),
        "headcount":            hc.get("total") or raw.get("headcount"),
        "headcount_growth_pct": hc.get("growth_percent"),
        "founded_year":         bi.get("year_founded") or raw.get("founded_year"),
        "hq_country":           locs.get("hq_country") or raw.get("hq_country"),
        "hq_city":              locs.get("hq_city"),
        "total_funding_usd":    fund.get("total_investment_usd") or raw.get("total_funding_raised_usd"),
        "last_round_type":      fund.get("last_round_type") or raw.get("latest_funding_stage"),
        "investors":            fund.get("investors") or [],
        "people_founders":       ppl.get("founders") or [],
        "people_decision_makers": ppl.get("decision_makers") or [],
        "people_cxos":           ppl.get("cxos") or [],
        "raw":                  raw,
    }


def _normalize_person_from_company_people(raw: dict) -> dict:
    """
    Normalise a person from company.people.founders / decision_makers / cxos.

    people.founders uses the SAME nested shape as /person/search:
      {"basic_profile": {"name": "Abhilash Chowdhary", "current_title": "Co-Founder & CEO"},
       "social_handles": {"professional_network_identifier": {"profile_url": "https://linkedin.com/in/..."}}}

    Also handles legacy flat shape:
      {"name": "...", "title": "...", "linkedin_profile_url": "..."}
    """
    if not isinstance(raw, dict):
        logger.warning("[Company] _normalize_person_from_company_people received non-dict: %s", type(raw))
        return {}

    bp  = raw.get("basic_profile") or {}
    soc = raw.get("social_handles") or {}

    # ENHANCED: Try multiple name extraction paths with logging
    name = (
        bp.get("name")                                          # nested (confirmed shape)
        or raw.get("name") or raw.get("full_name")              # flat
        or raw.get("first_last_name")
        or (raw.get("first_name", "") + " " + raw.get("last_name", "")).strip()
    ) or "Unknown"

    if name == "Unknown":
        logger.debug("[Company] Failed to extract name from founder data, keys: %s", list(raw.keys()))
        logger.debug("[Company] basic_profile keys: %s", list(bp.keys()) if isinstance(bp, dict) else type(bp))

    title = (
        bp.get("current_title") or bp.get("headline")          # nested
        or raw.get("title") or raw.get("current_title")         # flat
    )

    linkedin_url = (
        _deep_get(soc, "professional_network_identifier", "profile_url")  # nested
        or raw.get("linkedin_profile_url")                                  # flat
        or raw.get("flagship_profile_url")
        or raw.get("linkedin_url")
    )

    person_id = raw.get("crustdata_person_id") or raw.get("person_id") or raw.get("id")

    return {
        "name":        name,
        "title":       title,
        "linkedin_url": linkedin_url,
        "person_id":   person_id,
        "education":   [],
        "work_history": [],
        "skills":      [],
    }


def _deep_get(obj: dict, *keys: str):
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(obj, dict): return None
        obj = obj.get(key)
    return obj


def _deduplicate_people(people: list[dict]) -> list[dict]:
    """
    Deduplicate a list of person dicts by person_id → linkedin_url → name.
    Preserves first occurrence order.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for p in people:
        if not isinstance(p, dict):
            continue
        key = (
            str(p.get("person_id") or "")
            or (p.get("linkedin_url") or "").lower().rstrip("/")
            or (p.get("name") or "unknown").lower().strip()
        )
        if key and key not in seen and key != "unknown":
            seen.add(key)
            result.append(p)
        elif key == "unknown" or not key:
            result.append(p)  # keep unknowns without dedup
    return result


def _extract_list(data, keys: list[str]) -> list:
    """Extract list from bare list or wrapped dict."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in keys:
            val = data.get(key)
            if isinstance(val, list) and val:
                return val
    return []
