"""
Job Search Service
Wraps POST /job/search with clean query-building helpers.
Integrates with investor and HR analysis flows.
"""

from __future__ import annotations
import logging
from typing import Optional

from crustdata_client import CrustDataClient, CrustDataError

logger = logging.getLogger(__name__)

# Job categories from CrustData
JOB_CATEGORIES = [
    "Engineering", "Sales", "Marketing", "Operations", "Finance",
    "Human Resources", "Product", "Design", "Data Science", "Legal",
    "Customer Success", "Business Development", "Research",
]


async def search_company_jobs(
    client: CrustDataClient,
    company_id: Optional[int] = None,
    company_name: Optional[str] = None,
    category: Optional[str] = None,
    title_keywords: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """
    Search active job listings for a company.
    Returns normalised dict with job list + hiring signal analysis.
    """
    if not company_id and not company_name:
        return {"jobs": [], "total": 0, "signals": {}}

    try:
        if company_id:
            raw = await client.job_search_by_company(company_id, category=category, limit=limit)
        else:
            raw = await client.job_search_by_company_name(company_name, title_keywords, limit=limit)

        jobs = _extract_jobs(raw)
        total = raw.get("total_count", 0) if isinstance(raw, dict) else len(jobs)
        signals = _analyze_hiring_signals(jobs, total)
        return {"jobs": jobs, "total": total, "signals": signals, "next_cursor": raw.get("next_cursor") if isinstance(raw, dict) else None}

    except CrustDataError as e:
        logger.warning("job_search failed for company=%s/%s: %s", company_id, company_name, e)
        return {"jobs": [], "total": 0, "signals": {}}


async def search_jobs_by_skills(
    client: CrustDataClient,
    skills: list[str],
    country: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """Search jobs matching a list of skill keywords."""
    if not skills:
        return {"jobs": [], "total": 0, "signals": {}}

    skill_regex = "|".join(skills[:10])
    conditions = [
        {"field": "job_details.description", "type": "(.)", "value": skill_regex}
    ]
    if country:
        conditions.append({"field": "location.country", "type": "=", "value": country})

    try:
        raw = await client.job_search(
            filters={"op": "and", "conditions": conditions},
            limit=limit,
        )
        jobs = _extract_jobs(raw)
        total = raw.get("total_count", 0) if isinstance(raw, dict) else len(jobs)
        return {"jobs": jobs, "total": total, "signals": _analyze_hiring_signals(jobs, total)}
    except CrustDataError as e:
        logger.warning("job_search_by_skills failed: %s", e)
        return {"jobs": [], "total": 0, "signals": {}}


# ── Response parsing ───────────────────────────────────────────────────────────

def _extract_jobs(raw) -> list[dict]:
    """Extract job listings from /job/search response."""
    if isinstance(raw, dict):
        listings = raw.get("job_listings") or raw.get("results") or raw.get("jobs") or []
    elif isinstance(raw, list):
        listings = raw
    else:
        return []
    return [_normalize_job(j) for j in listings if isinstance(j, dict)]


def _normalize_job(raw: dict) -> dict:
    """Flatten a job listing to a consistent internal shape."""
    jd  = raw.get("job_details") or {}
    loc = raw.get("location") or {}
    meta = raw.get("metadata") or {}
    comp = raw.get("company") or {}
    comp_info = comp.get("basic_info") or {}

    return {
        "title":        jd.get("title") or "Unknown",
        "url":          jd.get("url"),
        "category":     jd.get("category"),
        "description":  (jd.get("description") or "")[:500],
        "country":      loc.get("country"),
        "city":         loc.get("city"),
        "date_added":   meta.get("date_added"),
        "company_name": comp_info.get("name"),
        "company_id":   comp_info.get("company_id"),
        "is_remote":    loc.get("is_remote") or False,
    }


def _analyze_hiring_signals(jobs: list[dict], total: int) -> dict:
    """
    Derive hiring signal insights from a set of job listings.
    Returns dict ready for LLM context or chart data.
    """
    if not jobs:
        return {
            "total_open_roles": total,
            "category_breakdown": {},
            "top_titles": [],
            "top_locations": [],
            "hiring_velocity": "unknown",
        }

    # Category breakdown
    cat_counts: dict[str, int] = {}
    for j in jobs:
        cat = j.get("category") or "Other"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Top titles
    title_counts: dict[str, int] = {}
    for j in jobs:
        t = j.get("title") or ""
        if t:
            title_counts[t] = title_counts.get(t, 0) + 1
    top_titles = sorted(title_counts, key=lambda k: title_counts[k], reverse=True)[:10]

    # Top locations
    loc_counts: dict[str, int] = {}
    for j in jobs:
        c = j.get("country") or j.get("city") or "Unknown"
        loc_counts[c] = loc_counts.get(c, 0) + 1
    top_locations = sorted(loc_counts, key=lambda k: loc_counts[k], reverse=True)[:5]

    # Velocity label
    if total > 200:
        velocity = "aggressive"
    elif total > 50:
        velocity = "moderate"
    elif total > 10:
        velocity = "slow"
    else:
        velocity = "minimal"

    # Engineering ratio
    eng_count = cat_counts.get("Engineering", 0)
    eng_ratio = f"{eng_count / len(jobs) * 100:.0f}%" if jobs else "0%"

    return {
        "total_open_roles":    total,
        "category_breakdown":  dict(sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)),
        "top_titles":          top_titles,
        "top_locations":       top_locations,
        "hiring_velocity":     velocity,
        "engineering_ratio":   eng_ratio,
    }
