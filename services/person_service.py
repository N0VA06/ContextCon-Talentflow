"""
Person Service — correct per CrustData 2025-11-01 docs.

/person/enrich response shape:
  [
    {
      "matched_on": "https://linkedin.com/in/...",
      "match_type": "professional_network_profile_url",
      "matches": [
        {
          "confidence_score": 1.0,
          "person_data": {
            "basic_profile":  {name, headline, current_title, summary, location, languages},
            "professional_network": {connections, followers, profile_picture_permalink},
            "experience":     {employment_details: {current:[{company_name,title,start_date}], past:[...]}},
            "education":      {schools:[{school, degree, field_of_study, start_date, end_date}]},
            "skills":         {professional_network_skills:[...]},
            "contact":        {business_emails, personal_emails, phone_numbers, websites},
            "social_handles": {professional_network_identifier:{profile_url,...}}
          }
        }
      ]
    }
  ]

/person/search response shape:
  {
    "profiles": [
      {
        "basic_profile": {name, headline, current_title, location},
        "experience": {employment_details: {current:[...], past:[...]}},
        "education": {schools:[...]},
        "skills": {professional_network_skills:[...]},
        "social_handles": {professional_network_identifier:{profile_url}},
        "professional_network": {connections}
      }
    ],
    "total_count": N,
    "next_cursor": "..."
  }
"""

from __future__ import annotations
import logging
import re
from typing import Any, Optional

from crustdata_client import CrustDataClient, CrustDataError
from models import EducationEntry, WorkEntry

logger = logging.getLogger(__name__)

# ── Pedigree constants ─────────────────────────────────────────────────────────

TIER1_SCHOOLS = {
    "mit", "stanford", "harvard", "caltech", "carnegie mellon", "uc berkeley",
    "iit", "iim", "oxford", "cambridge", "eth zurich", "columbia", "yale",
    "princeton", "upenn", "wharton", "lse", "imperial college", "ntu", "nus",
    "iit bombay", "iit delhi", "iit madras", "bits pilani", "iisc",
    "georgia tech", "university of waterloo", "epfl", "tsinghua", "peking",
}

NOTABLE_COMPANIES = {
    "google", "meta", "apple", "amazon", "microsoft", "stripe", "airbnb", "uber",
    "netflix", "openai", "anthropic", "palantir", "spacex", "tesla", "linkedin",
    "twitter", "x", "dropbox", "slack", "zoom", "salesforce", "databricks",
    "snowflake", "figma", "notion", "robinhood", "coinbase", "sequoia", "a16z",
    "ycombinator", "y combinator", "mckinsey", "goldman sachs", "blackrock",
    "flipkart", "razorpay", "swiggy", "zomato", "freshworks", "zepto",
    "meesho", "cred", "groww", "phonepe", "nvidia", "intel", "ibm",
}

# Tight founder-level titles only.
# Director/VP/Head/Partner are intentionally excluded — too broad, matches investors and VCs.
FOUNDER_TITLE_REGEX = "Founder|Co-Founder|CEO|CTO|CPO|Chief Executive|Chief Technology|President"


# ── High-level person operations ───────────────────────────────────────────────

async def enrich_person(client: CrustDataClient, person: dict) -> dict:
    """
    Enrich a person dict using /person/enrich.
    Requires a linkedin URL (professional_network_profile_url).
    Falls back gracefully — never raises, always returns normalised dict.
    """
    # Extract linkedin URL — handle multiple field name variants
    linkedin_url = _get_linkedin_url(person)
    email = person.get("email") or person.get("business_email")

    base = _normalize_person_minimal(person)

    if linkedin_url:
        try:
            raw = await client.person_enrich_by_urls([linkedin_url])
            person_data = _extract_person_data_from_enrich(raw, linkedin_url)
            if person_data:
                return _normalize_person_from_enrich(person_data, base)
        except CrustDataError as e:
            logger.warning("person_enrich_by_urls failed for %s: %s", linkedin_url, e)

    if email:
        try:
            raw = await client.person_enrich_by_emails([email])
            person_data = _extract_person_data_from_enrich(raw, email)
            if person_data:
                return _normalize_person_from_enrich(person_data, base)
        except CrustDataError as e:
            logger.warning("person_enrich_by_emails failed for %s: %s", email, e)

    return base


async def enrich_people_batch(
    client: CrustDataClient, people: list[dict]
) -> list[dict]:
    """
    Batch enrich up to 25 people at once using their LinkedIn URLs.
    THEN: Perform secondary name-based search on ALL founders to get complete profile data.
    This ensures we get both /person/enrich data AND /person/search data for richness.
    """
    url_people = [(i, p) for i, p in enumerate(people) if _get_linkedin_url(p)]
    no_url_people = [(i, p) for i, p in enumerate(people) if not _get_linkedin_url(p)]

    results: list[dict] = [_normalize_person_minimal(p) for p in people]
    
    logger.info("[Person] enrich_people_batch: %d with URL, %d without", len(url_people), len(no_url_people))

    # STAGE 1: Batch enrich those with URLs (max 25 per call)
    for batch_start in range(0, len(url_people), 25):
        batch = url_people[batch_start:batch_start + 25]
        urls = [_get_linkedin_url(p) for _, p in batch]
        logger.debug("[Person] Batch enriching %d people: %s", len(urls), [u[:40] for u in urls if u])
        try:
            raw_list = await client.person_enrich_by_urls(urls)
            logger.debug("[Person] Got response with %d entries", len(raw_list) if isinstance(raw_list, list) else 1)
            
            for idx, (orig_idx, person) in enumerate(batch):
                url = urls[idx]
                logger.debug("[Person] Processing person %d/%d (orig_idx=%d): %s", 
                            idx+1, len(batch), orig_idx, url[:50] if url else "no-url")
                person_data = _extract_person_data_from_enrich(raw_list, url)
                if person_data:
                    old_name = results[orig_idx].get("name", "unknown")
                    results[orig_idx] = _normalize_person_from_enrich(
                        person_data, results[orig_idx]
                    )
                    new_name = results[orig_idx].get("name", "unknown")
                    logger.info("[Person] Enriched person at idx %d: %s → %s (URL-based)", orig_idx, old_name, new_name)
                else:
                    logger.warning("[Person] No person_data found for idx %d: %s", orig_idx, url[:50] if url else "no-url")
        except CrustDataError as e:
            logger.warning("Batch person enrich failed: %s", e)

    # STAGE 2: Secondary enrichment by name search for ALL people (to get complete data)
    # This uses /person/search with exact name match to get full profile data
    logger.info("[Person] Starting secondary name-based search for all %d founders", len(people))
    for orig_idx, person in enumerate(people):
        name = person.get("name") or results[orig_idx].get("name")
        if name and name.lower() not in ("unknown", ""):
            logger.debug("[Person] Name-based search for founder: %s (idx=%d)", name, orig_idx)
            enriched_by_name = await enrich_person_by_name(client, name)
            if enriched_by_name:
                # Merge: prefer data from name search, fall back to URL enrich
                old_result = dict(results[orig_idx])
                merged = _merge_person_data(old_result, enriched_by_name)
                results[orig_idx] = merged
                logger.info("[Person] Enhanced person at idx %d via name search: %s", orig_idx, name)
            else:
                logger.debug("[Person] Name search returned no results for: %s", name)

    return results


async def enrich_person_by_name(client: CrustDataClient, name: str) -> Optional[dict]:
    """
    Search for a person by exact name match using /person/search.
    This is useful for founders without LinkedIn URLs or to get fresh data.
    
    Example curl:
    curl --request POST \\
      --url https://api.crustdata.com/person/search \\
      --header 'authorization: Bearer <KEY>' \\
      --header 'content-type: application/json' \\
      --header 'x-api-version: 2025-11-01' \\
      --data '{
        "filters": {
          "field": "basic_profile.name",
          "type": "=",
          "value": "Abhilash Chowdhary"
        },
        "limit": 1
      }'
    """
    if not name or name.lower() in ("unknown", ""):
        return None
    
    try:
        logger.debug("[Person] Searching by name: %s", name)
        # Use /person/search with exact name match
        raw = await client.person_search(
            filters={
                "field": "basic_profile.name",
                "type": "=",
                "value": name,
            },
            limit=1,
        )
        
        profiles = _extract_profiles(raw)
        if profiles:
            best_profile = profiles[0]
            logger.info("[Person] Found person by name search: %s", name)
            return _normalize_person_from_search(best_profile)
        else:
            logger.debug("[Person] No results for name search: %s", name)
            return None
    except CrustDataError as e:
        logger.warning("[Person] Name search failed for '%s': %s", name, e)
        return None


def _merge_person_data(url_enriched: dict, name_searched: dict) -> dict:
    """
    Merge person data from two sources:
    - url_enriched: data from /person/enrich (deep nested structure)
    - name_searched: data from /person/search (normalized structure)
    
    Preference: Take name_searched data as primary (it's typically more complete),
    fall back to url_enriched for missing fields.
    """
    if not url_enriched and not name_searched:
        return {}
    
    if name_searched and not url_enriched:
        return name_searched
    
    if url_enriched and not name_searched:
        return url_enriched
    
    # Merge: name_searched takes priority for completeness
    merged = dict(url_enriched)
    
    # Override with name_searched data for key fields
    for key in ("name", "title", "headline", "location", "linkedin_url", 
                "education", "work_history", "skills"):
        if key in name_searched and name_searched[key]:
            merged[key] = name_searched[key]
    
    # Ensure all name_searched fields are present
    for key, val in name_searched.items():
        if key not in merged or not merged[key]:
            merged[key] = val
    
    logger.debug("[Person] Merged person data: %s", merged.get("name", "unknown"))
    return merged


async def search_leaders_at_company(
    client: CrustDataClient, company_name: str, limit: int = 10
) -> list[dict]:
    """
    Find current founders/execs at a company using /person/search.
    """
    if not company_name:
        return []
    try:
        raw = await client.person_search_current_company(
            company_name, title_regex=FOUNDER_TITLE_REGEX, limit=limit
        )
        profiles = _extract_profiles(raw)
        logger.info("leaders at '%s': %d found", company_name, len(profiles))
        return [_normalize_person_from_search(p) for p in profiles]
    except CrustDataError as e:
        logger.warning("search_leaders_at_company '%s' failed: %s", company_name, e)
        return []


async def search_talent_at_company(
    client: CrustDataClient, 
    company_name: str, 
    title_keywords: Optional[list[str]] = None,
    limit: int = 20
) -> list[dict]:
    """
    Search for ANY talent currently at a company (not just founders).
    Used for talent pool mining in HR analysis.
    
    Args:
      company_name: Company to search at (e.g., "Google", "Meta")
      title_keywords: Optional job titles to filter by (e.g., ["Product Manager", "Engineer"])
      limit: Max results to return
    
    Returns:
      List of normalized person dicts with current roles at the company
    
    Example: Search for Product Managers at Google
      talent = await search_talent_at_company(client, "Google", ["Product Manager"])
    """
    if not company_name:
        return []
    
    try:
        logger.info("[Person] Searching talent at '%s' with keywords: %s (limit=%d)", 
                    company_name, title_keywords, limit)
        
        # Build flexible title regex from keywords if provided
        if title_keywords and len(title_keywords) > 0:
            # Create regex that matches any of the keywords
            title_regex = "|".join(title_keywords)
        else:
            # No filter — search ALL current employees (very broad)
            title_regex = ".*"  # Match any title
        
        raw = await client.person_search_current_company(
            company_name, title_regex=title_regex, limit=limit
        )
        profiles = _extract_profiles(raw)
        logger.info("[Person] Found %d talent at '%s'", len(profiles), company_name)
        
        if len(profiles) == 0:
            logger.warning("[Person] No results for company='%s', title_regex='%s'", 
                          company_name, title_regex)
        
        return [_normalize_person_from_search(p) for p in profiles]
    except CrustDataError as e:
        logger.warning("[Person] search_talent_at_company '%s' failed: %s", company_name, e)
        return []



async def search_alumni(
    client: CrustDataClient,
    institution: str,
    role_keywords: list[str],   # kept for interface compatibility, not used directly
    limit: int = 10,
) -> list[dict]:
    """
    Search alumni from a school who are now in founder/exec roles.
    Uses education.schools.school filter from /person/search.
    """
    if not institution or institution.lower() in ("unknown", ""):
        return []
    try:
        raw = await client.person_search_school(institution, title_regex=FOUNDER_TITLE_REGEX, limit=limit)
        profiles = _extract_profiles(raw)
        logger.info("school alumni '%s': %d found", institution, len(profiles))
        return [_normalize_person_from_search(p) for p in profiles]
    except CrustDataError as e:
        logger.warning("search_alumni '%s' failed: %s", institution, e)
        return []


async def search_company_alumni(
    client: CrustDataClient, company_name: str, limit: int = 10
) -> list[dict]:
    """
    Search past employees of a company who are now in founder/exec roles.
    Uses experience.employment_details.past.company_name filter.
    """
    if not company_name or company_name.lower() in ("unknown", ""):
        return []
    try:
        raw = await client.person_search_past_company(
            company_name, title_regex=FOUNDER_TITLE_REGEX, limit=limit
        )
        profiles = _extract_profiles(raw)
        logger.info("company alumni '%s': %d found", company_name, len(profiles))
        return [_normalize_person_from_search(p) for p in profiles]
    except CrustDataError as e:
        logger.warning("search_company_alumni '%s' failed: %s", company_name, e)
        return []


# ── Response extraction ────────────────────────────────────────────────────────

def _extract_profiles(data: Any) -> list[dict]:
    """
    Extract person profiles from /person/search response.
    Response: {"profiles":[...], "total_count":N, "next_cursor":"..."}
    """
    if isinstance(data, list):
        return [p for p in data if isinstance(p, dict)]
    if isinstance(data, dict):
        for key in ("profiles", "people", "results", "data"):
            val = data.get(key)
            if isinstance(val, list) and val:
                return [p for p in val if isinstance(p, dict)]
    return []


def _extract_person_list(data: Any) -> list[dict]:
    """Alias for _extract_profiles — used by company_service."""
    return _extract_profiles(data)


def _extract_person_data_from_enrich(raw: Any, identifier: str) -> Optional[dict]:
    """
    Extract the best person_data from /person/enrich response.
    Response: [{matched_on, match_type, matches:[{confidence_score, person_data}]}]
    
    CRITICAL FIX: Match by identifier (LinkedIn URL or email) to ensure we get the RIGHT person,
    not just the first best match from all entries.
    """
    if not isinstance(raw, list):
        raw = [raw] if isinstance(raw, dict) else []

    logger.debug("[Person] _extract_person_data_from_enrich: looking for identifier=%s in %d entries", 
                 identifier[:50] if identifier else "none", len(raw))

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        
        # Check if this entry matches our identifier (matched_on field)
        matched_on = entry.get("matched_on", "").lower()
        identifier_lower = (identifier or "").lower()
        
        # Compare by URL (remove trailing /)
        url_match = (
            matched_on.rstrip("/") == identifier_lower.rstrip("/") 
            or identifier_lower in matched_on
            or matched_on in identifier_lower
        )
        
        if url_match or not identifier:  # If no identifier, take first
            matches = entry.get("matches") or []
            if matches:
                # Sort by confidence and take best
                best = max(matches, key=lambda m: m.get("confidence_score", 0) if isinstance(m, dict) else 0)
                pd = best.get("person_data") if isinstance(best, dict) else None
                if isinstance(pd, dict):
                    logger.debug("[Person] Found matching person for %s", identifier[:50] if identifier else "unknown")
                    return pd
    
    logger.debug("[Person] No matching person found for identifier=%s", identifier[:50] if identifier else "none")
    return None


# ── Normalisation — /person/enrich response ──────────────────────────────────

def _normalize_person_from_enrich(person_data: dict, base: dict) -> dict:
    """
    Map /person/enrich person_data (nested under matches[].person_data) to internal schema.
    Structure: basic_profile, experience.employment_details, education.schools, skills.professional_network_skills
    """
    bp = person_data.get("basic_profile") or {}
    exp = person_data.get("experience") or {}
    emp_details = exp.get("employment_details") or {}
    edu_section = person_data.get("education") or {}
    skills_section = person_data.get("skills") or {}
    soc = person_data.get("social_handles") or {}
    pn  = person_data.get("professional_network") or {}

    # Education
    education = []
    for school in (edu_section.get("schools") or []):
        if not isinstance(school, dict):
            continue
        inst = school.get("school") or school.get("name") or ""
        if not inst:
            continue
        education.append(EducationEntry(
            institution=inst,
            degree=school.get("degree"),
            field=school.get("field_of_study"),
            year=_parse_year(school.get("end_date")),
            prestige_tier=_tier(inst),
        ).model_dump())

    # Work history — current + past
    work_history = []
    current_jobs = emp_details.get("current") or []
    past_jobs    = emp_details.get("past") or []
    if not isinstance(current_jobs, list): current_jobs = [current_jobs]
    if not isinstance(past_jobs, list):    past_jobs = [past_jobs]

    for job in current_jobs + past_jobs:
        if not isinstance(job, dict): continue
        company = job.get("company_name") or job.get("company") or ""
        work_history.append(WorkEntry(
            company=company,
            role=job.get("title") or job.get("role") or "Unknown",
            start_year=_parse_year(job.get("start_date")),
            end_year=_parse_year(job.get("end_date")),
            is_notable=_is_notable(company),
        ).model_dump())

    # Skills
    skills = [s for s in (skills_section.get("professional_network_skills") or [])
              if isinstance(s, str)]

    # LinkedIn URL from social_handles
    linkedin_url = (
        _deep_get(soc, "professional_network_identifier", "profile_url")
        or base.get("linkedin_url")
    )

    result = dict(base)
    result.update({
        "name":        bp.get("name") or base.get("name") or "Unknown",
        "title":       bp.get("current_title") or bp.get("headline") or base.get("title"),
        "linkedin_url": linkedin_url,
        "location":    _flatten_location(bp.get("location")) or base.get("location"),
        "summary":     bp.get("summary") or base.get("summary"),
        "connections": pn.get("connections") or base.get("connections"),
        "skills":      skills or base.get("skills", []),
        "education":   education or base.get("education", []),
        "work_history": work_history or base.get("work_history", []),
    })
    return result


# ── Normalisation — /person/search response ──────────────────────────────────

def _normalize_person_from_search(profile: dict) -> dict:
    """
    Map /person/search profile to internal schema.
    NEW API (/person/search 2025-11-01): nested basic_profile, experience.employment_details
    OLD screener (/screener/person/search): flat name, employer[], flagship_profile_url

    Handles BOTH shapes defensively.
    """
    # ── Detect shape ──────────────────────────────────────────────────────────
    has_basic_profile = isinstance(profile.get("basic_profile"), dict)

    if has_basic_profile:
        # NEW 2025-11-01 /person/search nested shape
        bp  = profile.get("basic_profile") or {}
        exp = profile.get("experience") or {}
        emp = exp.get("employment_details") or {}
        edu = profile.get("education") or {}
        sk  = profile.get("skills") or {}
        soc = profile.get("social_handles") or {}
        pn  = profile.get("professional_network") or {}

        education = _parse_education_nested(edu.get("schools") or [])
        current_jobs = emp.get("current") or []
        past_jobs    = emp.get("past") or []
        if not isinstance(current_jobs, list): current_jobs = [current_jobs]
        if not isinstance(past_jobs, list):    past_jobs = [past_jobs]
        work_history = _parse_jobs(current_jobs + past_jobs)
        skills = [s for s in (sk.get("professional_network_skills") or []) if isinstance(s, str)]
        linkedin_url = _deep_get(soc, "professional_network_identifier", "profile_url")
        name = bp.get("name") or "Unknown"
        title = bp.get("current_title") or bp.get("headline")
        location = _flatten_location(bp.get("location"))
        connections = pn.get("connections")
        summary = bp.get("summary")

    else:
        # OLD screener flat shape: name, employer[], flagship_profile_url, skills[], etc.
        name     = profile.get("name") or profile.get("full_name") or "Unknown"
        title    = profile.get("current_title") or profile.get("headline") or profile.get("title")
        location = profile.get("location")
        connections = profile.get("num_of_connections") or profile.get("connections")
        summary  = profile.get("summary") or profile.get("about")
        skills   = [s for s in (profile.get("skills") or []) if isinstance(s, str)]
        linkedin_url = (
            profile.get("flagship_profile_url")
            or profile.get("linkedin_profile_url")
            or profile.get("linkedin_url")
        )
        # employer[] in screener shape
        education = _parse_education_nested(profile.get("education_background") or profile.get("education") or [])
        work_history = _parse_jobs(profile.get("employer") or profile.get("experience") or [])

    return {
        "person_id":    profile.get("crustdata_person_id") or profile.get("person_id") or profile.get("id"),
        "name":         name,
        "title":        title,
        "linkedin_url": linkedin_url,
        "location":     _flatten_location(location) if not isinstance(location, str) else location,
        "summary":      summary,
        "connections":  connections,
        "skills":       skills,
        "education":    education,
        "work_history": work_history,
    }


def _normalize_person_minimal(person: dict) -> dict:
    """Create a minimal normalised person dict from whatever fields exist."""
    bp = person.get("basic_profile") or {}
    soc = person.get("social_handles") or {}

    name = (
        bp.get("name")                                     # nested company.people shape
        or person.get("name") or person.get("full_name")   # flat
        or person.get("first_last_name")
        or (person.get("first_name","") + " " + person.get("last_name","")).strip()
    ) or "Unknown"

    linkedin_url = (
        _deep_get(soc, "professional_network_identifier", "profile_url")
        or person.get("linkedin_profile_url")
        or person.get("flagship_profile_url")
        or person.get("linkedin_url")
    )

    return {
        "person_id":   person.get("crustdata_person_id") or person.get("person_id") or person.get("id"),
        "name":        name,
        "title":       bp.get("current_title") or bp.get("headline") or person.get("title") or person.get("current_title"),
        "linkedin_url": linkedin_url,
        "email":       person.get("email") or person.get("business_email"),
        "location":    _flatten_location(bp.get("location")) or person.get("location") or person.get("country"),
        "summary":     bp.get("summary") or person.get("summary") or person.get("about"),
        "connections": (person.get("professional_network") or {}).get("connections") or person.get("connections"),
        "skills":      (person.get("skills") or []) if isinstance(person.get("skills"), list) else [],
        "education":   (person.get("education") or []) if isinstance(person.get("education"), list) else [],
        "work_history": (person.get("work_history") or []) if isinstance(person.get("work_history"), list) else [],
    }


# ── Utility helpers ────────────────────────────────────────────────────────────

def _get_linkedin_url(person: dict) -> Optional[str]:
    """Try all common field names for a LinkedIn URL."""
    return (
        person.get("flagship_profile_url")
        or person.get("linkedin_profile_url")
        or person.get("linkedin_url")
        or person.get("professional_network_profile_url")
        or _deep_get(person, "social_handles", "professional_network_identifier", "profile_url")
    )


def _deep_get(obj: dict, *keys: str) -> Any:
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _flatten_location(loc: Any) -> Optional[str]:
    """Flatten CrustData location object to a string."""
    if not loc:
        return None
    if isinstance(loc, str):
        return loc
    if isinstance(loc, dict):
        return (
            loc.get("full_location")
            or loc.get("city")
            or loc.get("country")
            or loc.get("raw")
        )
    return None


def _tier(institution: str) -> str:
    lower = institution.lower()
    return "Tier-1" if any(s in lower for s in TIER1_SCHOOLS) else "Tier-2"


def _is_notable(company: str) -> bool:
    lower = company.lower()
    return any(nc in lower for nc in NOTABLE_COMPANIES)


def _parse_year(date_str: Optional[Any]) -> Optional[int]:
    if not date_str:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", str(date_str))
    return int(m.group()) if m else None


def _parse_education_nested(schools: list) -> list:
    """Parse education from schools array (nested shape: school.school, school.degree)."""
    result = []
    for school in (schools or []):
        if not isinstance(school, dict): continue
        inst = (school.get("school") or school.get("institute_name")
                or school.get("institution") or school.get("name") or "")
        if not inst: continue
        result.append(EducationEntry(
            institution=inst,
            degree=school.get("degree_name") or school.get("degree"),
            field=school.get("field_of_study") or school.get("field"),
            year=_parse_year(school.get("end_date") or school.get("year")),
            prestige_tier=_tier(inst),
        ).model_dump())
    return result


def _parse_jobs(jobs: list) -> list:
    """
    Parse work history from either employer[] or employment_details[] items.

    NEW /person/search API (2025-11-01) response shape:
      {"name": "Crustdata (YC F24)", "title": "Co-Founder & CEO", "crustdata_company_id": 6036032}
      ← company name is in "name" field, NOT "company_name"

    OLD screener shape:
      {"company_name": "Crustdata", "title": "Co-Founder"}
    """
    result = []
    for job in (jobs or []):
        if not isinstance(job, dict):
            continue
        company = (
            job.get("company_name")   # OLD screener
            or job.get("company")
            or job.get("name")        # NEW /person/search — confirmed from docs
            or job.get("organization")
            or ""
        )
        result.append(WorkEntry(
            company=company,
            role=job.get("title") or job.get("role") or "Unknown",
            start_year=_parse_year(job.get("start_date") or job.get("start_year")),
            end_year=_parse_year(job.get("end_date") or job.get("end_year")),
            is_notable=_is_notable(company),
        ).model_dump())
    return result
