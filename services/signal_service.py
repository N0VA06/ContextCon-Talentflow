"""
Signal Service
Aggregates raw CrustData signals into rich context bundles for LLM analysis.
The richer the bundle, the more signal the LLM has to produce accurate scores.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Optional

from crustdata_client import CrustDataClient, CrustDataError

logger = logging.getLogger(__name__)


async def gather_web_signals(
    client: CrustDataClient,
    company_name: str,
    topics: Optional[list[str]] = None,
) -> list[dict]:
    """
    POST /screener/web-search for company context.
    Stops early on credit exhaustion. Always returns list (never raises).
    """
    if topics is None:
        topics = [
            f"{company_name} founders background",
            f"{company_name} funding product launch",
        ]

    all_results: list[dict] = []
    seen_urls: set[str] = set()

    for topic in topics[:3]:
        try:
            data = await client.web_search(topic, sources=["web", "news"])
            items: list = []
            if isinstance(data, dict):
                items = data.get("results") or []
            elif isinstance(data, list):
                items = data

            for item in items:
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or ""
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "title":   item.get("title", ""),
                        "snippet": item.get("snippet") or item.get("description", ""),
                        "url":     url,
                        "source":  item.get("source") or "",
                    })
        except CrustDataError as e:
            err = str(e).lower()
            if "insufficient credits" in err or "credits" in err or "402" in err:
                logger.warning("web_search: credits exhausted — stopping")
                break
            logger.warning("web_search '%s' failed: %s", topic, e)
        except Exception as e:
            logger.warning("web_search '%s' unexpected error: %s", topic, e)

    return all_results


def build_investor_signal_bundle(
    company: dict,
    founders: list[dict],
    web_signals: list[dict],
    job_signals: Optional[dict] = None,
) -> str:
    """
    Build rich JSON context bundle for the investor LLM prompt.
    Includes: company profile, headcount trend, funding, job openings,
    employee reviews, founders (education + work history + scores), web signals.
    """
    # ── Company section ────────────────────────────────────────────────────────
    screener = company.get("screener") or {}
    company_section: dict[str, Any] = {
        "name":                 company.get("name"),
        "domain":               company.get("domain"),
        "industry":             company.get("industry"),
        "description":          company.get("description"),
        "headcount":            company.get("headcount"),
        "founded_year":         company.get("founded_year"),
        "hq_country":           company.get("hq_country"),
        "total_funding_usd":    company.get("total_funding_usd"),
        "latest_funding_stage": company.get("latest_funding_stage"),
    }

    # Headcount timeseries — last 12 data points
    hc_ts = company.get("headcount_timeseries") or {}
    hc_points = _trim(hc_ts, 12)
    if hc_points:
        company_section["headcount_timeseries"] = hc_points
        # Compute growth rate for LLM context
        vals = [_safe_int(p.get("headcount") or p.get("employee_count") or p.get("value")) for p in hc_points]
        vals = [v for v in vals if v]
        if len(vals) >= 2 and vals[0]:
            growth_pct = round((vals[-1] - vals[0]) / vals[0] * 100, 1)
            company_section["headcount_growth_pct"] = growth_pct

    # Funding milestones
    fund_ts = company.get("funding_timeseries") or {}
    fund_points = _trim(fund_ts, 10)
    if fund_points:
        company_section["funding_milestones"] = fund_points

    # Job openings timeseries — key hiring velocity signal
    job_ts = company.get("job_openings") or {}
    job_points = _trim(job_ts, 8)
    if job_points:
        company_section["job_openings_timeseries"] = job_points

    # Employee reviews — internal sentiment signal
    reviews = company.get("reviews")
    if reviews:
        company_section["employee_sentiment"] = _extract_review_summary(reviews)

    # Screener extra data
    if isinstance(screener, dict):
        for key in ("headcount_by_role_percent", "hiring", "web_traffic"):
            val = screener.get(key)
            if val:
                company_section[f"screener_{key}"] = val

    # ── Job signals from /job/search (if available) ────────────────────────────
    if job_signals:
        company_section["live_job_openings"] = {
            "total_open_roles":  job_signals.get("total", 0),
            "category_breakdown": job_signals.get("signals", {}).get("category_breakdown", {}),
            "top_titles":         job_signals.get("signals", {}).get("top_titles", [])[:10],
            "hiring_velocity":    job_signals.get("signals", {}).get("hiring_velocity", "unknown"),
            "engineering_ratio":  job_signals.get("signals", {}).get("engineering_ratio"),
        }

    # ── Founders section ───────────────────────────────────────────────────────
    founders_section = []
    for f in founders:
        founders_section.append({
            "name":         f.get("name"),
            "title":        f.get("title"),
            "education":    (f.get("education") or [])[:5],
            "work_history": (f.get("work_history") or [])[:8],
            "skills":       (f.get("skills") or [])[:20],
            "summary":      (f.get("summary") or "")[:600],
            "connections":  f.get("connections"),
            # Per-founder LLM scores (from pedigree step)
            "pedigree_scores": f.get("scores"),
            "key_insights":    f.get("key_insights") or [],
            "builder_signals": f.get("builder_signals") or [],
            "red_flags":       f.get("red_flags") or [],
        })

    bundle = {
        "company":       company_section,
        "founders":      founders_section,
        "web_signals":   web_signals[:10],
    }
    return json.dumps(bundle, default=str, indent=2)


def build_hr_signal_bundle(
    jd_text: Optional[str],
    jd_structured: Optional[dict],
    candidates: list[dict],
    web_signals: list[dict],
) -> str:
    """
    Build rich HR signal bundle for LLM analysis.
    Includes: JD, structured candidate profiles with full work history,
    education, skills, and market signals.
    """
    # Normalise JD
    jd_section: Any = jd_text or jd_structured or "Not provided"

    # Candidate profiles — richer than before, include work history for cultural signals
    candidate_section = []
    for c in (candidates or []):
        candidate_section.append({
            "name":             c.get("name"),
            "current_role":     c.get("current_role") or c.get("title"),
            "years_experience": c.get("years_experience"),
            "skills":           (c.get("skills") or [])[:25],
            "education": [
                {
                    "institution": e.get("institution") if isinstance(e, dict) else str(e),
                    "degree":      e.get("degree") if isinstance(e, dict) else None,
                    "prestige_tier": e.get("prestige_tier") if isinstance(e, dict) else None,
                }
                for e in (c.get("education") or [])[:3]
            ],
            "work_history": [
                {
                    "company":    w.get("company") if isinstance(w, dict) else str(w),
                    "role":       w.get("role") if isinstance(w, dict) else None,
                    "is_notable": w.get("is_notable") if isinstance(w, dict) else False,
                    "start_year": w.get("start_year") if isinstance(w, dict) else None,
                    "end_year":   w.get("end_year") if isinstance(w, dict) else None,
                }
                for w in (c.get("work_history") or [])[:6]
            ],
            "resume_text":  (c.get("resume_text") or "")[:1200],
            "location":     c.get("location"),
            "linkedin_url": c.get("linkedin_url"),
        })

    bundle = {
        "job_description": jd_section,
        "total_candidates": len(candidate_section),
        "candidates":       candidate_section,
        "market_signals":   web_signals[:5],
    }
    return json.dumps(bundle, default=str, indent=2)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _trim(data: Any, max_points: int = 10) -> list:
    if not data:
        return []
    if isinstance(data, dict):
        pts = data.get("data") or data.get("timeseries") or data.get("results") or []
    elif isinstance(data, list):
        pts = data
    else:
        return []
    return pts[-max_points:] if len(pts) > max_points else pts


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _extract_review_summary(reviews: Any) -> Optional[dict]:
    if not isinstance(reviews, dict):
        return None
    return {
        "overall_rating":    reviews.get("overall_rating"),
        "ceo_approval":      reviews.get("ceo_approval"),
        "recommend_pct":     reviews.get("recommend_to_friend_pct"),
        "review_count":      reviews.get("review_count"),
        "culture_rating":    reviews.get("culture_and_values_rating"),
        "work_life_balance": reviews.get("work_life_balance_rating"),
        "career_growth":     reviews.get("career_opportunities_rating"),
    }
