"""
HR Analyzer
Complete HR Intelligence per spec:

  ✅ Candidate & Talent Pool Analysis (skill match, past company perf, growth trajectory)
  ✅ Cultural & Performance Signals (sentiment trends, retention/attrition, past company signals)
  ✅ Talent Shift Intelligence (industry movement patterns, emerging skill demand)
  ✅ Visualization outputs:
       - Talent Pool Quality Distribution
       - Skill Match Score Histogram
       - Talent Source Map (which companies talent is from)
       - Industry Shift Graph
  ✅ HR Recommendations (best source companies, ideal archetypes, hidden talent pools)

Input modes: structured candidates | free-text resumes | target company pool
"""

from __future__ import annotations
import json
import logging
from typing import Optional

from crustdata_client import CrustDataClient
from llm_client import BedrockLLM
from models import CandidateInput, CandidateScore, HROutput, TalentShiftInsight
from services.person_service import enrich_person
from services.signal_service import build_hr_signal_bundle, gather_web_signals

logger = logging.getLogger(__name__)


# ── System prompt ──────────────────────────────────────────────────────────────

HR_SYSTEM = """
You are a world-class technical recruiter and talent intelligence analyst.
You assess candidates against job descriptions with precision, fairness, and data rigour.
You detect macro talent movement patterns and surface hidden talent pools.
You prioritise real signals over vanity metrics.
ALWAYS return ONLY valid JSON — no markdown fences, no prose, no preamble.
"""


# ── Main HR prompt — ALL required outputs explicitly requested ─────────────────

HR_ANALYSIS_PROMPT = """
Analyse the HR signal bundle below and return a complete talent intelligence assessment.
Return ONLY this exact JSON structure — every field is required:

{{
  "jd_summary": "<2-sentence JD summary highlighting key technical and experience requirements>",

  "candidate_scores": [
    {{
      "name": "<candidate name>",
      "skill_match_score": <0-100>,
      "experience_alignment": <0-100>,
      "cultural_fit_score": <0-100>,
      "overall_score": <0-100, weighted: skill*0.40 + exp*0.35 + culture*0.25>,
      "strengths": ["<specific strength backed by data>", "<second>"],
      "gaps": ["<specific gap vs JD>"],
      "recommendation": "<Strong Yes|Yes|Maybe|No>",
      "growth_trajectory": "<accelerating|steady|plateauing — evidence>",
      "past_company_quality": "<strong|moderate|weak — name notable companies>",
      "cultural_signals": "<specific evidence of cultural alignment or concern>"
    }}
  ],

  "talent_pool_analysis": "<Analytical paragraph: overall quality, depth, diversity of skills, notable patterns. Not generic — cite specific candidates.>",

  "skill_gap_analysis": "<What critical skills are missing or underrepresented in this pool vs the JD? Be specific.>",

  "retention_attrition_signals": {{
    "avg_tenure_years": <estimated average tenure from work histories>,
    "job_hopper_count": <candidates with average tenure < 18 months>,
    "signal": "<positive|neutral|negative — explain>",
    "patterns": ["<pattern 1>", "<pattern 2>"]
  }},

  "talent_shift_insights": [
    {{
      "pattern": "<specific talent movement pattern observed or inferred>",
      "source_industry": "<industry talent is moving FROM>",
      "target_industry": "<industry talent is moving TO>",
      "signal_strength": "<Strong|Moderate|Weak>",
      "relevance_to_jd": "<why this shift matters for this specific role>",
      "action": "<concrete action for recruiter to take based on this signal>"
    }}
  ],

  "talent_source_map": [
    {{
      "company": "<company name>",
      "candidate_count": <number of candidates from this company in the pool>,
      "quality_signal": "<high|medium|low — based on company reputation and candidate quality>",
      "recommended_for_sourcing": <true|false>
    }}
  ],

  "recommendations": [
    "<Specific, actionable recommendation 1 — name companies, titles, or pools>",
    "<Recommendation 2>",
    "<Recommendation 3>",
    "<Recommendation 4>"
  ],

  "ideal_candidate_archetype": {{
    "title": "<ideal target title>",
    "background": "<ideal company background>",
    "education": "<ideal education background>",
    "skills_must": ["<must-have skill 1>", "<must-have 2>", "<must-have 3>"],
    "skills_nice": ["<nice-to-have 1>", "<nice-to-have 2>"],
    "experience_years": "<range e.g. 5-8>",
    "where_to_find": "<specific LinkedIn search strategy or sourcing approach>"
  }},

  "hidden_talent_pools": [
    {{
      "pool_description": "<specific underrated talent pool>",
      "why_relevant": "<why they would excel in this role>",
      "sourcing_approach": "<concrete sourcing tactic>"
    }}
  ],

  "top_source_companies": ["<company1>", "<company2>", "<company3>", "<company4>", "<company5>"],

  "skill_clusters": {{
    "ML/AI": <count of candidates with ML/AI skills>,
    "Backend": <count>,
    "Frontend": <count>,
    "Data Engineering": <count>,
    "Product": <count>,
    "DevOps/Infra": <count>
  }}
}}

IMPORTANT RULES:
- Prioritise real signals over vanity metrics
- Always connect insights to specific signals in the data
- Avoid generic statements — name specific companies, skills, patterns
- If data is incomplete → say "based on available signals" but still provide analysis
- Talent shift insights must explain WHY the movement is a signal of emerging demand

HR Signal Bundle:
{signal_bundle}
"""


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_hr_analysis(
    client: CrustDataClient,
    llm: BedrockLLM,
    jd_text: Optional[str],
    jd_structured: Optional[dict],
    candidates: Optional[list[CandidateInput]],
    target_company: Optional[str],
    include_visual: bool = True,
    include_recommendations: bool = True,
) -> HROutput:

    # ── Step 1: Build candidate data (enrich if linkedin/email available) ─────
    candidate_dicts: list[dict] = []
    if candidates:
        for c in candidates:
            cd = c.model_dump()
            if c.linkedin_url or c.email:
                try:
                    enriched = await enrich_person(
                        client,
                        {"linkedin_url": c.linkedin_url, "email": c.email, "name": c.name},
                    )
                    for k, v in enriched.items():
                        if v and not cd.get(k):
                            cd[k] = v
                except Exception as e:
                    logger.warning("Could not enrich candidate %s: %s", c.name, e)
            candidate_dicts.append(cd)

    # ── Step 2: Mine talent pool from target company if no candidates ─────────
    if target_company and not candidate_dicts:
        try:
            from services.person_service import search_leaders_at_company
            pool = await search_leaders_at_company(client, target_company, limit=10)
            candidate_dicts = pool
            logger.info("[HR] Mined %d candidates from %s", len(pool), target_company)
        except Exception as e:
            logger.warning("[HR] Could not mine pool from %s: %s", target_company, e)

    # ── Step 3: Web market signals ────────────────────────────────────────────
    role_hint = _extract_role(jd_text or jd_structured or {})
    web_signals = await gather_web_signals(
        client, role_hint or "software engineer",
        topics=[
            f"{role_hint} talent market hiring trends 2025" if role_hint else "tech talent market 2025",
            "talent shift AI ML fintech 2025 emerging skills",
        ],
    )

    # ── Step 4: Build signal bundle ───────────────────────────────────────────
    signal_bundle = build_hr_signal_bundle(
        jd_text=jd_text,
        jd_structured=jd_structured,
        candidates=candidate_dicts,
        web_signals=web_signals,
    )
    logger.info("[HR] Signal bundle: %d chars, %d candidates", len(signal_bundle), len(candidate_dicts))

    # ── Step 5: Bedrock Claude LLM analysis ───────────────────────────────────
    prompt = HR_ANALYSIS_PROMPT.format(signal_bundle=signal_bundle)
    llm_result: dict = {}
    logger.info("[HR] Calling Bedrock Claude for HR analysis...")
    try:
        llm_result = await llm.ainvoke_json(prompt, system=HR_SYSTEM, max_tokens=8192)
        logger.info("[HR] Bedrock analysis complete. Candidates scored: %d",
                   len(llm_result.get("candidate_scores") or []))
    except Exception as e:
        logger.error("[HR] Bedrock call failed: %s", e)

    # ── Step 6: Parse LLM results ─────────────────────────────────────────────
    candidate_scores: list[CandidateScore] = []
    for s in (llm_result.get("candidate_scores") or []):
        try:
            candidate_scores.append(CandidateScore(**{
                "name":                  s.get("name", ""),
                "skill_match_score":     float(s.get("skill_match_score", 50)),
                "experience_alignment":  float(s.get("experience_alignment", 50)),
                "cultural_fit_score":    float(s.get("cultural_fit_score", 50)),
                "overall_score":         float(s.get("overall_score", 50)),
                "strengths":             s.get("strengths") or [],
                "gaps":                  s.get("gaps") or [],
                "recommendation":        s.get("recommendation", "Maybe"),
            }))
        except Exception as ex:
            logger.warning("Could not parse candidate score: %s", ex)

    talent_shifts: list[TalentShiftInsight] = []
    for ts in (llm_result.get("talent_shift_insights") or []):
        try:
            talent_shifts.append(TalentShiftInsight(**{
                "pattern":          ts.get("pattern", ""),
                "source_industry":  ts.get("source_industry", ""),
                "target_industry":  ts.get("target_industry", ""),
                "signal_strength":  ts.get("signal_strength", "Moderate"),
                "relevance_to_jd":  ts.get("relevance_to_jd", ""),
            }))
        except Exception:
            pass

    # ── Step 7: Build visualizations ─────────────────────────────────────────
    charts: Optional[dict] = None
    if include_visual:
        charts = _build_hr_charts(
            candidate_scores=candidate_scores,
            talent_shifts=talent_shifts,
            llm_result=llm_result,
        )

    recommendations = llm_result.get("recommendations") or [] if include_recommendations else []

    return HROutput(
        jd_summary=llm_result.get("jd_summary"),
        candidate_scores=candidate_scores,
        talent_pool_analysis=llm_result.get("talent_pool_analysis"),
        skill_gap_analysis=llm_result.get("skill_gap_analysis"),
        talent_shift_insights=talent_shifts,
        visualizations=charts,
        recommendations=recommendations,
        top_source_companies=llm_result.get("top_source_companies") or [],
    )


# ── Chart assembly ─────────────────────────────────────────────────────────────

def _build_hr_charts(
    candidate_scores: list[CandidateScore],
    talent_shifts: list[TalentShiftInsight],
    llm_result: dict,
) -> dict:
    from visualization.chart_builder import (
        candidate_score_bar_chart,
        talent_pool_distribution_chart,
        skill_match_histogram,
        industry_shift_chart,
        talent_source_map_chart,
        recommendation_radar_chart,
    )

    charts: dict = {}
    score_dicts = [s.model_dump() for s in candidate_scores]

    # Candidate scores grouped bar
    if score_dicts:
        charts["candidate_scores"] = candidate_score_bar_chart(score_dicts)
        charts["skill_match_histogram"] = skill_match_histogram(score_dicts)

    # Talent pool distribution by skill cluster
    skill_clusters = llm_result.get("skill_clusters") or {}
    if skill_clusters:
        charts["talent_pool_distribution"] = talent_pool_distribution_chart(skill_clusters)

    # Industry shift chart
    if talent_shifts:
        charts["industry_shifts"] = industry_shift_chart([t.model_dump() for t in talent_shifts])

    # Talent source map (which companies candidates are from)
    source_map = llm_result.get("talent_source_map") or []
    if source_map:
        charts["talent_source_map"] = talent_source_map_chart(source_map)

    # Per-candidate radar for top candidate
    if score_dicts:
        top = max(score_dicts, key=lambda x: x.get("overall_score", 0))
        charts["top_candidate_radar"] = recommendation_radar_chart(top)

    return charts


def _extract_role(jd) -> Optional[str]:
    if isinstance(jd, str):
        lines = [l.strip() for l in jd.split("\n") if l.strip()]
        return lines[0][:60] if lines else None
    if isinstance(jd, dict):
        return jd.get("title") or jd.get("role") or jd.get("position")
    return None
