"""
Investor Analyzer
Full investor intelligence pipeline.

LLM (Bedrock Claude) judges ALL scoring and generates all visualization data.
Required outputs (per spec):
  ✅ Founder Score Breakdown (radar) — 5 dimensions
  ✅ Talent Flow Graph (inflow vs outflow over time)
  ✅ Skill Heatmap (hiring vs losing)
  ✅ Alumni Success Distribution (% successful founders per background)
  ✅ Investment Score 0-100 with exact weights:
       Founder Strength 40%
       Talent Quality & Movement 25%
       Hiring Signals 15%
       Market & Skill Alignment 10%
       External / Sentiment Signals 10%
  ✅ Score Breakdown + Confidence Level
"""

from __future__ import annotations
import json
import logging
from typing import Optional

from crustdata_client import CrustDataClient
from llm_client import BedrockLLM
from models import (
    InvestorOutput,
    InvestmentScore,
    ScoreBreakdown,
    TalentSignals,
)
from analyzers.pedigree_analyzer import analyze_pedigree
from services.signal_service import build_investor_signal_bundle, gather_web_signals
from services.job_service import search_company_jobs

logger = logging.getLogger(__name__)


# ── System prompt ──────────────────────────────────────────────────────────────

INVESTOR_SYSTEM = """
You are a Managing Partner at a top-tier VC fund (Sequoia / a16z / Benchmark calibre).
You make investment decisions based on quantitative talent signals and founder pattern recognition.
You prioritise founder quality over vanity metrics.
You are rigorous, contrarian, and specific — never generic.
If data is incomplete, say "based on available signals" but still provide a score.
ALWAYS return ONLY valid JSON — no markdown fences, no prose, no preamble.
"""


# ── Main investor prompt — requests ALL required chart data from LLM ──────────

INVESTOR_ANALYSIS_PROMPT = """
Analyse the talent intelligence bundle below and return a complete investor assessment.
Return ONLY this exact JSON structure — every field is required:

{{
  "company_summary": "<2-sentence factual company description>",

  "key_insight": "<The single most important signal for an investor in ~60 words. Be specific, not generic.>",

  "talent_signals": {{
    "headcount_trend": "<growing X%/shrinking X%/flat — cite the data>",
    "hiring_velocity": "<aggressive|moderate|slow|minimal — cite open roles count>",
    "top_hiring_roles": ["<role1>", "<role2>", "<role3>", "<role4>", "<role5>"],
    "engineering_ratio": "<estimated % of engineering in total headcount>",
    "recent_leadership_changes": ["<change1 with date if known>", "<change2>"],
    "attrition_signals": "<high|medium|low — evidence from reviews/headcount data>"
  }},

  "talent_flow": {{
    "description": "<1-2 sentences on inflow vs outflow patterns>",
    "inflow_signals": ["<specific signal of talent joining>", "<second signal>"],
    "outflow_signals": ["<specific signal of talent leaving>", "<second signal>"],
    "net_talent_quality": "<improving|stable|declining — explain why>",
    "chart_data": {{
      "labels": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024", "Q1 2025", "Q2 2025"],
      "inflow":  [<estimated inflow Q1 2024>, <Q2>, <Q3>, <Q4>, <Q1 2025>, <Q2 2025>],
      "outflow": [<estimated outflow Q1 2024>, <Q2>, <Q3>, <Q4>, <Q1 2025>, <Q2 2025>]
    }}
  }},

  "skill_signals": {{
    "hiring_skills": ["<skill being actively hired 1>", "<2>", "<3>", "<4>", "<5>", "<6>"],
    "losing_skills": ["<skill being lost or deprioritised 1>", "<2>", "<3>"],
    "emerging_skills": ["<new skill cluster appearing in JDs>"],
    "skill_verdict": "<what the hiring pattern signals about company direction>"
  }},

  "alumni_success_distribution": [
    {{
      "source_name": "<school or company name>",
      "source_type": "<school|company>",
      "success_pct": <0-100>,
      "sample_size": "<e.g. '~40 founders tracked'>",
      "pattern": "<one sentence on what makes this network valuable or not>"
    }}
  ],

  "investment_score": {{
    "score": <0-100 weighted: founder_strength*0.40 + talent_quality*0.25 + hiring_signals*0.15 + market_alignment*0.10 + external_sentiment*0.10>,
    "breakdown": {{
      "founder_strength":    <0-100, weight 40%>,
      "talent_quality":      <0-100, weight 25%>,
      "hiring_signals":      <0-100, weight 15%>,
      "market_alignment":    <0-100, weight 10%>,
      "external_sentiment":  <0-100, weight 10%>
    }},
    "confidence": "<high|medium|low>",
    "confidence_reason": "<why this confidence level — e.g. 'limited founder data available'>",
    "verdict": "<one punchy investment verdict sentence>",
    "key_reasons": [
      "<specific evidence-backed reason to invest>",
      "<second reason>",
      "<third reason>"
    ],
    "risks": [
      "<specific risk with evidence>",
      "<second risk>"
    ]
  }}
}}

SCORING WEIGHTS (apply exactly):
  founder_strength   = 40% of final score
  talent_quality     = 25% of final score
  hiring_signals     = 15% of final score
  market_alignment   = 10% of final score
  external_sentiment = 10% of final score

Final score = (founder_strength × 0.40) + (talent_quality × 0.25) +
              (hiring_signals × 0.15) + (market_alignment × 0.10) +
              (external_sentiment × 0.10)

Signal Bundle:
{signal_bundle}
"""


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_investor_analysis(
    client: CrustDataClient,
    llm: BedrockLLM,
    company_name: str,
    company_domain: Optional[str] = None,
    include_visual: bool = True,
    include_rating: bool = True,
    max_founders: int = 3,
) -> InvestorOutput:

    # ── Step 1: Pedigree (company + founders + alumni) ────────────────────────
    logger.info("[Investor] Starting pedigree analysis for: %s", company_name)
    pedigree = await analyze_pedigree(
        client=client, llm=llm,
        company_name=company_name, company_domain=company_domain,
        include_schools=True, max_founders=max_founders,
    )
    company        = pedigree["company"]
    founders       = pedigree["founders"]
    alumni_insights = pedigree["alumni_insights"]
    web_signals    = pedigree["web_signals"]

    # ── Step 2: Live job signals (if company_id available) ────────────────────
    job_signals: Optional[dict] = None
    company_id = company.get("id")
    if company_id:
        try:
            job_signals = await search_company_jobs(client, company_id=company_id, limit=25)
            logger.info("[Investor] Job signals: %d open roles", job_signals.get("total", 0))
        except Exception as e:
            logger.warning("[Investor] Job signals failed: %s", e)

    # ── Step 3: Additional web signals ───────────────────────────────────────
    extra_web = await gather_web_signals(client, company_name, topics=[
        f"{company_name} hiring talent culture 2025",
        f"{company_name} employee reviews sentiment",
    ])
    all_web = (web_signals + extra_web)[:12]

    # ── Step 4: Build rich signal bundle for LLM ──────────────────────────────
    signal_bundle = build_investor_signal_bundle(company, founders, all_web, job_signals)
    logger.info("[Investor] Signal bundle built: %d chars", len(signal_bundle))

    # ── Step 5: LLM analysis via Bedrock Claude ───────────────────────────────
    llm_result: dict = {}
    if include_rating:
        prompt = INVESTOR_ANALYSIS_PROMPT.format(signal_bundle=signal_bundle)
        logger.info("[Investor] Calling Bedrock Claude for full investor analysis...")
        try:
            llm_result = await llm.ainvoke_json(prompt, system=INVESTOR_SYSTEM, max_tokens=8192)
            logger.info("[Investor] Bedrock analysis complete. Score: %s, Confidence: %s",
                       llm_result.get("investment_score", {}).get("score"),
                       llm_result.get("investment_score", {}).get("confidence"))
        except Exception as e:
            logger.error("[Investor] Bedrock call failed: %s", e)

    # ── Step 6: Parse LLM output ──────────────────────────────────────────────
    company_summary = llm_result.get("company_summary") or company.get("description") or ""
    key_insight     = llm_result.get("key_insight") or "Analysis based on available signals."

    ts_raw = llm_result.get("talent_signals") or {}
    talent_signals = TalentSignals(
        headcount_trend=ts_raw.get("headcount_trend"),
        hiring_velocity=ts_raw.get("hiring_velocity"),
        top_hiring_roles=ts_raw.get("top_hiring_roles") or [],
        engineering_ratio=ts_raw.get("engineering_ratio"),
        recent_leadership_changes=ts_raw.get("recent_leadership_changes") or [],
        attrition_signals=ts_raw.get("attrition_signals"),
    )

    final_score: Optional[InvestmentScore] = None
    if include_rating and llm_result:
        sc_raw = llm_result.get("investment_score") or {}
        bd_raw = sc_raw.get("breakdown") or {}

        # Compute weighted score ourselves as ground truth
        fs   = float(bd_raw.get("founder_strength", 50))
        tq   = float(bd_raw.get("talent_quality", 50))
        hs   = float(bd_raw.get("hiring_signals", 50))
        ma   = float(bd_raw.get("market_alignment", 50))
        es   = float(bd_raw.get("external_sentiment", 50))
        computed_score = round(fs*0.40 + tq*0.25 + hs*0.15 + ma*0.10 + es*0.10, 1)

        final_score = InvestmentScore(
            score=computed_score,
            breakdown=ScoreBreakdown(
                founder_strength=fs, talent_quality=tq, hiring_signals=hs,
                market_alignment=ma, external_sentiment=es,
            ),
            confidence=sc_raw.get("confidence", "medium"),
            verdict=sc_raw.get("verdict", "Insufficient data for definitive verdict"),
            key_reasons=sc_raw.get("key_reasons") or [],
            risks=sc_raw.get("risks") or [],
        )
        logger.info("[Investor] Computed score: %.1f (LLM suggested: %s)",
                   computed_score, sc_raw.get("score"))

    # ── Step 7: Build visualizations ─────────────────────────────────────────
    charts = None
    if include_visual:
        charts = _build_investor_charts(
            founders=founders,
            final_score=final_score,
            company=company,
            alumni_insights=alumni_insights,
            llm_result=llm_result,
        )

    # ── Step 8: Assemble typed objects ────────────────────────────────────────
    from models import FounderProfile, EducationEntry, WorkEntry, FounderScores, AlumniInsight

    founder_objs = []
    for f in founders:
        try:
            scores_raw = f.get("scores")
            scores = FounderScores(**scores_raw) if isinstance(scores_raw, dict) else None
            founder_objs.append(FounderProfile(
                name=f.get("name") or "Unknown",
                title=f.get("title"),
                linkedin_url=f.get("linkedin_url"),
                education=[EducationEntry(**e) if isinstance(e, dict) else e for e in (f.get("education") or [])],
                work_history=[WorkEntry(**w) if isinstance(w, dict) else w for w in (f.get("work_history") or [])],
                scores=scores,
                key_insights=f.get("key_insights") or [],
                builder_signals=f.get("builder_signals") or [],
                red_flags=f.get("red_flags") or [],
            ))
        except Exception as e:
            logger.warning("Could not build FounderProfile: %s", e)

    alumni_objs = []
    for a in alumni_insights:
        try:
            alumni_objs.append(AlumniInsight(**a) if isinstance(a, dict) else a)
        except Exception:
            pass

    # Enrich alumni with success_score from LLM alumni_success_distribution
    alumni_dist = {
        d.get("source_name"): d.get("success_pct", 55)
        for d in (llm_result.get("alumni_success_distribution") or [])
        if isinstance(d, dict)
    }
    for a in alumni_objs:
        if hasattr(a, "source_name") and a.source_name in alumni_dist:
            object.__setattr__(a, "success_score", alumni_dist[a.source_name])

    return InvestorOutput(
        company_name=company.get("name") or company_name,
        company_summary=company_summary,
        founders=founder_objs,
        talent_signals=talent_signals,
        alumni_insights=alumni_objs,
        key_insight=key_insight,
        visualizations=charts,
        final_score=final_score,
        data_confidence=(
            "high" if company.get("id") and len(founder_objs) >= 2
            else "medium" if company.get("id")
            else "low"
        ),
        sources_used=list({"company_identify", "person_search", "person_enrich",
                           "web_search", "job_search"} |
                          ({"headcount_timeseries", "funding_timeseries"} if company.get("id") else set())),
    )


# ── Chart assembly ─────────────────────────────────────────────────────────────

def _build_investor_charts(
    founders: list[dict],
    final_score: Optional[InvestmentScore],
    company: dict,
    alumni_insights: list,
    llm_result: dict,
) -> dict:
    from visualization.chart_builder import (
        founder_radar_chart,
        investment_score_breakdown_chart,
        headcount_trend_chart,
        funding_timeline_chart,
        alumni_success_distribution_chart,
        talent_flow_chart,
        skill_heatmap_chart,
        founder_experience_timeline_chart,
    )

    charts: dict = {}

    # 1. Founder radar (5 dimensions per founder)
    charts["founder_radar"] = founder_radar_chart(founders)

    # 2. Investment score breakdown (weighted doughnut)
    if final_score:
        charts["score_breakdown"] = investment_score_breakdown_chart(
            final_score.breakdown.model_dump()
        )

    # 3. Headcount trend (line chart)
    hc_ts = company.get("headcount_timeseries") or {}
    hc_points = _flatten_ts(hc_ts)
    if hc_points:
        charts["headcount_trend"] = headcount_trend_chart(hc_points)

    # 4. Funding timeline (bar chart)
    fund_ts = company.get("funding_timeseries") or {}
    fund_points = _flatten_ts(fund_ts)
    if fund_points:
        charts["funding_timeline"] = funding_timeline_chart(fund_points)

    # 5. Alumni success distribution (% from LLM)
    alumni_dist_data = llm_result.get("alumni_success_distribution") or []
    # Fallback: use pedigree alumni_insights with default score
    if not alumni_dist_data and alumni_insights:
        alumni_dist_data = [
            {
                "source_name": a.get("source_name") if isinstance(a, dict) else getattr(a, "source_name", ""),
                "source_type": a.get("source_type") if isinstance(a, dict) else getattr(a, "source_type", ""),
                "success_pct": a.get("success_score", 55) if isinstance(a, dict) else 55,
                "pattern": "",
            }
            for a in alumni_insights[:6]
        ]
    if alumni_dist_data:
        charts["alumni_success_distribution"] = alumni_success_distribution_chart(alumni_dist_data)

    # 6. Talent flow (inflow vs outflow over time) — from LLM chart_data
    tf = llm_result.get("talent_flow") or {}
    tf_chart_data = tf.get("chart_data") or {}
    if tf_chart_data.get("labels") and tf_chart_data.get("inflow"):
        charts["talent_flow"] = talent_flow_chart(tf_chart_data)

    # 7. Skill heatmap (hiring vs losing)
    skill_signals = llm_result.get("skill_signals") or {}
    hiring_skills = skill_signals.get("hiring_skills") or llm_result.get("hiring_skills") or []
    losing_skills = skill_signals.get("losing_skills") or llm_result.get("losing_skills") or []
    ts_raw = llm_result.get("talent_signals") or {}
    if not hiring_skills:
        hiring_skills = ts_raw.get("top_hiring_roles") or []
    if hiring_skills or losing_skills:
        charts["skill_heatmap"] = skill_heatmap_chart(hiring_skills, losing_skills)

    # 8. Founder experience timeline
    if founders:
        charts["founder_experience"] = founder_experience_timeline_chart(founders)

    return charts


def _flatten_ts(data) -> list:
    if isinstance(data, list): return data
    if isinstance(data, dict):
        for k in ("data", "timeseries", "results", "milestones"):
            v = data.get(k)
            if isinstance(v, list) and v: return v
    return []
