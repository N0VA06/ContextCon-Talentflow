"""
Pedigree Analyzer
Deep-dives into founder pedigree:
  - Educational background + prestige tier
  - Previous company history and exit outcomes
  - Alumni networks (school + company)
  - Builder signals and public presence
  - LLM-judged per-founder scores across 5 dimensions

LLM scores each founder via Bedrock Claude with a rich signal bundle.
"""

from __future__ import annotations
import json
import logging
from typing import Optional

from crustdata_client import CrustDataClient, CrustDataError
from llm_client import BedrockLLM
from models import AlumniInsight, FounderProfile, FounderScores, EducationEntry, WorkEntry
from services.company_service import resolve_company, enrich_company, get_company_leaders
from services.person_service import enrich_people_batch, search_alumni, search_company_alumni
from services.signal_service import gather_web_signals

logger = logging.getLogger(__name__)

FOUNDER_TITLE_REGEX = "founder|co-founder|CEO|CTO|CPO|chief executive|chief technology|president"


# ── System prompt — shared across all pedigree LLM calls ──────────────────────

PEDIGREE_SYSTEM = """
You are a senior partner at a top-tier venture capital firm (Sequoia / a16z calibre).
You evaluate founder pedigree with rigour, data-driven reasoning, and healthy skepticism.
You look past vanity metrics and focus on evidence of real execution, technical depth,
and network quality that predicts founder success.
ALWAYS respond with valid JSON only — no markdown fences, no prose, no preamble.
"""


# ── Prompt 1: Per-founder score (5 dimensions) ────────────────────────────────

FOUNDER_SCORE_PROMPT = """
Score this founder across five quality dimensions based on the signal bundle below.
Be analytical, not descriptive. Reference specific signals as evidence.
DO NOT penalize first-time founders or those without prior CEO experience.
Return ONLY this JSON — every field is required:

{{
  "technical_depth": <0-100>,
  "execution": <0-100>,
  "network_strength": <0-100>,
  "market_understanding": <0-100>,
  "public_signal_strength": <0-100>,
  "overall": <0-100>,
  "key_insights": [
    "<specific data-backed insight about this founder>",
    "<second insight>",
    "<third insight>"
  ],
  "builder_signals": [
    "<evidence of shipping velocity or product thinking>",
    "<second builder signal>"
  ],
  "red_flags": [
    "<any concern or missing signal — say 'None detected' if clean>"
  ]
}}

SCORING GUIDE (be rigorous, default 50 if data is absent):
IMPORTANT: Do NOT penalize first-time founders. First-time founders can score high if they have
strong technical depth, domain expertise, or demonstrated execution at scale in IC roles.

- technical_depth (0-100): 
  * IC/Senior/Staff Engineer roles at top companies (70-90)
  * Engineering degree from strong school (add 10-15)
  * OSS contributions or technical publications (add 5-10)
  * CTO/VP Eng experience (add 5, but not required)
  * Domain expertise in relevant area (e.g., robotics, payments, ML) (70-85)
  * Shipping shipped products, built systems from scratch (70+)
  * First-time founders WITHOUT prior IC roles: look for self-taught, independent projects, hackathons
  
- execution (0-100):
  * Built/shipped products at scale (70-85)
  * Led teams of 5+ people (60-75)
  * Scaled systems (infrastructure, product, business) (70-85)
  * Handled hard problems with limited resources (65-80)
  * Prior exits/acquisitions (add 10-15)
  * Prior successful fundraising rounds (add 5-10)
  * Prior CEO/founder experience (add 10, but NOT required)
  * First-time founders: score on demonstrated project ownership, shipping velocity, problem-solving
  
- network_strength (0-100): 
  * Tier-1 school alumni (add 15-20)
  * Ex-FAANG/top startup experience (60-75)
  * YC acceptance or top accelerator (add 20)
  * Notable investor relationships (add 10-15)
  * LinkedIn connections 5000+ (add 5-10)
  * Industry visibility/reputation (60-80)
  
- market_understanding (0-100):
  * 3+ years domain expertise in relevant area (70-85)
  * Customer-facing roles (sales, PM, partnerships) (60-75)
  * Evidence of market validation or customer discovery (65-80)
  * Published insights/thought leadership (add 10-15)
  * Evidence of pivoting intelligently based on market signals (65-75)
  * Technical depth in niche domain counts as market understanding (60-75)
  
- public_signal_strength (0-100):
  * LinkedIn followers 1000+ (60-75)
  * Published articles/talks on relevant topics (add 10-15)
  * Open source projects with traction (add 10-20)
  * Podcast appearances or media mentions (add 5-10)
  * Twitter/X presence with engaged followers (add 5-15)
  * Academic publications or research (add 10-15)
  * Data gaps are OK — default 50 if unknown

Founder Signal Bundle:
{signal_bundle}
"""


# ── Prompt 2: Alumni network insight ─────────────────────────────────────────

ALUMNI_INSIGHT_PROMPT = """
You are generating an alumni intelligence report for a venture investor.
Analyse the alumni data below and return a concise, evidence-based assessment.
Return ONLY this JSON:

{{
  "success_patterns": [
    "<specific pattern observed in this alumni network>",
    "<second pattern>"
  ],
  "success_rate_estimate": "<e.g. '~40% of ex-Google PMs who became founders raised Series A+'>",
  "notable_alumni": ["<name>", "<name2>"],
  "success_score": <0-100>,
  "pedigree_verdict": "<one sentence: why this network matters or doesn't for an investor>"
}}

success_score guide:
  90-100: World-class network (ex-Google/Stripe/OpenAI founders, Tier-1 school with strong startup output)
  70-89:  Strong network with clear success patterns
  50-69:  Moderate — some signal but limited data
  30-49:  Weak signal — few notable alumni
  0-29:   Unknown or no data

Source type: {source_type}
Source name: {source_name}
Alumni data:
{alumni_data}
"""


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def analyze_pedigree(
    client: CrustDataClient,
    llm: BedrockLLM,
    company_name: str,
    company_domain: Optional[str] = None,
    include_schools: bool = True,
    max_founders: int = 3,
) -> dict:
    """
    Full pedigree pipeline. Returns dict:
      {company, founders (with LLM scores), alumni_insights, web_signals}
    """

    # Step 1 — Resolve + enrich company
    logger.info("[Pedigree] Resolving: %s", company_name)
    company = await resolve_company(client, name=company_name, domain=company_domain)
    company = await enrich_company(client, company)

    # Step 2 — Get leaders
    logger.info("[Pedigree] Fetching leaders for: %s", company.get("name"))
    raw_leaders = await get_company_leaders(client, company, max_founders=max_founders)
    raw_leaders = raw_leaders[:max_founders]
    logger.info("[Pedigree] Got %d raw leaders: %s", 
                len(raw_leaders), 
                [{"name": r.get("name"), "title": r.get("title")} for r in raw_leaders])

    # Step 3 — Batch enrich founders via /person/enrich
    logger.info("[Pedigree] Enriching %d founders", len(raw_leaders))
    from services.person_service import enrich_people_batch
    enriched_founders = await enrich_people_batch(client, raw_leaders)
    logger.info("[Pedigree] Enriched founders: %s", 
                [{"name": f.get("name"), "title": f.get("title")} for f in enriched_founders])

    # Step 4 — Web signals for context
    web_signals = await gather_web_signals(
        client, company_name,
        topics=[f"{company_name} founders background story", f"{company_name} team building"]
    )

    # Step 5 — LLM score each founder (Bedrock Claude call)
    founder_profiles = []
    for ef in enriched_founders:
        fp = await _score_founder_llm(llm, ef, company_name)
        founder_profiles.append(fp)

    # Step 6 — Alumni analysis (school + notable past companies)
    alumni_insights: list[AlumniInsight] = []
    seen_sources: set[str] = set()

    for fp in founder_profiles:
        if include_schools:
            for edu in fp.education[:2]:
                src = edu.institution
                if src and src not in seen_sources:
                    seen_sources.add(src)
                    insight = await _analyze_alumni(client, llm, "school", src)
                    if insight:
                        alumni_insights.append(insight)

        for job in fp.work_history[:2]:
            if job.is_notable and job.company not in seen_sources:
                seen_sources.add(job.company)
                insight = await _analyze_alumni(client, llm, "company", job.company)
                if insight:
                    alumni_insights.append(insight)

    return {
        "company":        company,
        "founders":       [f.model_dump() for f in founder_profiles],
        "alumni_insights": [a.model_dump() for a in alumni_insights],
        "web_signals":    web_signals,
    }


# ── LLM founder scoring ────────────────────────────────────────────────────────

async def _score_founder_llm(
    llm: BedrockLLM, founder_data: dict, company_name: str
) -> FounderProfile:
    """
    Call Bedrock Claude to score a founder across 5 dimensions.
    Builds a rich signal bundle from all available CrustData fields.
    """
    signal_bundle = json.dumps({
        "company_being_analyzed": company_name,
        "name":         founder_data.get("name"),
        "current_title": founder_data.get("title"),
        "location":     founder_data.get("location"),
        "connections":  founder_data.get("connections"),
        "summary":      (founder_data.get("summary") or "")[:800],
        "skills":       (founder_data.get("skills") or [])[:25],
        "education": [
            {
                "institution":   e.get("institution") if isinstance(e, dict) else str(e),
                "degree":        e.get("degree") if isinstance(e, dict) else None,
                "field":         e.get("field") if isinstance(e, dict) else None,
                "year":          e.get("year") if isinstance(e, dict) else None,
                "prestige_tier": e.get("prestige_tier") if isinstance(e, dict) else None,
            }
            for e in (founder_data.get("education") or [])[:5]
        ],
        "work_history": [
            {
                "company":    w.get("company") if isinstance(w, dict) else str(w),
                "role":       w.get("role") if isinstance(w, dict) else None,
                "start_year": w.get("start_year") if isinstance(w, dict) else None,
                "end_year":   w.get("end_year") if isinstance(w, dict) else None,
                "is_notable": w.get("is_notable") if isinstance(w, dict) else False,
            }
            for w in (founder_data.get("work_history") or [])[:8]
        ],
    }, indent=2)

    prompt = FOUNDER_SCORE_PROMPT.format(signal_bundle=signal_bundle)

    try:
        logger.info("[Pedigree] Calling Bedrock for founder: %s", founder_data.get("name"))
        scored = await llm.ainvoke_json(prompt, system=PEDIGREE_SYSTEM, max_tokens=4096)
        logger.info("[Pedigree] Bedrock founder score complete: overall=%s", scored.get("overall"))
    except Exception as e:
        logger.error("[Pedigree] Bedrock founder scoring failed: %s", e)
        scored = {}

    scores = FounderScores(
        technical_depth=float(scored.get("technical_depth", 50)),
        execution=float(scored.get("execution", 50)),
        network_strength=float(scored.get("network_strength", 50)),
        market_understanding=float(scored.get("market_understanding", 50)),
        public_signal_strength=float(scored.get("public_signal_strength", 50)),
        overall=float(scored.get("overall", 50)),
    )

    education = [
        EducationEntry(**e) if isinstance(e, dict) else e
        for e in (founder_data.get("education") or [])
    ]
    work_history = [
        WorkEntry(**w) if isinstance(w, dict) else w
        for w in (founder_data.get("work_history") or [])
    ]

    return FounderProfile(
        name=founder_data.get("name") or "Unknown",
        title=founder_data.get("title"),
        linkedin_url=founder_data.get("linkedin_url"),
        education=education,
        work_history=work_history,
        scores=scores,
        key_insights=scored.get("key_insights") or [],
        builder_signals=scored.get("builder_signals") or [],
        red_flags=scored.get("red_flags") or [],
    )


# ── Alumni analysis ────────────────────────────────────────────────────────────

async def _analyze_alumni(
    client: CrustDataClient,
    llm: BedrockLLM,
    source_type: str,   # "school" | "company"
    source_name: str,
) -> Optional[AlumniInsight]:
    if not source_name or source_name.lower() in ("unknown", ""):
        return None

    # Fetch alumni data from CrustData
    if source_type == "school":
        alumni = await search_alumni(client, source_name, [], limit=8)
    else:
        alumni = await search_company_alumni(client, source_name, limit=8)

    alumni_data = json.dumps([
        {"name": a.get("name"), "title": a.get("title"),
         "current_company": _get_current_company(a)}
        for a in alumni[:8]
    ], indent=2)

    prompt = ALUMNI_INSIGHT_PROMPT.format(
        source_type=source_type,
        source_name=source_name,
        alumni_data=alumni_data,
    )

    try:
        logger.info("[Pedigree] Calling Bedrock for %s alumni: %s", source_type, source_name)
        scored = await llm.ainvoke_json(prompt, system=PEDIGREE_SYSTEM, max_tokens=1024)
        logger.info("[Pedigree] Bedrock alumni score: %s = %s", source_name, scored.get("success_score"))
    except Exception as e:
        logger.warning("[Pedigree] Bedrock alumni scoring failed for %s: %s", source_name, e)
        scored = {}

    return AlumniInsight(
        source_type=source_type,
        source_name=source_name,
        notable_alumni=scored.get("notable_alumni") or [a.get("name", "") for a in alumni[:3]],
        success_patterns=scored.get("success_patterns") or [],
        success_rate_estimate=scored.get("success_rate_estimate"),
    )


def _get_current_company(person: dict) -> Optional[str]:
    wh = person.get("work_history") or []
    for job in wh:
        if isinstance(job, dict) and not job.get("end_year"):
            return job.get("company")
    return None
