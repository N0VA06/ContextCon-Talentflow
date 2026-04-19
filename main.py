"""
TalentFlow API — Main FastAPI Application
Routes:
  GET  /                          → Frontend (index.html)
  POST /api/v1/pedigree/analyze   → Founder pedigree (schools + companies)
  POST /api/v1/investor/analyze   → Full investor intelligence
  POST /api/v1/hr/analyze         → HR / recruiter intelligence
  GET  /api/v1/health             → Health check
"""

from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from crustdata_client import CrustDataClient, CrustDataError
from llm_client import BedrockLLM
from models import (
    PedigreeRequest,
    InvestorAnalysisRequest,
    HRAnalysisRequest,
    JobSearchRequest,
    JobSearchResponse,
    JobListing,
    HiringSignals,
    TalentFlowResponse,
    InvestorOutput,
    HROutput,
    AlumniInsight,
    FounderProfile,
    EducationEntry,
    WorkEntry,
    FounderScores,
)
from analyzers.pedigree_analyzer import analyze_pedigree
from analyzers.investor_analyzer import run_investor_analysis
from analyzers.hr_analyzer import run_hr_analysis

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Shared state (instantiated once at startup) ────────────────────────────────

class AppState:
    crustdata: CrustDataClient
    llm: BedrockLLM


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy clients once at startup."""
    logger.info("🚀 TalentFlow starting up…")
    try:
        state.crustdata = CrustDataClient(timeout=settings.REQUEST_TIMEOUT)
        logger.info("✅ CrustData client ready (version=%s)", settings.CRUSTDATA_API_VERSION)
    except ValueError as e:
        logger.error("❌ CrustData init failed: %s", e)
        raise

    try:
        state.llm = BedrockLLM()
        logger.info("✅ Bedrock LLM ready")
    except ValueError as e:
        logger.error("❌ Bedrock init failed: %s", e)
        raise

    yield
    logger.info("🛑 TalentFlow shutting down")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="Talent Intelligence Platform — Investor & HR Analysis powered by CrustData + Claude",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ───────────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info("✅ Static files mounted at /static")
else:
    logger.warning("⚠️ Static directory not found at %s", static_dir)


# ── Middleware: request timing ─────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed = time.monotonic() - t0
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(CrustDataError)
async def crustdata_error_handler(request: Request, exc: CrustDataError):
    logger.error("CrustData error: %s", exc)
    return JSONResponse(
        status_code=502,
        content={"detail": f"Upstream data error: {exc}"},
    )


@app.exception_handler(TimeoutError)
async def timeout_handler(request: Request, exc: TimeoutError):
    return JSONResponse(
        status_code=504,
        content={"detail": f"Analysis timed out: {exc}"},
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Frontend"])
async def root():
    """Serve the frontend HTML"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "TalentFlow API — see /docs for endpoints"}

@app.get("/api/v1/health", tags=["Meta"])
async def health():
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "crustdata_api_version": settings.CRUSTDATA_API_VERSION,
        "crustdata_configured": bool(settings.CRUSTDATA_API_KEY),
        "bedrock_configured": bool(settings.AWS_ACCESS_KEY_ID),
    }


# ── Bedrock LLM Test ──────────────────────────────────────────────────────────

@app.post("/api/v1/test/llm", tags=["Meta"], summary="Test Bedrock Claude connection")
async def test_llm():
    """
    Verify Bedrock Claude is reachable and responding.
    Returns model response + latency so you can confirm the integration is live.
    """
    import time
    t0 = time.monotonic()
    try:
        response = await state.llm.ainvoke(
            prompt='Return ONLY this JSON: {"status":"ok","model":"claude","message":"Bedrock LLM is connected and working."}',
            system="You are a test assistant. Return ONLY valid JSON, no preamble.",
            max_tokens=100,
        )
        elapsed = round(time.monotonic() - t0, 2)
        return {
            "bedrock_status": "connected",
            "latency_seconds": elapsed,
            "raw_response": response,
            "model_profile": settings.CLAUDE_INFERENCE_PROFILE,
        }
    except Exception as e:
        return {
            "bedrock_status": "error",
            "error": str(e),
            "hint": "Check AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION in .env",
        }


@app.get("/api/v1/test/crustdata", tags=["Meta"], summary="Test CrustData connection")
async def test_crustdata():
    """Verify CrustData API key and connectivity."""
    import time
    t0 = time.monotonic()
    try:
        data = await state.crustdata.company_identify(name="Stripe")
        elapsed = round(time.monotonic() - t0, 2)
        count = len(data) if isinstance(data, list) else 1
        return {
            "crustdata_status": "connected",
            "latency_seconds": elapsed,
            "api_version": settings.CRUSTDATA_API_VERSION,
            "test_query": "company_identify('Stripe')",
            "results_count": count,
        }
    except Exception as e:
        return {
            "crustdata_status": "error",
            "error": str(e),
            "hint": "Check CRUSTDATA_API_KEY in .env",
        }


@app.get("/api/v1/autocomplete/titles", tags=["Utilities"])
async def autocomplete_titles(q: str = "", limit: int = 10):
    """Discover valid job title values for person search filters."""
    vals = await state.crustdata.person_search_autocomplete(
        "experience.employment_details.current.title", query=q, limit=limit
    )
    return {"field": "title", "suggestions": vals}


@app.get("/api/v1/autocomplete/companies", tags=["Utilities"])
async def autocomplete_companies(q: str = "", limit: int = 10):
    """Discover valid company name values for person search filters."""
    vals = await state.crustdata.person_search_autocomplete(
        "experience.employment_details.current.company_name", query=q, limit=limit
    )
    return {"field": "company_name", "suggestions": vals}


@app.get("/api/v1/autocomplete/schools", tags=["Utilities"])
async def autocomplete_schools(q: str = "", limit: int = 10):
    """Discover valid school names for person search filters."""
    vals = await state.crustdata.person_search_autocomplete(
        "education.schools.school", query=q, limit=limit
    )
    return {"field": "school", "suggestions": vals}


# ── Pedigree ──────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/pedigree/analyze",
    response_model=TalentFlowResponse,
    tags=["Investor"],
    summary="Analyse founder pedigree — schools, previous companies, alumni networks",
)
async def pedigree_analyze(req: PedigreeRequest):
    """
    Deep-dives into:
    - Founder educational background + school alumni success patterns
    - Previous company history + company alumni performance
    - Builder signals (public presence, technical depth)
    - LLM-judged per-founder scores
    - Frontend-ready visualizations
    """
    logger.info("Pedigree analysis request: %s", req.company_name)
    try:
        pedigree = await analyze_pedigree(
            client=state.crustdata,
            llm=state.llm,
            company_name=req.company_name,
            company_domain=req.company_domain,
            include_schools=req.include_schools,
            max_founders=req.max_founders,
        )
    except CrustDataError as e:
        raise HTTPException(status_code=502, detail=f"Upstream data error: {e}")
    except Exception as e:
        logger.exception("Pedigree analysis failed")
        raise HTTPException(status_code=500, detail=str(e))

    company = pedigree["company"]
    founders_raw = pedigree["founders"]
    alumni_raw = pedigree["alumni_insights"]

    # Rebuild typed objects
    founder_objs = _rebuild_founders(founders_raw)
    alumni_objs = _rebuild_alumni(alumni_raw)

    # Optional investment score (pedigree-only, lighter than full investor)
    final_score = None
    if req.include_rating and founder_objs:
        final_score = await _quick_pedigree_score(state.llm, founder_objs, company)

    # Visualizations
    charts = None
    if req.include_visual:
        from visualization.chart_builder import founder_radar_chart, alumni_success_chart
        charts = {
            "founder_radar": founder_radar_chart(founders_raw),
        }
        if alumni_raw:
            enriched_alumni = [
                {**a, "success_score": 55} if "success_score" not in a else a
                for a in alumni_raw
            ]
            charts["alumni_success"] = alumni_success_chart(enriched_alumni)

    investor_out = InvestorOutput(
        company_name=company.get("name") or req.company_name,
        company_summary=company.get("description"),
        founders=founder_objs,
        alumni_insights=alumni_objs,
        key_insight=_pedigree_key_insight(founder_objs, alumni_objs),
        visualizations=charts,
        final_score=final_score,
        data_confidence="high" if company.get("id") else "medium",
        sources_used=["company_identify", "person_enrich", "web_search"],
    )

    return TalentFlowResponse(
        status="success",
        mode="pedigree",
        investor=investor_out,
    )


# ── Full Investor Analysis ─────────────────────────────────────────────────────

@app.post(
    "/api/v1/investor/analyze",
    response_model=TalentFlowResponse,
    tags=["Investor"],
    summary="Full investor intelligence — talent signals, hiring trends, LLM score",
)
async def investor_analyze(req: InvestorAnalysisRequest):
    """
    Complete investor analysis:
    - Pedigree (founders + alumni)
    - Talent signals (headcount, hiring velocity, attrition)
    - Funding timeline
    - LLM-judged investment score (0–100)
    - Frontend-ready charts
    """
    logger.info("Investor analysis request: %s", req.company_name)
    try:
        result = await run_investor_analysis(
            client=state.crustdata,
            llm=state.llm,
            company_name=req.company_name,
            company_domain=req.company_domain,
            include_visual=req.include_visual,
            include_rating=req.include_rating,
            max_founders=req.max_founders,
        )
    except CrustDataError as e:
        raise HTTPException(status_code=502, detail=f"Upstream data error: {e}")
    except Exception as e:
        logger.exception("Investor analysis failed")
        raise HTTPException(status_code=500, detail=str(e))

    return TalentFlowResponse(
        status="success",
        mode="investor",
        investor=result,
    )


# ── HR Analysis ───────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/hr/analyze",
    response_model=TalentFlowResponse,
    tags=["HR"],
    summary="HR intelligence — candidate scoring, talent shifts, pool analysis",
)
async def hr_analyze(req: HRAnalysisRequest):
    """
    HR / Recruiter analysis:
    - Score candidates against JD (structured or free-text resume)
    - Detect talent shift patterns across industries
    - Identify best source companies
    - Visualizations: score distribution, skill match histogram, industry shift map

    Input modes (can mix):
    - `jd_text`: paste the JD as plain text
    - `jd_structured`: structured dict with title, skills, etc.
    - `candidates`: list of CandidateInput (with optional resume_text)
    - `target_company`: mine talent pool from this company
    """
    logger.info(
        "HR analysis request — candidates=%d, target_company=%s",
        len(req.candidates or []),
        req.target_company,
    )

    if not req.jd_text and not req.jd_structured and not req.target_company:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: jd_text, jd_structured, or target_company",
        )

    try:
        result = await run_hr_analysis(
            client=state.crustdata,
            llm=state.llm,
            jd_text=req.jd_text,
            jd_structured=req.jd_structured,
            candidates=req.candidates,
            target_company=req.target_company,
            include_visual=req.include_visual,
            include_recommendations=req.include_recommendations,
        )
    except CrustDataError as e:
        raise HTTPException(status_code=502, detail=f"Upstream data error: {e}")
    except Exception as e:
        logger.exception("HR analysis failed")
        raise HTTPException(status_code=500, detail=str(e))

    return TalentFlowResponse(
        status="success",
        mode="hr",
        hr=result,
    )


# ── Job Search ────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/jobs/search",
    response_model=JobSearchResponse,
    tags=["Jobs"],
    summary="Search active job listings — hiring velocity, category breakdown, top titles",
)
async def jobs_search(req: JobSearchRequest):
    """
    Search job listings via POST /job/search.

    Input modes (at least one required):
    - `company_id`   — fastest, use CrustData numeric ID
    - `company_name` — resolved via name match
    - `skills`       — search JD text for skill keywords
    - `category`     — Engineering | Sales | Marketing | Product | …

    Returns structured job list + hiring signals + frontend-ready charts.
    """
    if not req.company_id and not req.company_name and not req.skills:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: company_id, company_name, or skills",
        )

    logger.info(
        "Job search — company=%s/%s  category=%s  limit=%d",
        req.company_id, req.company_name, req.category, req.limit,
    )

    from services.job_service import search_company_jobs, search_jobs_by_skills

    try:
        if req.skills and not req.company_id and not req.company_name:
            result = await search_jobs_by_skills(
                state.crustdata, req.skills, country=req.country, limit=req.limit
            )
        else:
            # If we have a name but no ID, try to resolve first
            company_id = req.company_id
            company_name = req.company_name
            if company_name and not company_id:
                from services.company_service import resolve_company
                company = await resolve_company(state.crustdata, name=company_name)
                company_id = company.get("id")
                company_name = company.get("name") or company_name

            result = await search_company_jobs(
                state.crustdata,
                company_id=company_id,
                company_name=company_name,
                category=req.category,
                title_keywords=req.title_keywords,
                limit=req.limit,
            )
    except CrustDataError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
    except Exception as e:
        logger.exception("Job search failed")
        raise HTTPException(status_code=500, detail=str(e))

    jobs   = result.get("jobs", [])
    total  = result.get("total", 0)
    signals_raw = result.get("signals", {})

    # Build typed job listings
    job_objs = []
    for j in jobs:
        try:
            job_objs.append(JobListing(**j))
        except Exception:
            pass

    hiring_signals = HiringSignals(
        total_open_roles   = signals_raw.get("total_open_roles", total),
        category_breakdown = signals_raw.get("category_breakdown", {}),
        top_titles         = signals_raw.get("top_titles", []),
        top_locations      = signals_raw.get("top_locations", []),
        hiring_velocity    = signals_raw.get("hiring_velocity", "unknown"),
        engineering_ratio  = signals_raw.get("engineering_ratio"),
    )

    # Charts
    charts = None
    if req.include_visual and signals_raw:
        from visualization.chart_builder import (
            job_category_chart, job_titles_chart, hiring_velocity_gauge
        )
        charts = {}
        if signals_raw.get("category_breakdown"):
            charts["category_breakdown"] = job_category_chart(signals_raw["category_breakdown"])
        if signals_raw.get("top_titles"):
            charts["top_titles"] = job_titles_chart(signals_raw["top_titles"])
        charts["hiring_velocity"] = hiring_velocity_gauge(
            signals_raw.get("total_open_roles", total),
            signals_raw.get("hiring_velocity", "unknown"),
        )

    return JobSearchResponse(
        status         = "success",
        company_name   = req.company_name,
        jobs           = job_objs,
        total          = total,
        hiring_signals = hiring_signals,
        visualizations = charts,
        next_cursor    = result.get("next_cursor"),
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _rebuild_founders(raw_list: list[dict]) -> list[FounderProfile]:
    out = []
    for f in raw_list:
        try:
            scores_raw = f.get("scores")
            scores = FounderScores(**scores_raw) if isinstance(scores_raw, dict) else None
            education = [
                EducationEntry(**e) if isinstance(e, dict) else e
                for e in f.get("education", [])
            ]
            work_history = [
                WorkEntry(**w) if isinstance(w, dict) else w
                for w in f.get("work_history", [])
            ]
            out.append(
                FounderProfile(
                    name=f.get("name") or "Unknown",
                    title=f.get("title"),
                    linkedin_url=f.get("linkedin_url"),
                    education=education,
                    work_history=work_history,
                    scores=scores,
                    key_insights=f.get("key_insights") or [],
                    builder_signals=f.get("builder_signals") or [],
                    red_flags=f.get("red_flags") or [],
                )
            )
        except Exception as e:
            logger.warning("Skipping malformed founder: %s", e)
    return out


def _rebuild_alumni(raw_list: list[dict]) -> list[AlumniInsight]:
    out = []
    for a in raw_list:
        try:
            out.append(AlumniInsight(**a) if isinstance(a, dict) else a)
        except Exception:
            pass
    return out


def _pedigree_key_insight(
    founders: list[FounderProfile],
    alumni: list[AlumniInsight],
) -> str:
    if not founders:
        return "Insufficient founder data available for pedigree assessment."
    f = founders[0]
    school_names = [e.institution for e in f.education if e.institution]
    company_names = [w.company for w in f.work_history if w.is_notable]
    parts = []
    if school_names:
        parts.append(f"Lead founder from {school_names[0]}")
    if company_names:
        parts.append(f"previously at {', '.join(company_names[:2])}")
    if f.scores:
        parts.append(f"overall founder score {f.scores.overall:.0f}/100")
    if alumni:
        parts.append(f"{len(alumni)} alumni network(s) analysed")
    return ". ".join(parts) + "." if parts else "Pedigree analysis complete."


async def _quick_pedigree_score(
    llm: BedrockLLM,
    founders: list[FounderProfile],
    company: dict,
) -> Optional[object]:
    """Light-weight LLM scoring for pedigree-only endpoint."""
    import json
    from models import InvestmentScore, ScoreBreakdown

    bundle = json.dumps(
        {
            "company": company.get("name"),
            "founders": [
                {
                    "name": f.name,
                    "education": [e.model_dump() for e in f.education[:3]],
                    "work_history": [w.model_dump() for w in f.work_history[:5]],
                    "scores": f.scores.model_dump() if f.scores else {},
                    "key_insights": f.key_insights,
                }
                for f in founders
            ],
        },
        indent=2,
    )
    prompt = f"""
Score this founding team for investment potential based ONLY on pedigree signals.
Return ONLY this JSON:
{{
  "score": <0-100>,
  "breakdown": {{
    "founder_strength": <0-100>,
    "talent_quality": <0-100>,
    "hiring_signals": 50,
    "market_alignment": 50,
    "external_sentiment": 50
  }},
  "confidence": "<high|medium|low>",
  "verdict": "<one-line verdict>",
  "key_reasons": ["<reason1>", "<reason2>"],
  "risks": ["<risk1>"]
}}

Pedigree bundle:
{bundle}
"""
    try:
        res = await llm.ainvoke_json(prompt)
        bd = res.get("breakdown") or {}
        return InvestmentScore(
            score=float(res.get("score", 50)),
            breakdown=ScoreBreakdown(
                founder_strength=float(bd.get("founder_strength", 50)),
                talent_quality=float(bd.get("talent_quality", 50)),
                hiring_signals=float(bd.get("hiring_signals", 50)),
                market_alignment=float(bd.get("market_alignment", 50)),
                external_sentiment=float(bd.get("external_sentiment", 50)),
            ),
            confidence=res.get("confidence", "medium"),
            verdict=res.get("verdict", ""),
            key_reasons=res.get("key_reasons") or [],
            risks=res.get("risks") or [],
        )
    except Exception as e:
        logger.warning("Quick pedigree score failed: %s", e)
        return None
