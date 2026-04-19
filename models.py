"""
TalentFlow — Pydantic models for requests and responses.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Common
# ─────────────────────────────────────────────────────────────────────────────

class ChartDataset(BaseModel):
    label: str
    data: List[Union[float, int]]
    backgroundColor: Optional[Union[str, List[str]]] = None
    borderColor: Optional[str] = None
    fill: Optional[bool] = None


class ChartData(BaseModel):
    chart_type: str          # "radar" | "bar" | "line" | "doughnut" | "scatter"
    title: str
    description: Optional[str] = None
    labels: List[str]
    datasets: List[ChartDataset]
    x_label: Optional[str] = None
    y_label: Optional[str] = None


class EducationEntry(BaseModel):
    institution: str
    degree: Optional[str] = None
    field: Optional[str] = None
    year: Optional[int] = None
    prestige_tier: Optional[str] = None  # "Tier-1" / "Tier-2" / "Unknown"


class WorkEntry(BaseModel):
    company: str
    role: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    duration_months: Optional[int] = None
    is_notable: bool = False
    exit_outcome: Optional[str] = None  # "IPO" | "Acquired" | "Still running" | "Shut down"


# ─────────────────────────────────────────────────────────────────────────────
# Pedigree / Investor — Requests
# ─────────────────────────────────────────────────────────────────────────────

class PedigreeRequest(BaseModel):
    company_name: str = Field(..., description="Company to analyse (e.g. 'Stripe')")
    company_domain: Optional[str] = Field(None, description="Domain for faster lookup (e.g. 'stripe.com')")
    include_schools: bool = Field(True, description="Include alumni analysis from founder schools")
    include_visual: bool = Field(True, description="Return frontend-ready chart data")
    include_rating: bool = Field(True, description="Include LLM-judged investment score")
    max_founders: int = Field(3, ge=1, le=10, description="Number of founders to deep-analyse")


class InvestorAnalysisRequest(BaseModel):
    company_name: str
    company_domain: Optional[str] = None
    include_visual: bool = True
    include_rating: bool = True
    max_founders: int = Field(3, ge=1, le=10)


# ─────────────────────────────────────────────────────────────────────────────
# HR — Requests
# ─────────────────────────────────────────────────────────────────────────────

class CandidateInput(BaseModel):
    name: str
    linkedin_url: Optional[str] = None
    email: Optional[str] = None
    resume_text: Optional[str] = None
    skills: Optional[List[str]] = None
    current_role: Optional[str] = None
    years_experience: Optional[int] = None
    education: Optional[List[EducationEntry]] = None
    work_history: Optional[List[WorkEntry]] = None


class HRAnalysisRequest(BaseModel):
    """
    HR analysis supports two input modes (can mix both):
      1. Structured candidates list
      2. Free-text resume(s) embedded in candidates[].resume_text
    """
    jd_text: Optional[str] = Field(None, description="Job description as free text")
    jd_structured: Optional[Dict[str, Any]] = Field(
        None, description="Structured JD: {title, skills, experience_years, ...}"
    )
    candidates: Optional[List[CandidateInput]] = Field(
        None, description="Candidate list (structured or with resume_text)"
    )
    target_company: Optional[str] = Field(
        None, description="Source company to mine for talent pool"
    )
    include_visual: bool = True
    include_recommendations: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Founder Profile
# ─────────────────────────────────────────────────────────────────────────────

class FounderScores(BaseModel):
    technical_depth: float = Field(..., ge=0, le=100)
    execution: float = Field(..., ge=0, le=100)
    network_strength: float = Field(..., ge=0, le=100)
    market_understanding: float = Field(..., ge=0, le=100)
    public_signal_strength: float = Field(..., ge=0, le=100)
    overall: float = Field(..., ge=0, le=100)


class FounderProfile(BaseModel):
    name: str
    title: Optional[str] = None
    linkedin_url: Optional[str] = None
    education: List[EducationEntry] = []
    work_history: List[WorkEntry] = []
    scores: Optional[FounderScores] = None
    key_insights: List[str] = []
    builder_signals: List[str] = []
    red_flags: List[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# Alumni Insight
# ─────────────────────────────────────────────────────────────────────────────

class AlumniInsight(BaseModel):
    source_type: str        # "school" | "company"
    source_name: str
    notable_alumni: List[str] = []
    success_patterns: List[str] = []
    success_rate_estimate: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Investment Score
# ─────────────────────────────────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    founder_strength: float
    talent_quality: float
    hiring_signals: float
    market_alignment: float
    external_sentiment: float


class InvestmentScore(BaseModel):
    score: float = Field(..., ge=0, le=100)
    breakdown: ScoreBreakdown
    confidence: str          # "high" | "medium" | "low"
    verdict: str
    key_reasons: List[str] = []
    risks: List[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# Investor Response
# ─────────────────────────────────────────────────────────────────────────────

class TalentSignals(BaseModel):
    headcount_trend: Optional[str] = None
    hiring_velocity: Optional[str] = None
    top_hiring_roles: List[str] = []
    engineering_ratio: Optional[str] = None
    recent_leadership_changes: List[str] = []
    attrition_signals: Optional[str] = None


class InvestorOutput(BaseModel):
    company_name: str
    company_summary: Optional[str] = None
    founders: List[FounderProfile] = []
    talent_signals: Optional[TalentSignals] = None
    alumni_insights: List[AlumniInsight] = []
    key_insight: str
    visualizations: Optional[Dict[str, ChartData]] = None
    final_score: Optional[InvestmentScore] = None
    data_confidence: str    # "high" | "medium" | "low"
    sources_used: List[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# HR Response
# ─────────────────────────────────────────────────────────────────────────────

class CandidateScore(BaseModel):
    name: str
    skill_match_score: float = Field(..., ge=0, le=100)
    experience_alignment: float = Field(..., ge=0, le=100)
    cultural_fit_score: float = Field(..., ge=0, le=100)
    overall_score: float = Field(..., ge=0, le=100)
    strengths: List[str] = []
    gaps: List[str] = []
    recommendation: str     # "Strong Yes" | "Yes" | "Maybe" | "No"


class TalentShiftInsight(BaseModel):
    pattern: str
    source_industry: str
    target_industry: str
    signal_strength: str    # "Strong" | "Moderate" | "Weak"
    relevance_to_jd: str


class HROutput(BaseModel):
    jd_summary: Optional[str] = None
    candidate_scores: List[CandidateScore] = []
    talent_pool_analysis: Optional[str] = None
    skill_gap_analysis: Optional[str] = None
    talent_shift_insights: List[TalentShiftInsight] = []
    visualizations: Optional[Dict[str, ChartData]] = None
    recommendations: List[str] = []
    top_source_companies: List[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# Job Search
# ─────────────────────────────────────────────────────────────────────────────

class JobSearchRequest(BaseModel):
    company_name:   Optional[str]  = Field(None, description="Company name")
    company_id:     Optional[int]  = Field(None, description="CrustData company_id (faster)")
    category:       Optional[str]  = Field(None, description="Engineering | Sales | Marketing …")
    title_keywords: Optional[str]  = Field(None, description="Regex keywords e.g. 'ML|AI|data'")
    skills:         Optional[List[str]] = Field(None, description="Skills to match in JD text")
    country:        Optional[str]  = Field(None, description="Country filter")
    limit:          int            = Field(20, ge=1, le=100)
    include_visual: bool           = True


class JobListing(BaseModel):
    title:        str
    url:          Optional[str] = None
    category:     Optional[str] = None
    description:  Optional[str] = None
    country:      Optional[str] = None
    city:         Optional[str] = None
    date_added:   Optional[str] = None
    company_name: Optional[str] = None
    company_id:   Optional[int] = None
    is_remote:    bool = False


class HiringSignals(BaseModel):
    total_open_roles:   int = 0
    category_breakdown: Dict[str, int] = {}
    top_titles:         List[str] = []
    top_locations:      List[str] = []
    hiring_velocity:    str = "unknown"
    engineering_ratio:  Optional[str] = None


class JobSearchResponse(BaseModel):
    status:         str = "success"
    company_name:   Optional[str] = None
    jobs:           List[JobListing] = []
    total:          int = 0
    hiring_signals: Optional[HiringSignals] = None
    visualizations: Optional[Dict[str, ChartData]] = None
    next_cursor:    Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Unified Response
# ─────────────────────────────────────────────────────────────────────────────

class TalentFlowResponse(BaseModel):
    status: str = "success"
    mode: str               # "investor" | "hr" | "pedigree"
    investor: Optional[InvestorOutput] = None
    hr: Optional[HROutput] = None
    error: Optional[str] = None
