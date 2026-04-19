"""
Chart Builder — complete per spec.

Required investor charts:
  ✅ founder_radar_chart               (5 dimensions per founder)
  ✅ investment_score_breakdown_chart  (weighted doughnut)
  ✅ talent_flow_chart                 (inflow vs outflow over time)
  ✅ skill_heatmap_chart               (hiring vs losing)
  ✅ alumni_success_distribution_chart (% successful founders per background)
  ✅ headcount_trend_chart             (line)
  ✅ funding_timeline_chart            (bar)
  ✅ founder_experience_timeline_chart

Required HR charts:
  ✅ candidate_score_bar_chart         (grouped bar)
  ✅ skill_match_histogram             (histogram)
  ✅ talent_pool_distribution_chart    (doughnut)
  ✅ industry_shift_chart              (bar)
  ✅ talent_source_map_chart           (horizontal bar — which companies)
  ✅ recommendation_radar_chart        (single candidate radar)

All output is Chart.js / Recharts compatible.
"""

from __future__ import annotations
from typing import Any
from models import ChartData, ChartDataset

# ── Design tokens ──────────────────────────────────────────────────────────────
PALETTE = {
    "indigo":  {"solid": "#6366f1", "soft": "rgba(99,102,241,0.15)",  "mid": "rgba(99,102,241,0.7)"},
    "rose":    {"solid": "#f43f5e", "soft": "rgba(244,63,94,0.15)",   "mid": "rgba(244,63,94,0.7)"},
    "emerald": {"solid": "#10b981", "soft": "rgba(16,185,129,0.15)",  "mid": "rgba(16,185,129,0.7)"},
    "amber":   {"solid": "#f59e0b", "soft": "rgba(245,158,11,0.15)",  "mid": "rgba(245,158,11,0.7)"},
    "violet":  {"solid": "#8b5cf6", "soft": "rgba(139,92,246,0.15)",  "mid": "rgba(139,92,246,0.7)"},
    "sky":     {"solid": "#0ea5e9", "soft": "rgba(14,165,233,0.15)",  "mid": "rgba(14,165,233,0.7)"},
    "orange":  {"solid": "#f97316", "soft": "rgba(249,115,22,0.15)",  "mid": "rgba(249,115,22,0.7)"},
    "teal":    {"solid": "#14b8a6", "soft": "rgba(20,184,166,0.15)",  "mid": "rgba(20,184,166,0.7)"},
}
SEQ = [PALETTE[k] for k in ("indigo","rose","emerald","amber","violet","sky","orange","teal")]

def _solid(n: int) -> list[str]:  return [SEQ[i % len(SEQ)]["solid"] for i in range(n)]
def _mid(n: int) -> list[str]:    return [SEQ[i % len(SEQ)]["mid"] for i in range(n)]
def _clamp(v: Any, lo=0.0, hi=100.0) -> float:
    try: return max(lo, min(hi, float(v)))
    except: return 50.0


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTOR CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def founder_radar_chart(founders: list[dict]) -> ChartData:
    """
    Radar chart — 5-dimension quality score per founder.
    Frontend: Chart.js RadarChart / Recharts RadarChart.
    """
    labels = [
        "Technical Depth", "Execution Track Record",
        "Network Strength", "Market Understanding", "Public Signal"
    ]
    datasets = []
    for i, f in enumerate(founders[:4]):
        sc = f.get("scores") or {}
        if isinstance(sc, dict):
            data = [
                _clamp(sc.get("technical_depth", 50)),
                _clamp(sc.get("execution", 50)),
                _clamp(sc.get("network_strength", 50)),
                _clamp(sc.get("market_understanding", 50)),
                _clamp(sc.get("public_signal_strength", 50)),
            ]
        else:
            data = [50]*5
        c = SEQ[i % len(SEQ)]
        datasets.append(ChartDataset(
            label=f.get("name") or f"Founder {i+1}",
            data=data, backgroundColor=c["soft"], borderColor=c["solid"], fill=True,
        ))
    return ChartData(
        chart_type="radar",
        title="Founder Score Breakdown",
        description="Five-axis quality signal per founder (0–100). Source: Bedrock LLM analysis.",
        labels=labels, datasets=datasets,
    )


def investment_score_breakdown_chart(breakdown: dict) -> ChartData:
    """
    Doughnut — weighted contribution to final investment score.
    Weights: Founder 40%, Talent 25%, Hiring 15%, Market 10%, Sentiment 10%.
    """
    cats = [
        ("Founder Strength (40%)",   "founder_strength",   0.40, PALETTE["indigo"]["solid"]),
        ("Talent Quality (25%)",     "talent_quality",     0.25, PALETTE["emerald"]["solid"]),
        ("Hiring Signals (15%)",     "hiring_signals",     0.15, PALETTE["amber"]["solid"]),
        ("Market Alignment (10%)",   "market_alignment",   0.10, PALETTE["violet"]["solid"]),
        ("External Sentiment (10%)", "external_sentiment", 0.10, PALETTE["sky"]["solid"]),
    ]
    labels   = [c[0] for c in cats]
    weighted = [round(_clamp(breakdown.get(c[1], 50)) * c[2], 1) for c in cats]
    colors   = [c[3] for c in cats]
    total    = sum(weighted)
    return ChartData(
        chart_type="doughnut",
        title="Investment Score Breakdown",
        description=f"Weighted score = {total:.1f}/100. Each slice = category's contribution to final score.",
        labels=labels,
        datasets=[ChartDataset(label="Weighted Score", data=weighted, backgroundColor=colors)],
    )


def talent_flow_chart(chart_data: dict) -> ChartData:
    """
    Line chart — talent inflow vs outflow over time.
    chart_data: {labels:[...], inflow:[...], outflow:[...]}
    Frontend: Chart.js Line / Recharts LineChart with two series.
    """
    labels  = chart_data.get("labels") or []
    inflow  = [_clamp(v, 0, 9999) for v in (chart_data.get("inflow") or [])]
    outflow = [_clamp(v, 0, 9999) for v in (chart_data.get("outflow") or [])]

    # Net flow for annotation
    net = [round(i - o, 1) for i, o in zip(inflow, outflow)] if inflow and outflow else []
    net_label = f"Net trend: {'+' if net and net[-1] >= 0 else ''}{net[-1]:.0f}" if net else ""

    return ChartData(
        chart_type="line",
        title="Talent Flow — Inflow vs Outflow",
        description=f"Estimated talent movement over time. {net_label}. Source: LLM synthesis from headcount + hiring signals.",
        labels=labels,
        datasets=[
            ChartDataset(
                label="Inflow (new hires / talent joining)",
                data=inflow,
                borderColor=PALETTE["emerald"]["solid"],
                backgroundColor=PALETTE["emerald"]["soft"],
                fill=False,
            ),
            ChartDataset(
                label="Outflow (departures / attrition)",
                data=outflow,
                borderColor=PALETTE["rose"]["solid"],
                backgroundColor=PALETTE["rose"]["soft"],
                fill=False,
            ),
        ],
        x_label="Quarter", y_label="Talent Movement (estimated)",
    )


def skill_heatmap_chart(hiring_skills: list[str], losing_skills: list[str]) -> ChartData:
    """
    Grouped bar — skills being hired (+) vs lost (−).
    """
    all_skills = list(dict.fromkeys(hiring_skills[:8] + losing_skills[:8]))
    gaining = [1 if s in hiring_skills else 0 for s in all_skills]
    losing  = [-1 if s in losing_skills else 0 for s in all_skills]
    return ChartData(
        chart_type="bar",
        title="Skill Flow Heatmap",
        description="Skills actively being hired (+1) vs deprioritised or lost (−1). Source: LLM analysis of job postings.",
        labels=all_skills,
        datasets=[
            ChartDataset(label="Hiring",  data=gaining, backgroundColor=PALETTE["emerald"]["solid"]),
            ChartDataset(label="Losing",  data=losing,  backgroundColor=PALETTE["rose"]["solid"]),
        ],
        x_label="Skill", y_label="+1 hiring / −1 losing",
    )


def alumni_success_distribution_chart(alumni_dist: list[dict]) -> ChartData:
    """
    Horizontal bar — % of successful founders from each background (school / company).
    alumni_dist: [{source_name, source_type, success_pct, pattern}, ...]
    """
    sorted_data = sorted(alumni_dist, key=lambda x: x.get("success_pct", 0), reverse=True)[:8]
    labels = []
    values = []
    colors = []
    for d in sorted_data:
        src_type = d.get("source_type", "")
        icon = "🎓" if src_type == "school" else "🏢"
        labels.append(f"{icon} {d.get('source_name','Unknown')}")
        values.append(_clamp(d.get("success_pct", 50)))
        colors.append(
            PALETTE["indigo"]["solid"] if src_type == "school"
            else PALETTE["emerald"]["solid"]
        )
    return ChartData(
        chart_type="bar",
        title="Alumni Success Distribution",
        description="% of founders from each background who achieved notable startup outcomes. 🎓=school, 🏢=company.",
        labels=labels,
        datasets=[ChartDataset(
            label="Success Rate (%)", data=values, backgroundColor=colors,
        )],
        x_label="Success %", y_label="Background Source",
    )


def headcount_trend_chart(timeseries: list[dict]) -> ChartData:
    labels = [str(p.get("date") or p.get("month") or p.get("period") or p.get("as_of_date") or i)
              for i, p in enumerate(timeseries)]
    values = [int(p.get("headcount") or p.get("employee_count") or p.get("value") or 0) for p in timeseries]
    growth = ""
    if len(values) >= 2 and values[0]:
        pct = round((values[-1] - values[0]) / values[0] * 100, 1)
        growth = f"{'+' if pct >= 0 else ''}{pct}% over period"
    return ChartData(
        chart_type="line",
        title="Headcount Trend",
        description=f"Employee count over time. {growth}",
        labels=labels,
        datasets=[ChartDataset(
            label="Employees", data=values,
            borderColor=PALETTE["indigo"]["solid"],
            backgroundColor=PALETTE["indigo"]["soft"], fill=True,
        )],
        x_label="Date", y_label="Headcount",
    )


def funding_timeline_chart(milestones: list[dict]) -> ChartData:
    labels  = [str(m.get("announced_date") or m.get("date") or m.get("year") or f"Round {i+1}") for i, m in enumerate(milestones)]
    amounts = [round((m.get("amount_usd") or m.get("amount") or 0) / 1_000_000, 2) for m in milestones]
    total   = sum(amounts)
    return ChartData(
        chart_type="bar",
        title="Funding Milestones",
        description=f"Funding rounds in USD millions. Total raised: ${total:.1f}M",
        labels=labels,
        datasets=[ChartDataset(label="Amount ($M)", data=amounts, backgroundColor=_solid(len(milestones)))],
        x_label="Round Date", y_label="USD Millions",
    )


def founder_experience_timeline_chart(founders: list[dict]) -> ChartData:
    names, exp_years, notable_count = [], [], []
    for f in founders[:6]:
        wh = f.get("work_history") or []
        years, notable = 0, 0
        for job in wh:
            if isinstance(job, dict):
                sy, ey = job.get("start_year") or 0, job.get("end_year") or 2025
                if sy: years += max(0, ey - sy)
                if job.get("is_notable"): notable += 1
        names.append(f.get("name") or "Unknown")
        exp_years.append(min(years, 30))
        notable_count.append(notable)
    return ChartData(
        chart_type="bar",
        title="Founder Experience Profile",
        description="Total career years and count of notable company stints per founder.",
        labels=names,
        datasets=[
            ChartDataset(label="Years Experience", data=exp_years,     backgroundColor=PALETTE["indigo"]["solid"]),
            ChartDataset(label="Notable Companies", data=notable_count, backgroundColor=PALETTE["amber"]["solid"]),
        ],
        x_label="Founder", y_label="Count",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HR CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def candidate_score_bar_chart(candidates: list[dict]) -> ChartData:
    """Grouped bar — 4 score dimensions per candidate."""
    names = [c.get("name") or f"C{i+1}" for i, c in enumerate(candidates)]
    dims  = [
        ("Skill Match",     "skill_match_score",      PALETTE["indigo"]["solid"]),
        ("Experience Fit",  "experience_alignment",   PALETTE["emerald"]["solid"]),
        ("Cultural Fit",    "cultural_fit_score",     PALETTE["amber"]["solid"]),
        ("Overall",         "overall_score",          PALETTE["rose"]["solid"]),
    ]
    return ChartData(
        chart_type="bar",
        title="Candidate Score Comparison",
        description="Multi-dimensional LLM scoring (0–100) across all evaluated candidates.",
        labels=names,
        datasets=[ChartDataset(label=lbl, data=[_clamp(c.get(k,0)) for c in candidates], backgroundColor=col)
                  for lbl, k, col in dims],
        x_label="Candidate", y_label="Score (0–100)",
    )


def skill_match_histogram(candidates: list[dict]) -> ChartData:
    """Histogram — skill match score distribution."""
    buckets = ["0–20", "21–40", "41–60", "61–80", "81–100"]
    counts  = [0]*5
    for c in candidates:
        idx = min(int(_clamp(c.get("skill_match_score", 0)) // 20), 4)
        counts[idx] += 1
    return ChartData(
        chart_type="bar",
        title="Skill Match Score Distribution",
        description="How many candidates fall in each skill-match score band.",
        labels=buckets,
        datasets=[ChartDataset(
            label="Candidates", data=counts,
            backgroundColor=[PALETTE["rose"]["solid"], PALETTE["amber"]["solid"],
                             PALETTE["sky"]["solid"], PALETTE["emerald"]["solid"],
                             PALETTE["indigo"]["solid"]],
        )],
        x_label="Score Band", y_label="Candidates",
    )


def talent_pool_distribution_chart(skill_clusters: dict) -> ChartData:
    """Doughnut — talent pool by skill cluster."""
    top = sorted(skill_clusters.items(), key=lambda x: x[1], reverse=True)[:8]
    labels = [k for k, _ in top]
    values = [int(v) for _, v in top]
    return ChartData(
        chart_type="doughnut",
        title="Talent Pool Distribution by Skill Cluster",
        description="Candidate pool breakdown by primary skill area.",
        labels=labels,
        datasets=[ChartDataset(label="Candidates", data=values, backgroundColor=_solid(len(labels)))],
    )


def industry_shift_chart(shifts: list[dict]) -> ChartData:
    """Bar — strength of detected talent movements between industries."""
    strength_map = {"Strong": 90, "Moderate": 60, "Weak": 30}
    top = shifts[:8]
    labels = [f"{s.get('source_industry','?')} → {s.get('target_industry','?')}" for s in top]
    values = [strength_map.get(s.get("signal_strength","Moderate"), 60) for s in top]
    colors = [
        PALETTE["emerald"]["solid"] if v >= 80 else
        PALETTE["amber"]["solid"]   if v >= 50 else
        PALETTE["rose"]["solid"] for v in values
    ]
    return ChartData(
        chart_type="bar",
        title="Talent Industry Shift Signals",
        description="Detected talent movement between industries. Height = signal strength.",
        labels=labels,
        datasets=[ChartDataset(label="Signal Strength (0–100)", data=values, backgroundColor=colors)],
        x_label="Shift Direction", y_label="Strength",
    )


def talent_source_map_chart(source_map: list[dict]) -> ChartData:
    """
    Horizontal bar — which companies candidates are sourced from.
    source_map: [{company, candidate_count, quality_signal, recommended_for_sourcing}]
    """
    top = sorted(source_map, key=lambda x: x.get("candidate_count", 0), reverse=True)[:10]
    labels = [s.get("company", "Unknown") for s in top]
    values = [int(s.get("candidate_count", 1)) for s in top]
    colors = []
    for s in top:
        q = s.get("quality_signal", "medium")
        colors.append(
            PALETTE["emerald"]["solid"] if q == "high" else
            PALETTE["amber"]["solid"]   if q == "medium" else
            PALETTE["rose"]["solid"]
        )
    return ChartData(
        chart_type="bar",
        title="Talent Source Map",
        description="Which companies candidates are sourced from. Color: 🟢 high / 🟡 medium / 🔴 low quality signal.",
        labels=labels,
        datasets=[ChartDataset(label="Candidates from Company", data=values, backgroundColor=colors)],
        x_label="Candidates", y_label="Source Company",
    )


def recommendation_radar_chart(candidate: dict) -> ChartData:
    """Radar — single candidate across 4 evaluation dimensions."""
    labels = ["Skill Match", "Experience Fit", "Cultural Fit", "Overall"]
    data   = [_clamp(candidate.get(k,0)) for k in
              ("skill_match_score","experience_alignment","cultural_fit_score","overall_score")]
    rec   = candidate.get("recommendation", "Maybe")
    color = (PALETTE["emerald"] if "Yes" in rec else
             PALETTE["rose"]    if rec == "No"    else PALETTE["amber"])
    return ChartData(
        chart_type="radar",
        title=f"Candidate Profile — {candidate.get('name','Unknown')}",
        description=f"LLM recommendation: {rec}",
        labels=labels,
        datasets=[ChartDataset(
            label=candidate.get("name","Candidate"), data=data,
            backgroundColor=color["soft"], borderColor=color["solid"], fill=True,
        )],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# JOB CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def job_category_chart(category_breakdown: dict) -> ChartData:
    top = sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True)[:8]
    labels, values = [k for k,_ in top], [v for _,v in top]
    return ChartData(
        chart_type="doughnut",
        title="Open Roles by Category",
        description="Distribution of active job openings across departments.",
        labels=labels,
        datasets=[ChartDataset(label="Open Roles", data=values, backgroundColor=_solid(len(labels)))],
    )


def job_titles_chart(top_titles: list[str]) -> ChartData:
    labels = top_titles[:10]
    values = list(range(len(labels), 0, -1))
    return ChartData(
        chart_type="bar",
        title="Top Hiring Titles",
        description="Most frequently posted job titles (rank 1 = most common).",
        labels=labels,
        datasets=[ChartDataset(label="Frequency Rank", data=values, backgroundColor=_solid(len(labels)))],
        x_label="Title", y_label="Rank",
    )


def hiring_velocity_gauge(total_open: int, velocity: str) -> ChartData:
    cap = max(total_open, 1)
    label_map = {"aggressive":"🔥 Aggressive","moderate":"📈 Moderate",
                 "slow":"🐢 Slow","minimal":"⚠️ Minimal","unknown":"❓ Unknown"}
    color_map = {"aggressive": PALETTE["emerald"]["solid"], "moderate": PALETTE["indigo"]["solid"],
                 "slow": PALETTE["amber"]["solid"], "minimal": PALETTE["rose"]["solid"],
                 "unknown": PALETTE["sky"]["solid"]}
    return ChartData(
        chart_type="doughnut",
        title="Hiring Velocity",
        description=f"{label_map.get(velocity,'Unknown')} — {total_open:,} open roles detected.",
        labels=["Active Openings", ""],
        datasets=[ChartDataset(
            label="Open Roles", data=[cap, max(cap//5, 1)],
            backgroundColor=[color_map.get(velocity, PALETTE["sky"]["solid"]), "rgba(200,200,200,0.2)"],
        )],
    )


# Alias for backward compat
def alumni_success_chart(alumni_insights: list[dict]) -> ChartData:
    return alumni_success_distribution_chart(alumni_insights)
