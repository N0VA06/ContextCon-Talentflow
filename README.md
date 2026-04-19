# TalentFlow API

Talent Intelligence Platform powered by **CrustData** (real-time company & people data) and **AWS Bedrock Claude** (LLM analysis).

## Architecture

```
talentflow/
├── main.py                        # FastAPI app + all routes
├── config.py                      # Settings from .env
├── crustdata_client.py            # CrustData API client (fixed headers)
├── llm_client.py                  # AWS Bedrock Claude wrapper
├── models.py                      # All Pydantic request/response models
├── requirements.txt
├── .env.example
├── services/
│   ├── company_service.py         # Resolve + enrich company data
│   ├── person_service.py          # Enrich + search people
│   └── signal_service.py          # Bundle signals for LLM
├── analyzers/
│   ├── pedigree_analyzer.py       # School + company alumni pedigree
│   ├── investor_analyzer.py       # Full investor intelligence + scoring
│   └── hr_analyzer.py             # HR candidate scoring + talent shifts
└── visualization/
    └── chart_builder.py           # Frontend-ready Chart.js data builders
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| pip | latest |
| AWS account with Bedrock access | — |
| CrustData API key | https://app.crustdata.com |

---

## Setup

### 1. Clone and enter the directory

```bash
cd talentflow
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your real keys
nano .env
```

Required values in `.env`:
```
CRUSTDATA_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
```

### 5. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## API Reference

### Health Check

```bash
curl http://localhost:8000/api/v1/health | jq .
```

---

### 1. Pedigree Analysis (Founder Schools + Companies)

**`POST /api/v1/pedigree/analyze`**

Analyses founder educational pedigree and previous company alumni networks.

```bash
curl -X POST http://localhost:8000/api/v1/pedigree/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Stripe",
    "company_domain": "stripe.com",
    "include_schools": true,
    "include_visual": true,
    "include_rating": true,
    "max_founders": 3
  }' | jq .
```

**More examples:**

```bash
# OpenAI — AI-heavy founder pedigree
curl -X POST http://localhost:8000/api/v1/pedigree/analyze \
  -H "Content-Type: application/json" \
  -d '{"company_name": "OpenAI", "include_schools": true, "include_visual": true, "include_rating": true}' | jq .

# Indian unicorn example
curl -X POST http://localhost:8000/api/v1/pedigree/analyze \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Zepto", "include_schools": true, "include_visual": true, "include_rating": true}' | jq .
```

**Response structure:**
```json
{
  "status": "success",
  "mode": "pedigree",
  "investor": {
    "company_name": "Stripe",
    "company_summary": "...",
    "founders": [
      {
        "name": "Patrick Collison",
        "title": "CEO",
        "education": [{"institution": "MIT", "prestige_tier": "Tier-1", ...}],
        "work_history": [{"company": "...", "is_notable": true, ...}],
        "scores": {
          "technical_depth": 88,
          "execution": 92,
          "network_strength": 85,
          "market_understanding": 90,
          "public_signal_strength": 78,
          "overall": 87
        },
        "key_insights": ["..."],
        "builder_signals": ["..."],
        "red_flags": []
      }
    ],
    "alumni_insights": [
      {
        "source_type": "school",
        "source_name": "MIT",
        "success_patterns": ["High % of alumni founding B2B SaaS companies"],
        "success_rate_estimate": "~40% of MIT CS alumni who founded startups raised Series A+"
      }
    ],
    "visualizations": {
      "founder_radar": {
        "chart_type": "radar",
        "title": "Founder Score Breakdown",
        "labels": ["Technical Depth", "Execution", "Network Strength", "Market Understanding", "Public Signal"],
        "datasets": [{"label": "Patrick Collison", "data": [88, 92, 85, 90, 78]}]
      },
      "alumni_success": { "chart_type": "bar", "..." }
    },
    "final_score": {
      "score": 87,
      "breakdown": {"founder_strength": 90, "talent_quality": 82, ...},
      "confidence": "high",
      "verdict": "Strong founding team with exceptional pedigree"
    }
  }
}
```

---

### 2. Full Investor Analysis

**`POST /api/v1/investor/analyze`**

Full investment intelligence: pedigree + talent signals + hiring trends + funding + LLM score.

```bash
curl -X POST http://localhost:8000/api/v1/investor/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Databricks",
    "company_domain": "databricks.com",
    "include_visual": true,
    "include_rating": true,
    "max_founders": 3
  }' | jq .
```

**Additional examples:**

```bash
# Early-stage startup
curl -X POST http://localhost:8000/api/v1/investor/analyze \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Anduril", "include_visual": true, "include_rating": true}' | jq .

# Fintech
curl -X POST http://localhost:8000/api/v1/investor/analyze \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Razorpay", "include_visual": true, "include_rating": true}' | jq .
```

**Additional charts in full investor vs pedigree:**
- `headcount_trend` — company growth over time (line chart)
- `funding_timeline` — funding rounds in $M (bar chart)
- `score_breakdown` — weighted investment score breakdown (doughnut chart)
- `skill_heatmap` — skills being hired vs lost (grouped bar)

---

### 3. HR Analysis

**`POST /api/v1/hr/analyze`**

Supports two input modes — can mix both per request.

#### Mode A: Structured candidates

```bash
curl -X POST http://localhost:8000/api/v1/hr/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "jd_text": "Senior ML Engineer. Requirements: 5+ years Python, PyTorch/TensorFlow, distributed training, MLOps. Nice to have: LLM fine-tuning, RLHF, Kubernetes.",
    "candidates": [
      {
        "name": "Alice Chen",
        "linkedin_url": "https://linkedin.com/in/alice-chen",
        "skills": ["Python", "PyTorch", "Kubernetes", "LLM fine-tuning", "RLHF"],
        "years_experience": 7,
        "current_role": "ML Engineer at Google DeepMind",
        "education": [{"institution": "Stanford", "degree": "MS", "field": "Computer Science"}]
      },
      {
        "name": "Bob Sharma",
        "skills": ["Python", "TensorFlow", "MLOps", "Docker"],
        "years_experience": 4,
        "current_role": "Data Scientist at Flipkart"
      }
    ],
    "include_visual": true,
    "include_recommendations": true
  }' | jq .
```

#### Mode B: Free-text resumes

```bash
curl -X POST http://localhost:8000/api/v1/hr/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "jd_structured": {
      "title": "Backend Engineer",
      "skills": ["Go", "Rust", "distributed systems", "gRPC", "Kafka"],
      "experience_years": 4,
      "team": "Infrastructure"
    },
    "candidates": [
      {
        "name": "Carlos Mendez",
        "resume_text": "5 years building distributed systems at Uber. Led migration of payment service from Python to Go, reducing p99 latency by 60%. Expert in Kafka, gRPC, Kubernetes. MS CS from UT Austin."
      },
      {
        "name": "Priya Nair",
        "resume_text": "4 years at Swiggy as backend engineer. Built real-time order tracking in Go. Familiar with Kafka and Redis. B.Tech IIT Bombay."
      }
    ],
    "include_visual": true
  }' | jq .
```

#### Mode C: Mine talent pool from company

```bash
curl -X POST http://localhost:8000/api/v1/hr/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "jd_text": "AI Product Manager with LLM/GenAI product experience",
    "target_company": "OpenAI",
    "include_visual": true,
    "include_recommendations": true
  }' | jq .
```

**HR Response structure:**
```json
{
  "status": "success",
  "mode": "hr",
  "hr": {
    "jd_summary": "Senior ML role requiring PyTorch, distributed training, LLM expertise",
    "candidate_scores": [
      {
        "name": "Alice Chen",
        "skill_match_score": 92,
        "experience_alignment": 88,
        "cultural_fit_score": 80,
        "overall_score": 88,
        "strengths": ["Strong LLM/RLHF background", "7 years exceeds requirement"],
        "gaps": ["No MLOps certification mentioned"],
        "recommendation": "Strong Yes"
      }
    ],
    "talent_shift_insights": [
      {
        "pattern": "ML engineers moving from Big Tech to AI startups",
        "source_industry": "Big Tech",
        "target_industry": "AI Startups",
        "signal_strength": "Strong",
        "relevance_to_jd": "Rich talent pool available at below-FAANG salary"
      }
    ],
    "visualizations": {
      "candidate_scores": { "chart_type": "bar", "..." },
      "skill_match_histogram": { "chart_type": "bar", "..." },
      "talent_pool_distribution": { "chart_type": "doughnut", "..." },
      "industry_shifts": { "chart_type": "bar", "..." }
    },
    "recommendations": [
      "Prioritize Alice Chen — rare RLHF + production ML combination",
      "Source from ex-Google Brain / DeepMind alumni network",
      "Consider candidates shifting from fintech ML to pure AI roles"
    ],
    "top_source_companies": ["Google DeepMind", "OpenAI", "Anthropic", "Meta AI"]
  }
}
```

---

## Frontend Chart Integration

All `visualizations` objects are **Chart.js-compatible** out of the box.

```javascript
// React + Chart.js example
import { Radar, Bar, Doughnut, Line } from 'react-chartjs-2';

// The API returns chart data directly — just pass it in:
const { founder_radar, headcount_trend, score_breakdown } = response.investor.visualizations;

<Radar data={founder_radar} />
<Line data={headcount_trend} />
<Doughnut data={score_breakdown} />
```

**Recharts example:**
```javascript
import { RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts';

// Transform labels + datasets into recharts format:
const radarData = founder_radar.labels.map((label, i) => ({
  subject: label,
  ...Object.fromEntries(
    founder_radar.datasets.map(ds => [ds.label, ds.data[i]])
  )
}));
```

---

## CrustData API Notes

This implementation uses the **2025-11-01 API version** with the correct mandatory headers:

```
authorization: Bearer <CRUSTDATA_API_KEY>
x-api-version: 2025-11-01
content-type: application/json
```

**Why the original error occurred:**  
The original code was missing `x-api-version: 2025-11-01`. Without it, CrustData rejects with `400 — Missing required header: x-api-version`.

**Endpoint coverage:**
| Endpoint | Purpose |
|----------|---------|
| `POST /company/identify` | Resolve company name/domain → structured entity |
| `POST /company/enrich` | Full company profile |
| `POST /person/search` | Discover people by role/company |
| `POST /person/enrich` | Full individual profile |
| `POST /web/search/live` | Real-time web search |
| `GET /screener/company` | Legacy enrichment (fallback) |
| `GET /data_lab/headcount_timeseries` | Headcount over time |
| `GET /data_lab/funding_milestone_timeseries` | Funding rounds |
| `GET /data_lab/decision_makers` | Executive/founder data |
| `GET /employee_review/enrich` | Internal sentiment |

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `400 Missing required header: x-api-version` | Old client code | Fixed — `crustdata_client.py` always sends it |
| `401 Unauthorized` | Bad API key | Check `CRUSTDATA_API_KEY` in `.env` |
| `403 Forbidden` | Endpoint not on your plan | Check CrustData plan at `app.crustdata.com` |
| `502 Upstream data error` | CrustData returned error | Check logs; may be rate-limited |
| `504 Analysis timed out` | LLM/API too slow | Increase `BEDROCK_TIMEOUT` in `.env` |
| `ValueError: AWS credentials missing` | Bedrock not configured | Set `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
