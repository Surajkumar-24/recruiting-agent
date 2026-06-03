"""
AI Recruiting Agent — Flask Web App (Groq-powered)
---------------------------------------------------
Run:  python app.py
Open: http://localhost:5000

LLM: Groq (free, fast — https://console.groq.com)
Search: SerpAPI (free 100/mo — https://serpapi.com)
"""

import os, json, csv, re, time
from datetime import date
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, Response
import requests as req

app = Flask(__name__)
app.secret_key = os.urandom(24)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SERP_API_KEY = os.environ.get("SERP_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Groq API call ─────────────────────────────────────────────────────────────

def ask_groq(prompt, max_tokens=1500):
    """Call Groq LLM and return the text response."""
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    resp = req.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ── Helpers ───────────────────────────────────────────────────────────────────

def sse(event, data):
    payload = json.dumps(data) if isinstance(data, dict) else str(data)
    return f"event: {event}\ndata: {payload}\n\n"

def parse_json(text):
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r'[\[{].*[\]}]', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)

# ── Pipeline steps ────────────────────────────────────────────────────────────

def extract_jd_details(jd_text, overrides=None):
    prompt = f"""Extract structured details from this job description.
Return ONLY valid JSON, no markdown, no explanation, no extra text:
{{
  "title": "job title",
  "location": "primary city or region",
  "experience": "e.g. 5-8 years",
  "industry": "e.g. Fintech, SaaS",
  "academic": "e.g. B.Des, MBA, B.Tech",
  "techSkills": ["up to 6 technical skills"],
  "funcSkills": ["up to 6 functional skills"],
  "seniority": "Junior or Mid or Senior or Lead or Principal or Director"
}}

Job Description:
{jd_text[:3000]}"""
    result = parse_json(ask_groq(prompt, max_tokens=600))
    if overrides:
        for k, v in overrides.items():
            if v:
                result[k] = v
    return result

def generate_xray_strings(details, preferred_companies=None):
    comp_str = ", ".join(preferred_companies) if preferred_companies else "not specified"
    prompt = f"""You are an expert LinkedIn Boolean X-ray search specialist for recruiting.

Generate exactly 8 Google X-ray search strings to find LinkedIn profiles for this role:
- Job title: {details.get('title')}
- Location: {details.get('location', 'India')}
- Experience: {details.get('experience', 'any')}
- Seniority: {details.get('seniority', 'Senior')}
- Industry: {details.get('industry', 'any')}
- Technical skills: {', '.join(details.get('techSkills', []))}
- Functional skills: {', '.join(details.get('funcSkills', []))}
- Preferred companies: {comp_str}
- Academic: {details.get('academic', 'any')}

Rules:
1. Every string MUST start with: site:linkedin.com/in/
2. Use OR for alternatives, quotes for exact phrases
3. Keep each string under 200 characters
4. Make each string different: vary by skills, companies, academic, seniority

Return ONLY a JSON array of 8 objects, no markdown, no explanation:
[{{"label":"short label","string":"site:linkedin.com/in/ ..."}}]"""
    return parse_json(ask_groq(prompt, max_tokens=1200))

def search_google(query, num_results=10):
    params = {"q": query, "api_key": SERP_API_KEY, "num": num_results, "hl": "en", "gl": "in"}
    resp = req.get("https://serpapi.com/search", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("organic_results", [])

def extract_profiles(results):
    profiles = []
    for r in results:
        url = r.get("link", "")
        if "linkedin.com/in/" not in url:
            continue
        clean_url = re.sub(r"\?.*$", "", url).rstrip("/")
        title = r.get("title", "")
        name = title.split(" - ")[0].strip() if " - " in title else title.split("|")[0].strip()
        profiles.append({
            "name": name,
            "linkedin": clean_url,
            "snippet": r.get("snippet", ""),
            "raw_title": title
        })
    return profiles

def score_candidates(profiles, details, batch_size=6):
    scored = []
    for i in range(0, len(profiles), batch_size):
        batch = profiles[i:i + batch_size]
        profiles_text = "\n".join([
            f"{j+1}. Name: {p['name']}\n   Snippet: {p['snippet']}\n   URL: {p['linkedin']}"
            for j, p in enumerate(batch)
        ])
        prompt = f"""You are a strict senior recruiter scoring LinkedIn profiles for fit.

ROLE REQUIREMENTS:
- Title: {details.get('title')}
- Location: {details.get('location')}
- Required experience: {details.get('experience')}
- Seniority level: {details.get('seniority', 'any')}
- Required tech skills: {', '.join(details.get('techSkills', []))}
- Required functional skills: {', '.join(details.get('funcSkills', []))}

SCORING RULES — apply strictly:
1. If the required experience is 0-1 year or "intern" or "fresher", any profile showing 3+ years of experience or senior titles (Manager, Director, VP, Head, Lead, Senior) must score BELOW 30.
2. If required experience is 2-4 years, profiles with 8+ years or C-suite/Director titles score below 35.
3. If required experience is 5+ years, profiles with only 0-1 year experience score below 30.
4. Penalise heavily if current role is completely unrelated to the job title.
5. Reward profiles whose current or recent role closely matches the job title.

PROFILES TO SCORE:
{profiles_text}

Return ONLY a JSON array of {len(batch)} objects, no markdown, no explanation:
[{{"index":1,"score":75,"headline":"their current role and company","reason":"one sentence explaining the score"}}]"""
        try:
            result = parse_json(ask_groq(prompt, max_tokens=800))
            for r in result:
                idx = r["index"] - 1
                if 0 <= idx < len(batch):
                    batch[idx]["score"]    = r.get("score", 50)
                    batch[idx]["headline"] = r.get("headline", batch[idx].get("snippet", "")[:80])
                    batch[idx]["reason"]   = r.get("reason", "")
        except Exception:
            for p in batch:
                p.setdefault("score", 50)
                p.setdefault("headline", p.get("snippet", "")[:80])
                p.setdefault("reason", "")
        scored.extend(batch)
        time.sleep(0.5)
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored

# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/check-keys")
def check_keys():
    return jsonify({
        "groq": bool(GROQ_API_KEY),
        "serp": bool(SERP_API_KEY),
    })

@app.route("/api/extract", methods=["POST"])
def extract():
    data = request.json
    try:
        details = extract_jd_details(data.get("jd", ""), {
            "location": data.get("location") or None
        })
        return jsonify({"ok": True, "details": details})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/run")
def run_pipeline():
    jd        = request.args.get("jd", "")
    companies = [c.strip() for c in request.args.get("companies", "").split(",") if c.strip()]
    location  = request.args.get("location", "")
    max_s     = int(request.args.get("max_searches", 6))
    rps       = int(request.args.get("results_per_search", 10))
    overrides = {"location": location or None}

    manual = {}
    for field in ["title", "experience", "industry", "academic", "seniority"]:
        v = request.args.get(field, "")
        if v:
            manual[field] = v
    tech_raw = request.args.get("techSkills", "")
    func_raw = request.args.get("funcSkills", "")
    if tech_raw:
        manual["techSkills"] = [s.strip() for s in tech_raw.split(",") if s.strip()]
    if func_raw:
        manual["funcSkills"] = [s.strip() for s in func_raw.split(",") if s.strip()]

    def generate():
        try:
            yield sse("status", {"step": 1, "msg": "Extracting requirements from JD...", "pct": 5})
            details = extract_jd_details(jd, overrides)
            details.update(manual)
            yield sse("details", details)
            yield sse("status", {"step": 1, "msg": f"Role: {details.get('title')} · {details.get('location')}", "pct": 18})

            yield sse("status", {"step": 2, "msg": "Generating X-ray search strings...", "pct": 25})
            xrays = generate_xray_strings(details, companies)
            yield sse("xrays", xrays)
            yield sse("status", {"step": 2, "msg": f"{len(xrays)} X-ray strings ready", "pct": 38})

            all_profiles = []
            seen_urls    = set()
            to_run       = xrays[:max_s]
            for i, xray in enumerate(to_run, 1):
                pct = 38 + int((i / len(to_run)) * 32)
                yield sse("status", {"step": 3, "msg": f"Searching: {xray['label']} ({i}/{len(to_run)})...", "pct": pct})
                try:
                    results  = search_google(xray["string"], rps)
                    profiles = extract_profiles(results)
                    for p in profiles:
                        if p["linkedin"] not in seen_urls:
                            seen_urls.add(p["linkedin"])
                            p["source_string"] = xray["label"]
                            all_profiles.append(p)
                    yield sse("search_result", {"string_label": xray["label"], "found": len(profiles), "total": len(all_profiles)})
                    time.sleep(1.2)
                except Exception as e:
                    yield sse("warn", f"Search {i} failed: {e}")

            if not all_profiles:
                yield sse("error", "No profiles found. Check your SerpAPI key or try broader search settings.")
                return

            yield sse("status", {"step": 3, "msg": f"{len(all_profiles)} unique profiles found. Scoring...", "pct": 72})

            batch_size = 6
            for i in range(0, len(all_profiles), batch_size):
                pct = 72 + int(((i + batch_size) / len(all_profiles)) * 22)
                yield sse("status", {"step": 4, "msg": f"Scoring batch {i//batch_size+1}/{-(-len(all_profiles)//batch_size)}...", "pct": min(pct, 93)})

            candidates = score_candidates(all_profiles, details)
            yield sse("status", {"step": 4, "msg": "Scoring complete", "pct": 95})

            today    = date.today().isoformat()
            safe     = re.sub(r"[^\w\s-]", "", details.get("title", "candidates")).replace(" ", "_")
            filename = OUTPUT_DIR / f"{safe}_candidates_{today}.csv"
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Rank", "Name", "LinkedIn URL", "Headline", "Match Score", "Reason", "Source String", "Date Sourced"])
                writer.writeheader()
                for i, c in enumerate(candidates, 1):
                    writer.writerow({
                        "Rank": i,
                        "Name": c.get("name", ""),
                        "LinkedIn URL": c.get("linkedin", ""),
                        "Headline": c.get("headline", ""),
                        "Match Score": f"{c.get('score', 0)}%",
                        "Reason": c.get("reason", ""),
                        "Source String": c.get("source_string", ""),
                        "Date Sourced": today,
                    })

            yield sse("candidates", candidates)
            yield sse("status", {"step": 5, "msg": f"Done! {len(candidates)} candidates sourced.", "pct": 100})
            yield sse("done", {"file": filename.name, "count": len(candidates)})

        except Exception as e:
            yield sse("error", str(e))

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/download/<filename>")
def download(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return "File not found", 404
    return send_file(path, as_attachment=True, download_name=filename)

if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not set. Get free key at https://console.groq.com")
    if not SERP_API_KEY:
        print("WARNING: SERP_API_KEY not set. Get free key at https://serpapi.com")
    print("\n  AI Recruiting Agent running at http://localhost:5000\n")
    app.run(debug=True, threaded=True, port=5000)
