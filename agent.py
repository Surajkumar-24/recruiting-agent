"""
AI Recruiting Agent
-------------------
Sources real LinkedIn candidate profiles using:
  - Claude API  → JD analysis + X-ray string generation + candidate scoring
  - SerpAPI     → Real Google search results (LinkedIn profile URLs)

Usage:
  python agent.py --jd "path/to/jd.txt"
  python agent.py --jd "path/to/jd.txt" --companies "Google,Amazon,Flipkart" --location "Bangalore"
  python agent.py --interactive   (guided prompts)
"""

import os, sys, json, csv, time, re, argparse
from datetime import date
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic")

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SERP_API_KEY      = os.environ.get("SERP_API_KEY", "")       # https://serpapi.com  (free: 100 searches/mo)
MODEL             = "claude-opus-4-5"
OUTPUT_DIR        = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")

def step(msg):
    print(f"\n▸ {msg}")

def ok(msg):
    print(f"  ✓ {msg}")

def warn(msg):
    print(f"  ⚠ {msg}")

def ask_claude(prompt, max_tokens=1500):
    """Call Claude and return the text response."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()

def parse_json(text):
    """Strip markdown fences and parse JSON."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)

# ── Step 1: Extract JD details ────────────────────────────────────────────────

def extract_jd_details(jd_text, overrides=None):
    step("Extracting requirements from JD with Claude…")
    prompt = f"""Extract structured details from this job description.
Return ONLY valid JSON, no markdown, no explanation:
{{
  "title": "job title",
  "location": "primary city or region",
  "experience": "e.g. 5-8 years",
  "industry": "e.g. Fintech, SaaS, E-commerce",
  "academic": "e.g. B.Des, MBA, B.Tech",
  "techSkills": ["up to 6 specific technical skills"],
  "funcSkills": ["up to 6 functional/soft skills"],
  "seniority": "Junior|Mid|Senior|Lead|Principal|Director"
}}

Job Description:
{jd_text[:3000]}"""

    result = parse_json(ask_claude(prompt, max_tokens=600))

    # Apply CLI overrides
    if overrides:
        for k, v in overrides.items():
            if v:
                result[k] = v

    ok(f"Role: {result.get('title')} | {result.get('location')} | {result.get('experience')}")
    ok(f"Tech: {', '.join(result.get('techSkills', []))}")
    ok(f"Functional: {', '.join(result.get('funcSkills', []))}")
    return result

# ── Step 2: Generate X-ray strings ────────────────────────────────────────────

def generate_xray_strings(details, preferred_companies=None):
    step("Generating Google X-ray search strings…")
    comp_str = ", ".join(preferred_companies) if preferred_companies else "not specified"

    prompt = f"""You are an expert LinkedIn Boolean X-ray search specialist for recruiting.

Generate exactly 8 Google X-ray search strings to find LinkedIn profiles for:
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
2. Use OR for alternatives, AND for must-haves, quotes for exact phrases
3. Keep each string under 200 characters (Google limit)
4. Vary the strings: core skills, company targeting, academic, seniority, functional, tech stack
5. Do NOT use intitle: or inurl: — plain site: search only

Return ONLY a JSON array of 8 objects, no markdown:
[{{"label":"short descriptive label","string":"site:linkedin.com/in/ ..."}}]"""

    xrays = parse_json(ask_claude(prompt, max_tokens=1200))
    for i, x in enumerate(xrays, 1):
        ok(f"String {i}: {x['label']}")
    return xrays

# ── Step 3: Search Google via SerpAPI ─────────────────────────────────────────

def search_google(query, num_results=10):
    """Run one Google search via SerpAPI and return organic results."""
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "num": num_results,
        "hl": "en",
        "gl": "in",   # India locale — change to "us" if needed
    }
    resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("organic_results", [])

def extract_linkedin_profiles(search_results):
    """Pull LinkedIn profile URLs + names from search results."""
    profiles = []
    for r in search_results:
        url = r.get("link", "")
        if "linkedin.com/in/" not in url:
            continue
        # Clean URL — strip query params and trailing slashes
        clean_url = re.sub(r"\?.*$", "", url).rstrip("/")
        # Extract name from title (LinkedIn titles: "Name - Title - Company | LinkedIn")
        title = r.get("title", "")
        name = title.split(" - ")[0].strip() if " - " in title else title.split("|")[0].strip()
        snippet = r.get("snippet", "")
        profiles.append({
            "name": name,
            "linkedin": clean_url,
            "snippet": snippet,
            "raw_title": title,
        })
    return profiles

def run_searches(xrays, max_strings=6, results_per_search=10):
    step(f"Running {min(max_strings, len(xrays))} Google X-ray searches via SerpAPI…")
    all_profiles = []
    seen_urls = set()

    for i, xray in enumerate(xrays[:max_strings], 1):
        print(f"  [{i}/{min(max_strings, len(xrays))}] {xray['label']}…", end=" ", flush=True)
        try:
            results = search_google(xray["string"], num_results=results_per_search)
            profiles = extract_linkedin_profiles(results)
            new = 0
            for p in profiles:
                if p["linkedin"] not in seen_urls:
                    seen_urls.add(p["linkedin"])
                    p["source_string"] = xray["label"]
                    all_profiles.append(p)
                    new += 1
            print(f"{new} new profiles")
            time.sleep(1.2)  # Polite delay between requests
        except Exception as e:
            print(f"ERROR: {e}")
            warn(f"Skipping string {i} due to error")

    ok(f"Total unique profiles found: {len(all_profiles)}")
    return all_profiles

# ── Step 4: Score candidates against JD ───────────────────────────────────────

def score_candidates(profiles, details, jd_text, batch_size=6):
    step(f"Scoring {len(profiles)} candidates against JD with Claude…")
    scored = []

    for i in range(0, len(profiles), batch_size):
        batch = profiles[i:i+batch_size]
        print(f"  Scoring batch {i//batch_size + 1}/{-(-len(profiles)//batch_size)}…", end=" ", flush=True)

        profiles_text = "\n".join([
            f"{j+1}. Name: {p['name']}\n   Snippet: {p['snippet']}\n   URL: {p['linkedin']}"
            for j, p in enumerate(batch)
        ])

        prompt = f"""You are a senior recruiter scoring LinkedIn profiles against a job requirement.

Job: {details.get('title')} | {details.get('location')} | {details.get('experience')}
Required tech: {', '.join(details.get('techSkills', []))}
Required functional: {', '.join(details.get('funcSkills', []))}
Industry: {details.get('industry', 'any')}

Profiles to score (based on name + LinkedIn snippet only):
{profiles_text}

Score each profile 0-100 based on likely fit. Be realistic — most will score 50-85.
Return ONLY a JSON array of {len(batch)} objects, no markdown:
[{{"index":1,"score":75,"headline":"inferred current role from snippet","reason":"1 sentence why"}}]"""

        try:
            result = parse_json(ask_claude(prompt, max_tokens=800))
            for r in result:
                idx = r["index"] - 1
                if 0 <= idx < len(batch):
                    batch[idx]["score"] = r.get("score", 50)
                    batch[idx]["headline"] = r.get("headline", batch[idx].get("snippet", "")[:80])
                    batch[idx]["reason"] = r.get("reason", "")
            print(f"done")
        except Exception as e:
            print(f"ERROR: {e}")
            for p in batch:
                p.setdefault("score", 50)
                p.setdefault("headline", p.get("snippet", "")[:80])
                p.setdefault("reason", "")

        scored.extend(batch)
        time.sleep(0.5)

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored

# ── Step 5: Export to CSV ─────────────────────────────────────────────────────

def export_csv(candidates, job_title):
    today = date.today().isoformat()
    safe_title = re.sub(r"[^\w\s-]", "", job_title).replace(" ", "_")
    filename = OUTPUT_DIR / f"{safe_title}_candidates_{today}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Rank", "Name", "LinkedIn URL", "Headline",
            "Match Score", "Reason", "Source String", "Date Sourced"
        ])
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

    ok(f"Exported {len(candidates)} candidates → {filename}")
    return filename

# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive_mode():
    banner("AI Recruiting Agent — Interactive Mode")
    print("Paste your job description below.")
    print("When done, type END on a new line and press Enter:\n")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    jd_text = "\n".join(lines)

    companies_raw = input("\nPreferred companies (comma-separated, or leave blank): ").strip()
    preferred = [c.strip() for c in companies_raw.split(",")] if companies_raw else []
    location_override = input("Override location (or leave blank to auto-detect): ").strip()

    return jd_text, preferred, {"location": location_override or None}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Recruiting Agent")
    parser.add_argument("--jd", help="Path to JD text file")
    parser.add_argument("--companies", help="Comma-separated preferred companies")
    parser.add_argument("--location", help="Override location")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max-searches", type=int, default=6, help="Max X-ray strings to run (default 6)")
    parser.add_argument("--results-per-search", type=int, default=10, help="Results per search (default 10)")
    args = parser.parse_args()

    # Validate keys
    if not ANTHROPIC_API_KEY:
        sys.exit("ERROR: Set ANTHROPIC_API_KEY environment variable.")
    if not SERP_API_KEY:
        sys.exit("ERROR: Set SERP_API_KEY environment variable. Get a free key at https://serpapi.com")

    banner("AI Recruiting Agent")

    # Get JD text
    if args.interactive:
        jd_text, preferred_companies, overrides = interactive_mode()
    elif args.jd:
        jd_text = Path(args.jd).read_text(encoding="utf-8")
        preferred_companies = [c.strip() for c in args.companies.split(",")] if args.companies else []
        overrides = {"location": args.location}
    else:
        parser.print_help()
        sys.exit("\nProvide --jd <file> or --interactive")

    # Run pipeline
    details      = extract_jd_details(jd_text, overrides)
    xrays        = generate_xray_strings(details, preferred_companies)
    profiles     = run_searches(xrays, args.max_searches, args.results_per_search)

    if not profiles:
        sys.exit("\nNo profiles found. Try broader X-ray strings or check your SerpAPI key.")

    candidates   = score_candidates(profiles, details, jd_text)
    output_file  = export_csv(candidates, details.get("title", "candidates"))

    # Print summary
    banner(f"Done — Top 10 candidates for {details.get('title')}")
    print(f"{'Rank':<5} {'Score':<7} {'Name':<28} {'Headline':<40}")
    print("─" * 82)
    for c in candidates[:10]:
        name = c.get("name", "")[:27]
        headline = c.get("headline", "")[:39]
        print(f"{candidates.index(c)+1:<5} {c.get('score',0):<7} {name:<28} {headline:<40}")

    print(f"\n Full CSV saved to: {output_file}")

if __name__ == "__main__":
    main()
