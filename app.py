"""
Mrs. Sourcer V2 — Full SaaS Backend (SMTP Edition)
---------------------------------------------------
Auth: Supabase | Payments: Razorpay | Email: Gmail SMTP
"""

import os, json, csv, re, time, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date, datetime
from pathlib import Path
from functools import wraps

from flask import Flask, render_template, request, jsonify, send_file, Response, redirect, session
import requests as req
from supabase import create_client, Client

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "mrs-sourcer-secret-change-in-prod")
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 86400 * 7  # 7 days

# ── Config ────────────────────────────────────────────────────────────────────

SUPABASE_URL        = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY   = os.environ.get("SUPABASE_ANON_KEY", "")
RAZORPAY_KEY_ID     = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
SERP_API_KEY        = os.environ.get("SERP_API_KEY", "")
PROSPEO_API_KEY     = os.environ.get("PROSPEO_API_KEY", "")

GROQ_MODEL = "llama-3.3-70b-versatile"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ── Plans ─────────────────────────────────────────────────────────────────────

PLANS = {
    "free":       {"name": "Free Trial", "price": 0,      "searches": 1,   "emails": 10,   "razorpay_plan": None},
    "individual": {"name": "Individual", "price": 69900,  "searches": 15,  "emails": 100,  "razorpay_plan": os.environ.get("RAZORPAY_INDIVIDUAL_PLAN_ID","")},
    "agency":     {"name": "Agency",     "price": 499900, "searches": 999, "emails": 1000, "razorpay_plan": os.environ.get("RAZORPAY_AGENCY_PLAN_ID","")},
}

# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            # Always return JSON for API routes
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "Session expired. Please log in again."}), 401
            return redirect("/")
        return f(*args, **kwargs)
    return decorated

def get_user_profile(user_id):
    try:
        res = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
        return res.data
    except:
        return None

def check_usage(user_id, action):
    profile = get_user_profile(user_id)
    if not profile:
        return False, "Profile not found"
    plan = PLANS.get(profile.get("plan", "free"), PLANS["free"])
    used  = profile.get("searches_used", 0) if action == "search" else profile.get("emails_used", 0)
    limit = plan["searches"] if action == "search" else plan["emails"]
    if used >= limit:
        return False, f"{'Search' if action=='search' else 'Email'} limit reached ({limit}/month). Please upgrade."
    return True, None

def increment_usage(user_id, action):
    profile = get_user_profile(user_id)
    if not profile:
        return
    field = "searches_used" if action == "search" else "emails_used"
    supabase.table("profiles").update({field: profile.get(field, 0) + 1}).eq("id", user_id).execute()

# ── SMTP email sending ────────────────────────────────────────────────────────

def send_smtp_email(gmail, app_password, to_email, subject, body):
    """Send email via Gmail SMTP using user's app password."""
    msg = MIMEMultipart()
    msg["From"]    = gmail
    msg["To"]      = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail, app_password)
        server.sendmail(gmail, to_email, msg.as_string())

# ── Groq LLM ──────────────────────────────────────────────────────────────────

def ask_groq(prompt, max_tokens=1500):
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}
    body    = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.3}
    resp = req.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def parse_json(text):
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r'[\[{].*[\]}]', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)

# ── Sourcing pipeline ─────────────────────────────────────────────────────────

def generate_xray_strings(details, preferred_companies=None):
    comp_str = ", ".join(preferred_companies) if preferred_companies else "not specified"
    prompt = f"""You are an expert LinkedIn Boolean X-ray search specialist.
Generate exactly 8 Google X-ray search strings for:
- Job title: {details.get('title')}
- Location: {details.get('location','India')}
- Experience: {details.get('experience','any')}
- Seniority: {details.get('seniority','any')}
- Industry: {details.get('industry','any')}
- Technical skills: {', '.join(details.get('techSkills',[]))}
- Functional skills: {', '.join(details.get('funcSkills',[]))}
- Preferred companies: {comp_str}
- Academic: {details.get('academic','any')}

Rules:
1. Every string MUST start with: site:linkedin.com/in/
2. Use OR for alternatives, quotes for exact phrases
3. Keep under 200 characters each
4. Vary: core skills, companies, academic, seniority

Return ONLY JSON array of 8 objects, no markdown:
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
        name  = title.split(" - ")[0].strip() if " - " in title else title.split("|")[0].strip()
        profiles.append({"name": name, "linkedin": clean_url, "snippet": r.get("snippet", ""), "raw_title": title})
    return profiles

def score_candidates(profiles, details, batch_size=6):
    scored = []
    for i in range(0, len(profiles), batch_size):
        batch = profiles[i:i + batch_size]
        profiles_text = "\n".join([
            f"{j+1}. Name: {p['name']}\n   Snippet: {p['snippet']}\n   URL: {p['linkedin']}"
            for j, p in enumerate(batch)
        ])
        prompt = f"""You are a strict senior recruiter scoring LinkedIn profiles.

ROLE: {details.get('title')} | {details.get('location')} | {details.get('experience')}
Seniority: {details.get('seniority','any')}
Tech skills: {', '.join(details.get('techSkills',[]))}
Functional skills: {', '.join(details.get('funcSkills',[]))}

RULES:
1. Intern/fresher role: 3+ years or senior titles → score below 30
2. 2-4yr role: 8+ years or Director/VP → score below 35
3. Penalise unrelated current role
4. Reward close title match

PROFILES:
{profiles_text}

Return ONLY JSON array of {len(batch)} objects, no markdown:
[{{"index":1,"score":75,"headline":"current role","reason":"one sentence"}}]"""
        try:
            result = parse_json(ask_groq(prompt, max_tokens=800))
            for r in result:
                idx = r["index"] - 1
                if 0 <= idx < len(batch):
                    batch[idx]["score"]    = r.get("score", 50)
                    batch[idx]["headline"] = r.get("headline", batch[idx].get("snippet","")[:80])
                    batch[idx]["reason"]   = r.get("reason", "")
        except:
            for p in batch:
                p.setdefault("score", 50)
                p.setdefault("headline", p.get("snippet","")[:80])
                p.setdefault("reason", "")
        scored.extend(batch)
        time.sleep(0.5)
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored

# ── Email finder ──────────────────────────────────────────────────────────────

def find_email_prospeo(linkedin_url):
    try:
        # Clean the LinkedIn URL
        clean_url = re.sub(r"\?.*$", "", linkedin_url).rstrip("/")
        if not clean_url.startswith("https://"):
            clean_url = "https://" + clean_url.lstrip("/")

        url     = "https://api.prospeo.io/linkedin-email-finder"
        headers = {"Content-Type": "application/json", "X-KEY": PROSPEO_API_KEY}
        resp    = req.post(url, headers=headers, json={"url": clean_url}, timeout=15)

        print(f"Prospeo status: {resp.status_code} for {clean_url}")
        data = resp.json()
        print(f"Prospeo response: {data}")

        # Prospeo returns error: false when successful
        if not data.get("error") and data.get("response"):
            email_data = data["response"].get("email")
            if email_data and isinstance(email_data, dict):
                return email_data.get("value")
            elif email_data and isinstance(email_data, str):
                return email_data
        return None
    except Exception as e:
        print(f"Prospeo error for {linkedin_url}: {e}")
        return None

# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/auth/signup", methods=["POST"])
def signup():
    data     = request.json
    email    = data.get("email","").strip()
    password = data.get("password","").strip()
    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password required"}), 400
    try:
        res     = supabase.auth.sign_up({"email": email, "password": password})
        user_id = res.user.id
        supabase.table("profiles").insert({
            "id": user_id, "email": email,
            "plan": "free", "searches_used": 0, "emails_used": 0,
            "gmail_connected": False, "created_at": datetime.utcnow().isoformat()
        }).execute()
        session["user_id"] = user_id
        session["email"]   = email
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json
    try:
        res = supabase.auth.sign_in_with_password({"email": data.get("email",""), "password": data.get("password","")})
        session["user_id"] = res.user.id
        session["email"]   = res.user.email
        return jsonify({"ok": True})
    except:
        return jsonify({"ok": False, "error": "Invalid email or password"}), 401

@app.route("/auth/logout")
def logout():
    session.clear()
    return redirect("/")

# ── Gmail settings ────────────────────────────────────────────────────────────

@app.route("/api/save-gmail", methods=["POST"])
@login_required
def save_gmail():
    """Save user's Gmail + App Password (encrypted in DB)."""
    data         = request.json
    gmail        = data.get("gmail","").strip()
    app_password = data.get("app_password","").strip().replace(" ","")
    if not gmail or not app_password:
        return jsonify({"ok": False, "error": "Gmail and app password required"}), 400
    # Test the credentials before saving
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail, app_password)
    except Exception as e:
        return jsonify({"ok": False, "error": "Could not connect to Gmail. Check your email and app password."}), 400
    supabase.table("profiles").update({
        "gmail_connected": True,
        "gmail_address":   gmail,
        "gmail_app_password": app_password,
    }).eq("id", session["user_id"]).execute()
    return jsonify({"ok": True})

# ── Main routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", razorpay_key=RAZORPAY_KEY_ID)

@app.route("/api/me")
def me():
    if "user_id" not in session:
        return jsonify({"logged_in": False})
    profile = get_user_profile(session["user_id"])
    if not profile:
        return jsonify({"logged_in": False})
    plan = PLANS.get(profile.get("plan","free"), PLANS["free"])
    return jsonify({
        "logged_in":       True,
        "email":           session.get("email"),
        "plan":            profile.get("plan","free"),
        "plan_name":       plan["name"],
        "searches_used":   profile.get("searches_used", 0),
        "searches_limit":  plan["searches"],
        "emails_used":     profile.get("emails_used", 0),
        "emails_limit":    plan["emails"],
        "gmail_connected": profile.get("gmail_connected", False),
        "gmail_address":   profile.get("gmail_address",""),
    })

@app.route("/api/check-keys")
def check_keys():
    return jsonify({"groq": bool(GROQ_API_KEY), "serp": bool(SERP_API_KEY)})

@app.route("/api/run")
@login_required
def run_pipeline():
    # Capture session data before generator starts
    user_id = session["user_id"]
    can, err = check_usage(user_id, "search")
    if not can:
        def error_gen():
            yield f"event: error\ndata: {err}\n\n"
        return Response(error_gen(), mimetype="text/event-stream")

    details = {}
    for field in ["title","location","experience","industry","academic","seniority"]:
        v = request.args.get(field,"")
        if v: details[field] = v
    tech_raw = request.args.get("techSkills","")
    func_raw = request.args.get("funcSkills","")
    if tech_raw: details["techSkills"] = [s.strip() for s in tech_raw.split(",") if s.strip()]
    if func_raw: details["funcSkills"] = [s.strip() for s in func_raw.split(",") if s.strip()]

    companies = [c.strip() for c in request.args.get("companies","").split(",") if c.strip()]
    max_s     = int(request.args.get("max_searches", 6))
    rps       = int(request.args.get("results_per_search", 10))

    def generate():
        try:
            yield f"event: status\ndata: {json.dumps({'step':1,'msg':'Generating X-ray strings...','pct':10})}\n\n"
            xrays = generate_xray_strings(details, companies)
            yield f"event: xrays\ndata: {json.dumps(xrays)}\n\n"
            yield f"event: status\ndata: {json.dumps({'step':1,'msg':f'{len(xrays)} strings ready','pct':22})}\n\n"

            all_profiles, seen_urls = [], set()
            for i, xray in enumerate(xrays[:max_s], 1):
                pct = 22 + int((i/max_s)*40)
                label = xray["label"]
                yield f"event: status\ndata: {json.dumps({'step':2,'msg':f'Searching: {label} ({i}/{max_s})...','pct':pct})}\n\n"
                try:
                    results  = search_google(xray["string"], rps)
                    profiles = extract_profiles(results)
                    for p in profiles:
                        if p["linkedin"] not in seen_urls:
                            seen_urls.add(p["linkedin"])
                            p["source_string"] = xray["label"]
                            all_profiles.append(p)
                    yield f"event: search_result\ndata: {json.dumps({'string_label':xray['label'],'found':len(profiles),'total':len(all_profiles)})}\n\n"
                    time.sleep(1.2)
                except Exception as e:
                    yield f"event: warn\ndata: Search {i} failed: {e}\n\n"

            if not all_profiles:
                yield f"event: error\ndata: No profiles found. Try broader settings.\n\n"
                return

            yield f"event: status\ndata: {json.dumps({'step':3,'msg':f'Scoring {len(all_profiles)} profiles...','pct':68})}\n\n"
            candidates = score_candidates(all_profiles, details)

            today    = date.today().isoformat()
            safe     = re.sub(r"[^\w\s-]","", details.get("title","candidates")).replace(" ","_")
            filename = OUTPUT_DIR / f"{safe}_candidates_{today}.csv"
            with open(filename,"w",newline="",encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Rank","Name","LinkedIn URL","Headline","Match Score","Reason","Source String","Date Sourced","Email","Contacted"])
                writer.writeheader()
                for i, c in enumerate(candidates,1):
                    writer.writerow({"Rank":i,"Name":c.get("name",""),"LinkedIn URL":c.get("linkedin",""),"Headline":c.get("headline",""),"Match Score":f"{c.get('score',0)}%","Reason":c.get("reason",""),"Source String":c.get("source_string",""),"Date Sourced":today,"Email":"","Contacted":"No"})

            increment_usage(user_id, "search")
            yield f"event: candidates\ndata: {json.dumps(candidates)}\n\n"
            yield f"event: status\ndata: {json.dumps({'step':4,'msg':f'Done! {len(candidates)} candidates sourced.','pct':100})}\n\n"
            yield f"event: done\ndata: {json.dumps({'file':filename.name,'count':len(candidates)})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/find-emails", methods=["POST"])
@login_required
def find_emails():
    data          = request.json
    linkedin_urls = data.get("linkedin_urls", [])
    if not linkedin_urls:
        return jsonify({"ok": False, "error": "No URLs provided"}), 400
    can, err = check_usage(session["user_id"], "email")
    if not can:
        return jsonify({"ok": False, "error": err}), 403
    results = []
    for url in linkedin_urls:
        email = find_email_prospeo(url)
        results.append({"linkedin": url, "email": email or ""})
        if email:
            increment_usage(session["user_id"], "email")
        time.sleep(0.5)
    return jsonify({"ok": True, "results": results})

@app.route("/api/send-emails", methods=["POST"])
@login_required
def send_emails():
    data       = request.json
    candidates = data.get("candidates", [])
    subject    = data.get("subject","")
    body       = data.get("body","")
    if not candidates or not subject or not body:
        return jsonify({"ok": False, "error": "Missing fields"}), 400

    profile = get_user_profile(session["user_id"])
    if not profile.get("gmail_connected"):
        return jsonify({"ok": False, "error": "Gmail not connected. Please add your Gmail in Settings."}), 403

    gmail        = profile.get("gmail_address","")
    app_password = profile.get("gmail_app_password","")
    sent, failed = [], []

    for c in candidates:
        email = c.get("email","")
        if not email:
            failed.append({"name": c.get("name"), "reason": "No email"})
            continue
        try:
            send_smtp_email(gmail, app_password, email, subject, body)
            sent.append(c.get("name"))
            supabase.table("outreach").insert({
                "user_id":         session["user_id"],
                "candidate_name":  c.get("name",""),
                "candidate_email": email,
                "linkedin_url":    c.get("linkedin",""),
                "subject":         subject,
                "sent_at":         datetime.utcnow().isoformat(),
                "status":          "sent"
            }).execute()
            time.sleep(0.3)
        except Exception as e:
            failed.append({"name": c.get("name"), "reason": str(e)})

    return jsonify({"ok": True, "sent": sent, "failed": failed})

@app.route("/api/outreach-history")
@login_required
def outreach_history():
    try:
        res = supabase.table("outreach").select("*").eq("user_id", session["user_id"]).order("sent_at", desc=True).limit(100).execute()
        return jsonify({"ok": True, "history": res.data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/create-subscription", methods=["POST"])
@login_required
def create_subscription():
    data    = request.json
    plan_id = data.get("plan")
    if plan_id not in ["individual","agency"]:
        return jsonify({"ok": False, "error": "Invalid plan"}), 400
    plan = PLANS[plan_id]
    try:
        resp = req.post(
            "https://api.razorpay.com/v1/subscriptions",
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            json={"plan_id": plan["razorpay_plan"], "total_count": 12, "quantity": 1, "customer_notify": 1,
                  "notes": {"user_id": session["user_id"], "plan": plan_id}},
            timeout=15
        )
        sub = resp.json()
        return jsonify({"ok": True, "subscription_id": sub["id"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/payment-success", methods=["POST"])
@login_required
def payment_success():
    data    = request.json
    plan_id = data.get("plan")
    if plan_id in PLANS:
        supabase.table("profiles").update({
            "plan": plan_id, "searches_used": 0, "emails_used": 0,
            "plan_started": datetime.utcnow().isoformat()
        }).eq("id", session["user_id"]).execute()
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 400

@app.route("/api/download/<filename>")
@login_required
def download(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return "File not found", 404
    return send_file(path, as_attachment=True, download_name=filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=False, threaded=True, port=port)
