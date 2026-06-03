# AI Recruiting Agent — Web App

A browser-based recruiting tool. Paste a JD, AI extracts details, runs Google X-ray searches via SerpAPI, scores candidates with Claude, and exports a real CSV with working LinkedIn links.

## Setup (5 minutes)

### 1. Install dependencies
  pip install -r requirements.txt

### 2. Get API keys
  - Anthropic: https://console.anthropic.com -> API Keys (~$0.03 per run)
  - SerpAPI:   https://serpapi.com           (free: 100 searches/mo)

### 3. Set environment variables
  Mac/Linux:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export SERP_API_KEY="your-key"

  Windows:
    set ANTHROPIC_API_KEY=sk-ant-...
    set SERP_API_KEY=your-key

### 4. Run
  python app.py
  Open: http://localhost:5000

## Share with your team

  Same network: change app.run(...) to host='0.0.0.0' — team opens http://YOUR_IP:5000
  Internet:     Deploy free on Railway (railway.app) or Render (render.com) via GitHub
