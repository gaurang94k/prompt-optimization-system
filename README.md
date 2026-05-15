# Prompt Optimization System

AI-powered prompt refinement pipeline that critiques, clarifies, and optimizes your prompts before executing them — producing higher quality LLM outputs.

Built as a single-file standalone application using **Python + FastAPI + Google Gemini**.

## Quick Start (Run Locally)

### 1. Prerequisites
- Python 3.9+ installed ([python.org](https://python.org))
- A Google Gemini API key ([get one free](https://aistudio.google.com))

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API key
Copy the example env file and add your key:
```bash
cp .env.example .env
```
Then open `.env` and replace `your-gemini-api-key-here` with your actual key.

### 4. Run
```bash
python main.py
```
Your browser opens automatically to `http://localhost:8000`.

## Deployment (Public Website)

### Option A: Render (Recommended — Free Tier)

1. Push your repo to GitHub (see below)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables in Render dashboard:
   - `GEMINI_API_KEY` = your key
   - `APP_PASSWORD` = your chosen password
6. Deploy

### Option B: Railway

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Add environment variables: `GEMINI_API_KEY`, `APP_PASSWORD`
3. Railway auto-detects Python and deploys

### Pushing to GitHub

```bash
# In your project folder:
git init
git add main.py requirements.txt .env.example .gitignore README.md
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

> **Important:** Never commit your `.env` file. The `.gitignore` already blocks it.

## How It Works

The system uses a two-phase pipeline:

1. **Prompt Clarification** — Extracts task, scope, role, and output format → user confirms → optional context enrichment
2. **Prompt Optimization** — Diagnoses defects → applies targeted prompt engineering techniques → segments into blocks → executes sequentially with quality filtering

## Security

- **Password Protection:** Set `APP_PASSWORD` to require login before using the app
- **Rate Limiting:** 40 requests/minute per authenticated session
- **API Key Safety:** Key stays server-side only, never exposed to the browser

## Tech Stack

- **Backend:** Python, FastAPI, Google Gemini 2.5 Flash
- **Frontend:** Vanilla HTML/CSS/JS (embedded in the Python file)
- **No database required** — sessions stored in memory
