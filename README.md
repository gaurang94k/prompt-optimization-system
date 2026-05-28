# Prompt Optimization System

A small web app that rewrites your prompt *before* it hits the model. You type a rough prompt, it asks a couple of clarifying questions, figures out what's weak about it, and rebuilds it using standard prompt-engineering techniques — then runs the improved version through Gemini and hands back the output.

I built it because I kept burning through API calls (and patience) tweaking the same prompt over and over. This front-loads that work so the first run is closer to what I actually wanted.

**Live demo:** https://prompt-optimization-system-1.onrender.com

## Stack

Python + FastAPI on the backend, Google Gemini 2.5 Flash for the model calls, and plain HTML/CSS/JS for the UI — all in a single `main.py`. No database; sessions live in memory.

## How it works

Two passes:

1. **Clarify** — pulls out the task, scope, role, and desired output format, confirms them with you, then optionally enriches with extra context.
2. **Optimize** — diagnoses the weak spots, applies targeted fixes, splits the work into blocks, and runs them in sequence with a quality check on each.

## Running it locally

You'll need Python 3.9+ and a Gemini API key (free from [Google AI Studio](https://aistudio.google.com)).

```bash
pip install -r requirements.txt
cp .env.example .env        # paste your key into .env
python main.py
```

Opens at `http://localhost:8000`.

## Notes

- Set `APP_PASSWORD` to put the app behind a login.
- Rate-limited to 40 requests/min per session.
- The API key stays server-side — it's never sent to the browser.
- Deployed on Render's free tier with `uvicorn main:app --host 0.0.0.0 --port $PORT`; `GEMINI_API_KEY` and `APP_PASSWORD` set as env vars.
