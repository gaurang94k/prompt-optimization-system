"""
Prompt Optimization System — Single-file Standalone Application
================================================================
Run:   python main.py
Open:  http://localhost:8000  (auto-opens in browser)

Dependencies:
    pip install fastapi uvicorn google-genai pydantic

Set your Gemini API key:
    export GEMINI_API_KEY="your-key-here"
"""

from __future__ import annotations

import json, logging, os, re, time, uuid, random, threading, webbrowser
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from google import genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ============================================================
# 0) LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ============================================================
# 1) APP + CORS
# ============================================================
app = FastAPI(title="Prompt Optimization System", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth middleware — protects pipeline endpoints
OPEN_PATHS = {"/", "/health", "/api/status", "/api/login", "/api/set-key"}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if request.method == "GET" or path in OPEN_PATHS:
        return await call_next(request)
    # Check auth for all other POST endpoints
    if APP_PASSWORD:
        token = (request.headers.get("authorization") or "").replace("Bearer ", "")
        if not _check_auth(token):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized — please log in."})
        if not _check_rate(token):
            return JSONResponse(status_code=429, content={"detail": "Rate limited — max 40 requests/minute."})
    return await call_next(request)

# ============================================================
# 2) GEMINI SETUP — supports .env file, env var, or UI input
# ============================================================
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")

# Try loading from .env file in same folder (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
    log.info("Loaded .env file")
except ImportError:
    pass

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
APP_PASSWORD: str = os.getenv("APP_PASSWORD", "")  # If set, users must enter this to access
client: Optional[genai.Client] = None

def init_gemini(api_key: Optional[str] = None) -> None:
    global client, GEMINI_API_KEY
    if api_key:
        GEMINI_API_KEY = api_key
    if not GEMINI_API_KEY:
        log.warning("No API key yet — waiting for user to provide via UI.")
        return
    client = genai.Client(api_key=GEMINI_API_KEY)
    log.info(f"Gemini client ready  model={MODEL_NAME}")

class LLMError(RuntimeError):
    pass

# ============================================================
# 2b) AUTH + RATE LIMITING
# ============================================================
AUTH_TOKENS: Dict[str, float] = {}  # token -> last_active timestamp
AUTH_TTL = 86400  # tokens valid for 24 hours
RATE_LIMIT: Dict[str, List[float]] = {}  # token -> list of request timestamps
RATE_MAX = 40  # max requests per minute
RATE_WINDOW = 60  # seconds

def _check_auth(token: Optional[str]) -> bool:
    """Verify auth token is valid. Returns True if auth not required or token valid."""
    if not APP_PASSWORD:
        return True  # no password set = open access (local dev)
    if not token:
        return False
    token = token.replace("Bearer ", "")
    if token not in AUTH_TOKENS:
        return False
    if time.time() - AUTH_TOKENS[token] > AUTH_TTL:
        del AUTH_TOKENS[token]
        return False
    AUTH_TOKENS[token] = time.time()  # refresh
    return True

def _check_rate(token: str) -> bool:
    """Simple sliding window rate limiter."""
    now = time.time()
    key = token or "anon"
    if key not in RATE_LIMIT:
        RATE_LIMIT[key] = []
    RATE_LIMIT[key] = [t for t in RATE_LIMIT[key] if now - t < RATE_WINDOW]
    if len(RATE_LIMIT[key]) >= RATE_MAX:
        return False
    RATE_LIMIT[key].append(now)
    return True

def require_auth(authorization: Optional[str] = None):
    """Call at start of protected endpoints."""
    if not _check_auth(authorization):
        raise HTTPException(401, "Unauthorized — please log in first.")
    token = (authorization or "").replace("Bearer ", "") or "anon"
    if not _check_rate(token):
        raise HTTPException(429, "Rate limited — please slow down. Max 40 requests/minute.")

# Auth endpoints
class LoginReq(BaseModel):
    password: str = Field(..., min_length=1)

class SetKeyReq(BaseModel):
    api_key: str = Field(..., min_length=1)

@app.get("/api/status")
def api_status():
    return {
        "has_key": bool(GEMINI_API_KEY and client),
        "needs_auth": bool(APP_PASSWORD),
        "model": MODEL_NAME,
    }

@app.post("/api/login")
def api_login(req: LoginReq):
    if not APP_PASSWORD:
        return {"ok": True, "token": "open"}
    if req.password.strip() != APP_PASSWORD:
        raise HTTPException(401, "Incorrect password.")
    token = str(uuid.uuid4())
    AUTH_TOKENS[token] = time.time()
    log.info(f"User authenticated — token {token[:8]}...")
    return {"ok": True, "token": token}

@app.post("/api/set-key")
def set_api_key(req: SetKeyReq):
    try:
        init_gemini(req.api_key.strip())
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, f"Invalid API key: {e}")

# ============================================================
# 3) UTILITIES
# ============================================================
def _sanitize_json_text(s: str) -> str:
    if not s:
        return s
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)

def _find_json_object(text: str) -> Optional[str]:
    """Find the first valid top-level JSON object using brace-depth scanning."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

def safe_json_parse(text: str) -> dict:
    if text is None:
        raise ValueError("LLM returned None (expected JSON).")
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = _sanitize_json_text(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        blob = _find_json_object(cleaned)
        if blob:
            return json.loads(_sanitize_json_text(blob))
        raise ValueError(f"LLM did not return valid JSON.\nRaw:\n{text[:300]}")

# ============================================================
# 4) LLM CALL — exponential backoff with jitter
# ============================================================
def call_llm(prompt: str, *, retries: int = 3, base_backoff: float = 0.5) -> str:
    last_err: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = (response.text or "").strip()
            if not text:
                raise LLMError("Empty model response.")
            return text
        except LLMError:
            raise
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                wait = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
                log.warning(f"LLM attempt {attempt} failed: {last_err}  retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            raise LLMError(last_err or "Unknown LLM error")

# ============================================================
# 5) CONSTANTS
# ============================================================
REQUIRED_FIELDS = ["role", "task", "scope", "output_format"]
FieldKey = Literal["role", "task", "scope", "output_format"]
IssueKey = Literal["unclear_task", "missing_constraints", "missing_examples", "excessive_complexity"]
ISSUE_ORDER: List[IssueKey] = ["unclear_task", "missing_constraints", "missing_examples", "excessive_complexity"]
ISSUE_TO_TECHNIQUE: Dict[IssueKey, str] = {
    "unclear_task": "explicit_instruction",
    "missing_constraints": "add_constraints",
    "missing_examples": "few_shot_prompting",
    "excessive_complexity": "step_by_step_decomposition",
}

TECHNIQUE_INSTRUCTIONS: Dict[str, str] = {
    "explicit_instruction": """Make the task definition crystal clear:
- Add a specific action verb if vague
- Define the expected deliverable precisely
- Specify the audience or reader
- Remove ambiguous qualifiers
Example improvement: "Write a report" → "Write a 2-page executive summary for C-suite leadership analyzing Q3 revenue decline, including root causes and recommended actions." """,

    "add_constraints": """Add necessary boundaries and constraints:
- Set scope limits (time period, geography, industry sector)
- Define length/format requirements if missing
- Specify what to include AND what to explicitly exclude
- Add measurability criteria where applicable
Example: Add "Focus on the last 3 fiscal years. Limit to North American markets. Present findings in a table with supporting narrative." """,

    "few_shot_prompting": """Add 1-2 brief examples showing the expected input→output pattern:
- Show the format, tone, and level of detail expected
- Keep examples concise but illustrative of the quality bar
- Make sure examples match the domain of the task
Example addition: "For reference, a good output looks like: 'Challenge: Legacy POS systems → Impact: 23% slower checkout times → Recommendation: Cloud-based unified commerce platform (est. ROI 18 months)'" """,

    "step_by_step_decomposition": """Break complex reasoning into sequential numbered steps:
- Number each step clearly (Step 1, Step 2, etc.)
- Each step should build logically on the previous one
- Make the analytical chain explicit so the model follows it
- Ensure final step synthesizes into actionable output
Example: "Step 1: Identify the top 3 root causes. Step 2: Quantify the business impact of each. Step 3: Propose solutions ranked by feasibility and ROI. Step 4: Present a phased implementation timeline." """,
}

# ============================================================
# 6) PHASE A HELPERS — Clarification
# ============================================================
def llm_impossible_prompt(user_prompt: str) -> Tuple[bool, str]:
    log.info("Running feasibility check...")
    resp = call_llm(f"""You are a strict JSON classifier.

Determine if the prompt should be BLOCKED. Block ONLY if it meets at least one condition:
1. Contains logically contradictory or mutually exclusive requirements
2. Requests something physically or logically impossible
3. Is completely meaningless or incoherent with no recoverable intent

Do NOT block prompts that are vague, missing context, informal, short, or simple.
When in doubt, do NOT block.

Return ONLY JSON: {{"blocked": true|false, "reason": "short reason"}}

Prompt: \"\"\"{user_prompt}\"\"\"""")
    data = safe_json_parse(resp)
    blocked = bool(data.get("blocked", False))
    reason = str(data.get("reason", "")).strip()
    log.info(f"Feasibility: blocked={blocked}")
    return blocked, (reason or "Blocked by safety/impossibility filter.")


def llm_detect_prompt_fields(prompt: str) -> Dict[str, Optional[str]]:
    log.info("Extracting prompt parameters...")
    resp = call_llm(f"""You are a strict JSON generator.

Analyze the prompt and determine whether it explicitly specifies:
- Role: WHO the model should act as
- Task: WHAT ACTION to perform (must contain an action verb)
- Scope: DOMAIN / CONTEXT / BOUNDARY (must NOT contain action verbs)
- Output_format: Expected response structure

Rules: Do NOT infer missing info. Do NOT guess. If unclear, return null.
Return ONLY valid JSON:
{{"role": string|null, "task": string|null, "scope": string|null, "output_format": string|null}}

Prompt: \"\"\"{prompt}\"\"\"""")
    data = safe_json_parse(resp)
    out: Dict[str, Optional[str]] = {k: None for k in REQUIRED_FIELDS}
    for k in REQUIRED_FIELDS:
        v = data.get(k)
        if isinstance(v, str):
            v = v.strip()
            if not v or v.lower() == "null":
                v = None
        else:
            v = None
        out[k] = v
    return out


def check_missing(state: Dict[str, Optional[str]]) -> List[str]:
    return [k for k in REQUIRED_FIELDS if not (state.get(k) or "").strip()]


def llm_task_scope_collision(task: Optional[str], scope: Optional[str]) -> Tuple[bool, str]:
    if not task or not scope:
        return False, ""
    resp = call_llm(f"""You are validating prompt structure.
Task: "{task}"
Scope: "{scope}"
Do these represent the SAME conceptual meaning (scope written like an action/task)?
Return STRICT JSON: {{"collision": true|false, "reason": "short reason"}}""")
    data = safe_json_parse(resp)
    return bool(data.get("collision", False)), str(data.get("reason", "")).strip()


def build_structured_prompt(eff: Dict[str, str]) -> str:
    return (f"You are {eff['role']}.\nTask: {eff['task']}\n"
            f"Scope: {eff['scope']}\nOutput format: {eff['output_format']}").strip()


def meaning_changed(original: str, modified: str) -> bool:
    """JSON-based semantic comparison — more reliable than TRUE/FALSE parsing."""
    resp = call_llm(f"""Compare these two prompts. Has the core meaning or intent changed?
Minor wording improvements, added structure, or clarifications do NOT count as meaning changes.
Only flag TRUE if the fundamental goal, audience, or deliverable has shifted.

Prompt A: \"\"\"{original}\"\"\"
Prompt B: \"\"\"{modified}\"\"\"

Return ONLY JSON: {{"changed": true|false, "confidence": 0.0-1.0}}""")
    try:
        data = safe_json_parse(resp)
        changed = bool(data.get("changed", True))
        confidence = float(data.get("confidence", 0.5))
        if confidence < 0.6:
            return False
        return changed
    except Exception:
        return True


def merge_with_original(original: str, structured: str) -> str:
    log.info("Merging structured prompt with original...")
    resp = call_llm(f"""You are a prompt reconstruction system.

ORIGINAL: \"\"\"{original}\"\"\"
STRUCTURED: \"\"\"{structured}\"\"\"

Rules:
- Preserve structured format
- Reintegrate any lost constraints or requirements from ORIGINAL
- Do NOT invent new information or change intent
- Return ONLY the final prompt text""").strip()
    if meaning_changed(structured, resp):
        return structured
    return resp


def llm_context_questions(structured: str) -> List[str]:
    log.info("Generating context questions...")
    resp = call_llm(f"""You are evaluating whether additional context is needed.

Structured prompt: \"\"\"{structured}\"\"\"

Rules: Only ask questions that are NECESSARY. Max 3 questions. If nothing needed, return empty list.
Return STRICT JSON: {{"questions": ["list of strings"]}}""")
    data = safe_json_parse(resp)
    qs = data.get("questions", [])
    return [str(x).strip() for x in (qs if isinstance(qs, list) else []) if str(x).strip()]


def integrate_context(clarified: str, user_context: str) -> str:
    uc = (user_context or "").strip()
    if not uc:
        return clarified
    return f"{clarified}\n\nAdditional Context:\n{uc}".strip()

# ============================================================
# 7) PHASE B HELPERS — Optimization
# ============================================================
def diagnose_prompt(prompt: str) -> Dict[IssueKey, bool]:
    log.info("Diagnosing prompt defects...")
    resp = call_llm(f"""You are a prompt quality analyst.

Detect whether the prompt has each issue:
- unclear_task: The main objective is vague or ambiguous
- missing_constraints: No boundaries, limits, or scope defined
- missing_examples: Would benefit from example input/output patterns
- excessive_complexity: Multiple complex sub-tasks that need decomposition

Return ONLY JSON: {{"unclear_task": bool, "missing_constraints": bool, "missing_examples": bool, "excessive_complexity": bool}}

Prompt: \"\"\"{prompt}\"\"\"""")
    data = safe_json_parse(resp)
    out: Dict[IssueKey, bool] = {k: False for k in ISSUE_ORDER}
    for k in ISSUE_ORDER:
        out[k] = bool(data.get(k, False))
    log.info(f"Diagnosis: {out}")
    return out


def apply_technique(prompt: str, technique: str) -> str:
    instructions = TECHNIQUE_INSTRUCTIONS.get(technique, f"Apply the technique: {technique}")
    resp = call_llm(f"""You are a prompt optimization system.

Apply ONLY this prompting technique to improve the prompt:

TECHNIQUE: {technique}
INSTRUCTIONS:
{instructions}

CRITICAL RULES:
- Preserve original meaning strictly
- Do NOT answer the prompt — only improve its structure
- Do NOT add new goals or domain assumptions
- Return ONLY the updated prompt text

Original Prompt: \"\"\"{prompt}\"\"\"""").strip()
    return resp if resp else prompt

# ============================================================
# 8) PHASE C — Segmented Execution + Filtering
# ============================================================
def _parse_blocks(text: str, expected: int) -> List[str]:
    BLOCK_RE = re.compile(
        r"^\s*(?:\*{0,2}#{0,3}\s*)?(?:block\s*)?(\d+)[\.:\)]\s*(?:\*{0,2})?",
        re.IGNORECASE,
    )
    lines = [l.rstrip() for l in (text or "").split("\n")]
    blocks: List[str] = []
    cur: List[str] = []
    found_markers = False

    def flush():
        chunk = "\n".join(l for l in cur if l.strip()).strip()
        if chunk:
            blocks.append(chunk)
        cur.clear()

    for ln in lines:
        if BLOCK_RE.match(ln):
            found_markers = True
            flush()
            remainder = BLOCK_RE.sub("", ln, count=1).strip()
            if remainder:
                cur.append(remainder)
        else:
            cur.append(ln)
    flush()
    blocks = [b for b in blocks if b.strip()]

    if not found_markers or len(blocks) == 0:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
        if len(paragraphs) >= expected:
            blocks = paragraphs

    if len(blocks) < expected and expected > 1:
        all_lines = [l for l in lines if l.strip()]
        chunk_size = max(1, -(-len(all_lines) // expected))
        blocks = []
        for i in range(expected):
            start = i * chunk_size
            end = start + chunk_size if i < expected - 1 else len(all_lines)
            chunk = "\n".join(all_lines[start:end]).strip()
            if chunk:
                blocks.append(chunk)

    return blocks[:expected] if len(blocks) >= expected else blocks


def decide_segmentation(prompt: str) -> int:
    resp = call_llm(f"""Count the number of INDEPENDENT, SEPARABLE tasks in this prompt.
Each task must produce a distinct standalone output.
Sub-steps of a single task = still ONE task. Max 3, min 1. When in doubt, fewer.
Return STRICT JSON: {{"blocks": 1}} or {{"blocks": 2}} or {{"blocks": 3}}

Prompt: \"\"\"{prompt}\"\"\"""")
    try:
        return max(1, min(3, int(safe_json_parse(resp).get("blocks", 1))))
    except Exception:
        return 1


def segment_prompt(prompt: str) -> List[str]:
    n = decide_segmentation(prompt)
    log.info(f"Segmentation: {n} blocks")
    if n == 1:
        return [prompt]
    resp = call_llm(f"""Split this prompt into exactly {n} numbered blocks.
Each block = ONE self-contained task. Label: 1. <text>  2. <text>  etc.
Do NOT use markdown headers. Do not answer the prompt — only split it.

Prompt: \"\"\"{prompt}\"\"\"""")
    blocks = _parse_blocks(resp, expected=n)
    if len(blocks) < n:
        resp2 = call_llm(f"""Split into {n} parts. Number each: 1. ... 2. ... Return ONLY numbered parts.
Prompt: \"\"\"{prompt}\"\"\"""")
        blocks2 = _parse_blocks(resp2, expected=n)
        if len(blocks2) >= len(blocks):
            blocks = blocks2
    return blocks if blocks else [prompt]


def execute_block(block_prompt: str, prior: List[str]) -> str:
    ctx = "\n\n".join([f"[Prior Output {i+1}]\n{o}" for i, o in enumerate(prior)]).strip()
    return call_llm(f"""You are executing a segmented prompt in a sequential pipeline.

Prior context: {ctx if ctx else "(none)"}

Current block: \"\"\"{block_prompt}\"\"\"

Rules:
- Produce the best possible output for this block
- Use prior outputs for consistency, don't repeat them
- If facts are missing, state minimal assumptions briefly
- Return ONLY the block output""").strip()


def filter_block_output(block_prompt: str, raw: str, prior: List[str], max_retries: int = 2) -> Tuple[str, str]:
    ctx = "\n\n".join([f"[Prior Output {i+1}]\n{o}" for i, o in enumerate(prior)]).strip()
    candidate = raw
    last_notes = ""
    for attempt in range(max_retries):
        resp = call_llm(f"""You are a strict output filter.

Context: {ctx if ctx else "(none)"}
Block prompt: \"\"\"{block_prompt}\"\"\"
Raw output: \"\"\"{candidate}\"\"\"

Validate and minimally rewrite to satisfy:
1) Alignment with block prompt  2) Correctness  3) Conciseness  4) No drift
Return STRICT JSON: {{"pass": true|false, "filtered_output": "string", "notes": "short notes"}}""")
        data = safe_json_parse(resp)
        filtered = str(data.get("filtered_output", "")).strip() or candidate
        notes = str(data.get("notes", "")).strip()
        last_notes = notes or last_notes
        if bool(data.get("pass", False)):
            return filtered, notes or "pass"
        candidate = filtered
    return candidate, last_notes or "filter fallback"


def run_block_pipeline(prompt: str) -> Tuple[List[str], List[str], List[str], str]:
    blocks = segment_prompt(prompt)
    raw_outputs, filtered_outputs = [], []
    for i, blk in enumerate(blocks):
        log.info(f"Executing block {i+1}/{len(blocks)}...")
        raw = execute_block(blk, filtered_outputs)
        filtered, _ = filter_block_output(blk, raw, filtered_outputs)
        raw_outputs.append(raw)
        filtered_outputs.append(filtered)
    return blocks, raw_outputs, filtered_outputs, "\n\n".join(filtered_outputs).strip()

# ============================================================
# 9) PHASE D — Output Revision
# ============================================================
def revise_filtered_output(block_prompt: str, current: str, instruction: str) -> str:
    resp = call_llm(f"""You are an output editor.

BLOCK PROMPT (reference): \"\"\"{block_prompt}\"\"\"
CURRENT OUTPUT: \"\"\"{current}\"\"\"
USER INSTRUCTION: \"\"\"{instruction}\"\"\"

Apply the instruction to the output. Keep alignment with block prompt.
Do NOT introduce unrelated goals. Return ONLY the revised output.""").strip()
    return resp or current

# ============================================================
# 10) SESSION MANAGEMENT
# ============================================================
@dataclass
class Session:
    session_id: str
    original_prompt: str
    working_prompt: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    detected_state: Dict[str, Optional[str]] = field(default_factory=dict)
    overrides: Dict[str, Optional[str]] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    collision: bool = False
    collision_reason: Optional[str] = None
    structure_confirmed: bool = False
    context_questions: List[str] = field(default_factory=list)
    additional_context: str = ""
    clarified_prompt: Optional[str] = None
    semi_final_prompt: Optional[str] = None
    blocks: List[str] = field(default_factory=list)
    raw_outputs: List[str] = field(default_factory=list)
    filtered_outputs: List[str] = field(default_factory=list)
    final_output: Optional[str] = None
    revision_count: int = 0
    blocked: bool = False
    blocked_reason: Optional[str] = None
    optimization_log: List[str] = field(default_factory=list)

SESSIONS: Dict[str, Session] = {}
MAX_SESSIONS = 200
SESSION_TTL = 1800  # 30 minutes

def _cleanup_expired():
    now = time.time()
    expired = [k for k, v in SESSIONS.items() if now - v.last_active > SESSION_TTL]
    for k in expired:
        del SESSIONS[k]
    if expired:
        log.info(f"Cleaned up {len(expired)} expired sessions")

def _register(sess: Session):
    _cleanup_expired()
    if len(SESSIONS) >= MAX_SESSIONS:
        oldest = min(SESSIONS, key=lambda k: SESSIONS[k].last_active)
        del SESSIONS[oldest]
    SESSIONS[sess.session_id] = sess

def _get(sid: str) -> Session:
    sess = SESSIONS.get(sid)
    if not sess:
        raise HTTPException(404, "Session not found")
    sess.last_active = time.time()
    return sess

def effective_state(sess: Session) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for k in REQUIRED_FIELDS:
        ov = sess.overrides.get(k)
        if ov and str(ov).strip():
            out[k] = str(ov).strip()
        else:
            dv = sess.detected_state.get(k)
            out[k] = dv.strip() if isinstance(dv, str) and dv.strip() else None
    return out

# ============================================================
# 11) API SCHEMAS
# ============================================================
class StartReq(BaseModel):
    user_prompt: str = Field(..., min_length=1)

class StartResp(BaseModel):
    session_id: str
    working_prompt: str
    state: Dict[str, Optional[str]]
    missing: List[str]
    blocked: bool = False
    blocked_reason: Optional[str] = None
    collision: bool = False
    collision_reason: Optional[str] = None

class UpdateFieldsReq(BaseModel):
    session_id: str
    role: Optional[str] = None
    task: Optional[str] = None
    scope: Optional[str] = None
    output_format: Optional[str] = None

class PhaseAResp(BaseModel):
    session_id: str
    state: Dict[str, Optional[str]]
    missing: List[str]
    collision: bool = False
    collision_reason: Optional[str] = None

class SessionIdReq(BaseModel):
    session_id: str

class ContextQResp(BaseModel):
    session_id: str
    questions: List[str]

class SubmitCtxReq(BaseModel):
    session_id: str
    additional_context: str = ""

class FinalizeResp(BaseModel):
    session_id: str
    clarified: str
    context_questions: List[str]

class OptimizeResp(BaseModel):
    session_id: str
    semi_final: str
    blocks: List[str]
    raw_outputs: List[str]
    filtered_outputs: List[str]
    final: str
    optimization_log: List[str]

class ReviseReq(BaseModel):
    session_id: str
    block_index: int
    instruction: str = Field(..., min_length=1)

class ReviseResp(BaseModel):
    session_id: str
    filtered_outputs: List[str]
    final: str
    revision_count: int

# ============================================================
# 12) ENDPOINTS
# ============================================================
@app.on_event("startup")
def _startup():
    init_gemini()

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "sessions": len(SESSIONS)}

@app.post("/session/start", response_model=StartResp)
def session_start(req: StartReq):
    prompt = req.user_prompt.strip()
    if not prompt:
        raise HTTPException(400, "user_prompt is required")

    try:
        blocked, reason = llm_impossible_prompt(prompt)
    except LLMError as e:
        raise HTTPException(502, f"LLM error: {e}")

    sid = str(uuid.uuid4())
    sess = Session(
        session_id=sid, original_prompt=prompt, working_prompt=prompt,
        detected_state={k: None for k in REQUIRED_FIELDS},
        overrides={k: None for k in REQUIRED_FIELDS},
        blocked=blocked, blocked_reason=reason if blocked else None,
    )
    _register(sess)

    if blocked:
        return StartResp(session_id=sid, working_prompt=prompt,
                         state=effective_state(sess), missing=REQUIRED_FIELDS[:],
                         blocked=True, blocked_reason=reason)
    try:
        detected = llm_detect_prompt_fields(prompt)
    except LLMError as e:
        raise HTTPException(502, f"LLM error: {e}")
    sess.detected_state.update(detected)

    try:
        coll, cr = llm_task_scope_collision(sess.detected_state.get("task"), sess.detected_state.get("scope"))
    except LLMError as e:
        raise HTTPException(502, f"LLM error: {e}")
    sess.collision, sess.collision_reason = coll, cr if coll else None
    if coll:
        sess.detected_state["scope"] = None
    sess.missing = check_missing(effective_state(sess))
    log.info(f"Session {sid[:8]} started — missing={sess.missing}")

    return StartResp(session_id=sid, working_prompt=prompt,
                     state=effective_state(sess), missing=sess.missing,
                     collision=sess.collision, collision_reason=sess.collision_reason)

@app.post("/phase-a/update-fields", response_model=PhaseAResp)
def update_fields(req: UpdateFieldsReq):
    sess = _get(req.session_id)
    if sess.blocked:
        raise HTTPException(400, f"Blocked: {sess.blocked_reason}")
    for k, v in {"role": req.role, "task": req.task, "scope": req.scope, "output_format": req.output_format}.items():
        if v is not None:
            sess.overrides[k] = v.strip() if v.strip() else None
    try:
        coll, cr = llm_task_scope_collision(effective_state(sess).get("task"), effective_state(sess).get("scope"))
    except LLMError as e:
        raise HTTPException(502, f"LLM error: {e}")
    sess.collision, sess.collision_reason = coll, cr if coll else None
    if coll and not sess.overrides.get("scope"):
        sess.detected_state["scope"] = None
    sess.missing = check_missing(effective_state(sess))
    return PhaseAResp(session_id=sess.session_id, state=effective_state(sess),
                      missing=sess.missing, collision=sess.collision, collision_reason=sess.collision_reason)

@app.post("/phase-a/confirm-structure")
def confirm_structure(req: SessionIdReq):
    sess = _get(req.session_id)
    if sess.blocked:
        raise HTTPException(400, f"Blocked: {sess.blocked_reason}")
    eff = effective_state(sess)
    miss = check_missing(eff)
    if miss:
        raise HTTPException(400, f"Missing fields: {miss}")
    sess.structure_confirmed = True
    structured = build_structured_prompt({k: eff[k] or "" for k in REQUIRED_FIELDS})
    try:
        sess.context_questions = llm_context_questions(structured)
    except LLMError as e:
        raise HTTPException(502, f"LLM error: {e}")
    return {"ok": True}

@app.post("/phase-a/context-questions", response_model=ContextQResp)
def context_questions(req: SessionIdReq):
    sess = _get(req.session_id)
    return ContextQResp(session_id=sess.session_id, questions=sess.context_questions)

@app.post("/phase-a/submit-context")
def submit_context(req: SubmitCtxReq):
    sess = _get(req.session_id)
    sess.additional_context = (req.additional_context or "").strip()
    return {"ok": True}

@app.post("/phase-a/finalize", response_model=FinalizeResp)
def finalize(req: SessionIdReq):
    sess = _get(req.session_id)
    if sess.blocked:
        raise HTTPException(400, f"Blocked: {sess.blocked_reason}")
    if not sess.structure_confirmed:
        raise HTTPException(400, "Confirm structure first.")
    eff = effective_state(sess)
    structured = build_structured_prompt({k: eff[k] or "" for k in REQUIRED_FIELDS})
    try:
        merged = merge_with_original(sess.original_prompt, structured)
    except LLMError as e:
        raise HTTPException(502, f"LLM error: {e}")
    sess.clarified_prompt = integrate_context(merged, sess.additional_context)
    log.info(f"Session {sess.session_id[:8]} — Phase A complete")
    return FinalizeResp(session_id=sess.session_id, clarified=sess.clarified_prompt,
                        context_questions=sess.context_questions)

@app.post("/optimize", response_model=OptimizeResp)
def optimize(req: SessionIdReq):
    sess = _get(req.session_id)
    if sess.blocked:
        raise HTTPException(400, f"Blocked: {sess.blocked_reason}")
    if not sess.clarified_prompt:
        raise HTTPException(400, "Finalize Phase A first.")

    prompt = sess.clarified_prompt
    opt_log: List[str] = []

    try:
        diagnosis = diagnose_prompt(prompt)
    except (LLMError, ValueError) as e:
        raise HTTPException(502, f"LLM error: {e}")

    for issue in ISSUE_ORDER:
        if not diagnosis.get(issue, False):
            opt_log.append(f"✓ {issue}: not detected — skipped")
            continue
        technique = ISSUE_TO_TECHNIQUE[issue]
        applied = False
        for attempt in range(2):
            try:
                updated = apply_technique(prompt, technique)
            except LLMError as e:
                raise HTTPException(502, f"LLM error: {e}")
            if meaning_changed(prompt, updated):
                opt_log.append(f"⟳ {issue}: attempt {attempt+1} changed meaning — retrying")
                continue
            prompt = updated
            applied = True
            opt_log.append(f"✓ {issue}: applied {technique} (attempt {attempt+1})")
            break
        if not applied:
            opt_log.append(f"✗ {issue}: {technique} failed after 2 attempts — skipped")

    sess.semi_final_prompt = prompt
    log.info(f"Session {sess.session_id[:8]} — Optimization complete")

    try:
        blocks, raws, filtereds, final = run_block_pipeline(prompt)
    except (LLMError, ValueError) as e:
        raise HTTPException(502, f"LLM error: {e}")

    sess.blocks, sess.raw_outputs, sess.filtered_outputs = blocks, raws, filtereds
    sess.final_output = final
    sess.optimization_log = opt_log

    return OptimizeResp(session_id=sess.session_id, semi_final=prompt,
                        blocks=blocks, raw_outputs=raws, filtered_outputs=filtereds,
                        final=final, optimization_log=opt_log)

@app.post("/blocks/revise-output", response_model=ReviseResp)
def revise_output(req: ReviseReq):
    sess = _get(req.session_id)
    if not sess.blocks:
        raise HTTPException(400, "Run /optimize first.")
    if req.block_index < 0 or req.block_index >= len(sess.blocks):
        raise HTTPException(400, "Invalid block_index")
    instruction = req.instruction.strip()
    if not instruction:
        raise HTTPException(400, "instruction required")
    try:
        revised = revise_filtered_output(sess.blocks[req.block_index], sess.filtered_outputs[req.block_index], instruction)
    except LLMError as e:
        raise HTTPException(502, f"LLM error: {e}")
    sess.filtered_outputs[req.block_index] = revised
    sess.final_output = "\n\n".join(sess.filtered_outputs).strip()
    sess.revision_count += 1
    return ReviseResp(session_id=sess.session_id, filtered_outputs=sess.filtered_outputs,
                      final=sess.final_output, revision_count=sess.revision_count)

# ============================================================
# 13) EMBEDDED FRONTEND
# ============================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prompt Optimization System</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060816;--bg2:rgba(12,15,36,.92);--bg3:rgba(18,22,52,.88);--bg4:rgba(26,30,68,.85);
  --border:rgba(120,130,255,.12);--border-h:rgba(120,130,255,.3);--border-glow:rgba(124,58,237,.5);
  --text:#e8ecf4;--text2:#94a3b8;--text3:#5a6380;
  --grad1:linear-gradient(135deg,#8b5cf6,#3b82f6,#06b6d4);
  --grad2:linear-gradient(135deg,#06b6d4,#10b981);
  --grad3:linear-gradient(135deg,#f59e0b,#ef4444);
  --grad4:linear-gradient(135deg,#ec4899,#8b5cf6);
  --accent:#8b5cf6;--cyan:#06b6d4;--green:#10b981;--amber:#f59e0b;--red:#ef4444;--pink:#ec4899;
  --radius:16px;--radius-sm:10px;
  --font-h:'Sora',sans-serif;--font-b:'DM Sans',sans-serif;--font-m:'JetBrains Mono',monospace;
  --glass:rgba(255,255,255,.03);--glass-border:rgba(255,255,255,.08);
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--font-b);min-height:100vh;overflow-x:hidden}

/* NEURAL NETWORK CANVAS */
#neuralCanvas{position:fixed;inset:0;z-index:0;opacity:.6}

/* FLOATING PARTICLES (CSS-only) */
.particle{position:fixed;border-radius:50%;pointer-events:none;z-index:1;opacity:0;
  animation:floatUp linear infinite}
@keyframes floatUp{
  0%{opacity:0;transform:translateY(100vh) scale(0)}
  10%{opacity:1}
  90%{opacity:1}
  100%{opacity:0;transform:translateY(-10vh) scale(1)}
}

/* GRADIENT MESH OVERLAY */
.mesh-overlay{position:fixed;inset:0;z-index:1;pointer-events:none;
  background:radial-gradient(ellipse at 20% 0%,rgba(139,92,246,.08),transparent 50%),
             radial-gradient(ellipse at 80% 100%,rgba(6,182,212,.06),transparent 50%),
             radial-gradient(ellipse at 50% 50%,rgba(236,72,153,.03),transparent 40%)}

.container{max-width:1300px;margin:0 auto;padding:28px 24px;position:relative;z-index:10}

/* HEADER */
.header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap}
.logo{display:flex;align-items:center;gap:14px}
.logo-icon{width:44px;height:44px;border-radius:12px;background:var(--grad1);display:flex;align-items:center;
  justify-content:center;font-size:1.4rem;box-shadow:0 0 30px rgba(139,92,246,.3);animation:logoPulse 3s ease infinite}
@keyframes logoPulse{0%,100%{box-shadow:0 0 20px rgba(139,92,246,.3)}50%{box-shadow:0 0 40px rgba(139,92,246,.5)}}
.header h1{font-size:1.6rem;font-weight:800;font-family:var(--font-h);
  background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.header-sub{font-size:.8rem;color:var(--text3);letter-spacing:.02em}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.pill{display:inline-flex;align-items:center;gap:5px;padding:4px 14px;border-radius:20px;
  font-size:.72rem;font-weight:600;border:1px solid var(--glass-border);
  background:var(--glass);color:var(--text2);backdrop-filter:blur(8px)}

/* STEPPER */
.stepper{display:flex;align-items:center;justify-content:center;gap:0;margin:32px auto 24px;max-width:700px}
.step{display:flex;align-items:center;gap:0}
.step-circle{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:.8rem;font-weight:700;border:2px solid rgba(255,255,255,.08);color:var(--text3);
  background:rgba(255,255,255,.03);backdrop-filter:blur(10px);
  transition:all .5s cubic-bezier(.4,0,.2,1);position:relative;cursor:default;font-family:var(--font-h)}
.step-circle.active{border-color:var(--accent);color:#fff;
  background:linear-gradient(135deg,rgba(139,92,246,.8),rgba(59,130,246,.8));
  box-shadow:0 0 25px rgba(139,92,246,.5),0 0 50px rgba(139,92,246,.2);
  animation:stepGlow 2s ease infinite}
.step-circle.done{border-color:var(--green);color:#fff;
  background:linear-gradient(135deg,rgba(16,185,129,.8),rgba(6,182,212,.8));
  box-shadow:0 0 15px rgba(16,185,129,.3)}
.step-line{width:60px;height:2px;background:rgba(255,255,255,.06);transition:all .6s;position:relative;overflow:hidden}
.step-line.done{background:linear-gradient(90deg,var(--green),var(--cyan))}
.step-line.active::after{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);animation:lineSweep 1.5s ease infinite}
@keyframes lineSweep{to{left:100%}}
@keyframes stepGlow{0%,100%{box-shadow:0 0 25px rgba(139,92,246,.4)}50%{box-shadow:0 0 40px rgba(139,92,246,.7)}}
.step-label{position:absolute;top:48px;left:50%;transform:translateX(-50%);font-size:.62rem;
  white-space:nowrap;color:var(--text3);font-family:var(--font-h);font-weight:600;text-transform:uppercase;letter-spacing:.06em;
  transition:color .4s}
.step-circle.active .step-label{color:var(--accent)}.step-circle.done .step-label{color:var(--green)}

/* GRID */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:24px}
@media(max-width:960px){.grid{grid-template-columns:1fr}}

/* GLASS CARD */
.card{background:var(--bg2);border:1px solid var(--glass-border);border-radius:var(--radius);
  padding:26px;backdrop-filter:blur(16px);
  box-shadow:0 8px 32px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.04);
  transition:all .4s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden}
.card::before{content:'';position:absolute;inset:0;border-radius:var(--radius);padding:1px;
  background:linear-gradient(135deg,rgba(139,92,246,.15),transparent 40%,transparent 60%,rgba(6,182,212,.1));
  -webkit-mask:linear-gradient(#fff 0 0) content-box,linear-gradient(#fff 0 0);
  -webkit-mask-composite:xor;mask-composite:exclude;pointer-events:none;transition:opacity .4s;opacity:0}
.card:hover::before{opacity:1}
.card:hover{border-color:var(--border-h);transform:translateY(-2px);
  box-shadow:0 12px 40px rgba(0,0,0,.5),0 0 30px rgba(139,92,246,.05)}
.card-title{font-family:var(--font-h);font-size:1.05rem;font-weight:700;
  background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.card-sub{font-size:.82rem;color:var(--text2);margin-top:5px}
.label{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text3);font-weight:700;margin-bottom:6px}

/* INPUTS */
textarea{width:100%;border-radius:var(--radius-sm);border:1px solid var(--border);
  background:rgba(6,8,24,.7);backdrop-filter:blur(4px);
  color:var(--text);font-family:var(--font-m);font-size:.83rem;padding:14px;outline:none;resize:vertical;
  transition:all .3s cubic-bezier(.4,0,.2,1);line-height:1.65}
textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(139,92,246,.12),0 0 20px rgba(139,92,246,.08)}
textarea[readonly]{opacity:.8;cursor:default}
textarea::placeholder{color:var(--text3)}

/* BUTTONS */
.btn{display:inline-flex;align-items:center;gap:8px;padding:11px 24px;border-radius:var(--radius-sm);
  font-family:var(--font-h);font-size:.84rem;font-weight:700;border:none;cursor:pointer;
  transition:all .3s cubic-bezier(.4,0,.2,1);color:#fff;position:relative;overflow:hidden;
  letter-spacing:.02em}
.btn::after{content:'';position:absolute;inset:0;background:linear-gradient(rgba(255,255,255,.1),transparent);
  opacity:0;transition:opacity .3s}
.btn:hover::after{opacity:1}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(0,0,0,.3)}
.btn:active{transform:translateY(0);transition-duration:.1s}
.btn:disabled{opacity:.35;cursor:not-allowed;transform:none!important;box-shadow:none!important}
.btn-primary{background:var(--grad1);box-shadow:0 4px 15px rgba(139,92,246,.25)}
.btn-primary:hover{box-shadow:0 8px 30px rgba(139,92,246,.4)}
.btn-green{background:var(--grad2);box-shadow:0 4px 15px rgba(16,185,129,.2)}
.btn-amber{background:var(--grad3);color:#000;box-shadow:0 4px 15px rgba(245,158,11,.2)}
.btn-outline{background:var(--glass);border:1px solid var(--glass-border);color:var(--text2);backdrop-filter:blur(8px)}
.btn-outline:hover{border-color:var(--accent);color:var(--text);background:rgba(139,92,246,.08)}
.btn-sm{padding:8px 18px;font-size:.78rem}
.btn-copy{padding:5px 14px;font-size:.7rem;border-radius:8px;background:var(--glass);
  border:1px solid var(--glass-border);color:var(--text3);cursor:pointer;transition:all .25s;
  font-family:var(--font-h);font-weight:600;backdrop-filter:blur(8px)}
.btn-copy:hover{border-color:var(--cyan);color:var(--cyan);box-shadow:0 0 12px rgba(6,182,212,.15)}
.btn-copy.copied{border-color:var(--green);color:var(--green);box-shadow:0 0 12px rgba(16,185,129,.2)}

/* ALERTS */
.alert{border-radius:var(--radius-sm);padding:16px 20px;margin:14px 0;font-size:.84rem;
  border:1px solid;backdrop-filter:blur(10px);animation:slideDown .4s ease}
.alert-error{background:rgba(239,68,68,.06);border-color:rgba(239,68,68,.2);color:#fca5a5}
.alert-warn{background:rgba(245,158,11,.06);border-color:rgba(245,158,11,.2);color:#fcd34d}
.alert b{display:block;margin-bottom:4px;font-family:var(--font-h)}
@keyframes slideDown{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}

/* DETAILS */
details{border:1px solid var(--glass-border);border-radius:var(--radius-sm);background:var(--glass);
  backdrop-filter:blur(8px);overflow:hidden;transition:all .3s}
details summary{padding:12px 16px;cursor:pointer;font-size:.82rem;color:var(--text2);font-weight:600;
  list-style:none;display:flex;align-items:center;gap:8px;transition:all .25s;font-family:var(--font-h)}
details summary:hover{color:var(--text);background:rgba(255,255,255,.02)}
details summary::before{content:'▸';transition:transform .3s;font-size:.7rem}
details[open] summary::before{transform:rotate(90deg)}
details .inner{padding:4px 16px 16px;animation:fadeSlide .35s ease}
@keyframes fadeSlide{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}

/* TOAST */
.toast-container{position:fixed;top:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:10px}
.toast{padding:14px 22px;border-radius:var(--radius-sm);font-size:.84rem;color:#fff;
  backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.1);
  box-shadow:0 8px 32px rgba(0,0,0,.4);animation:toastIn .4s cubic-bezier(.4,0,.2,1);
  font-weight:600;font-family:var(--font-h)}
.toast-success{background:linear-gradient(135deg,rgba(5,150,105,.9),rgba(16,185,129,.9))}
.toast-error{background:linear-gradient(135deg,rgba(220,38,38,.9),rgba(239,68,68,.9))}
.toast-info{background:linear-gradient(135deg,rgba(2,132,199,.9),rgba(6,182,212,.9))}
@keyframes toastIn{from{transform:translateX(120%);opacity:0}to{transform:translateX(0);opacity:1}}

/* LOADING OVERLAY */
.loading-overlay{position:fixed;inset:0;background:rgba(6,8,22,.92);display:flex;flex-direction:column;
  align-items:center;justify-content:center;z-index:8888;backdrop-filter:blur(8px)}
.brain-loader{position:relative;width:80px;height:80px}
.brain-loader .ring{position:absolute;inset:0;border-radius:50%;border:3px solid transparent;
  animation:brainSpin 1.2s ease-in-out infinite}
.brain-loader .ring:nth-child(1){border-top-color:var(--accent);animation-delay:0s}
.brain-loader .ring:nth-child(2){inset:8px;border-right-color:var(--cyan);animation-delay:.15s;animation-direction:reverse}
.brain-loader .ring:nth-child(3){inset:16px;border-bottom-color:var(--pink);animation-delay:.3s}
.brain-core{position:absolute;inset:24px;border-radius:50%;background:var(--grad4);
  animation:corePulse 1.5s ease infinite;box-shadow:0 0 30px rgba(139,92,246,.4)}
@keyframes brainSpin{to{transform:rotate(360deg)}}
@keyframes corePulse{0%,100%{transform:scale(.8);opacity:.7}50%{transform:scale(1);opacity:1}}
.loading-msg{margin-top:24px;font-size:.95rem;color:var(--text2);font-family:var(--font-h);font-weight:600;
  animation:fadeSlide .5s ease;letter-spacing:.02em}
.loading-sub{margin-top:8px;font-size:.75rem;color:var(--text3);animation:fadeSlide .5s ease .2s both}

/* MODAL */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);display:flex;align-items:center;
  justify-content:center;z-index:9998;backdrop-filter:blur(6px);animation:fadeIn .25s}
.modal{background:var(--bg2);border:1px solid var(--glass-border);border-radius:var(--radius);padding:32px;
  max-width:420px;width:90%;backdrop-filter:blur(20px);
  box-shadow:0 24px 48px rgba(0,0,0,.5);animation:modalIn .35s cubic-bezier(.4,0,.2,1)}
.modal h3{font-size:1.1rem;font-family:var(--font-h);font-weight:700;
  background:var(--grad3);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.modal p{color:var(--text2);font-size:.86rem;margin:10px 0 22px;line-height:1.6}
.modal-btns{display:flex;gap:12px;justify-content:flex-end}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes modalIn{from{opacity:0;transform:scale(.92) translateY(10px)}to{opacity:1;transform:scale(1) translateY(0)}}

/* OPT LOG */
.opt-log{font-family:var(--font-m);font-size:.76rem;line-height:2;color:var(--text2);padding:8px 0}
.opt-log .log-ok{color:var(--green)}.opt-log .log-retry{color:var(--amber)}.opt-log .log-fail{color:var(--red)}

/* BLOCK CARD */
.block-card{background:var(--bg3);border:1px solid var(--glass-border);border-radius:var(--radius);padding:22px;
  margin-bottom:20px;transition:all .35s;position:relative;overflow:hidden}
.block-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--grad1);opacity:.5;transition:opacity .3s}
.block-card:hover{border-color:var(--border-h);transform:translateY(-1px)}
.block-card:hover::before{opacity:1}
.block-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.block-num{font-family:var(--font-h);font-weight:800;font-size:.9rem;
  background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}

/* FOOTER */
.footer{margin-top:32px;padding:16px 24px;border:1px solid var(--glass-border);border-radius:var(--radius-sm);
  background:var(--glass);backdrop-filter:blur(10px);font-size:.8rem;color:var(--text3);text-align:center;
  font-family:var(--font-h);letter-spacing:.02em}
.hidden{display:none!important}
.mt{margin-top:16px}.mt2{margin-top:12px}.mt3{margin-top:8px}
.gap{display:flex;flex-wrap:wrap;gap:10px}

/* STAGE TRANSITIONS */
.stage-enter{animation:stageIn .5s cubic-bezier(.4,0,.2,1) both}
@keyframes stageIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}

/* REQUIRED FIELD INDICATOR */
.req{color:var(--pink);font-weight:700;margin-left:4px}

/* SECTION DIVIDER */
.divider{height:1px;background:linear-gradient(90deg,transparent,var(--border-h),transparent);margin:16px 0}

/* API KEY SCREEN */
.key-screen{position:fixed;inset:0;z-index:9997;display:flex;align-items:center;justify-content:center;
  background:var(--bg)}
.key-card{background:var(--bg2);border:1px solid var(--glass-border);border-radius:20px;padding:40px;
  max-width:480px;width:90%;backdrop-filter:blur(20px);box-shadow:0 24px 48px rgba(0,0,0,.5);
  text-align:center;animation:modalIn .5s cubic-bezier(.4,0,.2,1)}
.key-icon{font-size:3rem;margin-bottom:16px;animation:logoPulse 3s ease infinite}
</style>
</head>
<body>

<canvas id="neuralCanvas"></canvas>
<div class="mesh-overlay"></div>

<div class="toast-container" id="toasts"></div>

<div id="loadingOverlay" class="loading-overlay hidden">
  <div class="brain-loader">
    <div class="ring"></div><div class="ring"></div><div class="ring"></div>
    <div class="brain-core"></div>
  </div>
  <div class="loading-msg" id="loadingMsg">Initializing AI Pipeline...</div>
  <div class="loading-sub" id="loadingSub">This may take a moment</div>
</div>

<div id="modalOverlay" class="modal-overlay hidden">
  <div class="modal">
    <h3>⚠ Reset Session?</h3>
    <p>All progress including optimized outputs will be lost. This action cannot be undone.</p>
    <div class="modal-btns">
      <button class="btn btn-outline btn-sm" onclick="closeModal()">Cancel</button>
      <button class="btn btn-amber btn-sm" onclick="doReset()">Reset Everything</button>
    </div>
  </div>
</div>

<!-- AUTH PASSWORD SCREEN -->
<div id="authScreen" class="key-screen hidden">
  <div class="key-card">
    <div class="key-icon">🔐</div>
    <h2 style="font-family:var(--font-h);font-size:1.4rem;font-weight:800;background:var(--grad4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
      Access Required</h2>
    <p style="color:var(--text2);font-size:.88rem;margin:12px 0 20px;line-height:1.7">
      This application is password-protected.<br>
      <span style="color:var(--text3);font-size:.78rem">Enter the access password to continue.</span></p>
    <input type="password" id="authInput" placeholder="Enter password..."
      style="width:100%;padding:14px;border-radius:var(--radius-sm);border:1px solid var(--border);
        background:rgba(6,8,24,.8);color:var(--text);font-family:var(--font-m);font-size:.85rem;outline:none;
        transition:all .3s" onfocus="this.style.borderColor='var(--accent)';this.style.boxShadow='0 0 0 3px rgba(139,92,246,.12)'"
        onblur="this.style.borderColor='var(--border)';this.style.boxShadow='none'">
    <div id="authError" style="color:var(--red);font-size:.8rem;margin-top:8px;display:none"></div>
    <button class="btn btn-primary" style="width:100%;margin-top:16px;justify-content:center" onclick="submitAuth()">
      🔓 Unlock</button>
  </div>
</div>

<!-- API KEY SCREEN -->
<div id="keyScreen" class="key-screen hidden">
  <div class="key-card">
    <div class="key-icon">🔑</div>
    <h2 style="font-family:var(--font-h);font-size:1.4rem;font-weight:800;background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
      Connect Your AI Engine</h2>
    <p style="color:var(--text2);font-size:.88rem;margin:12px 0 20px;line-height:1.7">
      Enter your Google Gemini API key to power the pipeline.<br>
      <span style="color:var(--text3);font-size:.78rem">Your key stays in memory only — never saved to disk or committed to code.<br>
      Get a free key at <a href="https://aistudio.google.com" target="_blank" style="color:var(--cyan);text-decoration:none">aistudio.google.com</a></span></p>
    <input type="password" id="keyInput" placeholder="Paste your Gemini API key here..."
      style="width:100%;padding:14px;border-radius:var(--radius-sm);border:1px solid var(--border);
        background:rgba(6,8,24,.8);color:var(--text);font-family:var(--font-m);font-size:.85rem;outline:none;
        transition:all .3s" onfocus="this.style.borderColor='var(--accent)';this.style.boxShadow='0 0 0 3px rgba(139,92,246,.12)'"
        onblur="this.style.borderColor='var(--border)';this.style.boxShadow='none'">
    <div id="keyError" style="color:var(--red);font-size:.8rem;margin-top:8px;display:none"></div>
    <button class="btn btn-primary" style="width:100%;margin-top:16px;justify-content:center" onclick="submitKey()">
      Connect & Launch →</button>
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid var(--glass-border);font-size:.75rem;color:var(--text3);text-align:center">
      Or set <code style="color:var(--cyan);background:var(--glass);padding:2px 6px;border-radius:4px">GEMINI_API_KEY</code> env variable or create a <code style="color:var(--cyan);background:var(--glass);padding:2px 6px;border-radius:4px">.env</code> file</div>
  </div>
</div>

<div class="container">
  <div class="header">
    <div>
      <div class="logo">
        <div class="logo-icon">🧠</div>
        <div>
          <h1>Prompt Optimization System</h1>
          <div class="header-sub">AI-Powered Critique → Clarify → Optimize → Execute Pipeline</div>
        </div>
      </div>
      <div class="pills" id="headerPills"></div>
    </div>
    <button class="btn btn-outline btn-sm" onclick="confirmReset()">↻ Reset</button>
  </div>

  <div class="stepper" id="stepper"></div>
  <div id="alerts"></div>

  <div class="grid">
    <div class="card" id="leftPanel"></div>
    <div class="card" id="rightPanel"></div>
  </div>

  <div class="footer" id="footer">Enter a prompt and click Start Session to begin the AI optimization pipeline.</div>
</div>

<!-- NEURAL NETWORK BACKGROUND -->
<script>
(function(){
  const canvas=document.getElementById('neuralCanvas');
  const ctx=canvas.getContext('2d');
  let W,H,nodes=[],mouse={x:-1000,y:-1000};
  const NODE_COUNT=65,CONNECT_DIST=160,SPEED=.3;

  function resize(){W=canvas.width=window.innerWidth;H=canvas.height=window.innerHeight}
  window.addEventListener('resize',resize);resize();

  for(let i=0;i<NODE_COUNT;i++){
    nodes.push({x:Math.random()*W,y:Math.random()*H,
      vx:(Math.random()-.5)*SPEED,vy:(Math.random()-.5)*SPEED,
      r:Math.random()*2+1,pulse:Math.random()*Math.PI*2});
  }

  document.addEventListener('mousemove',e=>{mouse.x=e.clientX;mouse.y=e.clientY});

  function draw(){
    ctx.clearRect(0,0,W,H);
    const t=Date.now()*.001;

    // Update positions
    nodes.forEach(n=>{
      n.x+=n.vx;n.y+=n.vy;n.pulse+=.02;
      if(n.x<0||n.x>W)n.vx*=-1;
      if(n.y<0||n.y>H)n.vy*=-1;
      // Mouse repulsion
      const dx=n.x-mouse.x,dy=n.y-mouse.y,dist=Math.sqrt(dx*dx+dy*dy);
      if(dist<150){const f=.5/dist;n.vx+=dx*f*.1;n.vy+=dy*f*.1;}
    });

    // Draw connections
    for(let i=0;i<nodes.length;i++){
      for(let j=i+1;j<nodes.length;j++){
        const dx=nodes[i].x-nodes[j].x,dy=nodes[i].y-nodes[j].y;
        const dist=Math.sqrt(dx*dx+dy*dy);
        if(dist<CONNECT_DIST){
          const alpha=(1-dist/CONNECT_DIST)*.15;
          ctx.beginPath();ctx.moveTo(nodes[i].x,nodes[i].y);ctx.lineTo(nodes[j].x,nodes[j].y);
          ctx.strokeStyle=`rgba(139,92,246,${alpha})`;ctx.lineWidth=.8;ctx.stroke();
        }
      }
    }

    // Draw nodes
    nodes.forEach(n=>{
      const glow=Math.sin(n.pulse)*.5+.5;
      const r=n.r+glow;
      ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2);
      ctx.fillStyle=`rgba(139,92,246,${.2+glow*.3})`;ctx.fill();
      ctx.beginPath();ctx.arc(n.x,n.y,r*.5,0,Math.PI*2);
      ctx.fillStyle=`rgba(200,180,255,${.3+glow*.4})`;ctx.fill();
    });

    requestAnimationFrame(draw);
  }
  draw();

  // Floating particles
  function spawnParticle(){
    const p=document.createElement('div');p.className='particle';
    const size=Math.random()*3+1;
    const colors=['rgba(139,92,246,.4)','rgba(6,182,212,.3)','rgba(236,72,153,.3)','rgba(59,130,246,.3)'];
    Object.assign(p.style,{
      width:size+'px',height:size+'px',left:Math.random()*100+'%',
      background:colors[Math.floor(Math.random()*colors.length)],
      animationDuration:(Math.random()*10+8)+'s',animationDelay:Math.random()*5+'s',
      boxShadow:`0 0 ${size*3}px ${colors[Math.floor(Math.random()*colors.length)]}`
    });
    document.body.appendChild(p);
    setTimeout(()=>p.remove(),(Math.random()*10+8)*1000+5000);
  }
  setInterval(spawnParticle,800);
  for(let i=0;i<8;i++)setTimeout(spawnParticle,i*200);
})();
</script>

<!-- APP LOGIC -->
<script>
const STAGES=['INPUT','STRUCTURE','CONTEXT','CLARIFIED','OUTPUT'];
const STAGE_LABELS=['Input','Clarify','Context','Optimize','Output'];
const STAGE_ICONS=['✏️','🔍','💬','⚡','✨'];
const LOAD_MESSAGES={
  start:['Analyzing prompt feasibility...','Running safety checks...'],
  structure:['Validating parameters...','Checking for collisions...'],
  confirm:['Confirming structure...','Generating context questions...'],
  context:['Building clarified prompt...','Merging with original intent...'],
  optimize:['Diagnosing prompt defects...','Applying optimization techniques...','Segmenting & executing blocks...','Filtering outputs for quality...']
};

let S={stage:'INPUT',sessionId:null,rawPrompt:'',form:{role:'',task:'',scope:'',output_format:''},
  serverMissing:[],collision:null,contextQuestions:[],contextInput:'',clarifiedPrompt:'',
  semiFinal:'',optLog:[],blocks:[],rawOutputs:[],filteredOutputs:[],finalOutput:'',
  editInstr:{},revisions:0,loading:false,error:null};

// HELPERS
function toast(msg,type='info'){
  const d=document.createElement('div');d.className='toast toast-'+type;d.textContent=msg;
  document.getElementById('toasts').appendChild(d);setTimeout(()=>{d.style.opacity='0';d.style.transform='translateX(50px)';d.style.transition='all .3s';setTimeout(()=>d.remove(),300)},3200);
}
let loadMsgIdx=0,loadInterval=null;
function showLoad(msgs){
  S.loading=true;loadMsgIdx=0;
  const arr=Array.isArray(msgs)?msgs:[msgs];
  const o=document.getElementById('loadingOverlay');o.classList.remove('hidden');
  document.getElementById('loadingMsg').textContent=arr[0];
  document.getElementById('loadingSub').textContent='AI pipeline processing...';
  if(loadInterval)clearInterval(loadInterval);
  if(arr.length>1){loadInterval=setInterval(()=>{loadMsgIdx=(loadMsgIdx+1)%arr.length;
    document.getElementById('loadingMsg').textContent=arr[loadMsgIdx];},2500);}
}
function hideLoad(){S.loading=false;if(loadInterval){clearInterval(loadInterval);loadInterval=null;}
  document.getElementById('loadingOverlay').classList.add('hidden');}
function confirmReset(){document.getElementById('modalOverlay').classList.remove('hidden')}
function closeModal(){document.getElementById('modalOverlay').classList.add('hidden')}
function doReset(){closeModal();
  S={stage:'INPUT',sessionId:null,rawPrompt:'',form:{role:'',task:'',scope:'',output_format:''},
    serverMissing:[],collision:null,contextQuestions:[],contextInput:'',clarifiedPrompt:'',
    semiFinal:'',optLog:[],blocks:[],rawOutputs:[],filteredOutputs:[],finalOutput:'',
    editInstr:{},revisions:0,loading:false,error:null};
  render();toast('Session reset','info');}
let authToken='';
async function api(path,body){
  const headers={'Content-Type':'application/json'};
  if(authToken)headers['Authorization']='Bearer '+authToken;
  const r=await fetch(path,{method:'POST',headers,body:JSON.stringify(body)});
  const t=await r.text();let d;try{d=JSON.parse(t)}catch{throw new Error(t||'HTTP '+r.status)}
  if(r.status===401){document.getElementById('authScreen').classList.remove('hidden');throw new Error('Session expired — please log in again.');}
  if(!r.ok)throw new Error(d?.detail||JSON.stringify(d));return d;}
function copyText(txt,btn){navigator.clipboard.writeText(txt).then(()=>{
  btn.textContent='✓ Copied!';btn.classList.add('copied');
  setTimeout(()=>{btn.textContent='⎘ Copy';btn.classList.remove('copied')},1500);});}
function esc(s){const d=document.createElement('div');d.textContent=s||'';return d.innerHTML;}

// ACTIONS
async function startSession(){
  if(!S.rawPrompt.trim())return;S.error=null;showLoad(LOAD_MESSAGES.start);
  try{const d=await api('/session/start',{user_prompt:S.rawPrompt});
    S.sessionId=d.session_id;S.serverMissing=d.missing||[];
    S.collision=d.collision?{on:true,reason:d.collision_reason}:null;
    S.form={role:d.state?.role||'',task:d.state?.task||'',scope:d.state?.scope||'',output_format:d.state?.output_format||''};
    if(d.blocked){S.error='Blocked: '+(d.blocked_reason||'Safety filter.');S.stage='INPUT';toast(S.error,'error');}
    else{S.stage='STRUCTURE';toast('Parameters extracted successfully','success');}
  }catch(e){S.error=e.message;toast(e.message,'error');}hideLoad();render();}

async function saveStructure(){
  if(!S.sessionId)return;S.error=null;showLoad(LOAD_MESSAGES.structure);
  try{const d=await api('/phase-a/update-fields',{session_id:S.sessionId,...S.form});
    S.serverMissing=d.missing||[];S.collision=d.collision?{on:true,reason:d.collision_reason}:null;
    toast('Structure saved','success');
  }catch(e){S.error=e.message;toast(e.message,'error');}hideLoad();render();}

async function confirmStructure(){
  if(!S.sessionId)return;S.error=null;showLoad(LOAD_MESSAGES.confirm);
  try{await api('/phase-a/update-fields',{session_id:S.sessionId,...S.form});
    await api('/phase-a/confirm-structure',{session_id:S.sessionId});
    const q=await api('/phase-a/context-questions',{session_id:S.sessionId});
    S.contextQuestions=q.questions||[];S.stage='CONTEXT';toast('Structure confirmed','success');
  }catch(e){S.error=e.message;toast(e.message,'error');}hideLoad();render();}

async function finalizeA(){
  if(!S.sessionId)return;S.error=null;showLoad(LOAD_MESSAGES.context);
  try{await api('/phase-a/submit-context',{session_id:S.sessionId,additional_context:S.contextInput});
    const d=await api('/phase-a/finalize',{session_id:S.sessionId});
    S.clarifiedPrompt=d.clarified||'';S.stage='CLARIFIED';toast('Clarification complete','success');
  }catch(e){S.error=e.message;toast(e.message,'error');}hideLoad();render();}

async function runOptimize(){
  if(!S.sessionId)return;S.error=null;showLoad(LOAD_MESSAGES.optimize);
  try{const d=await api('/optimize',{session_id:S.sessionId});
    S.semiFinal=d.semi_final||'';S.blocks=d.blocks||[];S.rawOutputs=d.raw_outputs||[];
    S.filteredOutputs=d.filtered_outputs||[];S.finalOutput=d.final||'';S.optLog=d.optimization_log||[];
    S.editInstr={};S.blocks.forEach((_,i)=>S.editInstr[i]='');
    S.stage='OUTPUT';toast('Optimization complete!','success');
  }catch(e){S.error=e.message;toast(e.message,'error');}hideLoad();render();}

async function reviseBlock(i){
  const instr=(S.editInstr[i]||'').trim();if(!instr||!S.sessionId)return;
  S.error=null;showLoad(['Revising output with AI...']);
  try{const d=await api('/blocks/revise-output',{session_id:S.sessionId,block_index:i,instruction:instr});
    S.filteredOutputs=d.filtered_outputs||[];S.finalOutput=d.final||'';
    S.revisions=d.revision_count||0;S.editInstr[i]='';toast('Output revised','success');
  }catch(e){S.error=e.message;toast(e.message,'error');}hideLoad();render();}

// RENDER
function render(){renderStepper();renderPills();renderAlerts();renderLeft();renderRight();renderFooter();}

function renderStepper(){
  const el=document.getElementById('stepper');const idx=STAGES.indexOf(S.stage);let h='';
  STAGES.forEach((s,i)=>{
    const cls=i<idx?'done':i===idx?'active':'';
    const lineActive=i===idx?'active':'';
    h+=`<div class="step"><div class="step-circle ${cls}">${i<idx?'✓':STAGE_ICONS[i]}<span class="step-label">${STAGE_LABELS[i]}</span></div></div>`;
    if(i<STAGES.length-1)h+=`<div class="step-line ${i<idx?'done':lineActive}"></div>`;
  });el.innerHTML=h;
}

function renderPills(){
  const el=document.getElementById('headerPills');const lbl=STAGE_LABELS[STAGES.indexOf(S.stage)]||'Ready';
  let h=`<span class="pill">${STAGE_ICONS[STAGES.indexOf(S.stage)]||''} ${lbl}</span>`;
  if(S.sessionId)h+=`<span class="pill">🔗 ${S.sessionId.slice(0,8)}…</span>`;
  if(S.revisions>0)h+=`<span class="pill">✍️ ${S.revisions} revision${S.revisions>1?'s':''}</span>`;
  el.innerHTML=h;
}

function renderAlerts(){
  const el=document.getElementById('alerts');let h='';
  if(S.error)h+=`<div class="alert alert-error"><b>⚠ Error</b>${esc(S.error)}</div>`;
  if(S.collision?.on)h+=`<div class="alert alert-warn"><b>🔀 Task/Scope Collision</b>${esc(S.collision.reason||'Scope was reset — please re-enter.')}</div>`;
  el.innerHTML=h;
}

function renderLeft(){
  const el=document.getElementById('leftPanel');
  if(S.stage==='INPUT'){
    el.innerHTML=`<div class="stage-enter">
      <div class="card-title">✏️ Workspace</div>
      <div class="card-sub">Enter your raw prompt to begin the AI-powered optimization pipeline.</div>
      <div class="mt"><div class="label">Raw Prompt</div>
        <textarea id="rawInput" rows="10" placeholder="Type or paste your prompt here...\n\nExample: Develop a digital transformation strategy for a retail company in Canada...">${esc(S.rawPrompt)}</textarea>
      </div>
      <div class="mt"><button class="btn btn-primary" onclick="S.rawPrompt=document.getElementById('rawInput').value;startSession()">🚀 Start Session</button></div>
    </div>`;
  } else if(S.stage==='STRUCTURE'){
    const fields=['role','task','scope','output_format'];
    const labels={role:'Role',task:'Task',scope:'Scope',output_format:'Output Format'};
    const icons={role:'👤',task:'🎯',scope:'📐',output_format:'📋'};
    const ph={role:'e.g., Strategy Consultant, Data Analyst, Marketing Expert',task:'What should the AI do? (use action verbs)',
      scope:'Domain, boundaries, constraints, context',output_format:'Report, table, bullet points, executive summary, etc.'};
    const localMiss=fields.filter(k=>!S.form[k]?.trim());
    let fh=fields.map(k=>`<div><div class="label">${icons[k]} ${labels[k]}${!S.form[k]?.trim()?'<span class="req">*</span>':''}</div>
      <textarea id="f_${k}" rows="${k==='task'||k==='scope'?3:2}" placeholder="${ph[k]}"
        oninput="S.form.${k}=this.value">${esc(S.form[k])}</textarea></div>`).join('');
    el.innerHTML=`<div class="stage-enter">
      <div class="card-title">🔍 Clarification Parameters</div>
      <div class="card-sub">Review and refine the AI-extracted parameters. All fields required.</div>
      <div class="mt" style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:.78rem;color:${localMiss.length?'var(--pink)':'var(--green)'};font-weight:600;font-family:var(--font-h)">
          ${localMiss.length?'Missing: '+localMiss.join(', '):'✓ All fields complete'}</span>
      </div>
      <div class="divider"></div>
      <div style="display:grid;gap:14px">${fh}</div>
      <div class="mt gap">
        <button class="btn btn-outline btn-sm" onclick="syncForm();saveStructure()">💾 Save</button>
        <button class="btn btn-primary btn-sm" ${localMiss.length?'disabled':''} onclick="syncForm();confirmStructure()">Confirm & Continue →</button>
      </div>
    </div>`;
  } else if(S.stage==='CONTEXT'){
    let qh=S.contextQuestions.length?S.contextQuestions.map(q=>`<div class="mt3" style="color:var(--text2);font-size:.84rem;padding:8px 12px;background:var(--glass);border-radius:8px;border:1px solid var(--glass-border)">💡 ${esc(q)}</div>`).join('')
      :'<div style="color:var(--text3);font-size:.84rem;padding:12px;background:var(--glass);border-radius:8px;text-align:center">✓ No additional context required</div>';
    el.innerHTML=`<div class="stage-enter">
      <div class="card-title">💬 Context Enrichment</div>
      <div class="card-sub">Optional : provide additional context to improve results, or skip ahead.</div>
      <div class="mt" style="display:grid;gap:8px">${qh}</div>
      <div class="divider"></div>
      <div><div class="label">📝 Additional Context</div>
        <textarea id="ctxInput" rows="6" placeholder="Add relevant context, data, or constraints here...">${esc(S.contextInput)}</textarea></div>
      <div class="mt"><button class="btn btn-green" onclick="S.contextInput=document.getElementById('ctxInput').value;finalizeA()">Continue →</button></div>
    </div>`;
  } else if(S.stage==='CLARIFIED'){
    el.innerHTML=`<div class="stage-enter">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div class="card-title">📋 Clarified Prompt</div>
        <button class="btn-copy" onclick="copyText(S.clarifiedPrompt,this)">⎘ Copy</button>
      </div>
      <div class="card-sub">Your prompt has been structured and enriched. Ready for optimization.</div>
      <div class="mt"><textarea readonly rows="12">${esc(S.clarifiedPrompt)}</textarea></div>
      <div class="mt"><button class="btn btn-primary" onclick="runOptimize()">⚡ Optimize & Execute</button></div>
    </div>`;
  } else if(S.stage==='OUTPUT'){
    let logHtml=S.optLog.map(l=>{
      let cls=l.startsWith('✓')?'log-ok':l.startsWith('⟳')?'log-retry':'log-fail';
      return `<div class="${cls}">${esc(l)}</div>`;}).join('');
    el.innerHTML=`<div class="stage-enter">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div class="card-title">✨ Final Integrated Output</div>
        <button class="btn-copy" onclick="copyText(S.finalOutput,this)">⎘ Copy</button>
      </div>
      <div class="card-sub">Combined output from all pipeline blocks.</div>
      <div class="mt"><textarea readonly rows="14">${esc(S.finalOutput)}</textarea></div>
      <details class="mt"><summary>🔧 Optimization Log</summary>
        <div class="inner opt-log">${logHtml||'No log.'}</div></details>
      <details class="mt"><summary>📝 Semi-final Prompt</summary>
        <div class="inner"><textarea readonly rows="8">${esc(S.semiFinal)}</textarea></div></details>
    </div>`;
  }
}

function renderRight(){
  const el=document.getElementById('rightPanel');
  if(S.stage!=='OUTPUT'){
    const tips={INPUT:'<b>🚀 Getting Started</b><br>Enter your raw prompt — it can be as vague or detailed as you like. The AI pipeline will analyze its structure, identify missing elements, and guide you through refinement before generating optimized output.',
      STRUCTURE:'<b>🔍 Parameter Refinement</b><br>The AI has extracted four key parameters from your prompt. Edit any that need adjustment. All four fields must be filled before proceeding. Click <em>Save</em> to validate, then <em>Confirm</em> to proceed.',
      CONTEXT:'<b>💬 Context Enrichment</b><br>The AI has identified areas where additional context could improve output quality. Answer what you can, or skip to continue with what you have.',
      CLARIFIED:'<b>⚡ Ready to Optimize</b><br>Your prompt has been clarified and structured. The next step runs defect diagnosis, applies prompt engineering techniques, segments the prompt into blocks, and executes them sequentially with quality filtering.'};
    el.innerHTML=`<div class="stage-enter">
      <div class="card-title">📖 Guide</div>
      <div class="mt" style="color:var(--text2);font-size:.86rem;line-height:1.8">${tips[S.stage]||''}</div>
      <div class="divider"></div>
      <div style="font-size:.78rem;color:var(--text3);line-height:1.8">
        <div style="font-weight:700;color:var(--text2);font-family:var(--font-h);margin-bottom:6px">🧠 Pipeline Architecture</div>
        Safety Filter → Parameter Extraction → User Clarification → Context Enrichment → Defect Diagnosis → Technique Application → Block Segmentation → Sequential Execution → Quality Filtering → Output
      </div>
    </div>`;
  } else {
    let bh=S.blocks.map((bp,i)=>{
      const ctx=i===0?'No prior context':`Uses output${i>1?'s':''} 1${i>1?'–'+i:''}`;
      return `<div class="block-card">
        <div class="block-header">
          <span class="block-num">⬡ Block ${i+1} of ${S.blocks.length}</span>
          <span class="pill">${i===0?'🔵':'🔗'} ${ctx}</span>
        </div>
        <div class="label">Prompt Segment</div>
        <textarea readonly rows="4">${esc(bp)}</textarea>
        <div class="mt2" style="display:flex;justify-content:space-between;align-items:center">
          <span class="label" style="margin:0">Filtered Output</span>
          <button class="btn-copy" onclick="copyText(S.filteredOutputs[${i}],this)">⎘ Copy</button>
        </div>
        <textarea readonly rows="6">${esc(S.filteredOutputs[i]||'')}</textarea>
        <details class="mt2"><summary>🔬 Raw output (debug)</summary>
          <div class="inner"><textarea readonly rows="5">${esc(S.rawOutputs[i]||'')}</textarea></div></details>
        <div class="mt2" style="padding:16px;background:var(--bg2);border-radius:var(--radius-sm);border:1px solid var(--glass-border)">
          <div style="font-size:.85rem;font-weight:700;color:var(--text);font-family:var(--font-h)">✍️ Revise This Output</div>
          <div class="mt3"><textarea id="edit_${i}" rows="2" placeholder='e.g. "Convert to a table" or "Make it more concise"'>${esc(S.editInstr[i]||'')}</textarea></div>
          <div class="mt3"><button class="btn btn-amber btn-sm" onclick="S.editInstr[${i}]=document.getElementById('edit_${i}').value;reviseBlock(${i})">Apply Edit →</button></div>
        </div>
      </div>`;}).join('');
    el.innerHTML=`<div class="stage-enter">
      <div class="card-title">🧱 Output Blocks</div>
      <div class="card-sub">Sequential execution with context chaining between blocks.</div>
      <div class="mt">${bh}</div>
    </div>`;
  }
}

function renderFooter(){
  const msgs={INPUT:'Enter a prompt and click Start Session to begin.',
    STRUCTURE:'Fill all four fields → Save → Confirm to proceed.',
    CONTEXT:'Add optional context or click Continue to skip.',
    CLARIFIED:'Review the clarified prompt → Click Optimize & Execute.',
    OUTPUT:'Review outputs. Use the revision tools to refine individual blocks.'};
  document.getElementById('footer').innerHTML=(msgs[S.stage]||'')+
    '<span style="float:right;opacity:.45;font-size:.72rem">Questions or feedback? <a href="https://www.linkedin.com/in/gaurangmakwana" target="_blank" style="color:var(--cyan);text-decoration:none;border-bottom:1px dotted var(--cyan)">Connect with me</a></span>';
}

function syncForm(){['role','task','scope','output_format'].forEach(k=>{
  const el=document.getElementById('f_'+k);if(el)S.form[k]=el.value;});}

// AUTH HANDLING
async function submitAuth(){
  const pw=document.getElementById('authInput').value.trim();
  const errEl=document.getElementById('authError');
  if(!pw){errEl.textContent='Please enter the password.';errEl.style.display='block';return;}
  errEl.style.display='none';
  try{
    const d=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({password:pw})});
    const j=await d.json();
    if(!d.ok)throw new Error(j.detail||'Login failed');
    authToken=j.token||'';
    document.getElementById('authScreen').classList.add('hidden');
    toast('Access granted!','success');
    checkKey();
  }catch(e){errEl.textContent=e.message||'Incorrect password.';errEl.style.display='block';}
}
document.getElementById('authInput')?.addEventListener('keydown',e=>{if(e.key==='Enter')submitAuth()});

// API KEY HANDLING
async function submitKey(){
  const key=document.getElementById('keyInput').value.trim();
  const errEl=document.getElementById('keyError');
  if(!key){errEl.textContent='Please enter an API key.';errEl.style.display='block';return;}
  errEl.style.display='none';
  try{
    await api('/api/set-key',{api_key:key});
    document.getElementById('keyScreen').classList.add('hidden');
    toast('API key connected!','success');render();
  }catch(e){errEl.textContent=e.message||'Failed to set key.';errEl.style.display='block';}
}
document.getElementById('keyInput')?.addEventListener('keydown',e=>{if(e.key==='Enter')submitKey()});

function checkKey(){
  fetch('/api/status').then(r=>r.json()).then(d=>{
    if(d.has_key){render();}
    else{document.getElementById('keyScreen').classList.remove('hidden');}
  }).catch(()=>document.getElementById('keyScreen').classList.remove('hidden'));
}

// STARTUP — check auth then key
async function startup(){
  try{
    const r=await fetch('/api/status');const d=await r.json();
    if(d.needs_auth){
      document.getElementById('authScreen').classList.remove('hidden');
    } else if(!d.has_key){
      document.getElementById('keyScreen').classList.remove('hidden');
    } else {
      render();
    }
  }catch{document.getElementById('keyScreen').classList.remove('hidden');}
}
startup();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML_PAGE

# ============================================================
# 14) ENTRY POINT
# ============================================================
def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    import uvicorn
    init_gemini()
    threading.Thread(target=open_browser, daemon=True).start()
    log.info("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
