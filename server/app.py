"""
FastAPI application for the Judicial Reasoning RL Environment.
Team ALACRITY — OpenEnv Hackathon

This module creates an HTTP server that exposes the JudicialEnv
over HTTP endpoints, compatible with the OpenEnv spec.

The server:
- Exposes /reset, /step, /state endpoints for the RL API
- Hosts the inference runner as a background task on startup
- Stays alive persistently on port 7860 for HF Spaces
"""

import asyncio
import concurrent.futures
import json
import os
import re
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from dotenv import load_dotenv
from openai import OpenAI

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import JudicialEnv, JudicialAction
from server.models import (
    ResetRequest, StepRequest,
    ResetResponse, StepResponse, StateResponse, HealthResponse, AIJudgeResponse,
    EscalateRequest, ChatRequest, ChatResponse,
    SummonsRequest, CaseStatusRequest
)

load_dotenv()

# ─── Configuration ────────────────────────────────────────────

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_BASE_URL  = "https://api-inference.huggingface.co/v1/"

# Primary API config (used for /chat endpoint)
if HF_TOKEN:
    API_KEY      = HF_TOKEN
    API_BASE_URL = HF_BASE_URL
    print("OK: Hugging Face Multi-Agent Council active via HF_TOKEN")
elif GROQ_API_KEY:
    API_KEY      = GROQ_API_KEY
    API_BASE_URL = "https://api.groq.com/openai/v1"
    print(f"OK: GROQ fallback active ({GROQ_API_KEY[:8]}...)")
else:
    API_KEY = ""
    API_BASE_URL = ""
    print("WARNING: No API keys. AI Judge will use offline demo mode.")

CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct" if HF_TOKEN else "llama-3.3-70b-versatile"

# ─── True Multi-Model Council ─────────────────────────────────
# Three completely independent AI models deliberate in parallel
COUNCIL_AGENTS = [
    {
        "name": "Agent 1 — The Precedent Analyst",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "persona": (
            "You are the Precedent Analyst on the AI Judicial Council. "
            "You reason EXCLUSIVELY through established Indian case law and statutory frameworks (like the Specific Relief Act 1963 for property/deposits, or the Limitation Act 1963). "
            "CRITICAL RULES: \n"
            "1. NEVER hallucinate or invent facts, damages, or monetary amounts not explicitly provided in the CASE FACTS. If the facts state Rs 10,000, you must strictly use Rs 10,000.\n"
            "2. Always consider if the claim is within the limitation period (e.g., 3 years for recovery of money).\n"
            "3. Do not cite overly broad landmark cases (like Hadley v Baxendale) for simple, direct refunds. Focus on specific, direct statutory rights."
        )
    },
    {
        "name": "Agent 2 — The Constitutional Scholar",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "persona": (
            "You are the Constitutional Scholar on the AI Judicial Council. "
            "You reason through Constitutional law, Fundamental Rights, and statutory compliance (BNS, CPC, Indian Contract Act). "
            "CRITICAL RULES: \n"
            "1. STRICTLY adhere to the facts provided. Do not invent pecuniary losses.\n"
            "2. Note that Section 89 of the CPC encourages mediation, but it is NOT strictly mandated for standard residential disputes unless specified by commercial courts.\n"
            "3. Ensure the legal basis directly matches the nature of the dispute (e.g., a security deposit is held in trust, it is not subject to market-price speculation)."
        )
    },
    {
        "name": "Agent 3 — The Legal Realist",
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "persona": (
            "You are the Legal Realist on the AI Judicial Council. "
            "You analyze cases through real-world impact, socioeconomic context, and the spirit of justice. "
            "CRITICAL RULES: \n"
            "1. Do NOT invent or alter the monetary values provided in the prompt.\n"
            "2. Provide practical, direct resolutions. Avoid over-complicating simple civil disputes (like unreturned rent deposits) with corporate legal theories.\n"
            "3. Balance legal technicality with equitable outcomes, ensuring the remedy makes logical sense based on the precise facts given."
        )
    }
]
# Chief Justice synthesizes all 3 arguments
CHIEF_JUSTICE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

MAX_TOTAL_REWARD       = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    {"name": "task1_contract", "domain": "contract", "difficulty": "easy"},
    {"name": "task2_tort",     "domain": "tort",     "difficulty": "medium"},
    {"name": "task3_property", "domain": "property", "difficulty": "hard"},
    {"name": "task4_petty_crime", "domain": "petty_crime", "difficulty": "hard"},
]

# Global results store
RESULTS: dict = {"status": "starting", "scores": {}, "overall": 0.0}

ESCALATED_CASES = []


# ─── FastAPI App ──────────────────────────────────────────────

app = FastAPI(
    title="Judicial Reasoning RL Environment",
    description="An RL environment where an LLM agent acts as a judge over Indian legal cases.",
    version="1.0.0",
)


ui_dir = os.path.join(os.path.dirname(__file__), "ui")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(ui_dir, "index.html"), headers={"Cache-Control": "no-cache, no-store"})

@app.get("/styles.css", include_in_schema=False)
def styles(v: str = None):
    return FileResponse(os.path.join(ui_dir, "styles.css"), media_type="text/css", headers={"Cache-Control": "no-cache, no-store"})

@app.get("/script.js", include_in_schema=False)
def script(v: str = None):
    return FileResponse(os.path.join(ui_dir, "script.js"), media_type="application/javascript", headers={"Cache-Control": "no-cache, no-store"})


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint — required by OpenEnv pre-submission checklist."""
    return HealthResponse(status="ok")


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    """
    Reset the environment and return the initial observation.
    Required by OpenEnv spec — must return HTTP 200.
    """
    domain     = request.domain     if request else "contract"
    difficulty = request.difficulty if request else "easy"
    env = JudicialEnv(domain=domain, difficulty=difficulty)
    obs, info = env.reset()
    
    if request and request.custom_facts:
        obs.fact_pattern = request.custom_facts
        obs.case_id = "USR-CUSTOM"
        if request.custom_evidence:
            obs.evidence_flags = request.custom_evidence
            
    return ResetResponse(observation=obs.model_dump(), info=info)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Execute one environment step with the provided action."""
    env = JudicialEnv(domain=request.domain, difficulty=request.difficulty)
    env.reset()
    action = JudicialAction(**request.action)
    obs, reward, done, truncated, info = env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        truncated=truncated,
        info=info,
    )


@app.post("/ai_judge", response_model=AIJudgeResponse)
def ai_judge(request: ResetRequest):
    """Generate an AI judgment for the requested domain/difficulty and evaluate it."""
    env = JudicialEnv(domain=request.domain, difficulty=request.difficulty)
    obs, _ = env.reset()

    # Override with custom user facts
    if request.custom_facts:
        obs.fact_pattern = request.custom_facts
        obs.case_id = "USR-CUSTOM"
        if request.custom_evidence:
            obs.evidence_flags = request.custom_evidence
            
        # Mock env.current_case so env.step() evaluation doesn't crash
        env.current_case["case_id"] = "USR-CUSTOM"
        env.current_case["fact_pattern"] = request.custom_facts
        env.current_case["evidence"] = request.custom_evidence or []
        env.current_case["gold_label_verdict"] = "forward_to_judge" if request.domain == "petty_crime" else "liable"
        env.current_case["expert_verdict"] = env.current_case["gold_label_verdict"]
        env.current_case["precedents"] = []

    if not API_KEY:
        # ─── Offline Demo Mode ─────────────────────────────────────────────────
        # Returns a realistic mock judgment using the user's actual facts when offline.
        is_criminal = obs.domain == "petty_crime"
        
        # Use the actual custom facts the user provided so context is not lost!
        facts_preview = (obs.fact_pattern[:200] + '...') if len(obs.fact_pattern) > 200 else obs.fact_pattern
        
        mock_action = JudicialAction(
            verdict          = "forward_to_judge" if is_criminal else "liable",
            confidence_score = 0.91,
            reasoning_chain  = (
                "[COUNCIL OF AI MAJORITY VOTE: 3/3 AGREED — OFFLINE FALLBACK]\n\n"
                f"Agent 1 (Fact Analyst): The user's case states: \"{facts_preview}\". This establishes a prima facie grievance.\n\n"
                f"Agent 2 (Legal Expert): Evaluating the provided facts against statutory law. The evidence supports the complainant's claim.\n\n"
                f"Agent 3 (Chief Justice): I concur. The defendant has failed in their legal obligations as established by the facts presented."
            ),
            cited_precedents = ["P001", "P002"] if not is_criminal else ["P-BNS-101"],
            ratio_decidendi  = (
                "When a party refuses to return a security deposit or breaches an agreement as stated in the facts, they are liable under Section 73 of the Indian Contract Act."
                if not is_criminal else
                "Criminal matters established in the facts are forwarded to a human judge per constitutional design."
            ),
            obiter_dicta     = "Parties are advised to attempt mediation before further legal proceedings.",
            refer_to_human_judge = is_criminal,
            case_status      = "forwarded_to_judge" if is_criminal else "resolved_by_ai",
        )
        obs_next, reward, done, truncated, info = env.step(mock_action)
        return AIJudgeResponse(
            action=mock_action.model_dump(),
            evaluation=StepResponse(observation=obs_next.model_dump(), reward=reward, done=done, truncated=truncated, info=info)
        )

    action, council_votes = get_agent_action(obs)
    obs_next, reward, done, truncated, info = env.step(action)
    return AIJudgeResponse(
        action=action.model_dump(),
        evaluation=StepResponse(observation=obs_next.model_dump(), reward=reward, done=done, truncated=truncated, info=info),
        council_deliberation=council_votes
    )


@app.post("/escalate")
def escalate_case(request: EscalateRequest):
    ESCALATED_CASES.append(request.model_dump())
    return {"status": "success", "appeal_type": request.appeal_type}

@app.get("/api/escalated-cases")
def get_escalated_cases():
    return {"cases": ESCALATED_CASES}

@app.post("/summons")
def generate_summons(request: SummonsRequest):
    """Generate a Summons Notice for the opposing party."""
    import datetime
    now = datetime.datetime.now()
    summons_id = f"SUM-{now.strftime('%Y%m%d%H%M%S')}"
    return {
        "status": "success",
        "summons_id": summons_id,
        "case_id": request.case_id,
        "issued_to": request.respondent_name,
        "issued_on": now.isoformat(),
        "message": f"Summons Notice {summons_id} generated for {request.respondent_name} in Case {request.case_id}."
    }

@app.post("/case_status")
def get_case_status(request: CaseStatusRequest):
    """Return the current status of a registered case."""
    # In production, this queries a DB. For demo: return mock status.
    return {
        "case_id": request.case_id,
        "status": "under_ai_analysis",
        "status_label": "Under AI Analysis — Council is deliberating",
        "last_updated": "2026-04-25T10:00:00",
        "cause_list": [
            {"step": "Case Registered", "done": True},
            {"step": "KYC Verified", "done": True},
            {"step": "Evidence Uploaded", "done": True},
            {"step": "AI Fact-Finding Complete", "done": True},
            {"step": "Council Judgment", "done": False},
            {"step": "AI Resolution Certificate Issued", "done": False},
        ]
    }

@app.get("/judge", include_in_schema=False)
def judge_dashboard():
    return FileResponse(os.path.join(ui_dir, "judge.html"))

@app.get("/judge.js", include_in_schema=False)
def judge_js():
    return FileResponse(os.path.join(ui_dir, "judge.js"))

@app.post("/chat", response_model=ChatResponse)
def fact_finding_chat(request: ChatRequest):
    """Real LLM-powered fact-finding chat using Groq/Llama-3.3."""
    if not API_KEY:
        return ChatResponse(response="API key not configured. Please add your GROQ_API_KEY to the .env file to enable AI fact-finding.")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    is_criminal = request.case_type == "criminal"

    if is_criminal:
        system_prompt = """You are JusticeEngine-01, an AI paralegal for Indian courts under the new Bharatiya Nyaya Sanhita (BNS).
Your task is to gather facts for a CRIMINAL case through targeted, specific questions.

Rules:
- Ask ONE short, specific question per response.
- Focus on: FIR number, BNS section that may apply, nature of offence, evidence available (CCTV, witnesses, medical reports, vehicle plate), date/time/location, whether police have been contacted.
- Ask if the accused is known or unknown.
- Once you have gathered enough facts (after 5-7 exchanges), say EXACTLY: "DOSSIER_COMPLETE: I have gathered sufficient information. You may now generate the AI Fact Bundle for the Judge."
- NEVER suggest guilt or innocence. You are gathering facts ONLY.
- Keep language simple and respectful."""
    else:
        system_prompt = """You are JusticeEngine-01, an AI legal analyst for Indian courts.
Your task is to gather facts about a CIVIL case through targeted, specific questions.

Rules:
- Ask ONE short, specific question per response.
- Questions should help clarify: written agreements/contracts, timeline of events, evidence available (receipts, messages, photos), witnesses, prior disputes, formal complaints filed, and whether an attempt at out-of-court settlement was made.
- Once you have gathered enough facts (after 4-6 exchanges), say EXACTLY: "DOSSIER_COMPLETE: I have gathered sufficient information. You may now generate the AI Judgment."
- Keep language simple, clear, and professional.
- Do NOT give legal opinions yet. Only gather facts."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add case context as the first assistant message
    messages.append({
        "role": "assistant",
        "content": f"I am reviewing your case. Here are the facts on file:\n\n{request.fact_pattern}\n\nTo help me build your complete legal dossier, I need to ask you a few targeted questions."
    })
    
    # Add the conversation history
    for msg in request.chat_history:
        role = "assistant" if msg.get("role") == "ai" else "user"
        messages.append({"role": role, "content": msg.get("content", "")})
    
    # Add the latest user message
    if request.user_message:
        messages.append({"role": "user", "content": request.user_message})

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=250,
        )
        reply = response.choices[0].message.content.strip()
        return ChatResponse(response=reply)
    except Exception as e:
        err = str(e)[:80]
        return ChatResponse(response=f"I encountered an issue connecting to the AI. Please try again. (Error: {err})")



@app.get("/state", response_model=StateResponse)
def get_state(domain: str = "contract", difficulty: str = "easy"):
    """Return current environment state."""
    env = JudicialEnv(domain=domain, difficulty=difficulty)
    env.reset()
    return StateResponse(state=env.state())


@app.get("/police")
async def serve_police_dashboard():
    return FileResponse(os.path.join(ui_dir, "police_dashboard.html"))


@app.get("/tasks")
def get_tasks():
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": "task1_contract",
                "difficulty": "easy",
                "domain": "contract",
                "description": "Contract breach and dispute resolution under Indian Contract Act",
                "expected_baseline": 0.9760,
            },
            {
                "id": "task2_tort",
                "difficulty": "medium",
                "domain": "tort",
                "description": "Tort and negligence cases with conflicting evidence",
                "expected_baseline": 0.6853,
            },
            {
                "id": "task3_property",
                "difficulty": "hard",
                "domain": "property",
                "description": "Property and inheritance disputes with adversarial ambiguous facts",
                "expected_baseline": 0.5520,
            },
            {
                "id": "task4_petty_crime",
                "difficulty": "easy",
                "domain": "petty_crime",
                "description": "Petty crimes using BNS and Constitutional law emphasizing restorative justice",
                "expected_baseline": 0.8500,
            },
        ]
    }


@app.get("/results")
def results():
    """Return inference results after baseline run completes."""
    return JSONResponse(content=RESULTS)


# ─── Inference Runner (background) ───────────────────────────

def log_start(task_name: str):
    print(f"[START] task={task_name}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


def _call_council_member(agent: dict, obs, is_criminal: bool) -> dict:
    """Call one council member model and return their structured legal argument."""
    client = OpenAI(api_key=HF_TOKEN or GROQ_API_KEY, base_url=HF_BASE_URL if HF_TOKEN else "https://api.groq.com/openai/v1")
    verdict_opts = '"forward_to_judge"' if is_criminal else '"liable", "not_liable", or "partial_liability"'
    task_note = (
        "TASK: This is a CRIMINAL case. Do NOT pass final judgment. Identify the BNS section, assess evidence, "
        "and recommend forward_to_judge with your reasoning."
        if is_criminal else
        "TASK: This is a CIVIL case. Analyze through your specialized legal perspective and state your verdict."
    )
    prompt = f"""{agent['persona']}

CASE FACTS:
{obs.fact_pattern}

APPLICABLE STATUTES:
{chr(10).join(obs.statutes)}

PRECEDENTS:
{json.dumps(obs.precedents[:3], indent=2)}

EVIDENCE: {', '.join(obs.evidence_flags) if obs.evidence_flags else 'None'}

{task_note}

Respond ONLY with valid JSON (no markdown, no preamble):
{{
  "verdict": {verdict_opts},
  "argument": "Your 2-3 sentence legal argument from your specialized perspective.",
  "key_statutes": ["statute1", "statute2"],
  "confidence": 0.0
}}"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=agent["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=500
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(m.group(0) if m else raw)
            return {"name": agent["name"], "model": agent["model"], **data}
        except Exception:
            time.sleep(2 ** attempt)
    return {
        "name": agent["name"], "model": agent["model"],
        "verdict": "forward_to_judge" if is_criminal else "liable",
        "argument": f"[{agent['name']} did not respond — fallback position adopted.]",
        "key_statutes": [], "confidence": 0.4
    }


def _synthesize_verdict(council_votes: list, obs, is_criminal: bool) -> dict:
    """Chief Justice DeepSeek reads all 3 arguments and delivers the final verdict."""
    client = OpenAI(api_key=HF_TOKEN or GROQ_API_KEY, base_url=HF_BASE_URL if HF_TOKEN else "https://api.groq.com/openai/v1")
    deliberation = ""
    for v in council_votes:
        deliberation += f"\n─── {v['name']} ({v['model']}) ───\n"
        deliberation += f"  Verdict: {v.get('verdict','?').upper()}\n"
        deliberation += f"  Confidence: {v.get('confidence', 0.5):.0%}\n"
        deliberation += f"  Argument: {v.get('argument', 'N/A')}\n"
        if v.get('key_statutes'):
            deliberation += f"  Key Statutes: {', '.join(v['key_statutes'])}\n"
    verdict_opts = '"forward_to_judge"' if is_criminal else '"liable", "not_liable", or "partial_liability"'
    extra = "Set verdict to forward_to_judge." if is_criminal else "Choose the verdict backed by the strongest legal arguments."
    synthesis_prompt = f"""You are the Chief Justice of the AI Judicial Council for Indian courts. Three expert agents have deliberated on this case.

CASE FACTS (use ONLY these facts — do NOT invent or modify any monetary amounts, names, or events):
{obs.fact_pattern[:600]}

COUNCIL DELIBERATION:{deliberation}

STRICT RULES FOR YOUR FINAL JUDGMENT:
1. NEVER invent or modify monetary amounts. Use ONLY the exact figures stated in the CASE FACTS above.
2. For civil money recovery (deposits, loans, dues): cite the Specific Relief Act 1963 alongside Section 73 of the Indian Contract Act 1872. Do NOT apply "Hadley v Baxendale" unless there is genuine consequential loss beyond the direct amount.
3. Section 89 CPC: This encourages mediation — it does NOT mandate it for standard residential civil disputes. Do not state it is mandatory unless it is a commercial court matter.
4. Always note if the claim is within the Limitation Act 1963 period (3 years for money recovery).
5. The ratio_decidendi must be specific to the FACTS, not a generic template.
{extra}

Your task:
1. Weigh the arguments from all three agents and select the most legally sound position.
2. Deliver the FINAL authoritative verdict with comprehensive, fact-specific reasoning.
3. Write a clear ratio_decidendi (binding legal principle based on these specific facts).

Respond ONLY with valid JSON:
{{
  "verdict": {verdict_opts},
  "confidence_score": 0.0,
  "reasoning_chain": "Fact-specific synthesis of all 3 council arguments. Quote exact monetary amounts from the facts. 3-5 sentences.",
  "cited_precedents": ["case1"],
  "ratio_decidendi": "The binding legal principle: ...",
  "obiter_dicta": "Non-binding observations for future cases."
}}"""
    for attempt in range(3):
        try:
            resp = OpenAI(api_key=HF_TOKEN or GROQ_API_KEY, base_url=HF_BASE_URL if HF_TOKEN else "https://api.groq.com/openai/v1").chat.completions.create(
                model=CHIEF_JUSTICE_MODEL,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.2, max_tokens=1000
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group(0) if m else raw)
        except Exception:
            time.sleep(2 ** attempt)
    # Fallback: majority vote
    counts = {}
    for v in council_votes:
        counts[v.get('verdict', 'liable')] = counts.get(v.get('verdict', 'liable'), 0) + 1
    maj = max(counts, key=counts.get)
    return {
        "verdict": maj, "confidence_score": 0.65,
        "reasoning_chain": f"Chief Justice synthesis unavailable. Majority council position ({maj}) adopted.",
        "cited_precedents": [], "ratio_decidendi": "Majority council verdict adopted per constitutional design.",
        "obiter_dicta": "Full synthesis will be available when the Chief Justice model is online."
    }


def get_agent_action(obs, _unused=None) -> tuple:
    """True multi-model judicial council: 3 agents in parallel, then Chief Justice synthesizes."""
    is_criminal = obs.domain == "petty_crime"

    # Phase 1: All 3 council members deliberate IN PARALLEL
    council_votes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_call_council_member, agent, obs, is_criminal): agent for agent in COUNCIL_AGENTS}
        for future in concurrent.futures.as_completed(futures):
            try:
                council_votes.append(future.result(timeout=45))
            except Exception as ex:
                a = futures[future]
                council_votes.append({
                    "name": a["name"], "model": a["model"],
                    "verdict": "forward_to_judge" if is_criminal else "liable",
                    "argument": f"[Agent timed out: {str(ex)[:60]}]",
                    "key_statutes": [], "confidence": 0.3
                })

    # Phase 2: Chief Justice DeepSeek synthesizes into final verdict
    synthesis = _synthesize_verdict(council_votes, obs, is_criminal)

    # Build full council transcript
    transcript = "[═══ JUDICIAL COUNCIL DELIBERATION ═══]\n\n"
    for vote in council_votes:
        transcript += f"◉ {vote['name']}\n"
        transcript += f"  Model: {vote['model']}\n"
        transcript += f"  Position: {vote.get('verdict','?').upper()} (Confidence: {vote.get('confidence',0.5):.0%})\n"
        transcript += f"  Argument: {vote.get('argument','N/A')}\n\n"
    transcript += "[═══ CHIEF JUSTICE SYNTHESIS (DeepSeek-R1) ═══]\n\n"
    transcript += synthesis.get("reasoning_chain", "")

    action = JudicialAction(
        verdict=synthesis["verdict"],
        confidence_score=float(synthesis.get("confidence_score", 0.8)),
        reasoning_chain=transcript,
        cited_precedents=synthesis.get("cited_precedents", []),
        ratio_decidendi=synthesis.get("ratio_decidendi", ""),
        obiter_dicta=synthesis.get("obiter_dicta", ""),
        case_status="resolved_by_ai" if synthesis["verdict"] != "forward_to_judge" else "forwarded_to_judge"
    )
    return action, council_votes



async def run_task(task_config: dict) -> float:
    task_name = task_config["name"]
    log_start(task_name)
    env = JudicialEnv(domain=task_config["domain"], difficulty=task_config["difficulty"])
    obs, _ = env.reset()
    rewards, steps_taken, success, score = [], 0, False, 0.0
    try:
        for step_num in range(1, 4):
            try:
                action, _ = get_agent_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                steps_taken = step_num
                log_step(step=step_num, action=action.verdict, reward=reward, done=done)
                if done:
                    obs, _ = env.reset()
            except Exception as e:
                log_step(step=step_num, action="ERROR", reward=0.0, done=False, error=str(e))
                break
        score = sum(rewards) / (len(rewards) * MAX_TOTAL_REWARD) if rewards else 0.001
        score = min(max(score, 0.001), 0.999)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score




async def run_all_tasks():
    global RESULTS
    RESULTS["status"] = "running"

    if not API_KEY:
        print("[WARN] No API key set (HF_TOKEN / GROQ_API_KEY). Skipping inference.", flush=True)
        RESULTS.update({"status": "skipped", "scores": {}, "overall": 0.0})
        return

    all_scores = {}
    for task in TASKS:
        score = await run_task(task)
        all_scores[task["name"]] = round(score, 4)

    overall = sum(all_scores.values()) / len(all_scores)

    print(f"\n=== BASELINE RESULTS ===", flush=True)
    for name, s in all_scores.items():
        print(f"{name}: {s:.4f}", flush=True)
    print(f"OVERALL AVERAGE: {overall:.4f}", flush=True)

    results_data = {
        "status": "complete",
        "scores": all_scores,
        "overall": round(overall, 4),
    }

    try:
        with open("results.json", "w") as f:
            json.dump(results_data, f, indent=2)
    except Exception:
        pass

    RESULTS.update(results_data)


def run_inference_background():
    asyncio.run(run_all_tasks())


@app.on_event("startup")
async def startup_event():
    """Launch inference in background when server starts."""
    thread = threading.Thread(target=run_inference_background, daemon=True)
    thread.start()


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
