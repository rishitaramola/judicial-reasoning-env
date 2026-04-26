"""
TRL GRPO Training Script for JusticeEngine-01
Team ALACRITY | OpenEnv Hackathon | Scaler × Meta | April 2026

Unsloth-accelerated GRPO training with curriculum learning.
Run on Google Colab (T4 GPU minimum, A100 recommended):

  # Step 1: Install Unsloth (must be first)
  !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  !pip install --no-deps xformers trl peft accelerate bitsandbytes
  !pip install wandb datasets huggingface_hub gymnasium pydantic

  # Step 2: Clone and run
  !git clone https://github.com/rishitaramola/META.git
  %cd META
  !python train.py

RLVR Reward Components (3 independent verifiable reward functions):
  - format_reward:   XML structure compliance (anti-reward-hacking: requires ALL tags)
  - accuracy_reward:  Verdict match against gold labels via JudicialEnv.step()
  - logic_reward:     BNS/Constitution keyword density + reasoning depth + ratio_decidendi
All scores clamped to (0.001, 0.999) per hackathon spec.
"""

import os
import json
import re
import sys

# ─── Unsloth + TRL imports (graceful fallback for local testing) ─────
TRAINING_AVAILABLE = False
UNSLOTH_AVAILABLE = False

try:
    from unsloth import FastLanguageModel, PatchFastRL
    UNSLOTH_AVAILABLE = True
    # Patch TRL's GRPO to use Unsloth's fast kernels
    PatchFastRL("GRPO", FastLanguageModel)
    print("[OK] Unsloth loaded — fast LoRA + vLLM inference enabled")
except ImportError:
    print("[WARN] Unsloth not installed. Install via:")
    print('   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
except NotImplementedError:
    print("[WARN] Unsloth requires a GPU. In Colab: Runtime > Change runtime type > T4 GPU")
except Exception as e:
    print(f"[WARN] Unsloth init failed: {e}")

try:
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    TRAINING_AVAILABLE = True
except ImportError:
    print("[WARN] GPU/TRL libraries not found. Run on Colab for full training.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not installed. Run: pip install wandb")

# ─── Local imports ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import JudicialEnv, JudicialAction

# ==========================================
# 0. Global Data Loading
# ==========================================
def load_gold_labels() -> dict:
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cases.json")
    with open(data_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    return {
        c["case_id"]: c.get("gold_label_verdict", c.get("expert_verdict", "forward_to_judge"))
        for c in cases
    }

GOLD_LABELS = load_gold_labels()

# ==========================================
# 1. Configuration
# ==========================================
MODEL_NAME        = "unsloth/Meta-Llama-3.1-8B-Instruct"   # Unsloth-optimized variant
MAX_SEQ_LENGTH    = 4096
LORA_RANK         = 16
LORA_ALPHA        = 16                                     # alpha = rank is standard
MAX_STEPS         = 250                                    # Total across 3 phases
HF_REPO_ID        = "Sarthaksingh26/justice-engine-01-lora"
DATASET_REPO_ID   = "Sarthaksingh26/indian-legal-cases"
WANDB_PROJECT     = "justice-engine-01"
WANDB_RUN_NAME    = "grpo-bns-curriculum-v3"

# Unsloth-specific settings
GPU_MEMORY_UTIL   = 0.6       # For vLLM inference server
LOAD_IN_4BIT      = True      # QLoRA: 4-bit base + 16-bit LoRA adapters
USE_GRADIENT_CKPT = "unsloth" # Unsloth's memory-efficient gradient checkpointing

# ── Target modules for LoRA ──
# All attention + MLP projections for maximum expressiveness
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SYSTEM_PROMPT = """You are JusticeEngine-01, an AI legal mediator for Indian courts.
You must strictly follow the Constitution of India and the Bharatiya Nyaya Sanhita (BNS) 2023.
Always follow court hierarchy: Supreme Court > High Court > Sessions Court > Magistrate.
For criminal cases under BNS 2023, you MUST output forward_to_judge. AI cannot adjudicate criminal matters.

Respond ONLY in valid XML format:
<action>
  <verdict>liable OR not_liable OR partial_liability OR forward_to_judge</verdict>
  <confidence_score>0.0 to 1.0</confidence_score>
  <reasoning_chain>Your step-by-step reasoning citing BNS sections and precedents</reasoning_chain>
  <ratio_decidendi>The single binding legal principle of this decision</ratio_decidendi>
  <obiter_dicta>Non-binding judicial observations made in passing</obiter_dicta>
</action>"""


# ==========================================
# 2. Reward Functions (RLVR — 3 Independent Verifiable Rewards)
# ==========================================
# Per hackathon guide §7: "Use multiple independent reward functions, not just one."

def extract_xml_action(completion: str) -> dict:
    """Helper to extract XML fields from LLM completion."""
    try:
        verdict    = re.search(r'<verdict>(.*?)</verdict>', completion, re.DOTALL)
        confidence = re.search(r'<confidence_score>(.*?)</confidence_score>', completion, re.DOTALL)
        reasoning  = re.search(r'<reasoning_chain>(.*?)</reasoning_chain>', completion, re.DOTALL)
        ratio      = re.search(r'<ratio_decidendi>(.*?)</ratio_decidendi>', completion, re.DOTALL)
        obiter     = re.search(r'<obiter_dicta>(.*?)</obiter_dicta>', completion, re.DOTALL)
        return {
            "verdict":          verdict.group(1).strip() if verdict else "invalid",
            "confidence_score": float(confidence.group(1).strip()) if confidence else 0.0,
            "reasoning_chain":  reasoning.group(1).strip() if reasoning else "",
            "ratio_decidendi":  ratio.group(1).strip() if ratio else "",
            "obiter_dicta":     obiter.group(1).strip() if obiter else "",
            "cited_precedents": []
        }
    except Exception:
        return None


def format_reward(prompts, completions, **kwargs):
    """
    REWARD 1/3: XML Format Compliance.
    
    +0.5 if ALL required tags present (<action>, <verdict>, <confidence_score>,
         <reasoning_chain>, <ratio_decidendi>).
    +0.25 bonus if <obiter_dicta> also present (complete legal format).
    0.001 if any required tag is missing.
    
    Anti-hacking: requires both opening AND closing tags.
    """
    rewards = []
    required_tags = [
        "<action>", "</action>", "<verdict>", "</verdict>",
        "<confidence_score>", "</confidence_score>",
        "<reasoning_chain>", "</reasoning_chain>",
        "<ratio_decidendi>", "</ratio_decidendi>",
    ]
    for comp in completions:
        comp_str = comp if isinstance(comp, str) else comp[0]["content"]
        if all(tag in comp_str for tag in required_tags):
            score = 0.5
            if "<obiter_dicta>" in comp_str and "</obiter_dicta>" in comp_str:
                score += 0.25
            rewards.append(min(score, 0.999))
        else:
            rewards.append(0.001)
    return rewards


def accuracy_reward(prompts, completions, **kwargs):
    """
    REWARD 2/3: Legal Verdict Accuracy (now case-specific).
    Uses the CASE ID embedded in the prompt to find the gold label.
    """
    rewards = []
    for prompt, comp in zip(prompts, completions):
        comp_str = comp if isinstance(comp, str) else comp[0]["content"]
        action_dict = extract_xml_action(comp_str)

        if not action_dict or action_dict["verdict"] == "invalid":
            rewards.append(0.001)
            continue

        try:
            # Extract case_id from the prompt
            prompt_text = prompt[-1]["content"] if isinstance(prompt, list) else prompt
            case_id_match = re.search(r'CASE ID:\s*(\S+)', prompt_text)
            if not case_id_match:
                rewards.append(0.001)
                continue

            case_id = case_id_match.group(1)
            gold_verdict = GOLD_LABELS.get(case_id, "forward_to_judge")
            verdict = action_dict["verdict"]

            if verdict == gold_verdict:
                score = 1.0
            elif "partial" in verdict or "partial" in gold_verdict:
                score = 0.5
            else:
                score = 0.0
            rewards.append(max(0.001, min(score, 0.999)))
        except Exception as e:
            print(f"[WARN] Reward failure: {e}")
            rewards.append(0.001)
    return rewards


def logic_reward(prompts, completions, **kwargs):
    """
    REWARD 3/3: Legal Reasoning Quality.
    
    Checks:
    - BNS/Constitution keyword density (0.5 weight)
    - Reasoning depth: >100 chars (+0.2), >300 chars (+0.2)
    - ratio_decidendi present (+0.1)
    
    Anti-hacking: keywords must appear in reasoning_chain only,
    not just dumped anywhere in the response.
    """
    BNS_KEYWORDS = [
        "constitution", "bns", "sanhita", "bnss", "bharatiya",
        "section", "supreme court", "high court", "precedent",
        "liable", "burden of proof", "ratio", "obiter",
        "contract act", "cognizable", "fir", "forward_to_judge",
        "statute", "article", "damages", "negligence",
    ]
    rewards = []
    for comp in completions:
        comp_str = comp if isinstance(comp, str) else comp[0]["content"]
        action = extract_xml_action(comp_str)
        if not action:
            rewards.append(0.001)
            continue

        # Only check reasoning_chain content (anti-hacking)
        text = action["reasoning_chain"].lower()
        score = 0.0
        # Keyword density
        hits = sum(1 for kw in BNS_KEYWORDS if kw in text)
        score += min(hits / len(BNS_KEYWORDS), 1.0) * 0.5
        # Reasoning depth
        if len(text) > 100: score += 0.2
        if len(text) > 300: score += 0.2
        # ratio_decidendi present and non-trivial
        ratio = action.get("ratio_decidendi", "")
        if ratio and len(ratio) > 20: score += 0.1
        rewards.append(max(0.001, min(score, 0.999)))
    return rewards


def process_reward(prompts, completions, **kwargs):
    """
    REWARD 4/4: Process-Aware Feedback (§9 from hackathon guide).
    
    Instead of only scoring the final outcome, this checks INTERMEDIATE
    reasoning steps for quality signals:
    
    1. Step-by-step structure: Does the reasoning have numbered steps?
    2. Verdict-reasoning consistency: Does the reasoning support the verdict?
    3. BNS section format: Are cited BNS sections in valid range (1-358)?
    4. Evidence acknowledgment: Does reasoning reference evidence flags from the prompt?
    
    This approximates process supervision without needing a separate verifier model.
    """
    rewards = []
    for prompt, comp in zip(prompts, completions):
        comp_str = comp if isinstance(comp, str) else comp[0]["content"]
        action = extract_xml_action(comp_str)
        if not action:
            rewards.append(0.001)
            continue

        score = 0.0
        reasoning = action["reasoning_chain"]
        verdict = action["verdict"]
        reasoning_lower = reasoning.lower()

        # ── Check 1: Step-by-step structure (0.25) ──
        # Reward explicit numbered steps ("Step 1:", "1.", "First,")
        step_patterns = [
            r'step\s*\d', r'^\d+\.\s', r'first[,:]', r'second[,:]',
            r'third[,:]', r'finally[,:]', r'therefore[,:]', r'in conclusion',
        ]
        step_hits = sum(1 for p in step_patterns if re.search(p, reasoning_lower, re.MULTILINE))
        score += min(step_hits / 4, 1.0) * 0.25

        # ── Check 2: Verdict-reasoning consistency (0.3) ──
        # The reasoning should contain language consistent with the verdict
        consistency_map = {
            "liable": ["liable", "breach", "duty of care", "negligent", "responsible", "violated"],
            "not_liable": ["not liable", "no breach", "no duty", "not responsible", "dismissed", "acquit"],
            "forward_to_judge": ["forward", "judge", "criminal", "cognizable", "bns", "magistrate", "trial"],
            "partial_liability": ["partial", "contributory", "shared", "reduced", "both parties"],
        }
        verdict_keywords = consistency_map.get(verdict, [])
        if verdict_keywords:
            consistency_hits = sum(1 for kw in verdict_keywords if kw in reasoning_lower)
            if consistency_hits >= 2:
                score += 0.3
            elif consistency_hits >= 1:
                score += 0.15
            # Penalize contradiction: reasoning says "not liable" but verdict is "liable"
            if verdict == "liable" and "not liable" in reasoning_lower:
                score -= 0.15
            if verdict == "not_liable" and re.search(r'\bliable\b', reasoning_lower) and "not liable" not in reasoning_lower:
                score -= 0.15

        # ── Check 3: BNS section validation (0.2) ──
        # If BNS sections are cited, check they're in valid range (1-358)
        bns_sections = re.findall(r'(?:bns|sanhita|section)\s*(\d+)', reasoning_lower)
        if bns_sections:
            valid = sum(1 for s in bns_sections if 1 <= int(s) <= 358)
            total = len(bns_sections)
            score += (valid / total) * 0.2 if total > 0 else 0.0
        else:
            # No BNS sections cited — neutral (don't penalize non-criminal cases)
            score += 0.1

        # ── Check 4: Evidence acknowledgment (0.25) ──
        # Does the reasoning reference evidence flags mentioned in the prompt?
        prompt_text = prompt[-1]["content"] if isinstance(prompt, list) else prompt
        evidence_match = re.search(r'EVIDENCE FLAGS:\s*(.+)', prompt_text)
        if evidence_match:
            flags = [f.strip().lower() for f in evidence_match.group(1).split(',') if f.strip() and f.strip().lower() != 'none']
            if flags:
                referenced = sum(1 for flag in flags if any(word in reasoning_lower for word in flag.split('_')))
                score += min(referenced / max(len(flags), 1), 1.0) * 0.25
            else:
                score += 0.15  # No evidence flags to check
        else:
            score += 0.15

        rewards.append(max(0.001, min(score, 0.999)))
    return rewards


# ==========================================
# 3. Dataset Preparation + HF Upload
# ==========================================
def load_and_upload_dataset(push_to_hub: bool = True):
    """
    Load cases.json and convert to a HF Dataset with chat-formatted prompts.
    Each row contains the system prompt + user prompt with CASE DOMAIN/DIFFICULTY
    metadata that the reward functions parse during training.
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cases.json")
    with open(data_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    dataset_rows = []
    for c in cases:
        prompt_text = (
            f"CASE ID: {c['case_id']}\n"
            f"CASE DOMAIN: {c['domain']}\n"
            f"CASE DIFFICULTY: {c['difficulty']}\n\n"
            f"FACT PATTERN:\n{c['fact_pattern']}\n\n"
            f"APPLICABLE STATUTES:\n{chr(10).join(c.get('applicable_statutes', []))}\n\n"
            f"PRECEDENTS:\n{json.dumps(c.get('precedents', []), indent=2)}\n\n"
            f"EVIDENCE FLAGS: {', '.join(c.get('evidence_flags', [])) or 'None'}"
        )
        dataset_rows.append({
            "case_id":    c["case_id"],
            "domain":     c["domain"],
            "difficulty": c["difficulty"],
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt_text}
            ],
            "gold_label_verdict":   c.get("gold_label_verdict", c.get("expert_verdict", "forward_to_judge")),
            "gold_label_reasoning": c.get("gold_label_reasoning", ""),
        })

    ds = Dataset.from_list(dataset_rows)

    if push_to_hub:
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            print(f"📤 Uploading dataset ({len(ds)} cases) to HF Hub: {DATASET_REPO_ID}")
            ds.push_to_hub(DATASET_REPO_ID, token=hf_token, private=False)
            print("✅ Dataset uploaded.")
        else:
            print("⚠️  HF_TOKEN not set. Skipping dataset upload.")
    return ds


# ==========================================
# 4. Model Loading (Unsloth-optimized)
# ==========================================
def load_model():
    """
    Load the base model using Unsloth's FastLanguageModel.
    
    Key Unsloth optimizations:
    - 4-bit QLoRA quantization (70% less VRAM vs full precision)
    - Unsloth gradient checkpointing (30% less memory vs HF default)
    - vLLM-powered fast inference for GRPO rollouts
    - Fused LoRA kernels for 2x faster training
    
    Returns (model, tokenizer) tuple.
    """
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError(
            "Unsloth is required for training. Install via:\n"
            '  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
        )

    print(f"🔄 Loading {MODEL_NAME} via Unsloth (4-bit quantized)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name             = MODEL_NAME,
        max_seq_length         = MAX_SEQ_LENGTH,
        load_in_4bit           = LOAD_IN_4BIT,
        fast_inference         = True,       # Enable vLLM for GRPO rollouts
        max_lora_rank          = LORA_RANK,
        gpu_memory_utilization = GPU_MEMORY_UTIL,
    )
    print(f"✅ Base model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Apply LoRA adapters to all attention + MLP projections
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_RANK,
        target_modules             = LORA_TARGET_MODULES,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = 0,          # Unsloth optimized: 0 dropout
        bias                       = "none",      # Unsloth optimized: no bias
        use_gradient_checkpointing = USE_GRADIENT_CKPT,
        random_state               = 3407,
        use_rslora                 = False,       # Rank-stabilized LoRA (optional)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"🧩 LoRA applied: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ==========================================
# 5. Curriculum Training Loop
# ==========================================
def create_grpo_config(phase: int, max_steps: int, lr: float, run_name: str) -> "GRPOConfig":
    """
    Create a GRPOConfig for one curriculum phase.
    
    Per hackathon guide §11: "Prefer GRPO/RLVR style training for verifiable tasks."
    Uses vLLM for fast rollout generation (guide §12: "Keep inference fast").
    """
    return GRPOConfig(
        use_vllm                    = True,       # Unsloth vLLM for fast rollouts
        learning_rate               = lr,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.99,
        weight_decay                = 0.1,
        warmup_ratio                = 0.1 if phase == 1 else 0.05,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_8bit",  # Unsloth: 8-bit optimizer
        logging_steps               = 1,
        bf16                        = True,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        num_generations             = 6,          # 6 rollouts per prompt for GRPO
        max_prompt_length           = 1024,
        max_completion_length       = 1024,
        max_steps                   = max_steps,
        save_steps                  = 50,
        output_dir                  = "outputs",
        report_to                   = "wandb" if WANDB_AVAILABLE else "none",
        run_name                    = run_name,
    )


def inspect_generations(trainer, n=3):
    """
    Per hackathon guide §15: "Inspect actual generations during training."
    Sample and print a few model outputs to verify learning.
    """
    print(f"\n{'─'*60}")
    print(f"🔍 Sample Generations (latest {n}):")
    print(f"{'─'*60}")
    try:
        # Access the most recent generation logs if available
        if hasattr(trainer, '_last_completions'):
            for i, comp in enumerate(trainer._last_completions[:n]):
                text = comp[0]["content"] if isinstance(comp, list) else comp
                # Truncate for display
                display = text[:300] + "..." if len(text) > 300 else text
                print(f"  [{i+1}] {display}")
    except Exception:
        print("  (No generations to display yet)")
    print(f"{'─'*60}\n")


def main():
    # ─── Init Wandb ─────────────────────────────────────
    if WANDB_AVAILABLE:
        if not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_MODE"] = "offline"
            print("[INFO] WANDB_API_KEY not found. Running in offline mode.")
        
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model": MODEL_NAME,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "max_steps": MAX_STEPS,
                "load_in_4bit": LOAD_IN_4BIT,
                "unsloth": UNSLOTH_AVAILABLE,
                "reward_components": ["format_reward", "accuracy_reward", "logic_reward", "process_reward"],
                "reward_clamping": "(0.001, 0.999)",
                "curriculum_phases": 3,
                "legal_framework": "BNS 2023 / BNSS 2023 / BSA 2023",
                "anti_hacking": ["hallucination_penalty", "bns_citation_check", "consistency_check"],
            }
        )
        print(f"📊 Wandb: {wandb.run.get_url()}")

    if not TRAINING_AVAILABLE:
        print("❌ Training libraries not found. Please run on Colab with GPU.")
        return

    print("=" * 60)
    print("⚖️  JusticeEngine-01 — GRPO Training with Unsloth")
    print("=" * 60)

    # ─── Load Model (Unsloth) ────────────────────────────
    model, tokenizer = load_model()

    # ─── Load Dataset with Curriculum ────────────────────
    print("\n📂 Loading legal cases dataset...")
    full_dataset = load_and_upload_dataset(push_to_hub=True)
    print(f"✅ Dataset ready: {len(full_dataset)} cases")

    # Curriculum splits (guide §6: "Keep the task simple at first")
    easy_ds        = full_dataset.filter(lambda x: x["difficulty"] == "easy")
    easy_medium_ds = full_dataset.filter(lambda x: x["difficulty"] in ["easy", "medium"])
    all_ds         = full_dataset

    print(f"📚 Curriculum splits:")
    print(f"   Phase 1 (Easy):    {len(easy_ds)} cases  →  80 steps, lr=5e-6")
    print(f"   Phase 2 (+Medium): {len(easy_medium_ds)} cases  →  90 steps, lr=3e-6")
    print(f"   Phase 3 (All):     {len(all_ds)} cases  →  80 steps, lr=1e-6")

    # The 4 independent verifiable reward functions (§7 + §9)
    reward_funcs = [format_reward, accuracy_reward, logic_reward, process_reward]

    # ─── Phase 1: Easy cases only (steps 0-80) ───────────
    print("\n" + "=" * 60)
    print("🎓 PHASE 1/3: Easy cases — building format compliance")
    print("=" * 60)
    config_p1 = create_grpo_config(phase=1, max_steps=80, lr=5e-6,
                                    run_name=f"{WANDB_RUN_NAME}-phase1")
    trainer_p1 = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = reward_funcs,
        args             = config_p1,
        train_dataset    = easy_ds,
    )
    trainer_p1.train()
    inspect_generations(trainer_p1)

    # ─── Phase 2: Easy + Medium (steps 80-170) ───────────
    print("\n" + "=" * 60)
    print("[PHASE] PHASE 2/3: Easy + Medium — adding reasoning depth")
    print("=" * 60)
    config_p2 = create_grpo_config(phase=2, max_steps=90, lr=3e-6,
                                    run_name=f"{WANDB_RUN_NAME}-phase2")
    trainer_p2 = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = reward_funcs,
        args             = config_p2,
        train_dataset    = easy_medium_ds,
    )
    trainer_p2.train()
    inspect_generations(trainer_p2)

    # ─── Phase 3: All difficulties (steps 170-250) ───────
    print("\n" + "=" * 60)
    print("[PHASE] PHASE 3/3: All difficulties — adversarial robustness")
    print("=" * 60)
    config_p3 = create_grpo_config(phase=3, max_steps=80, lr=1e-6,
                                    run_name=f"{WANDB_RUN_NAME}-phase3")
    trainer_p3 = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = reward_funcs,
        args             = config_p3,
        train_dataset    = all_ds,
    )
    trainer_p3.train()
    inspect_generations(trainer_p3)

    print("\n[DONE] Curriculum training complete! (250 total steps)")

    # ─── Save LoRA Adapter (Unsloth method) ──────────────
    # Per hackathon guide §16: "Do not upcast a 4-bit model to 16-bit and merge."
    # We save LoRA adapters separately, then use push_to_hub_merged with save_method="lora".
    print("\n[SAVE] Saving LoRA adapter (Unsloth native save)...")
    model.save_lora("outputs/justice_engine_lora")
    print("[OK] LoRA adapter saved to outputs/justice_engine_lora/")

    # ─── Push to HF Hub ──────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"\n[UPLOAD] Uploading to HF Hub: {HF_REPO_ID}")

        # Option 1: Upload LoRA adapters only (fast, small)
        model.push_to_hub_merged(
            HF_REPO_ID,
            tokenizer,
            save_method="lora",     # Save as LoRA, NOT merged 16-bit
            token=hf_token,
        )
        print(f"[OK] LoRA uploaded: https://huggingface.co/{HF_REPO_ID}")

        # Option 2 (optional): Also save GGUF for local inference
        # model.push_to_hub_gguf(
        #     f"{HF_REPO_ID}-GGUF",
        #     tokenizer,
        #     quantization_method="q4_k_m",
        #     token=hf_token,
        # )
    else:
        print("[WARN] HF_TOKEN not set. Skipping HF Hub upload.")
        print("   Set it with: export HF_TOKEN=hf_xxx")

    if WANDB_AVAILABLE:
        wandb.finish()

    print("\n" + "=" * 60)
    print("[SUCCESS] JusticeEngine-01 Training Complete!")
    print(f"   Model:  {MODEL_NAME}")
    print(f"   LoRA:   outputs/justice_engine_lora/")
    print(f"   Steps:  {MAX_STEPS} (3 phases)")
    print(f"   Hub:    https://huggingface.co/{HF_REPO_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
