---
title: "JusticeEngine-01: Training Legal AI with Verifiable Rewards to Clear India's 50M Case Backlog"
thumbnail: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlvr/thumbnail.png"
authors:
  - user: rishitaramola
  - user: Sarthaksingh2005
---

# JusticeEngine-01: RLVR for India's Judicial Crisis

*A Meta-PyTorch OpenEnv Hackathon Submission — Team ALACRITY*

---

## 1. The Crisis: Is this the image of Vishwaguru?

We call ourselves *Vishwaguru* (Teacher of the World) — and there is no doubt to that. We come from modern classrooms filled with Apple computers and state-of-the-art tech. But for a minute, imagine an Indian courtroom. Has it changed in 100 years? 

If you step into the storeroom or record room of a district court, you will see piles upon piles of files and paperwork. Long lines of decades of litigation, endless arguments, and millions of unresolved or pending cases.

The **National Judicial Data Grid (NJDG)** tells a sobering story:
- **50+ million pending cases** across all Indian courts.
- India has only about **21 judges per million people**.
- Millions of undertrials spend years behind bars, only to be ultimately acquitted because they were never guilty. 

What a waste of human life. What a waste of resources. Now tell me, is this the *Vishwaguru* image you have in your mind?

This is not a statistics problem. It is a human tragedy. And this is exactly where we present our vision for the future judiciary of India. We would like to present **Judge AI (JusticeEngine-01)** — the solution which saves hassle, saves time, and saves human life.

---

## 2. The New Laws: BNS, BNSS, and BSA (2024)

In July 2024, India replaced its entire criminal legal framework:

| Old Law (Colonial Era) | New Law (2024) |
|---|---|
| Indian Penal Code 1860 | **Bharatiya Nyaya Sanhita (BNS) 2023** |
| Code of Criminal Procedure 1973 | **Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023** |
| Indian Evidence Act 1872 | **Bharatiya Sakshya Adhiniyam (BSA) 2023** |

The entire judiciary — judges, lawyers, clerks — is simultaneously managing a 50-million case backlog and learning a completely new legal code. JusticeEngine-01 is designed to help.

**Our hypothesis:** If even 10% of simple civil cases (minor contract disputes, rent disagreements, small consumer claims) could be resolved by a verified AI mediator, that is **5 million cases cleared**. Human judges could then focus on what machines should never decide: criminal guilt, fundamental rights, and matters of life and liberty.

---

## 3. RLVR Explained: Teaching AI to Be a Fair Judge

**Reinforcement Learning with Verifiable Rewards (RLVR)** means we train the AI using a reward function that programmatically verifies whether the LLM's answer is actually correct — not just fluent.

For JusticeEngine-01, our reward function evaluates four independent components every time the AI makes a decision:

```
R = 0.3 × logic + 0.4 × accuracy + 0.2 × fairness + 0.1 × citation
  − 0.2 × (per hallucinated precedent, max −0.4)
  + 0.1 × (adversarial bonus: hard case + disputed evidence + deep reasoning)
  + 0.05 × (SC alignment: verdict matches Supreme Court record)
  − 0.15 × (hierarchy violation: chose High Court over available SC verdict)
```

**Why this is hard to game:** An LLM cannot simply generate confident-sounding text and score well. It must:
1. Match the **correct verdict** against expert gold labels
2. **Cite real precedents** from the provided case file (not hallucinated ones)
3. Be **consistent** across similar cases in the same domain
4. Follow **court hierarchy** — Supreme Court always overrides High Court

---

## 4. The Multi-Agent Council: Democratic AI Justice

JusticeEngine-01 does not use a single LLM call. It instantiates **three distinct AI personas** simultaneously:

| Persona | Philosophy |
|---|---|
| **The Strict Constitutionalist** | Adheres to the letter of BNS/BNSS/BSA, constitutional provisions, and binding Supreme Court precedents |
| **The Empathetic Mediator** | Focuses on restorative justice, equity, and the human context of the parties |
| **The Precedent Analyst** | Weights past case outcomes for maximum consistency; prioritises most recent cases from the highest court |

**Voting Rules:**
- All three vote independently
- **2/3 or 3/3 majority wins** — output is stamped `[COUNCIL OF AI MAJORITY VOTE: X/3 AGREED]`
- **3-way split (all different verdicts):** Case is **automatically escalated** to a Human Judge — no AI verdict issued

This 3-way split auto-escalation is emergent multi-agent behavior: no single agent was programmed to escalate; it arises from the disagreement between three independent reasoners.

---

## 5. The Ethical Line: Why We Refuse to Judge Criminal Cases

This is the most important design decision we made.

**Civil Cases (Contract / Tort / Property / Consumer):**
The AI may issue a full verdict: `liable`, `not_liable`, or `partial_liability`. It provides a `ratio_decidendi` (binding legal principle) and `obiter_dicta` (non-binding observations). Citizens can Accept, Appeal, Review (CPC Order 47), Revise, or Escalate.

**Criminal Cases (BNS Offences):**
The AI is **strictly forbidden** from issuing any verdict of `guilty` or `not_guilty`. Instead, it acts as a digital paralegal:
1. Routes evidence to the Police Verification Module (mandatory human pause)
2. Identifies the precise BNS section applicable
3. Determines whether the act is a **punishable offence** (yes/no)
4. States the range of possible sentences if proven
5. Bundles all facts in judge-readable order
6. Mandatorily outputs `forward_to_judge`

> *"Based on the facts provided, the described act falls under BNS Section 281 (rash driving). This is a cognizable and non-bailable offence. If proven, the court may award up to 6 months imprisonment and/or a fine up to ₹1,000. All facts are bundled for your Honourable Judge's review. No AI verdict is being issued."*

We know our constitutional place. AI should not decide criminal guilt. That is for human judges.

---

## 6. Training Results: The Model Learns

We trained an 8B parameter model for 250 steps using `GRPOTrainer` with Unsloth.

**Before training (Epoch 0):**
- Format Reward: 0.0 (free-form text, no XML structure)
- Logic Reward: 0.0 (no BNS citations, no legal keywords)
- Accuracy Reward: 0.1 (mostly wrong verdicts)

**After training (Epoch 250):**
- Format Reward: 0.5 (strict XML with all required tags)
- Logic Reward: 1.0 (BNS, Constitution, court hierarchy in every reasoning chain)
- Accuracy Reward: 0.9 (correct verdict matches expert gold label)

The model stopped hallucinating precedents and started correctly defaulting criminal cases to `forward_to_judge` with properly cited BNS sections.

![Training Curves](training_curve.png)

---

## 7. What's Next

- **10,000 case dataset:** Scale from 15 cases to 10,000 real Indian legal cases from IndianKanoon and NJDG district court data
- **Accuracy target:** ≥75% match with human judge verdicts on held-out test set
- **Court hierarchy training:** SC verdict always overrides — train the model to emulate the highest court's reasoning
- **NJDG integration:** Direct API connection to live court data for real-time triage
- **Multilingual expansion:** Full support for all 22 scheduled languages of India

---

**Try it yourself:**
- **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/RishitaRamola42/judicial-reasoning-env)
- **Train it:** [Colab Notebook](https://colab.research.google.com/github/Sarthaksingh2005/Judicial-Reasoning-RL-Environment/blob/master/training_notebook.ipynb)
- **Code:** [GitHub Repository](https://github.com/Sarthaksingh2005/Judicial-Reasoning-RL-Environment)

*Team ALACRITY — Rishita Ramola & Sarthak Singh | OpenEnv Hackathon | Scaler × Meta | April 2026*
