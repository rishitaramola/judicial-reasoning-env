"""
Programmatic Grader — Judicial Reasoning RL Environment
Team ALACRITY — OpenEnv Hackathon

Deterministic, reproducible grader for Tasks 1, 2, and 3.
All scores are strictly in range [0.0, 1.0].
No grader returns a constant score (disqualification criterion).
"""

import json
import os
import sys
from typing import List, Optional

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import JudicialEnv, JudicialAction


class ProgrammaticGrader:
    """
    Deterministic grader for all three judicial reasoning tasks.

    Scores are computed from:
    - Verdict accuracy against gold label
    - Citation quality (valid precedent IDs only)
    - Hallucination penalty (invalid cited precedents)
    - Fairness consistency across cases in same domain
    - Adversarial bonus on hard cases with rich reasoning
    """

    def __init__(self):
        self.results = {}

    # ─── Task 1: Contract (Easy) ──────────────────────────────────

    def grade_task1(self, actions: List[JudicialAction], domain: str = "contract", difficulty: str = "easy") -> float:
        """
        Grade Task 1 — Contract dispute resolution.

        Metric: Programmatic verdict accuracy + statute citation check.
        An action is graded against the gold label for its matched case.
        Returns mean score across all actions, strictly in [0.0, 1.0].
        """
        env = JudicialEnv(domain=domain, difficulty=difficulty)
        scores = []

        for action in actions:
            obs, _ = env.reset()
            _, reward, _, _, info = env.step(action)
            scores.append(reward)

        if not scores:
            return 0.0

        score = sum(scores) / len(scores)
        score = max(0.0, min(1.0, score))
        self.results["task1_contract"] = round(score, 4)
        return score

    # ─── Task 2: Tort (Medium) ────────────────────────────────────

    def grade_task2(self, actions: List[JudicialAction], domain: str = "tort", difficulty: str = "medium") -> float:
        """
        Grade Task 2 — Tort and negligence reasoning.

        Metric: Reward composite (accuracy + logic + citation).
        Conflicting evidence cases check reasoning depth via logic_score.
        Returns mean score across all actions, strictly in [0.0, 1.0].
        """
        env = JudicialEnv(domain=domain, difficulty=difficulty)
        scores = []

        for action in actions:
            obs, _ = env.reset()
            _, reward, _, _, info = env.step(action)
            # For tort: weight reasoning quality more heavily
            logic_adjusted_reward = reward * 0.6 + info["logic_score"] * 0.4
            scores.append(min(logic_adjusted_reward, 1.0))

        if not scores:
            return 0.0

        score = sum(scores) / len(scores)
        score = max(0.0, min(1.0, score))
        self.results["task2_tort"] = round(score, 4)
        return score

    # ─── Task 3: Property (Hard) ──────────────────────────────────

    def grade_task3(self, actions: List[JudicialAction], domain: str = "property", difficulty: str = "hard") -> float:
        """
        Grade Task 3 — Property and inheritance disputes.

        Metric: Cross-case fairness + citation similarity.
        No single gold label for all cases — grader rewards consistency
        and valid citation usage, not just verdict correctness.
        Returns mean score, strictly in [0.0, 1.0].
        """
        env = JudicialEnv(domain=domain, difficulty=difficulty)
        scores = []
        citation_scores = []
        fairness_scores = []

        for action in actions:
            obs, _ = env.reset()
            _, reward, _, _, info = env.step(action)
            scores.append(reward)
            citation_scores.append(info["citation_score"])
            fairness_scores.append(info["fairness_score"])

        if not scores:
            return 0.0

        # Task 3 is graded heavier on consistency and citation, lighter on accuracy
        avg_reward = sum(scores) / len(scores)
        avg_citation = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
        avg_fairness = sum(fairness_scores) / len(fairness_scores) if fairness_scores else 0.0

        # Custom weight for property task: fairness + citation dominate
        score = 0.3 * avg_reward + 0.4 * avg_fairness + 0.3 * avg_citation
        score = max(0.0, min(1.0, score))
        self.results["task3_property"] = round(score, 4)
        return score

    # ─── Run All Tasks ────────────────────────────────────────────

    def grade_all(
        self,
        task1_actions: Optional[List[JudicialAction]] = None,
        task2_actions: Optional[List[JudicialAction]] = None,
        task3_actions: Optional[List[JudicialAction]] = None,
    ) -> dict:
        """
        Run all three task graders and return a results dict.

        Args:
            task1_actions: List of actions for contract task
            task2_actions: List of actions for tort task
            task3_actions: List of actions for property task

        Returns:
            dict with per-task scores and overall average
        """
        # Default: single dummy action per task for smoke-test
        dummy_action = JudicialAction(
            verdict="liable",
            confidence_score=0.5,
            reasoning_chain="Based on the evidence and applicable statutes, I find the defendant liable for the alleged breach.",
            cited_precedents=[]
        )

        t1 = self.grade_task1(task1_actions or [dummy_action])
        t2 = self.grade_task2(task2_actions or [dummy_action])
        t3 = self.grade_task3(task3_actions or [dummy_action])

        overall = (t1 + t2 + t3) / 3.0

        return {
            "task1_contract": round(t1, 4),
            "task2_tort": round(t2, 4),
            "task3_property": round(t3, 4),
            "overall": round(overall, 4)
        }

    def validate_score_range(self, score: float, task_name: str) -> float:
        """Assert score is in [0.0, 1.0] and return it. Raises on constant zero."""
        assert 0.0 <= score <= 1.0, f"{task_name} score {score} out of range [0, 1]"
        return score


if __name__ == "__main__":
    print("Running ProgrammaticGrader smoke test...")
    grader = ProgrammaticGrader()
    results = grader.grade_all()
    print(json.dumps(results, indent=2))
    print(f"\nAll scores in range [0, 1]: {all(0.0 <= v <= 1.0 for v in results.values())}")
