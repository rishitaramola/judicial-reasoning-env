"""
Task 2 — Tort and Negligence Reasoning (Medium)
Team ALACRITY — OpenEnv Hackathon

Resolve a tort/negligence case with conflicting witness accounts and evidence.
Agent must reason through ambiguity and apply duty-of-care principles.
Expected baseline: 0.6853
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import JudicialEnv, JudicialAction


class TortTask:
    """
    Task 2: Tort and negligence with conflicting evidence.

    Difficulty: Medium
    Domain: Tort law — Negligence, duty of care, consumer protection
    Grader: LLM rubric scoring on reasoning chain logical steps + accuracy
    Expected baseline score: ~0.68
    """

    name = "task2_tort"
    difficulty = "medium"
    domain = "tort"
    description = "Resolve a tort/negligence case with conflicting evidence. Agent must reason through ambiguity."

    def __init__(self):
        self.env = JudicialEnv(domain=self.domain, difficulty=self.difficulty)

    def run(self, agent_fn) -> float:
        """
        Run one episode of the tort task.

        Args:
            agent_fn: Callable accepting JudicialObservation, returning JudicialAction

        Returns:
            float: Episode score in [0.0, 1.0]
        """
        obs, info = self.env.reset()
        action = agent_fn(obs)
        _, reward, done, truncated, info = self.env.step(action)
        return max(0.0, min(1.0, reward))