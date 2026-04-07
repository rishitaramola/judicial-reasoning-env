"""
Task 3 — Property and Inheritance Dispute (Hard)
Team ALACRITY — OpenEnv Hackathon

Resolve a property/inheritance dispute with adversarial, ambiguous facts.
No single clear gold label — grader rewards consistency, citation, and reasoning depth.
Expected baseline: 0.5520
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import JudicialEnv, JudicialAction


class PropertyTask:
    """
    Task 3: Property and inheritance dispute with adversarial facts.

    Difficulty: Hard
    Domain: Property law — adverse possession, succession, tenancy, wills
    Grader: Cross-case fairness + citation similarity score
    Expected baseline score: ~0.55
    """

    name = "task3_property"
    difficulty = "hard"
    domain = "property"
    description = "Resolve a property/inheritance dispute with adversarial ambiguous facts. No single clear gold label."

    def __init__(self):
        self.env = JudicialEnv(domain=self.domain, difficulty=self.difficulty)

    def run(self, agent_fn) -> float:
        """
        Run one episode of the property task.

        Args:
            agent_fn: Callable accepting JudicialObservation, returning JudicialAction

        Returns:
            float: Episode score in [0.0, 1.0]
        """
        obs, info = self.env.reset()
        action = agent_fn(obs)
        _, reward, done, truncated, info = self.env.step(action)
        return max(0.0, min(1.0, reward))