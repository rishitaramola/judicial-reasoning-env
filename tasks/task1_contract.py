"""
Task 1 — Contract Dispute Resolution (Easy)
Team ALACRITY — OpenEnv Hackathon

Resolve a contract breach dispute under Indian Contract Act.
Agent must identify breach and assign liability.
Expected baseline: 0.9760
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import JudicialEnv, JudicialAction


class ContractTask:
    """
    Task 1: Contract breach and dispute resolution.

    Difficulty: Easy
    Domain: Contract law — Indian Contract Act 1872
    Grader: Programmatic verdict check + citation quality
    Expected baseline score: ~0.97
    """

    name = "task1_contract"
    difficulty = "easy"
    domain = "contract"
    description = "Resolve a contract dispute. Agent must identify breach and assign liability under Indian Contract Act."

    def __init__(self):
        self.env = JudicialEnv(domain=self.domain, difficulty=self.difficulty)

    def run(self, agent_fn) -> float:
        """
        Run one episode of the contract task.

        Args:
            agent_fn: Callable accepting JudicialObservation, returning JudicialAction

        Returns:
            float: Episode score in [0.0, 1.0]
        """
        obs, info = self.env.reset()
        action = agent_fn(obs)
        _, reward, done, truncated, info = self.env.step(action)
        return max(0.0, min(1.0, reward))