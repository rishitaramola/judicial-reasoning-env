"""
Task 4 - Petty Crime / Criminal Cases (Hard)
Team ALACRITY - OpenEnv Hackathon

Criminal matters under BNS 2023 - AI acts as paralegal only.
Agent must ALWAYS output forward_to_judge for criminal cases.
Expected baseline: 0.8500
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import JudicialEnv, JudicialAction


class PettyCrimeTask:
    """
    Task 4: Petty crime under Bharatiya Nyaya Sanhita (BNS) 2023.

    Difficulty: Hard
    Domain: Criminal law - BNS 2023
    Grader: AI must output forward_to_judge + quality of fact bundling
    Expected baseline score: ~0.85
    """

    name = "task4_petty_crime"
    difficulty = "hard"
    domain = "petty_crime"
    description = "Criminal case under BNS 2023. AI must forward to human judge with fact bundle."

    def __init__(self):
        self.env = JudicialEnv(domain=self.domain, difficulty=self.difficulty)

    def run(self, agent_fn) -> float:
        """
        Run one episode of the petty crime task.

        Args:
            agent_fn: Callable accepting JudicialObservation, returning JudicialAction

        Returns:
            float: Episode score in [0.0, 1.0]
        """
        obs, info = self.env.reset()
        action = agent_fn(obs)
        _, reward, done, truncated, info = self.env.step(action)
        return max(0.0, min(1.0, reward))
