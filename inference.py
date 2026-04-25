import asyncio
import sys
import os
import time
from environment import JudicialEnv
from dotenv import load_dotenv

# Ensure we can import from server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.app import get_agent_action
MAX_TOTAL_REWARD       = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    {"name": "task1_contract", "domain": "contract", "difficulty": "easy"},
    {"name": "task2_tort",     "domain": "tort",     "difficulty": "medium"},
    {"name": "task3_property", "domain": "property", "difficulty": "hard"},
]


def log_start(task_name: str):
    print(f"[START] task={task_name}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)



async def run_task(task_config: dict) -> float:
    task_name = task_config["name"]
    log_start(task_name)

    env = JudicialEnv(domain=task_config["domain"], difficulty=task_config["difficulty"])
    obs, _ = env.reset()

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        for step in range(1, 4):
            try:
                action, council_votes = get_agent_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action.verdict, reward=reward, done=done, error=None)
                if done:
                    obs, _ = env.reset()
            except Exception as e:
                error = str(e)
                log_step(step=step, action="ERROR", reward=0.0, done=False, error=error)
                break

        score = sum(rewards) / (len(rewards) * MAX_TOTAL_REWARD) if rewards else 0.001
        score = min(max(score, 0.001), 0.999)  # strictly between 0 and 1
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    all_scores = []
    for task in TASKS:
        score = await run_task(task)
        all_scores.append(score)

    print(f"\n=== BASELINE RESULTS ===", flush=True)
    for task, score in zip(TASKS, all_scores):
        print(f"{task['name']}: {score:.4f}", flush=True)
    print(f"OVERALL AVERAGE: {sum(all_scores)/len(all_scores):.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())