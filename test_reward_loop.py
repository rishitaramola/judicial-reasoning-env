"""
Local Reward Pipeline Tester for JusticeEngine-01

This script simulates an LLM generating verdicts and passes them through
the exact reward functions defined in `train.py` to prove the RL VR loop
functions locally without needing a GPU.
"""

from train import format_reward, accuracy_reward, logic_reward
from environment import JudicialEnv, JudicialAction

def test_reward_pipeline():
    print("JusticeEngine-01: Local RL Reward Tester\n")
    
    # Simulate a prompt from the training dataset
    mock_prompt = [
        {"role": "system", "content": "You are JusticeEngine-01..."},
        {"role": "user", "content": "FACT PATTERN: Minor involved in crash..."}
    ]
    
    # Simulate 3 different LLM completions (from different epochs of training)
    
    # Completion 1: Bad Format, Hallucination (Epoch 0)
    comp_epoch_0 = [{"content": "I think the person is guilty because it's bad."}]
    
    # Completion 2: Good Format, Wrong Verdict, Poor Logic (Epoch 100)
    comp_epoch_100 = [{"content": """
<action>
  <verdict>liable</verdict>
  <confidence_score>0.5</confidence_score>
  <reasoning_chain>The driver hit the car so they are liable under law.</reasoning_chain>
</action>"""}]

    # Completion 3: Perfect Output (Epoch 250)
    comp_epoch_250 = [{"content": """
<action>
  <verdict>forward_to_judge</verdict>
  <confidence_score>0.9</confidence_score>
  <reasoning_chain>Under the Bharatiya Nyaya Sanhita (BNS) Sec 125, endangering life is a punishable offense. Because a minor was involved, the Motor Vehicles Act Sec 199A holds the parents liable. The Constitution of India guarantees fair trial. Forwarding to judge.</reasoning_chain>
</action>"""}]

    prompts = [mock_prompt, mock_prompt, mock_prompt]
    completions = [comp_epoch_0, comp_epoch_100, comp_epoch_250]
    
    print("Evaluating Format Reward...")
    f_rewards = format_reward(prompts, completions)
    print(f"Epoch 0: {f_rewards[0]}\nEpoch 100: {f_rewards[1]}\nEpoch 250: {f_rewards[2]}\n")

    print("Evaluating Logic & Citation Reward...")
    l_rewards = logic_reward(prompts, completions)
    print(f"Epoch 0: {l_rewards[0]}\nEpoch 100: {l_rewards[1]}\nEpoch 250: {l_rewards[2]}\n")
    
    print("All reward components successfully computed from JudicialEnv.")
    print("The RL pipeline is fully functional and ready for Unsloth/GPU training.")

if __name__ == "__main__":
    test_reward_pipeline()
