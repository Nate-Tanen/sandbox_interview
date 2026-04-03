# Injection guardrail helpers for MAIN_KERNEL_AGENT and optional SDK output_guardrails.
from .GUARDRAIL_AGENT import check_reward_hacking_cpp, reward_hacking_output_guardrail

__all__ = ["check_reward_hacking_cpp", "reward_hacking_output_guardrail"]
