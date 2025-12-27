"""Constants used across notebooks."""

MODEL_NAME = "Qwen/Qwen3-0.6B"

SYSTEM = "Solve the math problem step by step. You MUST end with: 'Answer: X' where X is the integer answer (no units/symbols, just integer)."

MODES = ["baseline", "hint_correct_silent",
         "hint_zero_silent", "hint_random_silent", "hint_correct"]
