"""Shared utilities for notebook experiments."""

from .constants import MODEL_NAME, SYSTEM, MODES
from .utils import (
    load_model_and_tokenizer,
    load_gsm8k_dataset, extract_gsm8k_answer,
    build_chat_input, build_user_msg,
    parse_answer,
    get_hint_value,
    generate_batch, generate_one,
    load_jsonl_results, save_results_jsonl,
    compute_answer_logprobs_from_tokens,
)
from .hooks import clear_all_hooks, register_steering, register_ablation

__all__ = [
    'MODEL_NAME', 'SYSTEM', 'MODES',
    'load_model_and_tokenizer',
    'load_gsm8k_dataset', 'extract_gsm8k_answer',
    'build_chat_input', 'build_user_msg',
    'parse_answer',
    'get_hint_value',
    'generate_batch', 'generate_one',
    'load_jsonl_results', 'save_results_jsonl',
    'compute_answer_logprobs_from_tokens',
    'clear_all_hooks', 'register_steering', 'register_ablation',
]
