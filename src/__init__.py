"""ZAC 2026 LifeMate Agent Package.

Provides core components for deterministic ReAct reasoning, schema-driven guardrails,
fast semantic tool routing, and context-grounded ASR language preprocessing.
"""
from src.agent import LifeMateAgent, LLMBackend, fix_all_seeds, render_chatml_preserving_think
from src.asr import Transcriber
from src.guardrail import PureSchemaStateTracker, normalize_arguments, parse_tool_calls
from src.preprocessor_a import ContextLanguagePreprocessorA
from src.preprocessor_b import ContextLanguagePreprocessorB
from src.prompt_builder import build_initial_messages, get_two_tier_allowed_tools
from src.router import SemanticToolRouter

__all__ = [
    "LifeMateAgent",
    "LLMBackend",
    "fix_all_seeds",
    "render_chatml_preserving_think",
    "Transcriber",
    "PureSchemaStateTracker",
    "normalize_arguments",
    "parse_tool_calls",
    "ContextLanguagePreprocessorA",
    "ContextLanguagePreprocessorB",
    "build_initial_messages",
    "get_two_tier_allowed_tools",
    "SemanticToolRouter",
]
