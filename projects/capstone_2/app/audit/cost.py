"""Token-to-cost arithmetic. Deterministic, and deliberately not an agent's job."""

from __future__ import annotations

from app.config import get_settings

# Tokens are priced per 1,000, so divide the raw count by this before multiplying.
_TOKENS_PER_PRICE_UNIT: int = 1000
_COST_DECIMALS: int = 6


def usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Return the USD cost of one model call, rounded to six decimal places."""
    settings = get_settings()
    cost = (
        prompt_tokens / _TOKENS_PER_PRICE_UNIT * settings.price_input_per_1k
        + completion_tokens / _TOKENS_PER_PRICE_UNIT * settings.price_output_per_1k
    )
    return round(cost, _COST_DECIMALS)
