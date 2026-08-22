"""Verity — KPI Intelligence-to-Action Engine.

Accenture Innovation Challenge 2026, Problem Track 3 (BusinessIntelligence.ai).
Team Verity Exchange.

Architectural rule enforced throughout this package: the LLM is never the
source of quantitative truth. Detection, attribution, forecasting, retrieval,
ranking, scoring, and permissions are all deterministic. The LLM receives an
Evidence Pack and nothing else.
"""

__version__ = "0.1.0"
