"""Synthetic world generation with recorded ground truth."""

from verity.datagen.documents import DOCUMENT_BY_ID, DOCUMENTS, Document, documents_for_role
from verity.datagen.entities import (
    CHANNELS,
    END_DATE,
    PRODUCTS,
    REGIONS,
    SCENARIO_BY_ID,
    SCENARIOS,
    SEGMENTS,
    START_DATE,
    Scenario,
)
from verity.datagen.generator import GeneratedData, generate

__all__ = [
    "DOCUMENTS",
    "DOCUMENT_BY_ID",
    "Document",
    "documents_for_role",
    "CHANNELS",
    "END_DATE",
    "PRODUCTS",
    "REGIONS",
    "SCENARIOS",
    "SCENARIO_BY_ID",
    "SEGMENTS",
    "START_DATE",
    "Scenario",
    "GeneratedData",
    "generate",
]
