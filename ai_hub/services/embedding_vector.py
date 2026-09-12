"""Pure vector validation and normalization, shared by corpus and query paths.

Extracted from S-20 so that a corpus vector and a query vector are put through
**the same code**, not through two implementations that agree today. If they
diverged even slightly - one applying L2 where the other did not, one tolerating
a value the other refused - every similarity score computed between them would be
meaningless while looking perfectly reasonable.

No I/O, no model imports beyond the normalization enum, no persistence, no
metric. Callers translate `EmbeddingVectorError` into their own bounded failure
category; the `category` values here are deliberately the same strings S-20
already used, so that translation is an identity mapping rather than a
re-interpretation.
"""

import math

from ai_hub.models import EmbeddingModelConfig


class VectorErrorCategory:
    VECTOR_DIMENSION_MISMATCH = "vector_dimension_mismatch"
    VECTOR_NON_FINITE = "vector_non_finite"
    ZERO_VECTOR_CANNOT_L2_NORMALIZE = "zero_vector_cannot_l2_normalize"


class EmbeddingVectorError(ValueError):
    """A vector that cannot be accepted, with a bounded category."""

    def __init__(self, category: str, message: str = ""):
        self.category = category
        super().__init__(message or category)


def validate_embedding_vector(values, *, expected_dimension: int) -> tuple:
    """Exact dimension, numeric non-bool components, all finite. Never repairs.

    Runs BEFORE normalization on purpose: a non-finite component would otherwise
    become a non-finite magnitude, and the failure would surface as something
    vaguer and further from its cause.
    """
    if len(values) != expected_dimension:
        raise EmbeddingVectorError(
            VectorErrorCategory.VECTOR_DIMENSION_MISMATCH,
            f"Provider returned {len(values)} components; "
            f"the contract requires {expected_dimension}.",
        )
    for index, raw in enumerate(values):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EmbeddingVectorError(
                VectorErrorCategory.VECTOR_NON_FINITE,
                f"Vector component {index} is not a number.",
            )
        if not math.isfinite(float(raw)):
            raise EmbeddingVectorError(
                VectorErrorCategory.VECTOR_NON_FINITE,
                f"Vector component {index} is not finite.",
            )
    return tuple(float(value) for value in values)


def normalize_embedding_vector(values, *, normalization) -> tuple:
    """Apply the S-18 normalization contract. Never infer it.

    Deliberately does not look at the distance metric, the provider or the model.
    Metric and normalization are two independent contract facts, and normalizing
    "because the metric is cosine" would make a stored vector disagree with the
    contract that describes it - which is exactly why the cosine scorer computes
    norms itself rather than assuming unit vectors.
    """
    if normalization == EmbeddingModelConfig.Normalization.NONE:
        return tuple(values)

    magnitude = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        # A zero vector has no direction, so it cannot be L2-normalized.
        # Refusing beats leaving it unchanged, substituting an epsilon or
        # inventing a unit vector - each of those yields a vector that lies
        # about what its contract says it is.
        raise EmbeddingVectorError(
            VectorErrorCategory.ZERO_VECTOR_CANNOT_L2_NORMALIZE,
            "A zero-magnitude vector cannot be L2-normalized.",
        )
    normalized = tuple(value / magnitude for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise EmbeddingVectorError(
            VectorErrorCategory.VECTOR_NON_FINITE,
            "Normalization produced a non-finite component.",
        )
    return normalized
