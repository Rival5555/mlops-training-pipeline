"""Unit tests for Stage 4: Promotion Gate logic and validation criteria.

This is the highest-risk piece in an automated pipeline. Tests cover all boundary
conditions, improvements, ties, regressions, and first-model cold-start behaviors.
"""

import pytest

from src.pipeline.config.schema import EvaluationGateConfig, GateStrategy
from src.pipeline.stages.s04_eval_gate import check_promotion_gate


def test_gate_passes_when_beating_baseline_by_threshold() -> None:
    """Candidate beats baseline by at least min_improvement and exceeds absolute floor."""
    passed, reason, required = check_promotion_gate(
        candidate_metric=0.88,
        production_metric=0.85,
        min_absolute_threshold=0.70,
        min_improvement=0.01,
        allow_first_model=True,
    )
    assert passed is True
    assert required == pytest.approx(0.86)
    assert "beat production (0.8500) by +0.0300" in reason


def test_gate_fails_when_improvement_is_insufficient() -> None:
    """Candidate is better than baseline, but below the required delta threshold."""
    passed, reason, required = check_promotion_gate(
        candidate_metric=0.855,
        production_metric=0.850,
        min_absolute_threshold=0.70,
        min_improvement=0.01,  # requires 0.860
        allow_first_model=True,
    )
    assert passed is False
    assert required == pytest.approx(0.860)
    assert "failed to beat production" in reason


def test_gate_fails_when_candidate_ties_baseline() -> None:
    """Candidate exactly matches baseline without positive delta -> FAIL."""
    passed, reason, required = check_promotion_gate(
        candidate_metric=0.850,
        production_metric=0.850,
        min_absolute_threshold=0.70,
        min_improvement=0.01,
        allow_first_model=True,
    )
    assert passed is False
    assert "failed to beat production" in reason


def test_gate_fails_when_candidate_is_worse_than_baseline() -> None:
    """Candidate metric regressed below baseline -> FAIL."""
    passed, reason, required = check_promotion_gate(
        candidate_metric=0.81,
        production_metric=0.85,
        min_absolute_threshold=0.70,
        min_improvement=0.01,
        allow_first_model=True,
    )
    assert passed is False
    assert "failed to beat production" in reason


def test_gate_fails_when_below_absolute_minimum_floor() -> None:
    """Candidate is lower than absolute acceptable quality floor, even if beating baseline."""
    passed, reason, required = check_promotion_gate(
        candidate_metric=0.65,
        production_metric=0.60,
        min_absolute_threshold=0.70,  # Floor is 0.70
        min_improvement=0.01,
        allow_first_model=True,
    )
    assert passed is False
    assert "failed to reach absolute minimum floor" in reason


def test_gate_cold_start_first_model_success() -> None:
    """When no production model exists, candidate above absolute floor passes."""
    passed, reason, required = check_promotion_gate(
        candidate_metric=0.82,
        production_metric=None,
        min_absolute_threshold=0.70,
        min_improvement=0.01,
        allow_first_model=True,
    )
    assert passed is True
    assert "No existing production model baseline found" in reason


def test_gate_cold_start_first_model_rejected_if_disallowed() -> None:
    """When allow_first_model is False and no production model exists -> FAIL."""
    passed, reason, required = check_promotion_gate(
        candidate_metric=0.85,
        production_metric=None,
        min_absolute_threshold=0.70,
        min_improvement=0.01,
        allow_first_model=False,
    )
    assert passed is False
    assert "allow_first_model is False" in reason


def test_gate_strategy_configuration() -> None:
    """Verify GateStrategy enum configurations."""
    cfg_fail = EvaluationGateConfig(gate_strategy=GateStrategy.FAIL)
    assert cfg_fail.gate_strategy == "fail"

    cfg_warn = EvaluationGateConfig(gate_strategy=GateStrategy.WARN)
    assert cfg_warn.gate_strategy == "warn"


def test_gate_passes_at_metric_ceiling_saturation() -> None:
    """When production model baseline is at maximum ceiling (1.0), candidate at 1.0 passes."""
    passed, reason, required = check_promotion_gate(
        candidate_metric=1.0,
        production_metric=1.0,
        min_absolute_threshold=0.70,
        min_improvement=0.01,
        allow_first_model=True,
    )
    assert passed is True
    assert required == 1.0
    assert "reached maximum quality ceiling" in reason


def test_gate_clamps_required_threshold_to_one() -> None:
    """Ensure required threshold does not exceed 1.0."""
    passed, reason, required = check_promotion_gate(
        candidate_metric=0.98,
        production_metric=0.995,
        min_absolute_threshold=0.70,
        min_improvement=0.01,
        allow_first_model=True,
    )
    assert passed is False
    assert required == 1.0
