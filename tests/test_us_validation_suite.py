from pathlib import Path

from scripts.run_us_strategy_validation_suite import (
    BENCHMARKS,
    STOCK_UNIVERSES,
    _generalization_gate,
)
from src.services.screening.strategy import load_all_strategies


def _baseline_cell() -> dict[str, object]:
    return {
        "metrics": {"total_return_pct": 20.0, "excess_return_pct": 5.0, "sharpe": 0.8},
        "validation_gate": {"checks": {"oos_excess_positive_in_majority": True}},
        "bootstrap_excess": {"ci95_low_pct": 0.1},
        "regime_metrics": [
            {"metrics": {"total_return_pct": 5.0, "excess_return_pct": 1.0}}
            for _ in range(5)
        ],
    }


def _stress_cell() -> dict[str, object]:
    return {"metrics": {"total_return_pct": 10.0, "excess_return_pct": 2.0}}


def _passing_inputs():
    baseline = {
        universe: {
            "strategy": {benchmark: _baseline_cell() for benchmark in BENCHMARKS}
        }
        for universe in STOCK_UNIVERSES
    }
    stress = {
        universe: {
            "strategy": {
                f"holding_{holding}": {
                    "round_trip_20bps": _stress_cell(),
                    "round_trip_50bps": _stress_cell(),
                }
                for holding in (5, 10, 20)
            }
        }
        for universe in STOCK_UNIVERSES
    }
    return baseline, stress


def test_generalization_gate_requires_absolute_and_relative_profitability():
    baseline, stress = _passing_inputs()
    assert _generalization_gate(baseline, stress, "strategy")["effective"] is True

    for benchmark in BENCHMARKS[:3]:
        baseline["diversified_60"]["strategy"][benchmark]["metrics"]["total_return_pct"] = -1.0
    stress["large_cap_22"]["strategy"]["holding_10"]["round_trip_50bps"]["metrics"]["total_return_pct"] = -1.0

    gate = _generalization_gate(baseline, stress, "strategy")
    assert gate["effective"] is False
    assert gate["checks"]["beats_majority_of_benchmarks_in_each_stock_universe"] is False
    assert gate["checks"]["survives_50bps_round_trip_cost_in_each_universe"] is False


def test_generalization_gate_requires_regime_robustness():
    baseline, stress = _passing_inputs()
    baseline["diversified_60"]["strategy"]["SPY"]["regime_metrics"] = [
        {"metrics": {"total_return_pct": 1.0, "excess_return_pct": -1.0}}
        for _ in range(5)
    ]

    gate = _generalization_gate(baseline, stress, "strategy")

    assert gate["effective"] is False
    assert gate["checks"]["positive_in_at_least_3_of_5_fixed_regime_epochs_in_each_universe"] is False


def test_rejected_v2_is_reproducible_but_not_production_enabled():
    production = load_all_strategies(Path("src/services/screening/strategies"))
    research = load_all_strategies(Path("research/us_strategy_candidates"))

    assert "us_quality_momentum_v2" not in production
    assert "us_quality_momentum_v2" in research
