import pytest

from pi_wbc.stats import wilson_interval


def test_wilson_known_values() -> None:
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))
    lo, hi = wilson_interval(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-4)
    assert hi == 1.0


def test_wilson_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
